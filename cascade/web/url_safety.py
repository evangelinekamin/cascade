"""SSRF protection for the web-fetch tool.

Ported from hermes-agent's url_safety.py (Python + stdlib), trimmed to
what cascade needs: resolve the hostname and block private, loopback,
link-local, reserved, multicast, and CGNAT addresses, plus an
always-blocked floor of cloud-metadata endpoints that no toggle can
open. Fails closed on DNS failure or any parse error.

Cascade carries this itself because, unlike Claude Code, it has no
blocklist-service backstop — the pre-flight and per-redirect checks are
the whole defense.

Known limitation: DNS rebinding (TOCTOU) is not defeated at pre-flight
level; the fetch tool re-validates every redirect hop and pins
same-origin, which covers the redirect vector.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from typing import Optional
from urllib.parse import parse_qsl, quote, unquote, urljoin, urlsplit, urlunsplit

logger = logging.getLogger("cascade.web")

MAX_URL_LENGTH = 2000

# Credential-named query params: block before fetching so signed URLs and
# magic links are never sent to a (possibly logging) remote host.
_SENSITIVE_QUERY_PARAM_NAMES = frozenset({
    "access_token", "api_key", "apikey", "auth_token", "authorization",
    "client_secret", "credential", "credentials", "jwt", "password",
    "passwd", "secret", "session_id", "signature", "token",
    "x_amz_security_token", "x_amz_signature",
})

# Vendor key prefixes to catch in the URL text itself.
_SECRET_PREFIXES = ("sk-", "sk_live_", "sk_test_", "ghp_", "gho_", "xoxb-", "xoxp-", "AKIA")

_BLOCKED_HOSTNAMES = frozenset({"metadata.google.internal", "metadata.goog"})

_ALWAYS_BLOCKED_IPS = frozenset({
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("169.254.170.2"),
    ipaddress.ip_address("169.254.169.253"),
    ipaddress.ip_address("fd00:ec2::254"),
    ipaddress.ip_address("100.100.100.200"),
    ipaddress.ip_address("::ffff:169.254.169.254"),
    ipaddress.ip_address("::ffff:169.254.170.2"),
})
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::ffff:169.254.0.0/112"),
)
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")

_ALLOW_PRIVATE_ENV = "CASCADE_ALLOW_PRIVATE_URLS"


def normalize_url(url: str) -> str:
    """ASCII-safe HTTP(S) URL from a possibly-IRI input; http upgraded to https."""
    if not isinstance(url, str):
        return ""
    raw = url.strip()
    if not raw:
        return ""
    raw = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)\s+", r"\1", raw)
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    scheme = parsed.scheme.lower()
    if scheme == "http":
        scheme = "https"
    if scheme not in {"http", "https"}:
        return raw
    netloc = parsed.netloc
    hostname = parsed.hostname
    if hostname:
        try:
            ascii_host = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_host = hostname
        if ascii_host != hostname:
            netloc = netloc.replace(hostname, ascii_host, 1)
    path = quote(parsed.path, safe="/%:@!$&'()*+,;=")
    query = quote(parsed.query, safe="/%:@!$&'()*+,;=?")
    return urlunsplit((scheme, netloc, path, query, ""))


def sensitive_query_param_name(url: str) -> Optional[str]:
    """First credential-named query parameter, if any."""
    if not isinstance(url, str) or "?" not in url:
        return None
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if not parsed.query:
        return None
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if value and unquote(key).lower() in _SENSITIVE_QUERY_PARAM_NAMES:
            return key
    return None


def contains_secret(url: str) -> Optional[str]:
    """Return a matched vendor key prefix in the URL (raw or decoded), if any."""
    for form in (url, unquote(url)):
        for prefix in _SECRET_PREFIXES:
            if prefix in form:
                return prefix
    if name := sensitive_query_param_name(url):
        return name
    return None


def _allow_private() -> bool:
    return os.getenv(_ALLOW_PRIVATE_ENV, "").strip().lower() in {"true", "1", "yes"}


def _is_blocked_ip(ip) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    if ip.is_multicast or ip.is_unspecified:
        return True
    return ip in _CGNAT_NETWORK


def url_safety_error(url: str) -> Optional[str]:
    """Return a human-readable reason the URL is unsafe to fetch, or None.

    Fails closed: DNS failure and parse errors are unsafe. Cloud-metadata
    endpoints are blocked regardless of the private-URL toggle.
    """
    if not url or len(url) > MAX_URL_LENGTH:
        return "URL missing or too long"
    try:
        parsed = urlsplit(url)
    except ValueError:
        return "URL could not be parsed"
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return f"unsupported scheme: {scheme or '(none)'}"
    if parsed.username or parsed.password:
        return "URL embeds credentials (userinfo)"
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return "URL has no host"
    if hostname in _BLOCKED_HOSTNAMES:
        return f"blocked internal hostname: {hostname}"

    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if literal in _ALWAYS_BLOCKED_IPS or any(
            literal in n for n in _ALWAYS_BLOCKED_NETWORKS
        ):
            return "blocked cloud-metadata address"
        if not _allow_private() and _is_blocked_ip(literal):
            return f"blocked private/internal address: {hostname}"
        return None

    try:
        addr_info = socket.getaddrinfo(
            hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except socket.gaierror:
        return f"DNS resolution failed for {hostname}"

    allow_private = _allow_private()
    for _family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0].split("%")[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return f"unparseable resolved address for {hostname}"
        if ip in _ALWAYS_BLOCKED_IPS or any(ip in n for n in _ALWAYS_BLOCKED_NETWORKS):
            return "blocked cloud-metadata address"
        if not allow_private and _is_blocked_ip(ip):
            return f"blocked private/internal address: {hostname} -> {ip_str}"
    return None


def is_safe_url(url: str) -> bool:
    """True when the URL is safe to fetch (no SSRF risk)."""
    return url_safety_error(url) is None


def same_origin(a: str, b: str) -> bool:
    """True when two URLs share scheme, host, and port (redirect pinning)."""
    try:
        pa, pb = urlsplit(a), urlsplit(b)
    except ValueError:
        return False
    return (
        pa.scheme == pb.scheme
        and (pa.hostname or "").lower() == (pb.hostname or "").lower()
        and pa.port == pb.port
    )


def resolve_redirect(base: str, location: str) -> str:
    """Absolute redirect target from a (possibly relative) Location header."""
    return urljoin(base, location)
