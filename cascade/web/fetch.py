"""Local, provider-agnostic web fetch: SSRF-hardened, bounded, paged to disk.

Pipeline: validate + normalize URL -> SSRF pre-check -> secret scan ->
stream with a hard byte cap, hand-walking redirects (same-origin only,
each hop re-validated) -> content-type gate -> HTML to markdown ->
token-bounded head+tail with the full text spilled to disk and a paging
pointer. Fetched text is untrusted content and is delimited as such by
the caller.

httpx is required (it already backs every provider). trafilatura is used
for extraction when installed; otherwise a stdlib HTML-to-text fallback.
"""

from __future__ import annotations

import hashlib
import re
import time
from html.parser import HTMLParser
from typing import Optional

import httpx

from .url_safety import (
    contains_secret,
    normalize_url,
    resolve_redirect,
    same_origin,
    url_safety_error,
)
from ..runtime import user_cache_path

_MAX_BYTES = 10 * 1024 * 1024          # hard stream cap
_MAX_REDIRECTS = 5
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 60.0
_USER_AGENT = "cascade-web-fetch/1.0"
# Head+tail budget when returning to the model; the full text is on disk.
_HEAD_CHARS = 6_000
_TAIL_CHARS = 2_000

class _TextExtractor(HTMLParser):
    """Stdlib fallback: strip scripts/styles, keep visible text + link text."""

    _SKIP = {"script", "style", "noscript", "svg", "head"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        joined = "".join(self._parts)
        joined = re.sub(r"[ \t]+", " ", joined)
        return re.sub(r"\n\s*\n+", "\n\n", joined).strip()


def _html_to_markdown(html: str, url: str) -> str:
    try:
        import trafilatura  # noqa: PLC0415

        extracted = trafilatura.extract(
            html, url=url, include_links=True, output_format="markdown",
        )
        if extracted and extracted.strip():
            return extracted.strip()
    except Exception:
        pass
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        return html
    return parser.text()


def _spill(url: str, text: str) -> Optional[str]:
    try:
        fetch_dir = user_cache_path("web-fetch")
        fetch_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(f"{time.time_ns()}:{url}".encode()).hexdigest()[:10]
        path = fetch_dir / f"{digest}.md"
        path.write_text(f"<!-- {url} -->\n\n{text}", errors="replace")
        return str(path)
    except Exception:
        return None


def _bound(text: str, url: str) -> str:
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text
    spilled = _spill(url, text)
    head = text[:_HEAD_CHARS]
    tail = text[-_TAIL_CHARS:]
    elided = len(text) - _HEAD_CHARS - _TAIL_CHARS
    pointer = f"read {spilled}" if spilled else "re-fetch for full text"
    return (
        f"{head}\n\n[… {elided} chars elided — full text: {pointer} …]\n\n{tail}"
    )


class FetchResult:
    def __init__(self, ok: bool, content: str, url: str, error: str = "") -> None:
        self.ok = ok
        self.content = content
        self.url = url
        self.error = error


def fetch_url(url: str, client: Optional[httpx.Client] = None) -> FetchResult:
    """Fetch and extract a URL to bounded markdown, or return an error result."""
    normalized = normalize_url(url)
    if not normalized:
        return FetchResult(False, "", url, "empty or invalid URL")
    if secret := contains_secret(normalized):
        return FetchResult(False, "", url, f"URL appears to contain a secret ({secret})")
    if reason := url_safety_error(normalized):
        return FetchResult(False, "", url, f"blocked: {reason}")

    owns_client = client is None
    client = client or httpx.Client(
        timeout=httpx.Timeout(_READ_TIMEOUT, connect=_CONNECT_TIMEOUT),
        follow_redirects=False,
    )
    try:
        current = normalized
        for _hop in range(_MAX_REDIRECTS + 1):
            try:
                with client.stream(
                    "GET", current, headers={"User-Agent": _USER_AGENT},
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location", "")
                        target = normalize_url(resolve_redirect(current, location))
                        if not target or not same_origin(current, target):
                            return FetchResult(
                                False, "", url,
                                "cross-origin redirect blocked",
                            )
                        if reason := url_safety_error(target):
                            return FetchResult(False, "", url, f"redirect blocked: {reason}")
                        current = target
                        continue

                    response.raise_for_status()
                    ctype = response.headers.get("content-type", "").lower()
                    if "text/html" not in ctype and "text/plain" not in ctype \
                            and "application/xhtml" not in ctype:
                        return FetchResult(
                            False, "", url,
                            f"unsupported content-type: {ctype or 'unknown'}",
                        )

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > _MAX_BYTES:
                            return FetchResult(
                                False, "", url, "response exceeds 10MB cap",
                            )
                        chunks.append(chunk)
                    body = b"".join(chunks).decode("utf-8", errors="replace")

                    if "text/plain" in ctype:
                        text = body.strip()
                    else:
                        text = _html_to_markdown(body, current)
                    return FetchResult(True, _bound(text, current), current)
            except httpx.HTTPStatusError as exc:
                return FetchResult(False, "", url, f"HTTP {exc.response.status_code}")
            except httpx.RequestError as exc:
                return FetchResult(False, "", url, f"request failed: {exc}")
        return FetchResult(False, "", url, "too many redirects")
    finally:
        if owns_client:
            client.close()
