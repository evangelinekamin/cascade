"""Permission verdict engine: the always-present gate on tool execution.

Design (from the Claude Code / Codex / rtk research, tuned for a
default-AUTO posture): safety comes from structure, not from prompting
the user about everything. Evaluation order is the load-bearing
property — deny and sacred checks run BEFORE any auto-allow, so no
posture, rule, or mode can silently touch credentials, cascade's own
config, or run structurally dangerous shell.

Postures:
- "auto"     (default): reads, workspace writes, and safe-shaped shell
  auto-approve; sacred paths and dangerous shell ask; everything is
  recorded in the audit trail so autonomy stays inspectable.
- "safe":    reads auto-approve; every mutation asks.
- "readonly": reads auto-approve; every mutation denies.

Rule grammar (config/permissions.yaml allow/deny/ask lists):
- "tool_name"             — whole tool
- "tool_name(exact arg)"  — exact primary-argument match
- "tool_name(prefix:*)"   — primary-argument prefix match
The primary argument is command for shell, path for file tools, url for
web tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import ToolDef


@dataclass(frozen=True)
class Verdict:
    decision: str  # "allow" | "ask" | "deny"
    reason: str
    rule: str = ""


# Paths no posture may touch without an explicit ask — evaluated before
# any auto-allow, immune to allow rules.
SACRED_PATTERNS = (
    "/.git/",
    "/.cascade/",
    "/.ssh",
    "/.aws",
    "/.gnupg",
    ".bashrc",
    ".zshrc",
    ".profile",
    ".bash_profile",
    "id_rsa",
    "id_ed25519",
    ".pem",
    ".env",
    "credentials",
    "authorized_keys",
)

# Shell shapes that never auto-approve regardless of posture or allow
# rules: command substitution defeats prefix matching entirely, and the
# rest are the classic irreversible/exfil moves.
_NEVER_AUTO_SHELL = (
    re.compile(r"\$\("),
    re.compile(r"`"),
    re.compile(r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+(/|~|\$HOME)(\s|$)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"curl[^|;&]*\|\s*(ba|z|da|fi)?sh\b"),
    re.compile(r"wget[^|;&]*\|\s*(ba|z|da|fi)?sh\b"),
    re.compile(r"\bgit\s+push\s[^;|&]*(--force\b|-f\b)"),
    re.compile(r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/"),
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+[^;|&]*of=/dev/"),
    re.compile(r":\(\)\s*\{"),
)

# Read-only documentation hosts auto-approved for web_fetch under the auto
# posture -- GET-only reference material, the low-risk egress case.
PREAPPROVED_FETCH_DOMAINS = (
    "docs.python.org",
    "developer.mozilla.org",
    "docs.rs",
    "pkg.go.dev",
    "readthedocs.io",
    "docs.github.com",
    "peps.python.org",
    "man7.org",
)

_SHELL_SPLIT = re.compile(r"(?<!\|)\|(?!\|)|\|\||&&|;")

_RULE_RE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*(?:\((.*)\))?\s*$")

# Escalation bounds for deny-and-continue in unattended lanes (rtk /
# Claude Code numbers): a compromised or confused agent grinding against
# the gate gets stopped, one honest false positive does not kill a run.
MAX_CONSECUTIVE_DENIALS = 3
MAX_TOTAL_DENIALS = 20


@dataclass(frozen=True)
class _Rule:
    tool: str
    content: str = ""
    prefix: bool = False

    def matches(self, tool_name: str, primary_arg: str) -> bool:
        if self.tool != tool_name:
            return False
        if not self.content:
            return True
        if self.prefix:
            return primary_arg.startswith(self.content)
        return primary_arg == self.content


def parse_rule(text: str) -> Optional[_Rule]:
    """Parse "tool", "tool(arg)", or "tool(prefix:*)" rule syntax."""
    m = _RULE_RE.match(text or "")
    if not m:
        return None
    tool, content = m.group(1), m.group(2) or ""
    content = content.strip()
    if content in ("", "*"):
        return _Rule(tool=tool)
    if content.endswith(":*"):
        return _Rule(tool=tool, content=content[:-2], prefix=True)
    return _Rule(tool=tool, content=content)


def _primary_arg(tool_name: str, arguments: dict) -> str:
    for key in ("command", "path", "file_path", "url", "query"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _url_host(url: str) -> str:
    """Lowercased hostname of a URL, for domain-scoped fetch rules."""
    from urllib.parse import urlsplit

    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _touches_sacred(text: str) -> Optional[str]:
    """Return the sacred pattern hit by *text* (path or shell command)."""
    if not text:
        return None
    expanded = str(Path(text.split()[0]).expanduser()) if " " not in text else text
    haystack = f"/{expanded}" if not expanded.startswith("/") else expanded
    lowered = f"{haystack} {text}".lower().replace("~", str(Path.home()).lower())
    for pattern in SACRED_PATTERNS:
        if pattern in lowered:
            return pattern
    return None


def _dangerous_shell(command: str) -> Optional[str]:
    """Return a description of the never-auto construct found, if any."""
    for pattern in _NEVER_AUTO_SHELL:
        if pattern.search(command):
            return pattern.pattern
    return None


def _shell_segments(command: str) -> list[str]:
    """Split a compound command; every segment must pass independently."""
    return [seg.strip() for seg in _SHELL_SPLIT.split(command) if seg.strip()]


class PermissionEngine:
    """Evaluate tool calls against rules, sacred paths, and posture.

    One instance is shared by every tool loop (wired onto providers by
    CascadeCore, like hook_runner). ``ask_handler`` resolves "ask"
    verdicts interactively; when None (headless lanes), asks become
    structured denials with deny-and-continue escalation bounds.
    """

    def __init__(
        self,
        posture: str = "auto",
        allow: tuple[str, ...] = (),
        deny: tuple[str, ...] = (),
        ask: tuple[str, ...] = (),
        workspace_root: Optional[str] = None,
        audit_limit: int = 200,
    ) -> None:
        self.posture = posture if posture in ("auto", "safe", "readonly") else "auto"
        self._allow = tuple(filter(None, (parse_rule(r) for r in allow)))
        self._deny = tuple(filter(None, (parse_rule(r) for r in deny)))
        self._ask = tuple(filter(None, (parse_rule(r) for r in ask)))
        self._workspace_root = Path(workspace_root or ".").expanduser().resolve()
        self._session_grants: set[str] = set()
        self.ask_handler: Optional[Callable[[str, dict, Verdict], str]] = None
        self.audit: list[tuple[str, str, str, str]] = []
        self._audit_limit = audit_limit
        self.consecutive_denials = 0
        self.total_denials = 0

    # -- rule store -----------------------------------------------------

    def grant_session(self, tool_name: str, primary_arg: str = "") -> None:
        """User chose "always allow (this session)" for this shape."""
        self._session_grants.add(self._grant_key(tool_name, primary_arg))

    @staticmethod
    def _grant_key(tool_name: str, primary_arg: str) -> str:
        head = primary_arg.split()[0] if primary_arg else ""
        return f"{tool_name}:{head}"

    # -- evaluation -----------------------------------------------------

    def evaluate(
        self,
        tool: Optional["ToolDef"],
        tool_name: str,
        arguments: dict,
    ) -> Verdict:
        """Pure verdict: deny > sacred > never-auto shell > ask rules >
        session grants > posture auto-allow > allow rules > posture default."""
        primary = _primary_arg(tool_name, arguments)

        for rule in self._deny:
            if rule.matches(tool_name, primary):
                return Verdict("deny", f"deny rule {rule.tool}", "deny-rule")

        is_fetch = tool_name in ("web_fetch", "web_search") or "url" in arguments
        is_shell = "command" in arguments and isinstance(arguments.get("command"), str)
        is_write = self._is_write_tool(tool, tool_name) and not is_fetch

        if (is_shell or is_write) and (hit := _touches_sacred(primary)):
            return Verdict(
                "ask", f"touches sacred path ({hit})", "sacred",
            )

        # Network egress: gated separately from filesystem writes (it is the
        # primary exfiltration lever). Auto-approve only preapproved
        # read-only docs domains; every other host asks once.
        if is_fetch:
            for rule in self._ask:
                if rule.matches(tool_name, primary):
                    return Verdict("ask", f"ask rule {rule.tool}", "ask-rule")
            host = _url_host(primary)
            if self._grant_key(tool_name, host) in self._session_grants:
                return Verdict("allow", "host granted this session", "session-grant")
            for rule in self._allow:
                if rule.matches(tool_name, primary) or rule.matches(tool_name, host):
                    return Verdict("allow", f"allow rule {rule.tool}", "allow-rule")
            if host and any(host == d or host.endswith("." + d) for d in PREAPPROVED_FETCH_DOMAINS):
                return Verdict("allow", f"preapproved docs domain ({host})", "docs-allowlist")
            if self.posture == "readonly":
                return Verdict("deny", "readonly posture blocks network egress", "posture")
            return Verdict("ask", f"fetch from {host or 'unknown host'}", "network")

        if is_shell:
            # Whole command first (pipe-spanning patterns like curl|sh are
            # invisible after splitting), then every compound segment.
            for candidate in [primary, *_shell_segments(primary)]:
                if danger := _dangerous_shell(candidate):
                    return Verdict(
                        "ask", f"dangerous shell construct ({danger})", "never-auto",
                    )

        for rule in self._ask:
            if rule.matches(tool_name, primary):
                return Verdict("ask", f"ask rule {rule.tool}", "ask-rule")

        if self._grant_key(tool_name, primary) in self._session_grants:
            return Verdict("allow", "granted for this session", "session-grant")

        if tool is not None and tool.is_read_only:
            return Verdict("allow", "read-only tool", "read-only")

        for rule in self._allow:
            if rule.matches(tool_name, primary):
                return Verdict("allow", f"allow rule {rule.tool}", "allow-rule")

        if self.posture == "readonly":
            return Verdict("deny", "readonly posture blocks mutations", "posture")
        if self.posture == "safe":
            return Verdict("ask", "safe posture asks for mutations", "posture")

        # posture == "auto"
        if is_write and not is_shell:
            path = arguments.get("path") or arguments.get("file_path") or ""
            if isinstance(path, str) and path and not self._in_workspace(path):
                return Verdict(
                    "ask", f"write outside workspace ({path})", "workspace",
                )
        return Verdict("allow", "auto posture", "posture")

    def _is_write_tool(self, tool: Optional["ToolDef"], tool_name: str) -> bool:
        if tool is not None and tool.is_read_only:
            return False
        return not tool_name.startswith(("read", "list", "search", "reflect", "get"))

    def _in_workspace(self, path: str) -> bool:
        try:
            resolved = Path(path).expanduser().resolve()
        except (OSError, ValueError):
            return False
        root = self._workspace_root
        return resolved == root or root in resolved.parents

    # -- resolution (ask handling, escalation, audit) -------------------

    def resolve(
        self,
        tool: Optional["ToolDef"],
        tool_name: str,
        arguments: dict,
    ) -> Verdict:
        """Evaluate and fully resolve: asks go to the handler or become
        structured denials with escalation bounds. Records the audit trail."""
        verdict = self.evaluate(tool, tool_name, arguments)
        primary = _primary_arg(tool_name, arguments)

        # Session grants for fetch are keyed by host, not the full URL, so
        # "always allow docs.foo.com" covers every page on that host.
        grant_arg = _url_host(primary) if verdict.rule == "network" else primary

        if verdict.decision == "ask":
            if self.ask_handler is not None:
                answer = "deny"
                try:
                    answer = self.ask_handler(tool_name, arguments, verdict)
                except Exception:
                    answer = "deny"
                if answer == "always":
                    self.grant_session(tool_name, grant_arg)
                    verdict = Verdict("allow", "approved (always this session)", verdict.rule)
                elif answer == "allow":
                    verdict = Verdict("allow", "approved by user", verdict.rule)
                else:
                    verdict = Verdict("deny", f"not approved: {verdict.reason}", verdict.rule)
            else:
                self.consecutive_denials += 1
                self.total_denials += 1
                if (
                    self.consecutive_denials >= MAX_CONSECUTIVE_DENIALS
                    or self.total_denials >= MAX_TOTAL_DENIALS
                ):
                    verdict = Verdict(
                        "deny",
                        f"{verdict.reason} — denial limit reached, stop and "
                        "report instead of retrying",
                        "escalation",
                    )
                else:
                    verdict = Verdict(
                        "deny",
                        f"{verdict.reason} — blocked in unattended mode; "
                        "choose a safer approach",
                        verdict.rule,
                    )

        if verdict.decision == "allow" and verdict.rule != "read-only":
            self.consecutive_denials = 0

        self._record(tool_name, primary, verdict)
        return verdict

    def _record(self, tool_name: str, primary: str, verdict: Verdict) -> None:
        hint = primary[:80]
        self.audit.append((tool_name, hint, verdict.decision, verdict.rule))
        if len(self.audit) > self._audit_limit:
            self.audit = self.audit[-self._audit_limit :]
