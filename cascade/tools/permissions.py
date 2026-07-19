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
import shlex
import threading
from collections import deque
from dataclasses import dataclass
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

# Shell interpreters that execute their stdin/argument as code. Piping
# into these, or invoking them on remote/temp content, is remote code
# execution regardless of how the payload is dressed up.
_INTERPRETERS = r"(?:ba|z|k|da|fi)?sh|python[0-9.]*|perl|ruby|node|php"

# Shell shapes that never auto-approve regardless of posture or allow
# rules. Detection is STRUCTURAL, not shape-specific: substitution defeats
# any downstream matching so it is flagged wherever it appears; RCE is
# "fetch anywhere + interpreter anywhere", not "curl|sh adjacency". Each
# entry is (pattern, human-readable label).
_NEVER_AUTO_SHELL = (
    (re.compile(r"\$\("), "command substitution $()"),
    (re.compile(r"[<>]\("), "process substitution <()/>()"),
    (re.compile(r"`"), "backtick substitution"),
    (re.compile(r"(?<![\w-])sudo(?![\w-])"), "sudo"),
    (re.compile(r"(?<![\w-])doas(?![\w-])"), "doas"),
    (re.compile(rf"\|\s*(?:{_INTERPRETERS})\b"), "pipe into an interpreter"),
    (re.compile(rf"(?<![\w-])(?:curl|wget|fetch)(?![\w-]).*(?:{_INTERPRETERS})\b", re.S),
     "remote fetch feeding an interpreter"),
    (re.compile(rf"(?<![\w-])(?:{_INTERPRETERS}|source|\.)\s+(?:-\S+\s+)*(?:/tmp/|/var/tmp/|/dev/shm/)"),
     "interpreter running a temp-dir script"),
    (re.compile(r"/dev/tcp/"), "reverse shell (/dev/tcp)"),
    (re.compile(r"(?<![\w-])n(?:et)?c(?:at)?(?![\w-])[^;|&\n]*\s-[a-z]*e"), "netcat exec"),
    (re.compile(r"(?<![\w-])(?:ba|z)?sh\s+-[a-z]*i\b[^;|&\n]*(?:/dev/tcp|>&|<&)"),
     "interactive reverse shell"),
    (re.compile(r"(?<![\w-])rm\s+(?:-\S+\s+)*(?:-[a-z]*[rf][a-z]*|--recursive|--force)\b[^;|&\n]*\s(?:/[\w.*/@-]*|~\S*|\$\{?HOME\}?\S*)"),
     "recursive rm on root/home"),
    (re.compile(r"(?<![\w-])mkfs(?![\w-])"), "mkfs"),
    (re.compile(r"(?<![\w-])dd\b[^;|&\n]*\sof=/dev/"), "dd to a device"),
    (re.compile(r">\s*/dev/(?:sd|nvme|hd|mmcblk|vd)[a-z0-9]"), "overwrite of a block device"),
    (re.compile(r":\s*\(\s*\)\s*\{"), "fork bomb"),
    (re.compile(r"(?<![\w-])chmod\s+(?:-\S+\s+)*0*777\b"), "chmod 777"),
    (re.compile(r"(?<![\w-])chown\s+[^;|&\n]*\s/(?:etc|usr|bin|boot|sys|var|root)\b"),
     "chown of a system path"),
    (re.compile(r"(?<![\w-])git\b[^;|&\n]*\bpush\b[^;|&\n]*(?:--force\b|--force-with-lease|\s-[a-zA-Z]*f[a-zA-Z]*\b|\s\+[\w./-]+)"),
     "git force push"),
    (re.compile(r"(?<![\w-])git\b[^;|&\n]*\b(?:reset\s+--hard|clean\s+-[a-z]*f|filter-branch)\b"),
     "git history/tree destruction"),
    (re.compile(r"(?<![\w-])eval(?![\w-])"), "eval"),
)

# Redirect / write operators whose targets must land in the workspace.
_REDIRECT_TARGET = re.compile(r"(?:>>?|(?<![\w-])tee(?![\w-])\s+(?:-\S+\s+)*)\s*([^\s;|&>]+)")

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

# Compound-command separators (newline included: models emit multi-line
# scripts, and a newline separates statements like ;).
_SHELL_SPLIT = re.compile(r"(?<!\|)\|(?!\|)|\|\||&&|;|\n")

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


def _path_hits_sacred(path: str) -> Optional[str]:
    """Return the sacred pattern a single filesystem path hits, if any."""
    if not path:
        return None
    home = str(Path.home()).lower()
    normalized = path.strip().strip("'\"")
    normalized = normalized.replace("~", home)
    lowered = normalized.lower()
    # Prefix with a slash so a bare ".env"/".ssh" still matches "/.env"/"/.ssh".
    candidate = lowered if lowered.startswith("/") else "/" + lowered
    for pattern in SACRED_PATTERNS:
        if pattern in candidate:
            return pattern
    return None


def _shell_tokens(command: str) -> list[str]:
    """Best-effort token split of a shell command for path inspection."""
    try:
        return shlex.split(command, comments=False, posix=True)
    except ValueError:
        return command.split()


def _command_hits_sacred(command: str) -> Optional[str]:
    """Return the sacred pattern any token OR redirect target of a shell
    command hits (a redirect like ``> .env`` writes a sacred file)."""
    for token in _shell_tokens(command):
        if hit := _path_hits_sacred(token):
            return hit
    for target in _REDIRECT_TARGET.findall(command):
        if hit := _path_hits_sacred(target):
            return hit
    return None


def _dangerous_shell(command: str) -> Optional[str]:
    """Return a description of the never-auto construct found, if any."""
    for pattern, label in _NEVER_AUTO_SHELL:
        if pattern.search(command):
            return label
    return None


def _shell_segments(command: str) -> list[str]:
    """Split a compound command; every segment must pass independently."""
    return [seg.strip() for seg in _SHELL_SPLIT.split(command) if seg.strip()]


def _redirect_targets(command: str) -> list[str]:
    """Filesystem paths a shell command writes to via > / >> / tee."""
    return [t.strip().strip("'\"") for t in _REDIRECT_TARGET.findall(command) if t.strip()]


class PermissionEngine:
    """Evaluate tool calls against rules, sacred paths, and posture.

    One instance is shared by every tool loop (wired onto providers by
    CascadeCore, like hook_runner). ``ask_handler`` resolves "ask"
    verdicts interactively; when None (headless lanes), asks become
    structured denials with deny-and-continue escalation bounds.
    """

    @staticmethod
    def normalize_posture(posture: object) -> str:
        """Coerce a posture value, failing CLOSED (to safe) on anything odd."""
        value = str(posture or "").lower()
        return value if value in ("auto", "safe", "readonly") else "safe"

    def __init__(
        self,
        posture: str = "auto",
        allow: tuple[str, ...] = (),
        deny: tuple[str, ...] = (),
        ask: tuple[str, ...] = (),
        workspace_root: Optional[str] = None,
        audit_limit: int = 200,
    ) -> None:
        self._posture = self.normalize_posture(posture)
        self._allow = tuple(filter(None, (parse_rule(r) for r in allow)))
        self._deny = tuple(filter(None, (parse_rule(r) for r in deny)))
        self._ask = tuple(filter(None, (parse_rule(r) for r in ask)))
        self._workspace_root = Path(workspace_root or ".").expanduser().resolve()
        self._session_grants: set[str] = set()
        self.ask_handler: Optional[Callable[[str, dict, Verdict], str]] = None
        # deque(maxlen) append is atomic under the GIL and self-bounds, so
        # parallel lanes cannot race an append against a reslice.
        self.audit: deque = deque(maxlen=audit_limit)
        self._lock = threading.Lock()
        self.consecutive_denials = 0
        self.total_denials = 0

    @property
    def posture(self) -> str:
        return self._posture

    @posture.setter
    def posture(self, value: object) -> None:
        # A posture switch revokes prior session grants and resets the
        # escalation counters, so "always allow" from a looser posture can
        # never outrank a newly-restrictive one.
        self._posture = self.normalize_posture(value)
        with self._lock:
            self._session_grants.clear()
            self.consecutive_denials = 0
            self.total_denials = 0

    def for_workspace(self, workspace_root: str) -> "PermissionEngine":
        """A sibling engine sharing posture + rules but scoped to a new root.

        Worktree lanes get one of these (root = the worktree) so in-worktree
        writes auto-approve while the sacred/dangerous floors still catch
        escapes; each lane's counters/grants/audit are its own, so concurrent
        lanes cannot poison one another or the chat loop.
        """
        clone = PermissionEngine(
            posture=self._posture,
            workspace_root=workspace_root,
            audit_limit=self.audit.maxlen or 200,
        )
        clone._allow = self._allow
        clone._deny = self._deny
        clone._ask = self._ask
        clone.ask_handler = self.ask_handler
        return clone

    # -- rule store -----------------------------------------------------

    def grant_session(self, tool_name: str, primary_arg: str = "") -> None:
        """User chose "always allow (this session)" for this shape."""
        with self._lock:
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

        # Sacred paths are immune to posture and rank above every auto-allow,
        # INCLUDING read-only auto-allow: reading ~/.ssh/id_rsa or .env is
        # exfiltration. Checked for every filesystem-touching tool (fetch is
        # URLs, not paths, and is gated separately below).
        if not is_fetch:
            if is_shell:
                sacred_hit = _command_hits_sacred(primary)
            else:
                sacred_hit = _path_hits_sacred(primary)
            if sacred_hit:
                return Verdict("ask", f"touches sacred path ({sacred_hit})", "sacred")

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

        is_write = self._is_write_tool(tool)

        if self._posture == "readonly":
            if is_write or is_shell:
                return Verdict("deny", "readonly posture blocks mutations", "posture")
            return Verdict("allow", "readonly posture allows reads", "posture")
        if self._posture == "safe":
            if is_write or is_shell:
                return Verdict("ask", "safe posture asks for mutations", "posture")
            return Verdict("allow", "safe posture allows reads", "posture")

        # posture == "auto": auto-approve, but keep writes inside the
        # workspace. Applies to file tools AND shell redirect targets.
        if is_write and not is_shell:
            path = arguments.get("path") or arguments.get("file_path") or ""
            if isinstance(path, str) and path and not self._in_workspace(path):
                return Verdict("ask", f"write outside workspace ({path})", "workspace")
        if is_shell:
            for target in _redirect_targets(primary):
                if not self._in_workspace(target):
                    return Verdict(
                        "ask", f"shell write outside workspace ({target})", "workspace",
                    )
        return Verdict("allow", "auto posture", "posture")

    @staticmethod
    def _is_write_tool(tool: Optional["ToolDef"]) -> bool:
        # Fail safe: anything not explicitly flagged read-only (including an
        # unknown tool with no ToolDef) is treated as a mutation, so the
        # sacred/workspace guards apply rather than being skipped.
        return not (tool is not None and tool.is_read_only)

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
                if answer == "always" and self._is_grantable(verdict.rule):
                    self.grant_session(tool_name, grant_arg)
                    verdict = Verdict("allow", "approved (always this session)", verdict.rule)
                elif answer in ("allow", "always"):
                    # "always" on a non-grantable tier (sacred/never-auto/ask
                    # rule) approves this one call only -- those tiers are
                    # never blanket-granted.
                    verdict = Verdict("allow", "approved by user", verdict.rule)
                else:
                    verdict = self._deny_and_escalate(
                        verdict, f"not approved: {verdict.reason}",
                    )
            else:
                verdict = self._deny_and_escalate(
                    verdict,
                    f"{verdict.reason} — blocked in unattended mode; "
                    "choose a safer approach",
                )

        if verdict.decision == "allow":
            # A genuine approval clears the streak; a read auto-allow does
            # not, so a compromised agent cannot alternate reads to dodge
            # the consecutive-denial escalation.
            if verdict.rule != "read-only":
                with self._lock:
                    self.consecutive_denials = 0

        self._record(tool_name, primary, verdict)
        return verdict

    @staticmethod
    def _is_grantable(rule: str) -> bool:
        """Whether an 'always' grant is allowed for this verdict's tier.

        Structural tiers (sacred paths, dangerous shell, explicit ask rules)
        are never blanket-granted -- only per-host fetch and posture-level
        asks are."""
        return rule in ("network", "posture", "workspace")

    def _deny_and_escalate(self, verdict: "Verdict", reason: str) -> "Verdict":
        """Turn an unapproved ask into a deny, counting toward escalation.

        Interactive denials count too, so a model cannot re-raise the same
        ask every round for unbounded modal fatigue."""
        with self._lock:
            self.consecutive_denials += 1
            self.total_denials += 1
            escalated = (
                self.consecutive_denials >= MAX_CONSECUTIVE_DENIALS
                or self.total_denials >= MAX_TOTAL_DENIALS
            )
        if escalated:
            return Verdict(
                "deny",
                f"{verdict.reason} — denial limit reached, stop and report "
                "instead of retrying",
                "escalation",
            )
        return Verdict("deny", reason, verdict.rule)

    def clear_session_grants(self) -> None:
        with self._lock:
            self._session_grants.clear()

    def _record(self, tool_name: str, primary: str, verdict: Verdict) -> None:
        self.audit.append((tool_name, primary[:80], verdict.decision, verdict.rule))
