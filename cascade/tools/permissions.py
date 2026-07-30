"""Permission verdict engine: the always-present gate on tool execution.

Design (from the Claude Code / Codex / rtk research, tuned for a
default-AUTO posture): safety comes from structure and background review,
not permission popups. Evaluation order is the load-bearing property:
explicit denies and the root/home deletion circuit breaker run before any
automatic approval.

Postures:
- "auto"     (default): reads, workspace writes, and TRANSPARENT shell
  auto-approve; ambiguous actions receive a tool-less background review.
- "yolo":    everything except explicit denies and the root/home deletion
  circuit breaker runs immediately, without background review.
- "safe":    compatibility posture: reads auto-approve and every mutation is
  background-reviewed. It never opens a prompt.
- "readonly": reads auto-approve; every mutation denies.

Threat model and residual risk (be honest -- default-auto trades some
safety for throughput): the floors reliably catch (a) a curated
catastrophic-command list, (b) sacred-path access by any tool, (c) shell
whose intent is not textually transparent (inline code, expansion-built
command words), and (d) writes outside the workspace. A shell blocklist
can NEVER be complete against an adversary -- shell is Turing-complete --
so the opaque-shell rule is the real structural defense: Cascade reviews
what it cannot understand rather than pretending to enumerate every
dangerous form. What still auto-approves under "auto":
transparent, in-workspace, non-catastrophic commands, some of which
could be mildly destructive (a plain `mv`/`truncate` of a project file).
That is the accepted cost of a default-auto posture. CLI-proxy providers
run their own tools in a subprocess Cascade cannot interpose on, so their
native no-prompt mode and sandbox form the corresponding boundary.

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
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import ToolDef


class PermissionAbort(Exception):
    """Raised when the denial-escalation limit is hit: the tool loop must
    stop and report rather than keep grinding against the gate."""


@dataclass(frozen=True)
class Verdict:
    decision: str  # "allow" | "review" | "deny"
    reason: str
    rule: str = ""


# Paths that require background review in auto mode. Explicit deny rules still
# outrank review, while yolo intentionally bypasses this list.
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
    # A remote fetch (any common downloader) anywhere in the command plus an
    # interpreter OR code-consuming build tool anywhere = download-then-run
    # RCE. Broadened past curl|sh because git clone && make, scp && sh, and
    # curl -o x.awk && awk -f x.awk are the same threat with more transparency.
    (re.compile(
        rf"(?<![\w-])(?:curl|wget|fetch|scp|sftp|rsync|ftp|git\s+clone|git\s+pull)(?![\w-])"
        rf".*(?:{_INTERPRETERS}|(?<![\w-])(?:make|awk\s+-f|sed\s+-f|gcc|cc|clang|npx?|pip3?\s+install)(?![\w-]))",
        re.S,
    ), "remote fetch then run/build (download-then-execute)"),
    # Recursive/catastrophic deletion that isn't a bare `rm` path.
    (re.compile(r"(?<![\w-])find\b[^;|&\n]*\s-(?:delete|exec\b)"), "find -delete/-exec"),
    (re.compile(r"(?<![\w-])xargs\b[^;|&\n]*\s(?:rm|shred|unlink|dd|mv)\b"), "xargs destructive"),
    (re.compile(rf"(?<![\w-])(?:{_INTERPRETERS}|source|\.)\s+(?:-\S+\s+)*(?:/tmp/|/var/tmp/|/dev/shm/)"),
     "interpreter running a temp-dir script"),
    (re.compile(r"/dev/tcp/"), "reverse shell (/dev/tcp)"),
    (re.compile(r"(?<![\w-])n(?:et)?c(?:at)?(?![\w-])[^;|&\n]*\s-[a-z]*e"), "netcat exec"),
    (re.compile(r"(?<![\w-])(?:ba|z)?sh\s+-[a-z]*i\b[^;|&\n]*(?:/dev/tcp|>&|<&)"),
     "interactive reverse shell"),
    (re.compile(r"(?<![\w-])rm\s+(?:-\S+\s+)*(?:-[a-z]*[rf][a-z]*|--recursive|--force)\b[^;|&\n]*\s(?:['\"]?/[\w.*/@-]*|~\S*|\$\{?HOME\}?\S*)"),
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
    # Persistence / boot-time footholds.
    (re.compile(r"(?<![\w-])crontab(?![\w-])"), "crontab (persistence)"),
    (re.compile(r"(?<![\w-])(?:systemctl|launchctl)\s+(?:enable|load)\b"),
     "service persistence"),
    (re.compile(r"(?<![\w-])at\s+(?:now|-f|\d)"), "at job (persistence)"),
    # Irreversible file destruction that isn't rm.
    (re.compile(r"(?<![\w-])(?:shred|srm)(?![\w-])"), "secure-delete"),
    (re.compile(r"(?<![\w-])truncate\s+-s\s*0"), "truncate to zero"),
    (re.compile(r"(?<![\w-])mv\b[^;|&\n]*\s/dev/null(?:\s|$)"),
     "move into /dev/null (data loss)"),
    # Global git config changes affect every repo (e.g. hijacking the editor
    # or a hook path).
    (re.compile(r"(?<![\w-])git\b[^;|&\n]*\bconfig\b[^;|&\n]*--global"),
     "global git config change"),
)

# Redirect / write operators whose targets must land in the workspace.
# Handles an optional leading fd digit and the `>|` noclobber-override form.
_REDIRECT_TARGET = re.compile(
    r"(?:\d*>>?\|?|&>\|?|(?<![\w-])tee(?![\w-])\s+(?:-\S+\s+)*)\s*([^\s;|&>]+)"
)

# `-t DIR` (cp/mv/install) puts the destination directory in a flag, not the
# last positional argument.
_DASH_T_TARGET = re.compile(r"(?<![\w-])-t\s+([^\s;|&]+)")

# dd of=PATH write destination.
_DD_TARGET = re.compile(r"(?<![\w-])dd\b[^;|&\n]*\sof=([^\s;|&]+)")

# Inline-code interpreter invocation (python -c, perl -e, node -e, sh -c,
# php -r) or a stdin/heredoc script -- the intent is NOT visible in the
# command text, so it can never be auto-approved.
_INLINE_CODE = re.compile(
    rf"(?<![\w-])(?:{_INTERPRETERS})\b[^;|&\n]*(?:\s-(?:e|c|r)\b|\s-\s|<<)"
)

# Command basenames that execute a script/code argument (so a dynamic
# argument to one means running computed code).
_CODE_RUNNERS = frozenset({
    "sh", "bash", "zsh", "ksh", "dash", "fish", "source", ".",
    "python", "python2", "python3", "perl", "ruby", "node", "php", "eval",
})

# Recursive/forced rm whose target is dangerous EVEN without an absolute
# path: current dir, parent, or a glob wipes broadly. (Absolute/home
# targets are handled by the never-auto rm rule.)
_RM_RELATIVE_BROAD = re.compile(
    r"(?<![\w-])rm\s+(?:-\S+\s+)*(?:-[a-z]*[rf][a-z]*|--recursive|--force)\b[^;|&\n]*\s(?:['\"]?\.{1,2}['\"]?(?:\s|$|/)|\S*\*)"
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

# Compound-command separators (newline included: models emit multi-line
# scripts, and a newline separates statements like ;). A `|` that is part
# of a `>|` / `N>|` noclobber-override redirect is NOT a pipe.
_SHELL_SPLIT = re.compile(r"(?<![|>&\d])\|(?!\|)|\|\||&&|;|\n")

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


def _root_or_home_deletion(command: str) -> Optional[str]:
    """Catch catastrophic recursive deletion even in yolo mode."""
    home = Path.home().resolve()

    def _is_root_or_home(target: str) -> bool:
        original = target.strip("'\"")
        raw = original.rstrip("/") or "/"
        broad_roots = {
            "/", "/*", "/.*", "/{*,.*}",
            "~", "~/*", "~/.*", "~/{*,.*}",
            "$HOME", "$HOME/*", "$HOME/.*", "$HOME/{*,.*}",
            "${HOME}", "${HOME}/*", "${HOME}/.*", "${HOME}/{*,.*}",
            str(home), f"{home}/*", f"{home}/.*", f"{home}/{{*,.*}}",
        }
        if original in broad_roots or raw in {"~", "$HOME", "${HOME}"}:
            return True
        # A wildcard/brace directly below root or home is effectively a broad
        # deletion of that anchor even though the literal target is not equal
        # to the anchor (`/[!.]*`, `~/D*`, `${HOME}/{*,.*}`).
        for anchor in ("/", "~", "$HOME", "${HOME}", str(home)):
            prefix = anchor if anchor == "/" else anchor + "/"
            if not original.startswith(prefix):
                continue
            leaf = original[len(prefix):].rstrip("/")
            if leaf and "/" not in leaf and any(char in leaf for char in "*?[{"):
                return True
        try:
            return Path(raw).expanduser().resolve() in {Path("/"), home}
        except (OSError, ValueError):
            return False

    wrappers = {"sudo", "doas", "command", "env", "nohup", "xargs"}
    options_with_values = {
        "-u", "--user", "-g", "--group", "-h", "--host",
        "-p", "--prompt", "-C", "--chdir", "-R", "--chroot",
        "-a", "--arg-file", "-E", "--eof", "-I", "--replace",
        "-L", "--max-lines", "-n", "--max-args", "-P", "--max-procs",
        "-s", "--max-chars",
    }

    def _command_index(tokens: list[str]) -> int:
        """Skip common execution wrappers without treating plain arguments
        (for example `echo rm -rf /`) as commands."""
        index = 0
        while index < len(tokens):
            name = tokens[index].rsplit("/", 1)[-1]
            if name not in wrappers:
                return index
            index += 1
            while index < len(tokens):
                token = tokens[index]
                if token == "--":
                    index += 1
                    break
                if name == "env" and "=" in token and not token.startswith("-"):
                    index += 1
                    continue
                if not token.startswith("-"):
                    break
                option = token.split("=", 1)[0]
                index += 1
                if option in options_with_values and "=" not in token:
                    index += 1
        return index

    def _scan(text: str, depth: int = 0) -> bool:
        for segment in _shell_segments(text):
            tokens = _shell_tokens(segment)
            if not tokens:
                continue
            index = _command_index(tokens)
            if index >= len(tokens):
                continue
            command_name = tokens[index].rsplit("/", 1)[-1]
            tail = tokens[index + 1:]
            if command_name in {"rm", "rmdir"}:
                recursive = command_name == "rmdir" or any(
                    option in {"--recursive", "--force"}
                    or (
                        option.startswith("-")
                        and any(flag in option[1:] for flag in ("r", "R", "f"))
                    )
                    for option in tail
                )
                targets = [item for item in tail if not item.startswith("-")]
                if recursive and any(_is_root_or_home(item) for item in targets):
                    return True
            if (
                command_name == "find"
                and any(option in {"-delete", "-exec"} for option in tail)
            ):
                roots = [item for item in tail if not item.startswith("-")]
                if roots and _is_root_or_home(roots[0]):
                    return True
            # `sh -c 'rm -rf /'` keeps the destructive command inside one
            # quoted argv item. Inspect that script once as well.
            if (
                depth == 0
                and command_name in {"sh", "bash", "zsh", "ksh", "dash", "fish"}
                and "-c" in tail
            ):
                script_index = tail.index("-c") + 1
                if script_index < len(tail) and _scan(tail[script_index], depth + 1):
                    return True
        return False

    if _scan(command):
        return "recursive deletion of filesystem root/home"
    return None


def _shell_segments(command: str) -> list[str]:
    """Split a compound command; every segment must pass independently."""
    return [seg.strip() for seg in _SHELL_SPLIT.split(command) if seg.strip()]


# Standard sinks that are always fine to "write" to.
_DEVICE_SINKS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"})


def _redirect_targets(command: str) -> list[str]:
    """Filesystem paths a shell command writes to (redirect, tee, dd, cp/mv)."""
    targets = [t for t in _REDIRECT_TARGET.findall(command)]
    targets += _DD_TARGET.findall(command)
    # cp/mv/install write to their final argument, or to a `-t DIR` flag.
    for seg in _shell_segments(command):
        toks = _shell_tokens(seg)
        if toks and toks[0] in ("cp", "mv", "install") and len(toks) >= 2:
            dash_t = _DASH_T_TARGET.findall(seg)
            if dash_t:
                targets.extend(dash_t)
            else:
                dest = toks[-1]
                if not dest.startswith("-"):
                    targets.append(dest)
    cleaned = [t.strip().strip("'\"") for t in targets if t and t.strip()]
    return [t for t in cleaned if t not in _DEVICE_SINKS]


def _opaque_shell_reason(command: str) -> Optional[str]:
    """Why a shell command's effect is NOT visible from its text, or None.

    The never-auto floor is a lexical blocklist and cannot be made complete
    against shell expansion, so the auto posture never auto-approves a
    command whose intent is hidden: inline interpreter code (python -c),
    or a command word assembled at runtime via variable/expansion. Such
    commands ASK instead. Common transparent dev commands (pytest, git
    status, python script.py, npm test) are unaffected.
    """
    if _INLINE_CODE.search(command):
        return "inline interpreter code"
    # Assemble-then-use: a command that assigns a shell variable and later
    # expands it hides its real target/effect from the lexical sacred and
    # workspace checks (`p=$HOME/.ssh; cat $p/id_rsa`, `d=/etc; cp x $d/`).
    # Bare $HOME/$PWD usage without a preceding assignment is NOT flagged.
    assigned = set(re.findall(r"(?:^|[;&|\n]|\bexport\s+)\s*(\w+)=", command))
    if assigned:
        for var in assigned:
            if re.search(rf"\$\{{?{re.escape(var)}\b", command):
                # Only if the expansion is used as/inside an argument, not
                # just re-assigned. Cheap check: it appears after its assign.
                return "variable assembled then expanded"
    for seg in _shell_segments(command):
        stripped = re.sub(r"^(?:\w+=\S*\s+)*", "", seg.strip())
        if not stripped:
            continue
        tokens = stripped.split()
        word = tokens[0]
        if "$" in word or "`" in word:
            return "command word built from an expansion"
        # An interpreter/sourcer running a dynamic argument executes a
        # computed script (`sh $p`, `source $x`, `python $mod`).
        base = word.rsplit("/", 1)[-1]
        if base in _CODE_RUNNERS and any(
            "$" in tok or "`" in tok for tok in tokens[1:]
        ):
            return "interpreter running a dynamic argument"
    return None


@dataclass(frozen=True)
class PermissionContext:
    """Trusted request context captured before tool results are appended."""

    objective: str = ""
    provider: str = ""
    model: str = ""
    mode: str = ""


@dataclass(frozen=True)
class PermissionReview:
    """One ambiguous action sent to the background safety reviewer."""

    tool_name: str
    arguments: dict[str, Any]
    reason: str
    rule: str
    workspace_root: str
    context: PermissionContext = PermissionContext()


@dataclass(frozen=True)
class ReviewDecision:
    """Normalized result from a tool-less background reviewer."""

    allow: bool
    reason: str


ReviewHandler = Callable[[PermissionReview], ReviewDecision]


def permission_context_from_messages(
    messages: list[dict],
    *,
    provider: str = "",
    model: str = "",
    mode: str = "",
) -> PermissionContext:
    """Capture the latest user objective without including tool results."""
    objective = ""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            objective = content.strip()[:6000]
            break
    return PermissionContext(
        objective=objective,
        provider=provider,
        model=model,
        mode=mode,
    )


class PermissionEngine:
    """Resolve tool actions without ever requiring interactive approval."""

    @staticmethod
    def normalize_posture(posture: object) -> str:
        """Coerce a posture value, failing closed to auto on anything odd."""
        value = str(posture or "").lower()
        return (
            value
            if value in ("auto", "yolo", "safe", "readonly")
            else "auto"
        )

    def __init__(
        self,
        posture: str = "auto",
        allow: tuple[str, ...] = (),
        deny: tuple[str, ...] = (),
        ask: tuple[str, ...] = (),
        workspace_root: Optional[str] = None,
        audit_limit: int = 200,
        review_handler: Optional[ReviewHandler] = None,
    ) -> None:
        self._posture = self.normalize_posture(posture)
        self._allow = tuple(filter(None, (parse_rule(r) for r in allow)))
        self._deny = tuple(filter(None, (parse_rule(r) for r in deny)))
        self._ask = tuple(filter(None, (parse_rule(r) for r in ask)))
        self._workspace_root = Path(workspace_root or ".").expanduser().resolve()
        self.review_handler = review_handler
        # deque(maxlen) append is atomic under the GIL and self-bounds, so
        # parallel lanes cannot race an append against a reslice.
        self.audit: deque = deque(maxlen=audit_limit)
        self._lock = threading.Lock()
        # Tool handlers may overlap, but reviews mutate shared counters and can
        # make provider calls. Keep that boundary single-file.
        self._resolution_lock = threading.RLock()
        self.consecutive_denials = 0
        self.total_denials = 0

    @property
    def posture(self) -> str:
        return self._posture

    @posture.setter
    def posture(self, value: object) -> None:
        # A posture switch resets denial escalation for the new policy.
        self._posture = self.normalize_posture(value)
        with self._lock:
            self.consecutive_denials = 0
            self.total_denials = 0

    def for_workspace(self, workspace_root: str) -> "PermissionEngine":
        """A sibling engine sharing posture + rules but scoped to a new root.

        Worktree lanes get one of these (root = the worktree) so in-worktree
        writes auto-approve while the sacred/dangerous floors still catch
        escapes; each lane's counters/audit are its own, so concurrent
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
        clone.review_handler = self.review_handler
        return clone

    def for_unattended_workspace(self, workspace_root: str) -> "PermissionEngine":
        """Return a popup-free worktree-scoped engine for autonomous lanes."""
        return self.for_workspace(workspace_root)

    # -- evaluation -----------------------------------------------------

    def evaluate(
        self,
        tool: Optional["ToolDef"],
        tool_name: str,
        arguments: dict,
    ) -> Verdict:
        """Pure verdict: deny > circuit breaker > review rules >
        posture auto-allow > allow rules > posture default."""
        primary = _primary_arg(tool_name, arguments)

        for rule in self._deny:
            if rule.matches(tool_name, primary):
                return Verdict("deny", f"deny rule {rule.tool}", "deny-rule")

        is_fetch = tool_name in ("web_fetch", "web_search") or "url" in arguments
        is_shell = "command" in arguments and isinstance(arguments.get("command"), str)
        if is_shell and (circuit_reason := _root_or_home_deletion(primary)):
            return Verdict("deny", circuit_reason, "circuit-breaker")
        if self._posture == "yolo":
            return Verdict("allow", "yolo posture", "yolo")

        # Secret/config paths require review before read-only auto-approval.
        if not is_fetch:
            if is_shell:
                sacred_hit = _command_hits_sacred(primary)
            else:
                sacred_hit = _path_hits_sacred(primary)
            if sacred_hit:
                return Verdict(
                    "review", f"touches protected path ({sacred_hit})", "sacred",
                )

        # Readonly is a hard posture, not a default that an allow rule or
        # dangerous-shell review can accidentally override.
        if self._posture == "readonly" and (
            is_fetch or is_shell or self._is_write_tool(tool)
        ):
            reason = (
                "readonly posture blocks network egress"
                if is_fetch
                else "readonly posture blocks mutations"
            )
            return Verdict("deny", reason, "posture")

        # Network egress: gated separately from filesystem writes (it is the
        # primary exfiltration lever). Auto-approve only preapproved
        # read-only docs domains; every other host receives review.
        if is_fetch:
            for rule in self._ask:
                if rule.matches(tool_name, primary):
                    return Verdict("review", f"review rule {rule.tool}", "ask-rule")
            # web_search contacts a single fixed search endpoint, not a
            # user-controlled host: the only thing leaving the machine is the
            # query text. Strictly safer than web_fetch's arbitrary-host GET --
            # auto-approve unless the posture forbids all egress.
            if tool_name == "web_search":
                return Verdict("allow", "web search (fixed endpoint)", "web-search")
            host = _url_host(primary)
            for rule in self._allow:
                if rule.matches(tool_name, primary) or rule.matches(tool_name, host):
                    return Verdict("allow", f"allow rule {rule.tool}", "allow-rule")
            if host and any(host == d or host.endswith("." + d) for d in PREAPPROVED_FETCH_DOMAINS):
                return Verdict("allow", f"preapproved docs domain ({host})", "docs-allowlist")
            return Verdict(
                "review", f"fetch from {host or 'unknown host'}", "network",
            )

        # Ambiguous shell actions reach the background reviewer before broad
        # allow rules can re-enable them.
        if is_shell:
            for candidate in [primary, *_shell_segments(primary)]:
                if danger := _dangerous_shell(candidate):
                    return Verdict(
                        "review",
                        f"dangerous shell construct ({danger})",
                        "never-auto",
                    )
                if _RM_RELATIVE_BROAD.search(candidate):
                    return Verdict(
                        "review", "recursive rm on '.'/'..'/glob", "never-auto",
                    )
            # Opaque and out-of-workspace shell writes are floors too, so a
            # loose allow rule cannot re-open inline-code RCE or an
            # out-of-workspace write. (Only meaningful under auto; safe and
            # readonly review/deny shell mutations at the posture branch.)
            if self._posture == "auto":
                if reason := _opaque_shell_reason(primary):
                    return Verdict(
                        "review", f"opaque shell ({reason})", "opaque-shell",
                    )
                for target in _redirect_targets(primary):
                    if not self._in_workspace(target):
                        return Verdict(
                            "review",
                            f"shell write outside workspace ({target})",
                            "workspace",
                        )

        for rule in self._ask:
            if rule.matches(tool_name, primary):
                return Verdict("review", f"review rule {rule.tool}", "ask-rule")

        if tool is not None and tool.is_read_only:
            return Verdict("allow", "read-only tool", "read-only")

        for rule in self._allow:
            if rule.matches(tool_name, primary):
                return Verdict("allow", f"allow rule {rule.tool}", "allow-rule")

        is_write = self._is_write_tool(tool)

        if self._posture == "safe":
            if is_write or is_shell:
                return Verdict(
                    "review", "safe posture reviews mutations", "posture",
                )
            return Verdict("allow", "safe posture allows reads", "posture")

        # posture == "auto": file writes must stay in the workspace (shell was
        # already floored above).
        if is_write and not is_shell:
            path = arguments.get("path") or arguments.get("file_path") or ""
            if isinstance(path, str) and path and not self._in_workspace(path):
                return Verdict(
                    "review", f"write outside workspace ({path})", "workspace",
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

    # -- resolution (background review, escalation, audit) ---------------

    def resolve(
        self,
        tool: Optional["ToolDef"],
        tool_name: str,
        arguments: dict,
        context: Optional[PermissionContext] = None,
    ) -> Verdict:
        """Resolve an action to allow/deny; this method never prompts."""
        with self._resolution_lock:
            return self._resolve_locked(
                tool,
                tool_name,
                arguments,
                context or PermissionContext(),
            )

    def _resolve_locked(
        self,
        tool: Optional["ToolDef"],
        tool_name: str,
        arguments: dict,
        context: PermissionContext,
    ) -> Verdict:
        """Resolve one verdict while the background-review gate is held."""
        verdict = self.evaluate(tool, tool_name, arguments)
        primary = _primary_arg(tool_name, arguments)

        reviewed_allow = False
        if verdict.decision == "review":
            review = PermissionReview(
                tool_name=tool_name,
                arguments=dict(arguments),
                reason=verdict.reason,
                rule=verdict.rule,
                workspace_root=str(self._workspace_root),
                context=context,
            )
            decision = None
            if self.review_handler is not None:
                try:
                    decision = self.review_handler(review)
                except Exception:
                    decision = None
            if isinstance(decision, ReviewDecision) and decision.allow:
                verdict = Verdict(
                    "allow",
                    f"background review approved: {decision.reason}",
                    verdict.rule,
                )
                reviewed_allow = True
            else:
                reason = (
                    decision.reason
                    if isinstance(decision, ReviewDecision) and decision.reason
                    else "background reviewer unavailable"
                )
                verdict = self._deny_and_escalate(
                    verdict,
                    f"{verdict.reason} — blocked by background review: {reason}",
                )
        elif verdict.decision == "deny":
            # Hard policy denials are just as capable of trapping an agent in
            # a retry loop as reviewer denials, so they share the same bound.
            verdict = self._deny_and_escalate(verdict, verdict.reason)

        # Only a reviewed approval clears the consecutive streak, so a
        # compromised agent cannot alternate auto-allowed reads/writes to
        # keep the escalation counter pinned at zero forever.
        if reviewed_allow:
            with self._lock:
                self.consecutive_denials = 0

        self._record(tool_name, primary, verdict)
        return verdict

    def _deny_and_escalate(self, verdict: "Verdict", reason: str) -> "Verdict":
        """Count a blocked action and bound agent retries."""
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

    def _record(self, tool_name: str, primary: str, verdict: Verdict) -> None:
        self.audit.append((tool_name, primary[:80], verdict.decision, verdict.rule))
