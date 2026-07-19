"""Output filters: compress high-volume command output before it hits context.

Keyed on argv[0] (+ subcommand). Two non-negotiables ported from rtk:

- never_worse: a filter's output is used only if it is actually shorter
  than the raw output. A filter that would expand output is discarded.
- tee: the full raw output is always available; filters append a
  recovery pointer so the model can request it. (Cascade spills to the
  run ledger's artifact store; the pointer names the run.)

Filters are pure ``(raw: str) -> str``. Structured-output requests
(``--json``, pipes into ``jq``/``wc``) bypass filtering entirely -- the
model asked for a machine shape and must get it verbatim.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# A filtered result is kept only if it saves at least this fraction of
# characters; a marginal win is not worth the shape change.
_MIN_SAVING = 0.15

_STRUCTURED_MARKERS = ("--json", "--format=json", "-o json", "--output json")
_STRUCTURED_PIPES = re.compile(r"\|\s*(jq|wc|awk|cut|head|tail|grep\s+-c)\b")


def _first_token(command: str) -> str:
    parts = command.strip().split()
    return parts[0] if parts else ""


def _subcommand(command: str) -> str:
    parts = command.strip().split()
    return parts[1] if len(parts) > 1 else ""


def wants_structured_output(command: str) -> bool:
    """True when the command explicitly asked for a machine-readable shape."""
    lowered = command.lower()
    if any(marker in lowered for marker in _STRUCTURED_MARKERS):
        return True
    return bool(_STRUCTURED_PIPES.search(command))


# -- individual filters -------------------------------------------------


def filter_pytest(raw: str) -> str:
    """Keep the failures and the summary line; drop the passing-dots noise."""
    lines = raw.splitlines()
    summary = [ln for ln in lines if re.search(r"=+ .*(passed|failed|error).* =+", ln)]
    if not any("fail" in ln.lower() or "error" in ln.lower() for ln in summary):
        # All green: the summary line alone is the whole story.
        return "\n".join(summary) if summary else raw

    kept: list[str] = []
    in_failure = False
    for ln in lines:
        if re.match(r"_+ .+ _+$", ln) or ln.startswith("FAILED") or ln.startswith("ERROR"):
            in_failure = True
        if re.search(r"=+ (short test summary|FAILURES|ERRORS) =+", ln):
            in_failure = True
        if in_failure:
            kept.append(ln)
    kept.extend(summary)
    return "\n".join(kept) if kept else raw


def filter_git_status(raw: str) -> str:
    """Collapse porcelain-ish status to counts plus the first changed paths."""
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    changed = [ln for ln in lines if re.match(r"\s*[MADRCU?]{1,2}\s", ln)]
    if len(changed) <= 12:
        return raw
    head = changed[:12]
    return "\n".join(head) + f"\n… and {len(changed) - 12} more changed paths"


def filter_grep(raw: str) -> str:
    """Group many matches per file into a per-file count with sample lines."""
    lines = raw.splitlines()
    if len(lines) <= 40:
        return raw
    by_file: dict[str, list[str]] = {}
    for ln in lines:
        m = re.match(r"([^:]+):\d+:", ln)
        key = m.group(1) if m else "(no file)"
        by_file.setdefault(key, []).append(ln)
    out: list[str] = []
    for path, hits in by_file.items():
        out.append(f"{path}: {len(hits)} matches")
        out.extend(f"  {h}" for h in hits[:3])
        if len(hits) > 3:
            out.append(f"  … {len(hits) - 3} more in this file")
    return "\n".join(out)


def filter_install(raw: str) -> str:
    """Package-manager chatter: keep the outcome, drop progress spam."""
    lines = raw.splitlines()
    kept = [
        ln for ln in lines
        if re.search(
            r"(error|warn|fail|added|removed|installed|up to date|"
            r"success|conflict|ERR!)",
            ln, re.IGNORECASE,
        )
    ]
    return "\n".join(kept) if kept else raw


# argv[0] (optionally + subcommand) -> filter
_FILTERS: dict[str, Callable[[str], str]] = {
    "pytest": filter_pytest,
    "py.test": filter_pytest,
    "git status": filter_git_status,
    "grep": filter_grep,
    "rg": filter_grep,
    "npm install": filter_install,
    "npm": filter_install,
    "pnpm install": filter_install,
    "yarn": filter_install,
    "pip install": filter_install,
    "uv": filter_install,
}


def _resolve_filter(command: str) -> Optional[Callable[[str], str]]:
    # "python -m pytest ..." and "uv run pytest ..." route to pytest too.
    if re.search(r"\bpytest\b", command) and "pytest" not in command.split()[:1]:
        return filter_pytest
    key2 = f"{_first_token(command)} {_subcommand(command)}".strip()
    if key2 in _FILTERS:
        return _FILTERS[key2]
    return _FILTERS.get(_first_token(command))


def apply_output_filter(command: str, raw: str, recovery_hint: str = "") -> str:
    """Filter *raw* for *command*, honoring never_worse and structured bypass.

    Returns raw unchanged when: the command wants structured output, no
    filter matches, or the filtered result is not meaningfully shorter.
    Otherwise returns the filtered text with a recovery pointer appended.
    """
    if not raw or wants_structured_output(command):
        return raw
    fn = _resolve_filter(command)
    if fn is None:
        return raw
    try:
        filtered = fn(raw)
    except Exception:
        return raw
    # never_worse: keep the filter only on a real saving.
    if len(filtered) >= len(raw) * (1 - _MIN_SAVING):
        return raw
    if recovery_hint:
        filtered = f"{filtered}\n\n[filtered output — full text: {recovery_hint}]"
    return filtered
