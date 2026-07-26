"""Score a `/compete --list` run's worktrees into a ranked leaderboard.

`/compete --list` leaves a `competition.json` manifest plus one retained git
worktree per competitor. This module re-inspects those worktrees -- an
objective build/test gate plus an LLM quality judge -- and ranks every
competitor into a `Leaderboard`, so picking a cheap daily-driver model is a
data-driven choice instead of a vibe.

The pure ranking/disqualification math lives in `cascade.swarm.scoring_rank`
and is re-exported here; `CompetitionScorer` below is the LLM/build
side-effecting half that produces the rows it ranks.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .scoring_rank import (
    Leaderboard,
    ScoredCompetitor,
    cache_hit_pct,
    disqualification,
    rank_leaderboard,
    tokens_per_second,
)
from .solve import _is_infra_failure, _project_verify_test, _run_tests_in
from .worktree import WorktreeManager

__all__ = [
    "Leaderboard",
    "ScoredCompetitor",
    "cache_hit_pct",
    "tokens_per_second",
    "rank_leaderboard",
    "resolve_manifest_path",
    "discover_latest_manifest",
    "load_competitors",
    "CompetitionScorer",
]

_GATE_TIMEOUT_DEFAULT = 300
_DIFF_CHAR_CAP_DEFAULT = 12_000
_GATE_TAIL_CHARS = 2_000

_QUALITY_JUDGE_SYSTEM = """\
You are a strict evaluator scoring one coding agent's diff against its task.

Judge strictly from the actual diff, not the agent's self-report. Score three
axes from 0 (not attempted / wrong / reckless) to 5 (complete / correct /
disciplined):
- spec_completeness: how much of the requested task is actually implemented
- correctness: whether the change looks correct and would work
- scope_discipline: whether the change stayed focused, without unrelated edits

Respond with JSON only:
{
  "spec_completeness": 0-5,
  "correctness": 0-5,
  "scope_discipline": 0-5,
  "summary": "one concise line assessing the change"
}
"""

_QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "spec_completeness": {"type": "integer", "minimum": 0, "maximum": 5},
        "correctness": {"type": "integer", "minimum": 0, "maximum": 5},
        "scope_discipline": {"type": "integer", "minimum": 0, "maximum": 5},
        "summary": {
            "type": "string",
            "description": "One concise line assessing the change",
        },
    },
    "required": ["spec_completeness", "correctness", "scope_discipline", "summary"],
    "additionalProperties": False,
}


# --- Manifest loading ----------------------------------------------------------


@dataclass(frozen=True)
class _ManifestCompetitor:
    """One competitor entry read back from a `competition.json` manifest."""

    label: str
    model: str
    objective: str
    success: bool
    changed_files: "tuple[str, ...]"
    worktree_path: str
    cost: float
    tokens: int
    duration_seconds: float
    cache_read: int
    prompt_total: int


def resolve_manifest_path(path: str) -> Optional[Path]:
    """The manifest file for *path*: itself if a file, else `<path>/competition.json`."""
    try:
        candidate = Path(path).expanduser()
    except (OSError, RuntimeError):
        return None
    if candidate.is_dir():
        manifest = candidate / "competition.json"
        return manifest if manifest.is_file() else None
    return candidate if candidate.is_file() else None


def discover_latest_manifest() -> Optional[str]:
    """The most recently written `competition.json` under the worktree cache root.

    Reuses `WorktreeManager`'s own cache-root resolution (env override, XDG,
    then the `~/.cache` default) so discovery agrees with wherever `/compete`
    actually wrote the run.
    """
    root = WorktreeManager._cache_root()
    if not root.is_dir():
        return None
    candidates = list(root.glob("cascade-compete-*/competition.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _competitor_from_dict(raw: dict, objective: str) -> Optional[_ManifestCompetitor]:
    """A `_ManifestCompetitor` from one manifest entry, or None if malformed."""
    label = str(raw.get("label", "") or "")
    if not label:
        return None
    return _ManifestCompetitor(
        label=label,
        model=str(raw.get("model", "") or ""),
        objective=objective,
        success=bool(raw.get("success", False)),
        changed_files=tuple(raw.get("changed_files") or ()),
        worktree_path=str(raw.get("worktree_path", "") or ""),
        cost=float(raw.get("cost", 0.0) or 0.0),
        tokens=int(raw.get("tokens", 0) or 0),
        duration_seconds=float(raw.get("duration_seconds", 0.0) or 0.0),
        cache_read=int(raw.get("cache_read", 0) or 0),
        prompt_total=int(raw.get("prompt_total", 0) or 0),
    )


def load_competitors(
    manifest_paths: Sequence[str],
) -> "tuple[tuple[str, ...], tuple[_ManifestCompetitor, ...]]":
    """Resolve and read every manifest; return (resolved paths, all competitors).

    Best-effort per path: a missing, unreadable, or malformed manifest is
    skipped rather than raising, so one bad path never blocks scoring the rest.
    """
    resolved: list[str] = []
    competitors: list[_ManifestCompetitor] = []
    for raw_path in manifest_paths:
        manifest_path = resolve_manifest_path(raw_path)
        if manifest_path is None:
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        raw_competitors = data.get("competitors")
        if not isinstance(raw_competitors, list):
            continue
        objective = str(data.get("objective", "") or "")
        resolved.append(str(manifest_path))
        for item in raw_competitors:
            if not isinstance(item, dict):
                continue
            parsed = _competitor_from_dict(item, objective)
            if parsed is not None:
                competitors.append(parsed)
    return tuple(resolved), tuple(competitors)


# --- Quality judge (LLM side effect) -----------------------------------------


@dataclass(frozen=True)
class _QualityScore:
    spec_completeness: Optional[int] = None
    correctness: Optional[int] = None
    scope_discipline: Optional[int] = None
    summary: str = ""
    error: str = ""

    @property
    def total(self) -> Optional[int]:
        parts = (self.spec_completeness, self.correctness, self.scope_discipline)
        if any(part is None for part in parts):
            return None
        return sum(parts)  # type: ignore[arg-type]


def _clamp_score(value: object) -> int:
    try:
        return max(0, min(5, int(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _parse_quality_payload(payload: object) -> _QualityScore:
    """Parse a judge response into a `_QualityScore`. Raises on unusable input."""
    if isinstance(payload, str):
        match = re.search(r"\{[\s\S]*\}", payload)
        if not match:
            raise ValueError("judge returned no JSON object")
        payload = json.loads(match.group())
    if not isinstance(payload, dict):
        raise ValueError("judge response was not a JSON object")
    return _QualityScore(
        spec_completeness=_clamp_score(payload.get("spec_completeness", 0)),
        correctness=_clamp_score(payload.get("correctness", 0)),
        scope_discipline=_clamp_score(payload.get("scope_discipline", 0)),
        summary=str(payload.get("summary", "") or "").strip(),
    )


def _output_tokens(total: int, prompt_total: int) -> Optional[int]:
    """The generated-output token count, or None when the split is unknown.

    ``total`` counts the whole request (prompt + output); subtracting the prompt
    isolates the output. A manifest that never recorded ``prompt_total`` (from
    before cache accounting) leaves the split unknowable, so return None rather
    than pass off the full total -- prompt-inclusive -- as output.
    """
    if prompt_total <= 0:
        return None
    return max(0, total - prompt_total)


def _read_handoff(worktree_path: str) -> str:
    """Best-effort contents of a worktree's handoff.md, or "" if absent/unreadable."""
    if not worktree_path:
        return ""
    candidate = Path(worktree_path) / "handoff.md"
    try:
        return candidate.read_text(encoding="utf-8") if candidate.is_file() else ""
    except OSError:
        return ""


def _build_judge_prompt(objective: str, handoff: str, diff: str) -> str:
    parts = [f"Task:\n{objective or '(no objective recorded in the manifest)'}"]
    if handoff.strip():
        parts.append(f"Target spec (worktree handoff.md):\n{handoff}")
    parts.append(f"Diff:\n{diff or '(no diff captured)'}")
    return "\n\n".join(parts)


# --- Scorer --------------------------------------------------------------------


class CompetitionScorer:
    """Turn one or more `/compete --list` manifests into a ranked `Leaderboard`.

    The gate and judge are best-effort per competitor: any failure there is
    recorded on the row (never raised), so one bad worktree never blocks
    scoring the rest of the heat. Competitors are scored one at a time --
    *judge_provider* is a single shared instance, and a provider's per-call
    usage tracking (``last_usage``/activity state) is not safe to drive from
    more than one thread at once, so this deliberately does not fan the judge
    calls out the way `CompetitionOrchestrator` does for independent provider
    instances.
    """

    def __init__(
        self,
        manifest_paths: Sequence[str],
        judge_provider,
        verify_command_override: Optional[str] = None,
        gate_timeout: int = _GATE_TIMEOUT_DEFAULT,
        diff_char_cap: int = _DIFF_CHAR_CAP_DEFAULT,
    ) -> None:
        self._manifest_paths = tuple(manifest_paths)
        self._judge_provider = judge_provider
        self._verify_command_override = verify_command_override
        self._gate_timeout = gate_timeout
        self._diff_char_cap = diff_char_cap

    def score(self) -> Leaderboard:
        """Load every manifest, score each competitor, and rank the result."""
        resolved_paths, competitors = load_competitors(self._manifest_paths)
        rows = tuple(self._score_one(competitor) for competitor in competitors)
        return Leaderboard(rows=rank_leaderboard(rows), manifest_paths=resolved_paths)

    def _score_one(self, competitor: _ManifestCompetitor) -> ScoredCompetitor:
        """Score one competitor; any unexpected error degrades to a DQ row."""
        try:
            return self._score_one_unsafe(competitor)
        except Exception as exc:
            return ScoredCompetitor(
                label=competitor.label,
                model=competitor.model,
                gate="inconclusive",
                gate_output_tail="",
                spec_completeness=None,
                correctness=None,
                scope_discipline=None,
                quality_total=None,
                quality_summary="",
                cost=competitor.cost,
                tokens=competitor.tokens,
                tok_per_s=tokens_per_second(
                    _output_tokens(competitor.tokens, competitor.prompt_total),
                    competitor.duration_seconds,
                ),
                cache_hit_pct=cache_hit_pct(competitor.cache_read, competitor.prompt_total),
                disqualified=True,
                changed_files=competitor.changed_files,
                notes=(f"scoring error: {exc}",),
            )

    def _score_one_unsafe(self, competitor: _ManifestCompetitor) -> ScoredCompetitor:
        # Capture the model's diff BEFORE the gate runs: a build/test can touch a
        # tracked file (a lockfile, a generated source), and that must not leak
        # into what the judge scores as the model's work.
        diff = self._capture_diff(competitor.worktree_path)
        gate, gate_tail = self._run_gate(competitor.worktree_path)
        quality = self._run_judge(competitor, diff)
        pct = cache_hit_pct(competitor.cache_read, competitor.prompt_total)
        disqualified, dq_notes = disqualification(
            pct, competitor.success, competitor.changed_files,
        )
        notes = list(dq_notes)
        if quality.error:
            notes.append(f"quality judge unavailable: {quality.error}")

        return ScoredCompetitor(
            label=competitor.label,
            model=competitor.model,
            gate=gate,
            gate_output_tail=gate_tail,
            spec_completeness=quality.spec_completeness,
            correctness=quality.correctness,
            scope_discipline=quality.scope_discipline,
            quality_total=quality.total,
            quality_summary=quality.summary,
            cost=competitor.cost,
            tokens=competitor.tokens,
            tok_per_s=tokens_per_second(
                _output_tokens(competitor.tokens, competitor.prompt_total),
                competitor.duration_seconds,
            ),
            cache_hit_pct=pct,
            disqualified=disqualified,
            changed_files=competitor.changed_files,
            notes=tuple(notes),
        )

    def _resolve_verify_command(self, worktree_path: str) -> Optional[str]:
        if self._verify_command_override:
            return self._verify_command_override
        try:
            return _project_verify_test(worktree_path)
        except Exception:
            return None

    def _run_gate(self, worktree_path: str) -> "tuple[str, str]":
        """The objective build/test gate for one worktree: (classification, tail).

        "inconclusive" covers a missing worktree, no resolvable verify command,
        a timeout, or a setup/environment failure (e.g. gitignored deps like
        node_modules that never made it into the worktree) -- never scored as
        a real fail.
        """
        if not worktree_path or not Path(worktree_path).is_dir():
            return "inconclusive", "worktree is missing (already cleaned up?)"
        test_cmd = self._resolve_verify_command(worktree_path)
        if not test_cmd:
            return "inconclusive", "no verify command could be resolved for this project"
        try:
            output, returncode = _run_tests_in(test_cmd, worktree_path, self._gate_timeout)
        except Exception as exc:
            return "inconclusive", f"verify command errored: {exc}"
        tail = (output or "")[-_GATE_TAIL_CHARS:]
        if returncode == -1:  # `_run_tests_in`'s sentinel for a timeout
            return "inconclusive", tail
        if _is_infra_failure(output, returncode):
            return "inconclusive", tail or f"exit code {returncode} (setup/environment issue)"
        return ("pass" if returncode == 0 else "fail"), tail

    def _run_judge(self, competitor: _ManifestCompetitor, diff: str) -> _QualityScore:
        """Score the competitor's diff against its task. Never raises.

        The diff is captured by the caller before the gate runs, so it reflects
        only the model's changes, not any build/test side effects.
        """
        try:
            handoff = _read_handoff(competitor.worktree_path)
            prompt = _build_judge_prompt(competitor.objective, handoff, diff)
            ask_structured = getattr(self._judge_provider, "ask_structured", None)
            if callable(ask_structured):
                payload = ask_structured(
                    prompt,
                    _QUALITY_SCHEMA,
                    system=_QUALITY_JUDGE_SYSTEM,
                    schema_name="cascade_quality_score",
                )
            else:
                payload = self._judge_provider.ask_single(prompt, system=_QUALITY_JUDGE_SYSTEM)
            return _parse_quality_payload(payload)
        except Exception as exc:
            return _QualityScore(error=str(exc))

    def _capture_diff(self, worktree_path: str) -> str:
        """A capped `git diff HEAD` of the worktree -- the model's uncommitted
        changes (the competition agents edit files without committing, so a model
        that committed its own work would be the rare case that shows empty here).
        """
        if not worktree_path or not Path(worktree_path).is_dir():
            return ""
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if result.returncode != 0:
            return ""
        diff = result.stdout or ""
        if len(diff) > self._diff_char_cap:
            diff = diff[: self._diff_char_cap] + "\n...[diff truncated]...\n"
        return diff
