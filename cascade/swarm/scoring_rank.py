"""Pure scoring math for `/score`: metrics, disqualification, and ranking.

Everything here is a deterministic function or frozen dataclass with no I/O --
no subprocess, no provider calls, no filesystem. `cascade.swarm.scoring` (the
LLM/build side-effecting half) computes rows and hands them to `rank_leaderboard`;
this split keeps the pure logic trivially unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

# A cache-hit rate below this is treated as "no real caching benefit" -- a model
# cheap on paper but reading nothing from cache is out regardless of its score.
_CACHE_HIT_DISQUALIFY_PCT = 1.0

_GATE_RANK = {"pass": 2, "inconclusive": 1, "fail": 0}


def cache_hit_pct(cache_read: int, prompt_total: int) -> Optional[float]:
    """Share of prompt tokens served from cache, or None when it is unknown.

    None (not 0) when prompt_total was not recorded -- e.g. a manifest written
    before cache accounting existed -- so a missing measurement is never mistaken
    for a proven zero and used to disqualify a model.
    """
    if prompt_total <= 0:
        return None
    return 100.0 * cache_read / prompt_total


def tokens_per_second(
    output_tokens: Optional[int], duration_seconds: float,
) -> Optional[float]:
    """Output-token generation throughput, or None when it cannot be measured.

    Uses OUTPUT tokens only: a coding agent's prompt (input + cache reads) dwarfs
    its output, so counting the whole context would wildly inflate the rate.
    None when the output split is unknown (a manifest without prompt_total) or
    the duration is non-positive.
    """
    if output_tokens is None or duration_seconds <= 0:
        return None
    return output_tokens / duration_seconds


def disqualification(
    pct: Optional[float], success: bool, changed_files: "tuple[str, ...]",
) -> "tuple[bool, tuple[str, ...]]":
    """Whether a row is disqualified, and the notes explaining why.

    Two independent triggers, both recorded when both apply: a *measured*
    cache-hit rate below the threshold (no real caching benefit, regardless of
    how cheap the model looks on paper -- but an unknown rate, ``pct is None``,
    never disqualifies) and a no-op run (failed with no file changes -- nothing
    to evaluate).
    """
    notes: list[str] = []
    disqualified = False
    if pct is not None and pct < _CACHE_HIT_DISQUALIFY_PCT:
        disqualified = True
        notes.append(
            f"disqualified: {pct:.1f}% cache-hit "
            f"(below the {_CACHE_HIT_DISQUALIFY_PCT:.0f}% threshold)"
        )
    if not success and not changed_files:
        disqualified = True
        notes.append("no-op: produced no changes")
    return disqualified, tuple(notes)


@dataclass(frozen=True)
class ScoredCompetitor:
    """One ranked row: a competitor's objective gate, quality, and cost/speed."""

    label: str
    model: str
    gate: str  # "pass" | "fail" | "inconclusive"
    gate_output_tail: str
    spec_completeness: Optional[int]
    correctness: Optional[int]
    scope_discipline: Optional[int]
    quality_total: Optional[int]
    quality_summary: str
    cost: float
    tokens: int
    tok_per_s: Optional[float]
    cache_hit_pct: Optional[float]
    disqualified: bool
    changed_files: "tuple[str, ...]" = ()
    notes: "tuple[str, ...]" = ()


@dataclass(frozen=True)
class Leaderboard:
    """A ranked set of scored competitors plus the manifests they came from."""

    rows: "tuple[ScoredCompetitor, ...]" = ()
    manifest_paths: "tuple[str, ...]" = ()


def _quality_sort_key(quality_total: Optional[int]) -> "tuple[bool, int]":
    """Descending by quality total; None (judge unavailable) always sorts last."""
    if quality_total is None:
        return (True, 0)
    return (False, -quality_total)


def _row_sort_key(row: ScoredCompetitor) -> tuple:
    return (
        row.disqualified,                      # non-disqualified sorts first
        -_GATE_RANK.get(row.gate, 0),           # pass > inconclusive > fail
        _quality_sort_key(row.quality_total),   # quality descending, None last
        row.cost,                               # cost ascending
        -(row.tok_per_s or 0.0),                # tok/s descending (unknown last)
    )


def rank_leaderboard(rows: Sequence[ScoredCompetitor]) -> "tuple[ScoredCompetitor, ...]":
    """Best-first ordering. Pure: no I/O, safe to unit test directly.

    Non-disqualified rows sort before disqualified ones; within each group the
    order is gate outcome, then quality total, then cost, then throughput --
    the same secondary keys apply on both sides of the disqualified split.
    """
    return tuple(sorted(rows, key=_row_sort_key))
