"""Swarm system: verified orchestration lanes and competitive execution.

The verified lanes (solve, pipeline, fanout) build in isolated git
worktrees and gate results behind tests. Competitive execution runs the
same task against multiple providers and judges the winner.
"""

from .schema import (
    CompetitionEntry,
    CompetitionJudgment,
    CompetitionResult,
)
from .competition import CompetitionOrchestrator

__all__ = [
    "CompetitionEntry",
    "CompetitionJudgment",
    "CompetitionOrchestrator",
    "CompetitionResult",
]
