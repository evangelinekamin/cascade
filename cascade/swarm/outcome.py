"""Shared terminal outcomes for verified orchestration runs."""

from enum import Enum


class RunOutcome(str, Enum):
    """Truthful user-facing result of an orchestration run."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
