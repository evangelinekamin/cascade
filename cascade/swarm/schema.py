"""Data types for competitive multi-provider execution."""

from dataclasses import dataclass, field


@dataclass
class CompetitionEntry:
    """Result from running the same task against one competitor.

    ``provider`` is the competitor label (a provider name, or a model label when
    several models compete through one provider). ``model`` records the exact
    model that ran, and ``cost`` the provider-reported USD when available
    (OpenRouter); both feed the per-competitor tokens/cost/speed comparison.
    """

    provider: str
    response: str
    tokens: int = 0
    duration_seconds: float = 0.0
    success: bool = True
    error: str = ""
    worktree_path: str = ""
    changed_files: list[str] = field(default_factory=list)
    diff_stat: str = ""
    diff_excerpt: str = ""
    retained: bool = False
    model: str = ""
    cost: float = 0.0


@dataclass(frozen=True)
class CompetitionJudgment:
    """Judge output for a competition run."""

    winner_provider: str
    rationale: str
    summary: str = ""


@dataclass
class CompetitionResult:
    """Final result of a parallel competition across providers."""

    objective: str
    entries: list[CompetitionEntry] = field(default_factory=list)
    judgment: CompetitionJudgment | None = None
    winner_provider: str = ""
    winner_response: str = ""
    total_tokens: int = 0
    judge_provider: str = ""
