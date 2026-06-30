"""Competitive multi-provider execution with judge-based winner selection."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, TYPE_CHECKING

from .schema import CompetitionEntry, CompetitionJudgment, CompetitionResult
from .workspace import run_agent_in_worktree
from .worktree import WorktreeManager, WorktreeSnapshot

if TYPE_CHECKING:
    from ..cli import CascadeCore


_JUDGE_SYSTEM = """\
You are a strict evaluator comparing multiple model outputs for the same task.

Choose exactly one winner from the successful provider results.
Prioritize correctness first, then completeness, then practical usefulness.

Respond with JSON only:
{
  "winner_provider": "provider_name",
  "rationale": "Why this response won",
  "summary": "Optional concise synthesis of the winning answer"
}
"""

_CODE_JUDGE_SYSTEM = """\
You are a strict evaluator comparing multiple coding-agent runs for the same task.

Choose exactly one winner from the successful provider results.
Prioritize the actual file changes first, then correctness, then scope control,
then stated verification, then response clarity.

Use the changed files and diff excerpts more than the agent's self-report.

Respond with JSON only:
{
  "winner_provider": "provider_name",
  "rationale": "Why this coding run won",
  "summary": "Optional concise synthesis of the winning changes"
}
"""

_CODE_SYSTEM = """\
You are competing against other coding agents in an isolated git worktree.

Make the requested code changes directly in the provided workspace.
Stay inside that workspace, keep the change set focused, and do not ask for
confirmation. If you cannot complete the task, explain exactly what blocked you.

Your final response must summarize:
- files changed
- what was implemented
- any verification you ran
- any remaining risks
"""

_JUDGE_PREFERENCE = ("claude", "gemini", "openai", "openrouter")
ProgressCallback = Optional[Callable[[str, str], None]]


class CompetitionOrchestrator:
    """Run the same task across providers and judge the best result."""

    def __init__(
        self,
        app: "CascadeCore",
        judge_provider: Optional[str] = None,
        max_workers: int = 4,
    ) -> None:
        self._app = app
        self._judge_provider = judge_provider or self._pick_judge_provider()
        self._max_workers = max_workers

    def _pick_judge_provider(self) -> str:
        """Pick the best available provider for judging."""
        for name in _JUDGE_PREFERENCE:
            if name in self._app.providers:
                return name
        available = list(self._app.providers.keys())
        if not available:
            raise RuntimeError("No providers available for competition")
        return available[0]

    @staticmethod
    def _result_blocks(entries: list[CompetitionEntry]) -> str:
        blocks = []
        for entry in entries:
            sections = [f"[{entry.provider}]"]
            if entry.success:
                sections.append(f"Response:\n{entry.response or '[no response]'}")
            else:
                sections.append(f"Response:\nFAILED: {entry.error}")
            if entry.changed_files:
                sections.append("Changed files:\n" + "\n".join(entry.changed_files))
            if entry.diff_stat:
                sections.append(f"Diff stat:\n{entry.diff_stat}")
            if entry.diff_excerpt:
                sections.append(f"Diff excerpt:\n{entry.diff_excerpt}")
            blocks.append("\n\n".join(sections))
        return "\n\n---\n\n".join(blocks)

    def _execute_provider(
        self,
        provider_name: str,
        objective: str,
        on_progress: ProgressCallback = None,
    ) -> CompetitionEntry:
        """Run one provider against the shared objective."""
        if on_progress:
            on_progress("competing", f"[{provider_name}] running")

        provider = self._app.providers.get(provider_name)
        if provider is None:
            return CompetitionEntry(
                provider=provider_name,
                response="",
                success=False,
                error=f"Provider '{provider_name}' not available",
            )

        start = time.monotonic()
        try:
            response = provider.ask_single(objective)
            usage = provider.last_usage or (0, 0)
            return CompetitionEntry(
                provider=provider_name,
                response=response,
                tokens=usage[0] + usage[1],
                duration_seconds=time.monotonic() - start,
                success=True,
            )
        except Exception as exc:
            return CompetitionEntry(
                provider=provider_name,
                response="",
                tokens=0,
                duration_seconds=time.monotonic() - start,
                success=False,
                error=str(exc),
            )

    @staticmethod
    def _parse_judgment(
        response: str,
        successful: set[str],
    ) -> Optional[CompetitionJudgment]:
        """Parse the judge's JSON response."""
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

        winner_provider = str(data.get("winner_provider", "")).strip()
        if winner_provider not in successful:
            return None

        rationale = str(data.get("rationale", "")).strip()
        summary = str(data.get("summary", "")).strip()
        return CompetitionJudgment(
            winner_provider=winner_provider,
            rationale=rationale or f"{winner_provider} selected by judge.",
            summary=summary,
        )

    def _fallback_judgment(self, entries: list[CompetitionEntry]) -> CompetitionJudgment:
        """Choose a winner when the judge response is invalid or unavailable."""
        successful = [entry for entry in entries if entry.success]
        if not successful:
            return CompetitionJudgment(
                winner_provider="",
                rationale="No providers produced a successful response.",
                summary="",
            )

        by_name = {entry.provider: entry for entry in successful}
        for provider_name in _JUDGE_PREFERENCE:
            if provider_name in by_name:
                return CompetitionJudgment(
                    winner_provider=provider_name,
                    rationale=(
                        "Judge result unavailable; selected the highest-priority "
                        "successful provider."
                    ),
                    summary="",
                )

        winner = successful[0]
        return CompetitionJudgment(
            winner_provider=winner.provider,
            rationale="Judge result unavailable; selected the first successful provider.",
            summary="",
        )

    def _judge(
        self,
        objective: str,
        entries: list[CompetitionEntry],
        on_progress: ProgressCallback = None,
        judge_system: str = _JUDGE_SYSTEM,
    ) -> CompetitionJudgment:
        """Run the judge pass over all provider results."""
        successful = {entry.provider for entry in entries if entry.success}
        if not successful:
            return self._fallback_judgment(entries)
        if len(successful) == 1:
            winner = next(iter(successful))
            return CompetitionJudgment(
                winner_provider=winner,
                rationale="Only one provider completed successfully.",
                summary="",
            )

        if on_progress:
            on_progress("judging", f"Judge ({self._judge_provider}) comparing outputs...")

        judge_provider = self._app.providers.get(self._judge_provider)
        if judge_provider is None:
            return self._fallback_judgment(entries)

        prompt = (
            f"Original objective:\n{objective}\n\n"
            f"Competition results:\n\n{self._result_blocks(entries)}"
        )

        try:
            response = judge_provider.ask_single(prompt, system=judge_system)
        except Exception:
            return self._fallback_judgment(entries)

        judgment = self._parse_judgment(response, successful)
        if judgment is None:
            return self._fallback_judgment(entries)
        return judgment

    @staticmethod
    def _safe_snapshot(manager: WorktreeManager, worktree_path: str) -> WorktreeSnapshot:
        try:
            return manager.capture_snapshot(worktree_path)
        except Exception:
            return WorktreeSnapshot()

    @staticmethod
    def _snapshot_fields(snapshot: WorktreeSnapshot) -> dict:
        return {
            "changed_files": list(snapshot.changed_files),
            "diff_stat": snapshot.diff_stat,
            "diff_excerpt": snapshot.diff_excerpt,
        }

    @staticmethod
    def _has_code_changes(snapshot: WorktreeSnapshot) -> bool:
        return bool(snapshot.changed_files or snapshot.diff_stat or snapshot.diff_excerpt)

    def _build_competition_system(self, extra: str = "") -> Optional[str]:
        pipeline = getattr(self._app, "prompt_pipeline", None)
        base = ""
        if pipeline is not None:
            try:
                built = pipeline.build()
                if isinstance(built, str):
                    base = built
            except Exception:
                base = ""
        parts = [part.strip() for part in (base, extra) if part and part.strip()]
        if not parts:
            return None
        return "\n\n".join(parts)

    @staticmethod
    def _build_code_prompt(objective: str, worktree_path: str) -> str:
        return (
            f"Task:\n{objective}\n\n"
            f"Workspace:\n{worktree_path}\n\n"
            "This workspace is an isolated git worktree prepared from the current repo state,\n"
            "including uncommitted tracked changes and untracked files from the source tree.\n"
            "Make your changes directly in this workspace. Prefer absolute paths rooted here if\n"
            "you call file tools."
        )

    def _execute_code_provider(
        self,
        provider_name: str,
        objective: str,
        worktree_path: str,
        manager: WorktreeManager,
        on_progress: ProgressCallback = None,
    ) -> CompetitionEntry:
        """Run one provider inside an isolated git worktree."""
        if on_progress:
            on_progress("competing", f"[{provider_name}] coding in {worktree_path}")

        provider = self._app.providers.get(provider_name)
        if provider is None:
            return CompetitionEntry(
                provider=provider_name,
                response="",
                success=False,
                error=f"Provider '{provider_name}' not available",
                worktree_path=worktree_path,
            )

        start = time.monotonic()
        response = ""
        try:
            system = self._build_competition_system(_CODE_SYSTEM)
            prompt = self._build_code_prompt(objective, worktree_path)
            response = run_agent_in_worktree(provider, prompt, worktree_path, system=system)
            usage = provider.last_usage or (0, 0)
            snapshot = self._safe_snapshot(manager, worktree_path)
            if not self._has_code_changes(snapshot):
                return CompetitionEntry(
                    provider=provider_name,
                    response=response,
                    tokens=usage[0] + usage[1],
                    duration_seconds=time.monotonic() - start,
                    success=False,
                    error="no changes produced",
                    worktree_path=worktree_path,
                    **self._snapshot_fields(snapshot),
                )
            return CompetitionEntry(
                provider=provider_name,
                response=response,
                tokens=usage[0] + usage[1],
                duration_seconds=time.monotonic() - start,
                success=True,
                worktree_path=worktree_path,
                **self._snapshot_fields(snapshot),
            )
        except Exception as exc:
            snapshot = self._safe_snapshot(manager, worktree_path)
            return CompetitionEntry(
                provider=provider_name,
                response=response,
                tokens=0,
                duration_seconds=time.monotonic() - start,
                success=False,
                error=str(exc),
                worktree_path=worktree_path,
                **self._snapshot_fields(snapshot),
            )

    @staticmethod
    def _apply_retention(entries: list[CompetitionEntry], winner_provider: str) -> None:
        if not entries:
            return
        if not winner_provider:
            for entry in entries:
                entry.retained = True
            return
        for entry in entries:
            entry.retained = entry.provider == winner_provider
            if not entry.retained:
                entry.worktree_path = ""

    def execute(
        self,
        objective: str,
        providers: Optional[list[str]] = None,
        on_progress: ProgressCallback = None,
    ) -> CompetitionResult:
        """Run the same objective across providers and judge a winner."""
        provider_names = providers or list(self._app.providers.keys())
        if len(provider_names) < 2:
            raise RuntimeError("Competition needs at least two providers")

        entries: list[CompetitionEntry] = []
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(provider_names))) as pool:
            futures = {
                pool.submit(
                    self._execute_provider,
                    provider_name,
                    objective,
                    on_progress,
                ): provider_name
                for provider_name in provider_names
            }
            for future in as_completed(futures):
                entry = future.result()
                entries.append(entry)
                if on_progress:
                    status = "done" if entry.success else f"failed: {entry.error}"
                    on_progress(
                        "result",
                        f"[{entry.provider}] {status} ({entry.duration_seconds:.1f}s)",
                    )

        entries.sort(key=lambda entry: provider_names.index(entry.provider))
        judgment = self._judge(objective, entries, on_progress)

        winner_response = ""
        if judgment.winner_provider:
            for entry in entries:
                if entry.provider == judgment.winner_provider:
                    winner_response = entry.response
                    break

        return CompetitionResult(
            objective=objective,
            entries=entries,
            judgment=judgment,
            winner_provider=judgment.winner_provider,
            winner_response=winner_response,
            total_tokens=sum(entry.tokens for entry in entries),
            judge_provider=self._judge_provider,
        )

    def execute_code(
        self,
        objective: str,
        providers: Optional[list[str]] = None,
        on_progress: ProgressCallback = None,
    ) -> CompetitionResult:
        """Run a code-edit competition in isolated git worktrees."""
        provider_names = providers or list(self._app.providers.keys())
        if len(provider_names) < 2:
            raise RuntimeError("Competition needs at least two providers")

        manager = WorktreeManager()
        prepared_paths: dict[str, str] = {}
        entries: list[CompetitionEntry] = []

        for provider_name in provider_names:
            try:
                prepared = manager.prepare(provider_name)
                prepared_paths[provider_name] = prepared.path
                if on_progress:
                    on_progress("workspace", f"[{provider_name}] {prepared.path}")
            except Exception as exc:
                entries.append(
                    CompetitionEntry(
                        provider=provider_name,
                        response="",
                        success=False,
                        error=f"worktree setup failed: {exc}",
                    )
                )

        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(prepared_paths) or 1)) as pool:
            futures = {
                pool.submit(
                    self._execute_code_provider,
                    provider_name,
                    objective,
                    worktree_path,
                    manager,
                    on_progress,
                ): provider_name
                for provider_name, worktree_path in prepared_paths.items()
            }
            for future in as_completed(futures):
                entry = future.result()
                entries.append(entry)
                if on_progress:
                    status = "done" if entry.success else f"failed: {entry.error}"
                    on_progress(
                        "result",
                        f"[{entry.provider}] {status} ({entry.duration_seconds:.1f}s)",
                    )

        entries.sort(key=lambda entry: provider_names.index(entry.provider))
        judgment = self._judge(
            objective,
            entries,
            on_progress,
            judge_system=_CODE_JUDGE_SYSTEM,
        )

        winner_response = ""
        if judgment.winner_provider:
            for entry in entries:
                if entry.provider == judgment.winner_provider:
                    winner_response = entry.response
                    break

        if judgment.winner_provider:
            manager.cleanup(keep_provider=judgment.winner_provider)
        self._apply_retention(entries, judgment.winner_provider)

        return CompetitionResult(
            objective=objective,
            entries=entries,
            judgment=judgment,
            winner_provider=judgment.winner_provider,
            winner_response=winner_response,
            total_tokens=sum(entry.tokens for entry in entries),
            judge_provider=self._judge_provider,
        )
