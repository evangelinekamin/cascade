"""VerifiedWorker: a test-gated iterate-until-correct loop.

Turns a single task into a verified diff. The loop is pure orchestration over
three injected callables, so its logic is independent of providers and the
filesystem:

    prepare_worktree() -> path
    run_agent(prompt, path) -> response          # agent edits files in the worktree
    run_tests(path) -> (output, returncode)      # 0 == pass

Each iteration the agent edits and the tests run; on failure the test output is
fed back into the next prompt, up to ``max_iterations``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(frozen=True)
class VerifyAttempt:
    """One pass through the loop."""

    iteration: int
    passed: bool
    test_output: str
    agent_response: str


@dataclass(frozen=True)
class WorkerResult:
    """Outcome of a VerifiedWorker run."""

    task: str
    passed: bool
    iterations: int
    attempts: tuple[VerifyAttempt, ...]
    worktree_path: str
    error: str = ""


RunAgent = Callable[[str, str], str]
RunTests = Callable[[str], "tuple[str, int]"]
PrepareWorktree = Callable[[], str]


_INITIAL_PROMPT = """\
Task:
{task}

Make the change directly in the workspace. The project's tests will be run to
verify your work, so make them pass.
"""

_RETRY_PROMPT = """\
Task:
{task}

Your previous attempt did not pass the tests. Here is the failing output:

{failure}

Fix the failures. Make the change directly in the workspace; the tests will be
run again to verify.
"""


class VerifiedWorker:
    """Run one task to a verified diff via a test-gated retry loop."""

    def __init__(
        self,
        run_agent: RunAgent,
        run_tests: RunTests,
        prepare_worktree: PrepareWorktree,
        max_iterations: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self._run_agent = run_agent
        self._run_tests = run_tests
        self._prepare_worktree = prepare_worktree
        self._max_iterations = max_iterations

    def run(
        self,
        task: str,
        on_attempt: Optional[Callable[[VerifyAttempt], None]] = None,
    ) -> WorkerResult:
        """Drive the loop until tests pass or iterations are exhausted.

        ``on_attempt`` is invoked with each VerifyAttempt as it completes, for
        progress reporting.
        """
        path = self._prepare_worktree()
        attempts: List[VerifyAttempt] = []

        for i in range(1, self._max_iterations + 1):
            prompt = self._build_prompt(task, attempts)
            response = self._run_agent(prompt, path)
            output, returncode = self._run_tests(path)
            attempt = VerifyAttempt(
                iteration=i,
                passed=returncode == 0,
                test_output=output,
                agent_response=response,
            )
            attempts.append(attempt)
            if on_attempt is not None:
                on_attempt(attempt)
            if returncode == 0:
                break

        return WorkerResult(
            task=task,
            passed=attempts[-1].passed,
            iterations=len(attempts),
            attempts=tuple(attempts),
            worktree_path=path,
        )

    @staticmethod
    def _build_prompt(task: str, attempts: "List[VerifyAttempt]") -> str:
        if not attempts:
            return _INITIAL_PROMPT.format(task=task)
        return _RETRY_PROMPT.format(task=task, failure=attempts[-1].test_output)
