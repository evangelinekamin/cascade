"""Contract tests for VerifiedWorker -- the test-gated iterate-until-correct loop.

The loop is pure orchestration over three injected callables (run_agent,
run_tests, prepare_worktree), so its logic is testable without real providers
or subprocesses.
"""

from cascade.swarm.verify_loop import VerifiedWorker, WorkerResult, VerifyAttempt


def _worker(run_agent, run_tests, *, max_iterations=3, path="/tmp/wt"):
    return VerifiedWorker(
        run_agent=run_agent,
        run_tests=run_tests,
        prepare_worktree=lambda: path,
        max_iterations=max_iterations,
    )


def test_passes_on_first_iteration_calls_agent_once():
    calls = []

    def agent(prompt, path):
        calls.append((prompt, path))
        return "made the change"

    worker = _worker(agent, lambda path: ("all tests passed", 0))
    result = worker.run("add a feature")

    assert isinstance(result, WorkerResult)
    assert result.passed is True
    assert result.iterations == 1
    assert len(calls) == 1
    assert result.worktree_path == "/tmp/wt"
    assert result.attempts[0].passed is True


def test_fixes_on_second_iteration():
    test_results = iter([("FAILED test_x", 1), ("ok", 0)])
    worker = _worker(lambda prompt, path: "edited", lambda path: next(test_results))

    result = worker.run("fix the bug")

    assert result.passed is True
    assert result.iterations == 2
    assert len(result.attempts) == 2
    assert result.attempts[0].passed is False
    assert result.attempts[1].passed is True


def test_gives_up_after_max_iterations():
    worker = _worker(
        lambda prompt, path: "edited",
        lambda path: ("still FAILED", 1),
        max_iterations=2,
    )
    result = worker.run("an impossible task")

    assert result.passed is False
    assert result.iterations == 2
    assert all(a.passed is False for a in result.attempts)


def test_failure_output_is_fed_into_the_retry_prompt():
    prompts = []

    def agent(prompt, path):
        prompts.append(prompt)
        return "edited"

    test_results = iter([("UNIQUE_FAILURE_TOKEN_42", 1), ("ok", 0)])
    worker = _worker(agent, lambda path: next(test_results))

    worker.run("implement parser")

    assert "implement parser" in prompts[0]
    # the retry prompt must carry the prior failure so the agent can fix it
    assert "UNIQUE_FAILURE_TOKEN_42" in prompts[1]


def test_attempt_records_agent_response_and_output():
    worker = _worker(lambda prompt, path: "I changed foo.py", lambda path: ("pytest: 1 passed", 0))
    result = worker.run("touch foo")

    attempt = result.attempts[0]
    assert isinstance(attempt, VerifyAttempt)
    assert attempt.agent_response == "I changed foo.py"
    assert attempt.test_output == "pytest: 1 passed"
    assert attempt.iteration == 1


def test_on_attempt_called_once_per_iteration():
    seen = []
    test_results = iter([("fail", 1), ("ok", 0)])
    worker = _worker(lambda prompt, path: "edited", lambda path: next(test_results))

    worker.run("task", on_attempt=lambda a: seen.append((a.iteration, a.passed)))

    assert seen == [(1, False), (2, True)]
