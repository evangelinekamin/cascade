"""Contract tests for VerifiedWorker -- the test-gated iterate-until-correct loop.

The loop is pure orchestration over three injected callables (run_agent,
run_tests, prepare_worktree), so its logic is testable without real providers
or subprocesses.
"""

from cascade.swarm.verify_loop import (
    VerifiedWorker,
    WorkerResult,
    VerifyAttempt,
    _compact_test_output,
)


def _worker(
    run_agent, run_tests, *, max_iterations=3, path="/tmp/wt",
    describe_changes=None, context="",
):
    return VerifiedWorker(
        run_agent=run_agent,
        run_tests=run_tests,
        prepare_worktree=lambda: path,
        max_iterations=max_iterations,
        describe_changes=describe_changes,
        context=context,
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


# --- Test-output compaction (feedback stays small on every retry) ---------------


def test_compact_test_output_passes_short_output_through():
    out = "1 failed, 3 passed in 0.2s"
    assert _compact_test_output(out, cap=4000) == out


def test_compact_test_output_keeps_the_failures_section():
    head = "collecting ...\n" + "NOISE_PREAMBLE line\n" * 2000  # long, irrelevant
    failures = (
        "=================================== FAILURES ===================================\n"
        "____________________________ test_widget ____________________________\n"
        "    assert result == 42\n"
        "E   assert 7 == 42\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_widget.py::test_widget - assert 7 == 42\n"
        "1 failed, 3 passed in 0.30s\n"
    )
    compacted = _compact_test_output(head + failures, cap=4000)

    assert "FAILURES" in compacted
    assert "E   assert 7 == 42" in compacted
    assert "1 failed, 3 passed" in compacted
    assert "NOISE_PREAMBLE" not in compacted  # the verbose preamble is dropped
    assert len(compacted) <= 4000


def test_compact_test_output_falls_back_to_tail_without_a_section():
    body = "".join(f"line {i}\n" for i in range(5000))
    tail_marker = "FINAL_SUMMARY_LINE 2 failed\n"
    compacted = _compact_test_output(body + tail_marker, cap=1000)

    assert tail_marker.strip() in compacted  # the salient tail is preserved
    assert len(compacted) <= 1000 + len(tail_marker)


def test_retry_prompt_uses_compacted_output_not_the_full_dump():
    huge_failure = "NOISE_PREAMBLE line\n" * 5000 + (
        "=================================== FAILURES ===================================\n"
        "E   assert TOKEN_XYZ\n"
        "1 failed in 0.1s\n"
    )
    prompts: list[str] = []

    def agent(prompt, path):
        prompts.append(prompt)
        return "edited"

    results = iter([(huge_failure, 1), ("ok", 0)])
    worker = _worker(agent, lambda path: next(results))
    worker.run("fix parser")

    retry = prompts[1]
    assert "TOKEN_XYZ" in retry  # the salient failure reached the agent
    assert "NOISE_PREAMBLE" not in retry  # the verbose dump did not
    assert len(retry) < len(huge_failure)


# --- Light iteration memory (build on prior work, don't restart cold) -----------


def test_retry_prompt_includes_change_summary_when_available():
    prompts: list[str] = []

    def agent(prompt, path):
        prompts.append(prompt)
        return "edited"

    results = iter([("FAILED", 1), ("ok", 0)])
    worker = _worker(
        agent,
        lambda path: next(results),
        describe_changes=lambda path: "SENTINEL_DIFFSTAT feature.py | 12 +++++",
    )
    worker.run("build feature")

    assert "SENTINEL_DIFFSTAT" not in prompts[0]  # nothing changed yet on iter 1
    assert "SENTINEL_DIFFSTAT" in prompts[1]  # iter 2 is told what already changed


def test_change_summary_is_omitted_when_describer_returns_empty():
    prompts: list[str] = []

    def agent(prompt, path):
        prompts.append(prompt)
        return "edited"

    results = iter([("FAILED thing", 1), ("ok", 0)])
    worker = _worker(
        agent,
        lambda path: next(results),
        describe_changes=lambda path: "",  # e.g. no diff / not a git repo
    )
    worker.run("build feature")

    # A retry still happens and still carries the failure, just no memory block.
    assert "FAILED thing" in prompts[1]


def test_change_summary_describer_errors_do_not_break_the_loop():
    def boom(path):
        raise RuntimeError("git exploded")

    results = iter([("FAILED", 1), ("ok", 0)])
    worker = _worker(
        lambda prompt, path: "edited",
        lambda path: next(results),
        describe_changes=boom,
    )
    result = worker.run("build feature")
    assert result.passed is True  # a broken describer must not sink the run


# --- Conversation context (referential tasks keep their referent) ---------------


def test_context_leads_the_initial_prompt():
    prompts: list[str] = []

    def agent(prompt, path):
        prompts.append(prompt)
        return "edited"

    worker = _worker(
        agent, lambda path: ("ok", 0),
        context="[Prior conversation]\ncodex: CTX_REFERENT the errors are in x.ts",
    )
    worker.run("fix the errors codex found")

    assert prompts[0].startswith("[Prior conversation]")
    assert "CTX_REFERENT" in prompts[0]
    assert "fix the errors codex found" in prompts[0]  # task still present


def test_context_persists_across_retries():
    prompts: list[str] = []

    def agent(prompt, path):
        prompts.append(prompt)
        return "edited"

    results = iter([("FAILED_ONCE", 1), ("ok", 0)])
    worker = _worker(agent, lambda path: next(results), context="CTX_REFERENT block")
    worker.run("do it")

    # The referent must survive into the retry, not just the first attempt.
    assert "CTX_REFERENT" in prompts[0]
    assert "CTX_REFERENT" in prompts[1]
    assert "FAILED_ONCE" in prompts[1]  # retry still carries the failure too


def test_empty_context_leaves_the_prompt_unchanged():
    prompts: list[str] = []

    def agent(prompt, path):
        prompts.append(prompt)
        return "edited"

    worker = _worker(agent, lambda path: ("ok", 0), context="")
    worker.run("plain task")

    assert prompts[0].startswith("Task:")  # no context preamble prepended
