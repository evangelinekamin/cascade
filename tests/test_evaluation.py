from pathlib import Path

import pytest

from cascade.evaluation import (
    EvalTask,
    load_eval_manifest,
    run_evaluation,
    select_eval_tasks,
)


def test_real_fixture_evaluation_verifies_result(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "before.txt").write_text("before")
    task = EvalTask(
        id="write-result",
        prompt="Create result.txt",
        fixture=str(fixture),
        verify=("test -f result.txt",),
        expected_files=("result.txt",),
    )

    def executor(_task, repo: Path, _provider, _mode):
        (repo / "result.txt").write_text("done")
        return {
            "outcome": "succeeded",
            "workflow": "solve",
            "provider": "fake",
            "worktree_path": str(repo),
            "changed_files": ["result.txt"],
            "duration_seconds": 0.1,
            "tool_metrics_available": True,
            "tool_calls": 3,
            "tool_errors": 0,
            "duplicate_reads": 1,
        }

    report = run_evaluation(
        (task,), executor=executor, temp_parent=tmp_path
    )

    assert report.passed == report.total == 1
    assert report.results[0].verification[0]["passed"]
    assert report.results[0].changed_files == ("result.txt",)
    assert report.results[0].tool_calls == 3


def test_manifest_rejects_duplicate_ids(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = tmp_path / "eval.yaml"
    manifest.write_text(
        "tasks:\n"
        f"  - {{id: same, fixture: {fixture}, prompt: one, verify: []}}\n"
        f"  - {{id: same, fixture: {fixture}, prompt: two, verify: []}}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_manifest(manifest)


def test_failed_run_carries_agent_reason_into_report(tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    task = EvalTask(
        id="blocked",
        prompt="Do the task",
        fixture=str(fixture),
        verify=(),
    )

    report = run_evaluation(
        (task,),
        executor=lambda *_args: {
            "outcome": "blocked",
            "workflow": "solve",
            "text": "Blocked because no provider was available.",
        },
        temp_parent=tmp_path,
    )

    assert not report.results[0].passed
    assert "no provider" in report.results[0].error


def test_manifest_protects_verification_files_by_default(tmp_path):
    fixture = tmp_path / "fixture"
    tests = fixture / "tests"
    tests.mkdir(parents=True)
    (tests / "test_result.py").write_text("def test_result(): pass\n")
    (fixture / "test_root.py").write_text("def test_root(): pass\n")
    (fixture / "pytest.ini").write_text("[pytest]\n")
    manifest = tmp_path / "eval.yaml"
    manifest.write_text(
        "tasks:\n"
        f"  - id: protected\n"
        f"    fixture: {fixture}\n"
        f"    prompt: do the work\n"
        f"    verify: [python3 -m pytest -q]\n"
    )

    task = load_eval_manifest(manifest)[0]

    assert task.protected_files == (
        "pytest.ini",
        "test_root.py",
        "tests/test_result.py",
    )


def test_evaluation_fails_when_agent_tampers_with_tests(tmp_path):
    fixture = tmp_path / "fixture"
    tests = fixture / "tests"
    tests.mkdir(parents=True)
    (tests / "test_result.py").write_text("def test_result(): assert True\n")
    task = EvalTask(
        id="tamper",
        prompt="Do the task",
        fixture=str(fixture),
        verify=("python3 -m pytest -q",),
        protected_files=("tests/test_result.py",),
    )

    def executor(_task, repo: Path, _provider, _mode):
        (repo / "tests" / "test_result.py").write_text(
            "def test_result(): assert True  # changed\n"
        )
        return {"outcome": "succeeded", "worktree_path": str(repo)}

    report = run_evaluation((task,), executor=executor, temp_parent=tmp_path)

    assert not report.results[0].passed
    assert report.results[0].modified_protected_files == ("tests/test_result.py",)
    assert "protected verification files changed" in report.results[0].error


def test_select_eval_tasks_preserves_manifest_order_and_rejects_unknown():
    tasks = tuple(
        EvalTask(id=task_id, prompt="x", fixture=".", verify=())
        for task_id in ("one", "two", "three")
    )

    assert [task.id for task in select_eval_tasks(tasks, ("three", "one"))] == [
        "one",
        "three",
    ]
    with pytest.raises(ValueError, match="unknown evaluation task"):
        select_eval_tasks(tasks, ("missing",))
