"""Tests for the sequential verified pipeline orchestrator."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import cascade.swarm.pipeline as pipe
from cascade.swarm.pipeline import (
    PipelineResult,
    PipelineTask,
    _parse_steps,
    _step_prompt,
    plan_steps,
    run_pipeline,
)


def test_parse_steps_valid_json():
    resp = (
        '{"steps": [{"id": "s1", "description": "add fn", "prompt": "write f()"},'
        ' {"id": "s2", "description": "wire it", "prompt": "call f()"}]}'
    )
    steps = _parse_steps("obj", resp)
    assert [s.id for s in steps] == ["s1", "s2"]
    assert steps[0].prompt == "write f()"


def test_parse_steps_fallback_on_non_json():
    steps = _parse_steps("build a thing", "I cannot do that")
    assert len(steps) == 1
    assert steps[0].prompt == "build a thing"


def test_parse_steps_fallback_on_empty():
    steps = _parse_steps("obj", '{"steps": []}')
    assert len(steps) == 1
    assert steps[0].description == "obj"


def test_parse_steps_replaces_unsafe_and_duplicate_director_ids():
    steps = _parse_steps(
        "obj",
        '{"steps": ['
        '{"id": "../escape", "description": "one", "prompt": "one"},'
        '{"id": "safe", "description": "two", "prompt": "two"},'
        '{"id": "safe", "description": "three", "prompt": "three"}'
        "]}",
    )
    assert [step.id for step in steps] == ["step_1", "safe", "step_3"]


def test_step_prompt_includes_objective_and_completed():
    task = PipelineTask("s2", "wire it", "call f()")
    done = [PipelineTask("s1", "add fn", "write f()")]
    prompt = _step_prompt("build X", task, done)
    assert "build X" in prompt
    assert "s1: add fn" in prompt
    assert "call f()" in prompt


def test_plan_steps_uses_frontier_model_and_parses():
    app = MagicMock()
    prov = MagicMock()
    prov.config = SimpleNamespace(model="full")
    used = {}

    def ask(prompt, system=None):
        used["model"] = prov.config.model
        return '{"steps": [{"id": "s1", "description": "do", "prompt": "do it"}]}'

    prov.ask_single = ask
    app.providers = {"openai": prov}
    app.config.get_model_for = MagicMock(
        side_effect=lambda n, mode_name=None, fast=False: "fast" if fast else "frontier"
    )
    app.config.get_escalation_target.return_value = None

    steps = plan_steps(app, "obj", "openai")

    assert used["model"] == "frontier"  # director runs on the frontier model
    assert prov.config.model == "full"  # restored afterward
    assert [s.id for s in steps] == ["s1"]


def test_plan_steps_fallback_on_exception():
    app = MagicMock()
    prov = MagicMock()
    prov.config = SimpleNamespace(model="full")
    prov.ask_single = MagicMock(side_effect=RuntimeError("boom"))
    app.providers = {"openai": prov}
    app.config.get_model_for = MagicMock(return_value="frontier")
    app.config.get_escalation_target.return_value = None

    steps = plan_steps(app, "build X", "openai")
    assert len(steps) == 1 and steps[0].prompt == "build X"


def _pipeline_app():
    app = MagicMock()
    app.config.get_default_provider.return_value = "openai"
    app.config.get_model_for = MagicMock(
        side_effect=lambda n, mode_name=None, fast=False: "fast" if fast else "frontier"
    )
    app.config.get_bulk_model.return_value = "bulk"
    app.config.get_escalation_target.return_value = None
    app.config.data = {}
    app.providers = {"openai": MagicMock()}
    return app


def test_run_pipeline_runs_each_step_in_one_worktree(monkeypatch):
    app = _pipeline_app()
    monkeypatch.setattr(
        pipe, "plan_steps",
        lambda app, obj, prov, on_progress=None, on_tokens=None: [
            PipelineTask("s1", "step one", "do one"),
            PipelineTask("s2", "step two", "do two"),
        ],
    )

    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt-pipe")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="2 files", diff_excerpt="+x", changed_files=("a.py", "b.py")
    )
    fm.diff_patch.side_effect = ["", "patch-1", "patch-1", "patch-2"]
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)

    seen_paths = []

    def fake_task(provider, worktree_path, prompt, test_cmd, **kw):
        seen_paths.append(worktree_path)
        return SimpleNamespace(passed=True, iterations=1), ["fast"], ["openai"]

    monkeypatch.setattr(pipe, "run_verified_task", fake_task)

    result = run_pipeline(app, "build X")

    assert isinstance(result, PipelineResult)
    assert [s.id for s in result.steps] == ["s1", "s2"]
    assert result.passed is True
    assert result.outcome == "succeeded"
    assert result.worktree_path == "/tmp/wt-pipe"
    assert result.changed_files == ("a.py", "b.py")
    # both steps ran in the SAME shared worktree
    assert seen_paths == ["/tmp/wt-pipe", "/tmp/wt-pipe"]


def test_run_pipeline_requires_every_step_to_pass(monkeypatch):
    app = _pipeline_app()
    monkeypatch.setattr(
        pipe, "plan_steps",
        lambda *a, **k: [PipelineTask("s1", "one", "one"), PipelineTask("s2", "two", "two")],
    )
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=("a.py",)
    )
    fm.diff_patch.side_effect = ["", "patch-1", "patch-1", "patch-2"]
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)
    outcomes = iter([
        (SimpleNamespace(passed=True, iterations=1), ["m"], ["openai"]),
        (SimpleNamespace(passed=False, iterations=3), ["m"], ["openai"]),
    ])
    monkeypatch.setattr(pipe, "run_verified_task", lambda *a, **k: next(outcomes))

    result = run_pipeline(app, "x")
    assert result.passed is False  # final step failed -> pipeline not green
    assert result.steps[0].passed is True
    assert result.steps[1].passed is False
    assert result.outcome == "partial"


def test_run_pipeline_stops_after_first_failed_step(monkeypatch):
    app = _pipeline_app()
    monkeypatch.setattr(
        pipe,
        "plan_steps",
        lambda *a, **k: [
            PipelineTask("s1", "fails", "one"),
            PipelineTask("s2", "would pass", "two"),
        ],
    )
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.diff_patch.side_effect = ["", "broken-patch"]
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=("a.py",)
    )
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)
    worker = MagicMock(
        return_value=(SimpleNamespace(passed=False, iterations=3, error="red"), ["m"], ["p"])
    )
    monkeypatch.setattr(pipe, "run_verified_task", worker)

    result = run_pipeline(app, "x")

    assert result.passed is False
    assert result.outcome == "failed"
    assert [step.id for step in result.steps] == ["s1"]
    worker.assert_called_once()
    assert "s1" in result.error


def test_run_pipeline_rejects_a_noop_step(monkeypatch):
    app = _pipeline_app()
    monkeypatch.setattr(
        pipe, "plan_steps", lambda *a, **k: [PipelineTask("s1", "noop", "do it")]
    )
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.diff_patch.side_effect = ["same", "same"]
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)
    monkeypatch.setattr(
        pipe,
        "run_verified_task",
        lambda *a, **k: (SimpleNamespace(passed=True, iterations=1, error=""), ["m"], ["p"]),
    )

    result = run_pipeline(app, "x")

    assert result.passed is False
    assert result.steps[0].changed is False
    assert "no changes" in result.error


def test_run_pipeline_rejects_green_sentinel_verification(monkeypatch):
    app = _pipeline_app()
    app.config.data = {"workflows": {"verify": {"test": "true"}}}
    monkeypatch.setattr(
        pipe, "plan_steps", lambda *a, **k: [PipelineTask("s1", "edit", "do it")]
    )
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.diff_patch.side_effect = ["", "patch"]
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=("a.py",)
    )
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)
    monkeypatch.setattr(
        pipe,
        "run_verified_task",
        lambda *a, **k: (
            SimpleNamespace(passed=True, iterations=1, error=""),
            ["m"],
            ["p"],
        ),
    )

    result = run_pipeline(app, "x")

    assert result.passed is False
    assert result.verification_kind == "sentinel"
    assert "does not prove behavior" in result.error


def test_run_pipeline_uses_bulk_model_not_ultrafast(monkeypatch):
    app = _pipeline_app()
    app.config.get_bulk_model.return_value = "deepseek-bulk"
    monkeypatch.setattr(
        pipe, "plan_steps", lambda *a, **k: [PipelineTask("s1", "one", "one")]
    )
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.diff_patch.side_effect = ["", "patch"]
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=("a.py",)
    )
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)
    captured = {}

    def fake_task(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(passed=True, iterations=1, error=""), ["m"], ["p"]

    monkeypatch.setattr(pipe, "run_verified_task", fake_task)

    assert run_pipeline(app, "x").passed is True
    assert captured["bulk_model"] == "deepseek-bulk"


def test_run_pipeline_missing_provider():
    app = MagicMock()
    app.config.get_default_provider.return_value = "ghost"
    app.providers = {}
    result = run_pipeline(app, "x")
    assert result.passed is False
    assert "not available" in result.error


def test_pipeline_counts_planner_and_worker_tokens(monkeypatch):
    app = _pipeline_app()

    def fake_plan(app, objective, provider_name, on_progress=None, on_tokens=None):
        on_tokens(7, 2)
        return [PipelineTask("s1", "one", "one")]

    monkeypatch.setattr(pipe, "plan_steps", fake_plan)
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.diff_patch.side_effect = ["", "patch"]
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=("a.py",)
    )
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)

    def fake_worker(*args, on_tokens=None, **kwargs):
        on_tokens(20, 5)
        return SimpleNamespace(passed=True, iterations=1, error=""), ["m"], ["p"]

    monkeypatch.setattr(pipe, "run_verified_task", fake_worker)

    result = run_pipeline(app, "x")

    assert (result.input_tokens, result.output_tokens) == (27, 7)
