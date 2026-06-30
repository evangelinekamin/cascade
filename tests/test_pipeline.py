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

    steps = plan_steps(app, "build X", "openai")
    assert len(steps) == 1 and steps[0].prompt == "build X"


def _pipeline_app():
    app = MagicMock()
    app.config.get_default_provider.return_value = "openai"
    app.config.get_model_for = MagicMock(
        side_effect=lambda n, mode_name=None, fast=False: "fast" if fast else "frontier"
    )
    app.config.data = {}
    app.providers = {"openai": MagicMock()}
    return app


def test_run_pipeline_runs_each_step_in_one_worktree(monkeypatch):
    app = _pipeline_app()
    monkeypatch.setattr(
        pipe, "plan_steps",
        lambda app, obj, prov, on_progress=None: [
            PipelineTask("s1", "step one", "do one"),
            PipelineTask("s2", "step two", "do two"),
        ],
    )

    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt-pipe")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="2 files", diff_excerpt="+x", changed_files=("a.py", "b.py")
    )
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)

    seen_paths = []

    def fake_task(provider, worktree_path, prompt, test_cmd, **kw):
        seen_paths.append(worktree_path)
        return SimpleNamespace(passed=True, iterations=1), ["fast"]

    monkeypatch.setattr(pipe, "run_verified_task", fake_task)

    result = run_pipeline(app, "build X")

    assert isinstance(result, PipelineResult)
    assert [s.id for s in result.steps] == ["s1", "s2"]
    assert result.passed is True
    assert result.worktree_path == "/tmp/wt-pipe"
    assert result.changed_files == ("a.py", "b.py")
    # both steps ran in the SAME shared worktree
    assert seen_paths == ["/tmp/wt-pipe", "/tmp/wt-pipe"]


def test_run_pipeline_passed_reflects_final_step(monkeypatch):
    app = _pipeline_app()
    monkeypatch.setattr(
        pipe, "plan_steps",
        lambda *a, **k: [PipelineTask("s1", "one", "one"), PipelineTask("s2", "two", "two")],
    )
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)
    outcomes = iter([
        (SimpleNamespace(passed=True, iterations=1), ["m"]),
        (SimpleNamespace(passed=False, iterations=3), ["m"]),
    ])
    monkeypatch.setattr(pipe, "run_verified_task", lambda *a, **k: next(outcomes))

    result = run_pipeline(app, "x")
    assert result.passed is False  # final step failed -> pipeline not green
    assert result.steps[0].passed is True
    assert result.steps[1].passed is False


def test_run_pipeline_missing_provider():
    app = MagicMock()
    app.config.get_default_provider.return_value = "ghost"
    app.providers = {}
    result = run_pipeline(app, "x")
    assert result.passed is False
    assert "not available" in result.error
