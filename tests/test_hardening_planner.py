"""Hardening: lane context must reach the decomposition planner, not just workers.

A referential objective ("fix the errors codex found across the stack") is only
resolvable if the director that SPLITS it into steps/subtasks sees the prior
conversation -- otherwise the split happens blind and each worker gets the
context too late. These tests pin that the planner prompt carries the bounded
digest, that run_pipeline/run_fanout thread it in, and that /pipeline + /fanout
build and pass it the same way /solve does.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cascade.swarm.fanout as fanout_mod
import cascade.swarm.pipeline as pipe
from cascade.commands import CommandHandler
from cascade.swarm.fanout import FanoutTask, plan_subtasks, run_fanout
from cascade.swarm.pipeline import PipelineResult, PipelineTask, plan_steps, run_pipeline


# --- planner prompt carries the context -------------------------------------


def _plan_app():
    app = MagicMock()
    prov = MagicMock()
    prov.config = SimpleNamespace(model="full")
    app.providers = {"openai": prov}
    app.config.get_model_for = MagicMock(
        side_effect=lambda n, mode_name=None, fast=False: "fast" if fast else "frontier"
    )
    app.config.get_escalation_target.return_value = None
    return app, prov


def test_plan_steps_prepends_context_to_director_prompt():
    app, prov = _plan_app()
    seen = {}

    def ask(prompt, system=None):
        seen["prompt"] = prompt
        return '{"steps": [{"id": "s1", "description": "do", "prompt": "do it"}]}'

    prov.ask_single = ask

    plan_steps(app, "fix what codex found", "openai", context="[prior] codex report")

    assert "[prior] codex report" in seen["prompt"]
    # Context leads, exactly as the workers receive it.
    assert seen["prompt"].startswith("[prior] codex report")
    assert seen["prompt"].index("[prior] codex report") < seen["prompt"].index(
        "fix what codex found"
    )


def test_plan_steps_omits_context_when_empty():
    app, prov = _plan_app()
    seen = {}

    def ask(prompt, system=None):
        seen["prompt"] = prompt
        return '{"steps": [{"id": "s1", "description": "do", "prompt": "do it"}]}'

    prov.ask_single = ask

    plan_steps(app, "build X", "openai")

    assert seen["prompt"].startswith("Decompose this objective")


def test_plan_subtasks_prepends_context_to_director_prompt():
    director = MagicMock()
    director.config = SimpleNamespace(model="frontier")
    seen = {}

    def ask(prompt, system=None):
        seen["prompt"] = prompt
        return '{"subtasks": [{"id": "sub_1", "description": "a", "prompt": "p", "files": ["a.py"]}]}'

    director.ask_single = ask

    plan_subtasks(
        MagicMock(),
        "fix what codex found",
        director,
        "frontier",
        context="[prior] codex report",
    )

    assert seen["prompt"].startswith("[prior] codex report")
    assert seen["prompt"].index("[prior] codex report") < seen["prompt"].index(
        "fix what codex found"
    )


def test_plan_subtasks_omits_context_when_empty():
    director = MagicMock()
    director.config = SimpleNamespace(model="frontier")
    seen = {}

    def ask(prompt, system=None):
        seen["prompt"] = prompt
        return '{"subtasks": [{"id": "sub_1", "description": "a", "prompt": "p", "files": ["a.py"]}]}'

    director.ask_single = ask

    plan_subtasks(MagicMock(), "build X", director, "frontier")

    assert seen["prompt"].startswith("Decompose this objective")


# --- run_pipeline / run_fanout thread the context into the planner ----------


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


def test_run_pipeline_threads_context_into_planner(monkeypatch):
    app = _pipeline_app()
    captured = {}

    def fake_plan(app_, objective, provider_name, on_progress=None, on_tokens=None,
                  on_cost=None, context=""):
        captured["context"] = context
        return [PipelineTask("s1", "one", "one")]

    monkeypatch.setattr(pipe, "plan_steps", fake_plan)
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.diff_patch.side_effect = ["", "patch"]
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="1 file", diff_excerpt="+x", changed_files=("a.py",)
    )
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)
    monkeypatch.setattr(
        pipe, "run_verified_task",
        lambda *a, **k: (SimpleNamespace(passed=True, iterations=1, error=""), ["m"], ["p"]),
    )

    run_pipeline(app, "fix what codex found", context="[prior] codex report")

    assert captured["context"] == "[prior] codex report"


def _fanout_app():
    app = MagicMock()
    prov = MagicMock()
    prov.config = SimpleNamespace(model="frontier")
    app.providers = {"openrouter": prov}
    app.config.get_default_provider.return_value = "openrouter"
    app.config.get_model_for.return_value = "frontier"
    app.config.get_bulk_model.return_value = "bulk"
    app.config.get_escalation_target.return_value = None
    app.config.data = {"workflows": {"verify": {"test": "pytest"}}}
    return app


def test_run_fanout_threads_context_into_planner(monkeypatch):
    app = _fanout_app()
    captured = {}

    def fake_plan(app_, objective, director, director_model, on_progress=None,
                  on_tokens=None, on_cost=None, context=""):
        captured["context"] = context
        return [FanoutTask("sub_1", "x", "p1", ("a.py",))]

    monkeypatch.setattr(fanout_mod, "plan_subtasks", fake_plan)

    fm = MagicMock()
    fm.prepare.side_effect = lambda name: SimpleNamespace(path=f"/tmp/wt/{name}")
    fm.diff_patch.side_effect = lambda path: f"patch::{path}"
    fm.apply_patch.return_value = True
    fm.capture_snapshot.side_effect = lambda path: SimpleNamespace(
        diff_stat="1 file", diff_excerpt="+x", changed_files=("a.py",)
    )
    monkeypatch.setattr(fanout_mod, "WorktreeManager", lambda *a, **k: fm)
    monkeypatch.setattr(
        fanout_mod, "run_verified_task",
        lambda *a, **k: (SimpleNamespace(passed=True), ["bulk"], ["openrouter"]),
    )
    monkeypatch.setattr(fanout_mod, "_run_tests_in", lambda c, w, t: ("ok", 0))

    run_fanout(app, "fix what codex found", context="[prior] codex report")

    assert captured["context"] == "[prior] codex report"


# --- /pipeline and /fanout build and pass lane context ----------------------


def _command_app():
    app = MagicMock()
    cli_app = MagicMock()
    cli_app.providers = {"openai": MagicMock()}
    app.cli_app = cli_app
    app.state = MagicMock()
    app.state.messages = []
    app.state.active_provider = "openai"
    app._db_session = None
    app.run_ledger = None
    app.call_from_thread.side_effect = lambda fn, *a: fn(*a)
    app.screen = MagicMock()
    app.screen.run_worker.side_effect = lambda fn, **kwargs: fn()
    return app


def test_pipeline_command_builds_and_passes_lane_context():
    app = _command_app()
    handler = CommandHandler(app)
    handler._post_system = lambda text, **kw: None
    handler._mount_progress_indicator = MagicMock(return_value=None)

    captured = {}

    def fake_run_pipeline(cli_app, objective, **kwargs):
        captured["context"] = kwargs.get("context")
        return PipelineResult(
            objective=objective, provider="openai", worktree_path="",
            steps=(), passed=False,
        )

    with patch("cascade.conversation.build_lane_context", return_value="LANE_CTX") as blc, \
            patch("cascade.swarm.pipeline.run_pipeline", side_effect=fake_run_pipeline):
        handler._cmd_pipeline(["fix", "what", "codex", "found"])

    assert blc.call_count == 1 and blc.call_args.args == ([], "openai")
    assert captured["context"] == "LANE_CTX"


def test_fanout_command_builds_and_passes_lane_context():
    app = _command_app()
    handler = CommandHandler(app)
    handler._post_system = lambda text, **kw: None
    handler._mount_progress_indicator = MagicMock(return_value=None)

    captured = {}

    def fake_run_fanout(cli_app, objective, **kwargs):
        captured["context"] = kwargs.get("context")
        return SimpleNamespace(
            outcome=SimpleNamespace(value="failed"),
            subs=(),
            provider="openai",
            error="",
            input_tokens=0,
            output_tokens=0,
            diff_stat="",
            changed_files=(),
            worktree_path="",
        )

    with patch("cascade.conversation.build_lane_context", return_value="LANE_CTX") as blc, \
            patch("cascade.swarm.fanout.run_fanout", side_effect=fake_run_fanout):
        handler._cmd_fanout(["fix", "what", "codex", "found"])

    assert blc.call_count == 1 and blc.call_args.args == ([], "openai")
    assert captured["context"] == "LANE_CTX"
