"""Per-provider token/cost attribution for the /pipeline and /fanout lanes.

/solve already breaks an escalation's usage out per provider. These lock the
same accounting into the multi-step and parallel orchestrators, where the run
summary used to lump an escalation's frontier tokens under the base provider
and label the whole bill with the base provider's name.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cascade.swarm.fanout as fanout_mod
import cascade.swarm.pipeline as pipe
from cascade.commands import _solve_token_lines
from cascade.swarm.fanout import FanoutResult, FanoutTask, run_fanout
from cascade.swarm.pipeline import PipelineResult, PipelineTask, run_pipeline


def _escalating_app(primary: str, escalation: str):
    """An app whose *primary* provider escalates (and directs) on *escalation*."""
    app = MagicMock()
    providers = {}
    for name in (primary, escalation):
        prov = MagicMock()
        prov.config = SimpleNamespace(model="frontier")
        providers[name] = prov
    app.providers = providers
    app.config.get_default_provider.return_value = primary
    app.config.get_model_for = MagicMock(
        side_effect=lambda n, mode_name=None, fast=False: "fast" if fast else "frontier"
    )
    app.config.get_bulk_model.return_value = "bulk"
    app.config.get_escalation_target.return_value = escalation
    app.config.data = {"workflows": {"verify": {"test": "pytest"}}}
    return app


def _worker(usage):
    """A run_verified_task double replaying *usage*: one list of splits per call.

    Each split is ``(provider, input_tokens, output_tokens, cost)``, reported
    live through the callbacks (as the real loop does) and returned as the
    per-provider breakdown.
    """
    calls = iter(usage)

    def fake(*args, **kwargs):
        splits = next(calls)
        for label, tokens_in, tokens_out, cost in splits:
            kwargs["on_tokens"](tokens_in, tokens_out)
            kwargs["on_cost"](cost)
        return (
            SimpleNamespace(passed=True, iterations=1, error=""),
            ["bulk"],
            [split[0] for split in splits],
            tuple((label, tin, tout) for label, tin, tout, _cost in splits),
            tuple((label, cost) for label, _tin, _tout, cost in splits),
        )

    return fake


def _by_provider(breakdown):
    return {label: (tokens_in, tokens_out) for label, tokens_in, tokens_out in breakdown}


# ----------------------------------------------------------------------
# /pipeline
# ----------------------------------------------------------------------


def _patch_pipeline_manager(monkeypatch):
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt-pipe")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="2 files", diff_excerpt="+x", changed_files=("a.py", "b.py")
    )
    fm.diff_patch.side_effect = ["", "patch-1", "patch-1", "patch-2"]
    monkeypatch.setattr(pipe, "WorktreeManager", lambda *a, **k: fm)
    return fm


def test_run_pipeline_attributes_tokens_and_cost_per_provider(monkeypatch):
    app = _escalating_app("openrouter", "claude")
    monkeypatch.setattr(
        pipe, "plan_steps",
        lambda *a, **k: [
            PipelineTask("s1", "step one", "do one"),
            PipelineTask("s2", "step two", "do two"),
        ],
    )
    _patch_pipeline_manager(monkeypatch)
    # Step two escalates: the frontier tokens belong to claude, not openrouter.
    monkeypatch.setattr(pipe, "run_verified_task", _worker([
        [("openrouter", 100, 10, 0.01)],
        [("openrouter", 50, 5, 0.005), ("claude", 2_000_000, 3_000, 1.25)],
    ]))

    result = run_pipeline(app, "build X")

    assert _by_provider(result.tokens_by_provider) == {
        "openrouter": (150, 15),
        "claude": (2_000_000, 3_000),
    }
    assert dict(result.cost_by_provider) == pytest.approx(
        {"openrouter": 0.015, "claude": 1.25}
    )
    # The breakdown accounts for every token (and credit) the flat totals claim.
    assert result.input_tokens == 2_000_150
    assert result.output_tokens == 3_015
    assert result.cost == pytest.approx(1.265)


def test_run_pipeline_attributes_planning_to_the_director(monkeypatch):
    app = _escalating_app("openrouter", "claude")

    def fake_plan(app, objective, provider_name, on_progress=None, on_tokens=None,
                  on_cost=None, cancel_token=None):
        on_tokens(1_000, 200)  # the director runs on the frontier model
        on_cost(0.5)
        return [PipelineTask("s1", "only step", "do it")]

    monkeypatch.setattr(pipe, "plan_steps", fake_plan)
    _patch_pipeline_manager(monkeypatch)
    monkeypatch.setattr(pipe, "run_verified_task", _worker([
        [("openrouter", 100, 10, 0.01)],
    ]))

    result = run_pipeline(app, "build X")

    assert _by_provider(result.tokens_by_provider) == {
        "claude": (1_000, 200),
        "openrouter": (100, 10),
    }
    assert dict(result.cost_by_provider) == pytest.approx(
        {"claude": 0.5, "openrouter": 0.01}
    )


def test_run_pipeline_survives_a_worker_without_a_breakdown(monkeypatch):
    """A worker that reports no per-provider split still yields a usable result."""
    app = _escalating_app("openrouter", "claude")
    monkeypatch.setattr(
        pipe, "plan_steps", lambda *a, **k: [PipelineTask("s1", "one", "do one")],
    )
    _patch_pipeline_manager(monkeypatch)
    monkeypatch.setattr(
        pipe, "run_verified_task",
        lambda *a, **k: (SimpleNamespace(passed=True, iterations=1, error=""), ["bulk"], ["openrouter"]),
    )

    result = run_pipeline(app, "build X")

    assert result.passed is True
    assert result.tokens_by_provider == ()
    assert result.cost_by_provider == ()


# ----------------------------------------------------------------------
# /fanout
# ----------------------------------------------------------------------


def _patch_fanout_manager(monkeypatch):
    fm = MagicMock()
    fm.prepare.side_effect = lambda name: SimpleNamespace(path=f"/tmp/wt/{name}")
    fm.diff_patch.side_effect = lambda path: f"patch::{path}"
    fm.apply_patch.return_value = True
    changed = {"task_01": ("a.py",), "task_02": ("b.py",)}
    fm.capture_snapshot.side_effect = lambda path: SimpleNamespace(
        diff_stat="2 files changed",
        diff_excerpt="+A",
        changed_files=changed.get(path.rsplit("/", 1)[-1], ("a.py", "b.py")),
    )
    monkeypatch.setattr(fanout_mod, "WorktreeManager", lambda *a, **k: fm)
    return fm


def test_run_fanout_attributes_tokens_and_cost_per_provider(monkeypatch):
    app = _escalating_app("openrouter", "claude")
    monkeypatch.setattr(fanout_mod, "plan_subtasks", lambda *a, **k: [
        FanoutTask("sub_1", "add A", "add A to a.py", ("a.py",)),
        FanoutTask("sub_2", "add B", "add B to b.py", ("b.py",)),
    ])
    _patch_fanout_manager(monkeypatch)
    monkeypatch.setattr(fanout_mod, "_run_tests_in", lambda *a: ("ok", 0))
    monkeypatch.setattr(fanout_mod, "run_verified_task", _worker([
        [("openrouter", 100, 10, 0.01)],
        [("openrouter", 50, 5, 0.005), ("claude", 900_000, 2_000, 0.75)],
    ]))

    result = run_fanout(app, "build A and B")

    assert result.passed is True
    assert _by_provider(result.tokens_by_provider) == {
        "openrouter": (150, 15),
        "claude": (900_000, 2_000),
    }
    assert dict(result.cost_by_provider) == pytest.approx(
        {"openrouter": 0.015, "claude": 0.75}
    )
    assert result.input_tokens == 900_150
    assert result.cost == pytest.approx(0.765)


def test_run_fanout_attributes_planning_to_the_director(monkeypatch):
    app = _escalating_app("openrouter", "claude")

    def fake_plan(app, objective, director, director_model, on_progress=None,
                  on_tokens=None, on_cost=None, cancel_token=None):
        on_tokens(1_000, 200)
        on_cost(0.5)
        return [FanoutTask("sub_1", "add A", "add A to a.py", ("a.py",))]

    monkeypatch.setattr(fanout_mod, "plan_subtasks", fake_plan)
    _patch_fanout_manager(monkeypatch)
    monkeypatch.setattr(fanout_mod, "_run_tests_in", lambda *a: ("ok", 0))
    monkeypatch.setattr(fanout_mod, "run_verified_task", _worker([
        [("openrouter", 100, 10, 0.01)],
    ]))

    result = run_fanout(app, "build A")

    assert _by_provider(result.tokens_by_provider) == {
        "claude": (1_000, 200),
        "openrouter": (100, 10),
    }
    assert dict(result.cost_by_provider) == pytest.approx(
        {"claude": 0.5, "openrouter": 0.01}
    )


# ----------------------------------------------------------------------
# The shared summary helper picks both lanes up unchanged
# ----------------------------------------------------------------------


def test_pipeline_summary_breaks_the_bill_out_per_provider():
    result = PipelineResult(
        objective="build X",
        provider="openrouter",
        worktree_path="/tmp/wt",
        steps=(),
        passed=True,
        input_tokens=2_000_150,
        output_tokens=3_015,
        cost=1.265,
        tokens_by_provider=(("openrouter", 150, 15), ("claude", 2_000_000, 3_000)),
        cost_by_provider=(("openrouter", 0.015), ("claude", 1.25)),
    )

    lines = _solve_token_lines(result)

    assert lines[0] == "Tokens by provider:"
    assert "  openrouter: 150 in / 15 out · 0.015000 credits" in lines
    assert "  claude: 2,000,000 in / 3,000 out · 1.250000 credits" in lines
    assert lines[-1] == "Total: 2,000,150 in / 3,015 out"


def test_fanout_summary_breaks_the_bill_out_per_provider():
    result = FanoutResult(
        objective="build A and B",
        provider="openrouter",
        worktree_path="/tmp/wt",
        subs=(),
        passed=True,
        input_tokens=900_150,
        output_tokens=2_015,
        cost=0.765,
        tokens_by_provider=(("openrouter", 150, 15), ("claude", 900_000, 2_000)),
        cost_by_provider=(("openrouter", 0.015), ("claude", 0.75)),
    )

    lines = _solve_token_lines(result)

    assert lines[0] == "Tokens by provider:"
    assert "  claude: 900,000 in / 2,000 out · 0.750000 credits" in lines
    assert lines[-1] == "Total: 900,150 in / 2,015 out"
