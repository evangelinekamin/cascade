"""Hardening tests for the AUTO orchestration route.

Covers four review findings:

  (L) per-provider token/cost attribution survives the AUTO route and reaches
      the UI, so an escalation's tokens are credited to the model that incurred
      them rather than lumped under the base provider;
  (Q) a blocked fanout's own planning spend is not erased when it reroutes to a
      pipeline;
  (M) test-mode recon runs its commands in a throwaway worktree, never the
      user's real checkout, and never leaves a writable-sandbox flag behind;
  (S) a CLI-proxy active provider may serve as the test-mode recon fallback.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import cascade.swarm.auto as auto
from cascade.screens.main import MainScreen
from cascade.swarm.auto import RouteDecision, WorkflowKind
from cascade.swarm.outcome import RunOutcome
from cascade.providers.usage import Usage


# --- shared fakes ----------------------------------------------------------------


class _Config:
    """Minimal orchestration config: no recon clone, no escalation target."""

    def __init__(self, **overrides):
        self.orchestration = {
            "recon_provider": "unavailable",  # forces the fallback path
            "recon_model": "m",
            "provider_preferences": {},
            "recon_max_rounds": 5,
        }
        self.orchestration.update(overrides)

    def get_orchestration_config(self):
        return self.orchestration

    def get_escalation_target(self, _provider):
        return None


class _FakeCliProxy:
    """A CLI-proxy provider: drives its own sandbox via ``working_directory``."""

    def __init__(self):
        self._use_cli_proxy = True
        self.config = SimpleNamespace(model="codex-model")
        self.last_usage = Usage(input=15, output=5)
        self.tools = None
        self.entered = []
        self.force_during = "unset"

    def working_directory(self, path):
        @contextmanager
        def _cm():
            self.entered.append(path)
            yield

        return _cm()

    def ask_with_tools(self, messages, tools, **kwargs):
        self.tools = tools
        self.force_during = getattr(self, "_force_repo_write", "unset")
        return "verified: the suite is green", []


class _FakeWorktreeManager:
    """A WorktreeManager stand-in that hands back a fixed path, doing no git."""

    def __init__(self, path):
        self._path = path
        self.cleaned = False
        self.prepared = []

    def __call__(self, *args, **kwargs):
        return self

    def prepare(self, provider):
        self.prepared.append(provider)
        return SimpleNamespace(provider=provider, path=self._path)

    def cleanup(self, keep_provider=""):
        self.cleaned = True


def _spy_workspace_tools(roots):
    real = auto.WorkspaceTools

    def _factory(root, **kwargs):
        roots["ws"] = root
        return real(root, **kwargs)

    return _factory


# --- (L) per-provider attribution through the AUTO route -------------------------


def test_solve_route_carries_per_provider_breakdown(monkeypatch):
    """An escalated solve's per-provider tokens/cost reach the AutoResult, merged
    with the router's own call -- not collapsed into a flat base-provider total."""
    solve_result = SimpleNamespace(
        outcome=RunOutcome.SUCCEEDED,
        iterations=2,
        provider="openai",
        models_used=("bulk", "frontier"),
        error="",
        diff_stat="1 file changed",
        changed_files=("a.py",),
        worktree_path="/tmp/wt",
        input_tokens=100,
        output_tokens=40,
        cost=0.05,
        tokens_by_provider=(("openai", 70, 20), ("claude", 30, 20)),
        cost_by_provider=(("claude", 0.05),),
    )
    monkeypatch.setattr(auto, "run_solve", MagicMock(return_value=solve_result))
    app = SimpleNamespace(config=_Config(), providers={"openai": MagicMock()})
    decision = RouteDecision(
        WorkflowKind.SOLVE,
        "edit",
        0.8,
        router_provider="openrouter",
        input_tokens=5,
        output_tokens=2,
        cost=0.001,
    )

    result = auto.execute_auto(app, "Fix a.py", "openai", decision)

    # Flat totals fold route + solve.
    assert (result.input_tokens, result.output_tokens) == (105, 42)
    assert result.cost == 0.001 + 0.05
    # The breakdown attributes each provider its own share (router included).
    assert result.tokens_by_provider == (
        ("openrouter", 5, 2),
        ("openai", 70, 20),
        ("claude", 30, 20),
    )
    assert result.cost_by_provider == (("openrouter", 0.001), ("claude", 0.05))
    # Invariant: the breakdown sums back to the flat totals.
    assert sum(i for _, i, _ in result.tokens_by_provider) == result.input_tokens
    assert sum(o for _, _, o in result.tokens_by_provider) == result.output_tokens


def test_route_without_spend_adds_no_phantom_breakdown_entry(monkeypatch):
    """A zero-cost local route contributes no entry to the breakdown."""
    solve_result = SimpleNamespace(
        outcome=RunOutcome.SUCCEEDED, iterations=1, provider="openai",
        models_used=("bulk",), error="", diff_stat="", changed_files=(),
        worktree_path="/tmp/wt", input_tokens=10, output_tokens=4,
        cost=0.0, tokens_by_provider=(("openai", 10, 4),), cost_by_provider=(),
    )
    monkeypatch.setattr(auto, "run_solve", MagicMock(return_value=solve_result))
    app = SimpleNamespace(config=_Config(), providers={"openai": MagicMock()})
    decision = RouteDecision(
        WorkflowKind.SOLVE, "edit", 0.9, router_provider="local",
    )  # no route tokens/cost

    result = auto.execute_auto(app, "fix it", "openai", decision)

    assert result.tokens_by_provider == (("openai", 10, 4),)
    assert result.cost_by_provider == ()


def test_on_stream_done_credits_tokens_per_provider():
    """main._on_stream_done credits each provider in the breakdown, not the
    active provider flat -- so an escalation is not miscredited."""
    calls = []
    state = SimpleNamespace(
        set_thinking=lambda *a, **k: None,
        add_message=lambda *a, **k: None,
        update_tokens=lambda provider, i, o: calls.append((provider, i, o)),
        provider_tokens={},
    )
    fake = SimpleNamespace(
        _stop_activity_poll=lambda: None,
        _thinking=None,
        _set_input_locked=lambda locked: None,
        _refresh_context_display=lambda: None,
        query_one=MagicMock(side_effect=Exception("no widget in unit test")),
        app=SimpleNamespace(state=state, record_message=lambda *a, **k: None),
    )
    breakdown = (("openai", 70, 20), ("claude", 30, 20))

    MainScreen._on_stream_done(fake, "openai", "text", 100, 40, breakdown)

    assert calls == [("openai", 70, 20), ("claude", 30, 20)]


def test_on_stream_done_flat_credit_when_no_breakdown():
    """Without a breakdown (the ordinary chat path) crediting stays flat."""
    calls = []
    state = SimpleNamespace(
        set_thinking=lambda *a, **k: None,
        add_message=lambda *a, **k: None,
        update_tokens=lambda provider, i, o: calls.append((provider, i, o)),
        provider_tokens={},
    )
    fake = SimpleNamespace(
        _stop_activity_poll=lambda: None,
        _thinking=None,
        _set_input_locked=lambda locked: None,
        _refresh_context_display=lambda: None,
        query_one=MagicMock(side_effect=Exception("no widget in unit test")),
        app=SimpleNamespace(state=state, record_message=lambda *a, **k: None),
    )

    MainScreen._on_stream_done(fake, "openai", "text", 100, 40)

    assert calls == [("openai", 100, 40)]


# --- (Q) blocked-fanout spend is not erased on reroute ---------------------------


def test_blocked_fanout_planning_spend_survives_reroute(monkeypatch):
    blocked = SimpleNamespace(
        outcome=RunOutcome.BLOCKED,
        error="invalid parallel plan: overlapping ownership",
        input_tokens=12,
        output_tokens=4,
        cost=0.002,
        tokens_by_provider=(("claude", 12, 4),),
        cost_by_provider=(("claude", 0.002),),
    )
    pipeline = SimpleNamespace(
        outcome=RunOutcome.SUCCEEDED,
        provider="openai",
        steps=(SimpleNamespace(
            passed=True, id="s1", iterations=1, description="seq", error="",
        ),),
        error="",
        diff_stat="1 file changed",
        worktree_path="/tmp/p",
        input_tokens=40,
        output_tokens=12,
        cost=0.01,
        tokens_by_provider=(("openai", 40, 12),),
        cost_by_provider=(("openai", 0.01),),
    )
    monkeypatch.setattr(auto, "run_fanout", MagicMock(return_value=blocked))
    monkeypatch.setattr(auto, "run_pipeline", MagicMock(return_value=pipeline))
    app = SimpleNamespace(config=_Config(), providers={"openai": MagicMock()})
    decision = RouteDecision(
        WorkflowKind.FANOUT,
        "looked independent",
        0.7,
        router_provider="openrouter",
        input_tokens=3,
        output_tokens=1,
        cost=0.0005,
    )

    result = auto.execute_auto(app, "Build both pieces", "openai", decision)

    assert result.outcome == RunOutcome.SUCCEEDED
    assert "fanout -> pipeline" in result.text
    # The blocked fanout's own director call (12/4, 0.002) is folded in, not lost.
    assert (result.input_tokens, result.output_tokens) == (3 + 12 + 40, 1 + 4 + 12)
    assert result.cost == 0.0005 + 0.002 + 0.01
    assert result.tokens_by_provider == (
        ("openrouter", 3, 1),
        ("claude", 12, 4),
        ("openai", 40, 12),
    )
    assert result.cost_by_provider == (
        ("openrouter", 0.0005),
        ("claude", 0.002),
        ("openai", 0.01),
    )


# --- (M)/(S) test-mode recon isolation + CLI-proxy fallback ----------------------


def _recon_app(active_provider):
    provider = _FakeCliProxy()
    return SimpleNamespace(
        config=_Config(),
        providers={active_provider: provider},
    ), provider


def test_test_mode_recon_runs_in_a_throwaway_worktree(monkeypatch, tmp_path):
    """(M) recon commands run in the isolated worktree, never the real cwd, and
    the writable-sandbox flag is removed afterwards. (S) a CLI-proxy active
    provider is accepted as the test-mode recon fallback."""
    app, codex = _recon_app("codex")
    worktree = tmp_path / "recon_wt"
    worktree.mkdir()
    manager = _FakeWorktreeManager(str(worktree))
    roots = {}
    monkeypatch.setattr(auto, "WorktreeManager", manager)
    monkeypatch.setattr(auto, "WorkspaceTools", _spy_workspace_tools(roots))
    decision = RouteDecision(WorkflowKind.RECON, "verify it works", 0.9)

    result = auto.execute_auto(
        app, "does it work?", "codex", decision, mode="test",
    )

    # (S) the CLI proxy served as recon.
    assert result.outcome == RunOutcome.SUCCEEDED
    assert result.execution_provider == "codex"
    assert "run_command" in codex.tools
    # (M) tools rooted at the worktree, the proxy aimed there, cleaned up after.
    assert roots["ws"] == str(worktree)
    assert codex.entered == [str(worktree)]
    assert manager.cleaned is True
    # (M) the writable-sandbox flag was set DURING the call and removed AFTER.
    assert codex.force_during is True
    assert not hasattr(codex, "_force_repo_write")


def test_non_test_mode_recon_still_refuses_a_cli_proxy(monkeypatch):
    """(S) outside test mode a CLI proxy cannot be constrained to read-only, so
    it is refused and recon blocks rather than running an unsandboxed proxy."""
    app, codex = _recon_app("codex")
    # No worktree should ever be created on this path.
    monkeypatch.setattr(
        auto, "WorktreeManager",
        MagicMock(side_effect=AssertionError("must not build a worktree")),
    )
    decision = RouteDecision(WorkflowKind.RECON, "inspect", 0.9)

    result = auto.execute_auto(app, "review the codebase", "codex", decision)

    assert result.outcome == RunOutcome.BLOCKED
    assert "no safe read-only provider" in result.text
    assert codex.tools is None  # the proxy was never invoked


def test_test_mode_recon_restores_a_preexisting_force_flag(monkeypatch, tmp_path):
    """A provider that already carried _force_repo_write keeps its prior value."""
    app, codex = _recon_app("codex")
    codex._force_repo_write = False  # a pre-existing value must be restored
    worktree = tmp_path / "wt"
    worktree.mkdir()
    monkeypatch.setattr(auto, "WorktreeManager", _FakeWorktreeManager(str(worktree)))
    monkeypatch.setattr(auto, "WorkspaceTools", _spy_workspace_tools({}))
    decision = RouteDecision(WorkflowKind.RECON, "verify", 0.9)

    auto.execute_auto(app, "does it work?", "codex", decision, mode="test")

    assert codex.force_during is True
    assert codex._force_repo_write is False


# --- helper unit coverage --------------------------------------------------------


def test_merge_token_breakdown_sums_shared_labels():
    merged = auto._merge_token_breakdown(
        (("openai", 5, 2), ("claude", 1, 1)),
        (("openai", 3, 4),),
    )
    assert merged == (("openai", 8, 6), ("claude", 1, 1))


def test_route_attribution_only_when_spent():
    spent = RouteDecision(
        WorkflowKind.SOLVE, "r", 0.5, router_provider="openrouter",
        input_tokens=4, output_tokens=1, cost=0.01,
    )
    assert auto._route_token_attribution(spent) == (("openrouter", 4, 1),)
    assert auto._route_cost_attribution(spent) == (("openrouter", 0.01),)
    free = RouteDecision(WorkflowKind.SOLVE, "r", 0.5, router_provider="local")
    assert auto._route_token_attribution(free) == ()
    assert auto._route_cost_attribution(free) == ()
