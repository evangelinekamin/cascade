"""Tests for the context-occupancy indicator plumbing (Phase B)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cascade.commands import CommandHandler
from cascade.providers.usage import Usage
from cascade.state import CascadeState
from cascade.widgets.status_bar import StatusBar


class TestStateContextAccounting:
    def test_anchor_set_and_cleared_by_compaction(self):
        state = CascadeState()
        anchor = Usage(input=10_000, output=500, cache_read=2_000)
        state.set_context_anchor(anchor)
        assert state.context_anchor is anchor

        state.mark_compaction()
        assert state.context_anchor is None
        assert state.compaction_count == 1

    def test_reset_session_clears_context_accounting(self):
        state = CascadeState()
        state.set_context_anchor(Usage(input=5))
        state.mark_compaction()
        state.set_compaction_summary("carried summary")
        state.reset_session()
        assert state.context_anchor is None
        assert state.compaction_count == 0
        assert state.compaction_summary == ""


class TestStatusBarContext:
    def _render(self, bar: StatusBar) -> str:
        return bar.render().plain

    def test_calm_band_under_warn(self):
        bar = StatusBar(cwd="~", branch="")
        bar._ctx = (17_100, 171_000, 151_000, 0)
        out = self._render(bar)
        assert "ctx 10%" in out
        assert "⟳" not in out

    def test_unknown_state_renders_question_mark(self):
        bar = StatusBar(cwd="~", branch="")
        bar._ctx = (None, 171_000, 151_000, 2)
        out = self._render(bar)
        assert "ctx ?" in out
        assert "⟳2" in out

    def test_warn_and_error_bands_style(self):
        from cascade.theme import PALETTE

        bar = StatusBar(cwd="~", branch="")
        bar._ctx = (160_000, 171_000, 151_000, 0)
        rendered = bar.render()
        assert "ctx 93%" in rendered.plain
        styles = [str(span.style) for span in rendered.spans]
        assert any(PALETTE.amber in s for s in styles)

        bar._ctx = (180_000, 171_000, 151_000, 1)
        rendered = bar.render()
        styles = [str(span.style) for span in rendered.spans]
        assert any(PALETTE.error in s for s in styles)


class TestContextCommandOccupancy:
    def _handler(self, state: CascadeState):
        app = MagicMock()
        app.state = state
        cli_app = MagicMock()
        from cascade.context.memory import ContextBuilder

        cli_app.context_builder = ContextBuilder()
        cli_app.providers = {
            "claude": SimpleNamespace(
                config=SimpleNamespace(model="claude-opus-4-8", context_window=None)
            )
        }
        app.cli_app = cli_app
        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda msg: posted.append(msg)
        return handler, posted

    def test_occupancy_block_with_anchor(self):
        state = CascadeState()
        state.active_provider = "claude"
        state.set_context_anchor(
            Usage(input=10_000, output=500, cache_read=2_000, cache_write=100)
        )
        handler, posted = self._handler(state)
        handler._cmd_context([])
        out = posted[0]
        assert "Context window (claude · claude-opus-4-8)" in out
        assert "12,600 tok" in out  # 10k + 2k cache_read + 100 write + 500 out
        assert "cache read 2,000" in out
        assert "compact at 171,000" in out

    def test_occupancy_unknown_after_compaction(self):
        state = CascadeState()
        state.active_provider = "claude"
        state.mark_compaction()
        handler, posted = self._handler(state)
        handler._cmd_context([])
        assert "unknown" in posted[0]
        assert "since compaction" in posted[0]


class TestSummaryBreaker:
    """MainScreen._generate_compaction_summary guard behavior (review gap)."""

    def _screen(self):
        from cascade.screens.main import MainScreen

        class _TestableScreen(MainScreen):
            def __init__(self):  # bypass Screen/Textual initialization
                self._compaction_summary_enabled = True
                self._summary_failures = 0
                self._mode = "plan"
                self._fake_app = None

            @property
            def app(self):  # Screen.app is a read-only Textual property
                return self._fake_app

        return _TestableScreen()

    def _prov_and_app(self, screen, ask_behavior):
        from types import SimpleNamespace
        from cascade.providers.base import ProviderConfig

        class FakeProv:
            def __init__(self, config):
                self.config = config

            def ask_single(self, prompt, system=None):
                return ask_behavior(prompt)

        prov = FakeProv(ProviderConfig(api_key="k", model="base-model"))
        state = SimpleNamespace(compaction_summary="")
        cli_app = SimpleNamespace(
            config=SimpleNamespace(get_model_for=lambda p, m, fast=True: "fast-model"),
        )
        screen._fake_app = SimpleNamespace(state=state, cli_app=cli_app)
        return prov

    def _big_range(self):
        from cascade.state import ChatMessage

        return [ChatMessage(role="you", content="z" * 6000)]

    def test_two_failures_trip_the_session_breaker(self):
        screen = self._screen()

        def fail(prompt):
            raise RuntimeError("provider down")

        prov = self._prov_and_app(screen, fail)
        assert screen._generate_compaction_summary(prov, "claude", self._big_range()) is None
        assert screen._compaction_summary_enabled is True
        assert screen._generate_compaction_summary(prov, "claude", self._big_range()) is None
        assert screen._compaction_summary_enabled is False
        # Disabled: returns None without touching the provider
        assert screen._generate_compaction_summary(prov, "claude", self._big_range()) is None

    def test_small_range_does_not_count_as_failure(self):
        from cascade.state import ChatMessage

        screen = self._screen()
        prov = self._prov_and_app(screen, lambda p: "x")
        small = [ChatMessage(role="you", content="tiny")]
        assert screen._generate_compaction_summary(prov, "claude", small) is None
        assert screen._summary_failures == 0

    def test_success_resets_failure_count_and_uses_fast_model(self):
        seen = {}
        screen = self._screen()

        def ok(prompt):
            return "1. Primary Request and Intent\n" + "Good content. " * 40

        prov = self._prov_and_app(screen, ok)

        original_type = type(prov)
        built = {}
        real_init = original_type.__init__

        def spy_init(self, config):
            built["model"] = config.model
            built["max_tokens"] = config.max_tokens
            real_init(self, config)

        original_type.__init__ = spy_init
        try:
            screen._summary_failures = 1
            result = screen._generate_compaction_summary(
                prov, "claude", self._big_range(),
            )
        finally:
            original_type.__init__ = real_init

        assert result is not None
        assert screen._summary_failures == 0
        assert built["model"] == "fast-model"
        assert built["max_tokens"] == 8000
