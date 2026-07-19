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
