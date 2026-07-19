"""Streaming prose renders each completed line once (no O(n^2) reparse)."""

from unittest.mock import patch

import cascade.widgets.stream_message as sm
from cascade.widgets.stream_message import _ProseBody


class TestFreezeRender:
    def test_completed_lines_rendered_exactly_once(self):
        calls = []
        real = sm.render_md_line

        def counting(line):
            calls.append(line)
            return real(line)

        with patch.object(sm, "render_md_line", counting):
            body = _ProseBody()
            # Simulate 100 streaming batches, each completing one new line
            # while re-showing a growing partial. Naive rendering would be
            # O(n^2) in render_md_line calls; frozen rendering is ~O(n).
            for i in range(100):
                body.append_lines([f"line {i}"])
                body.set_partial(f"partial {i}")
                body.render()
            body.set_partial("")
            body.render()

        completed = [c for c in calls if c.startswith("line ")]
        partials = [c for c in calls if c.startswith("partial ")]
        # Each completed line rendered exactly once (frozen thereafter).
        assert len(completed) == 100
        assert sorted(completed) == sorted(f"line {i}" for i in range(100))
        # Partials re-render per batch, but that is one line, not the whole
        # segment -- bounded by batch count, not batch count squared.
        assert len(partials) <= 100

    def test_render_content_matches_frozen_output(self):
        from cascade.widgets.message import render_content

        lines = ["# Title", "plain line", "- bullet", "**bold** end"]
        body = _ProseBody()
        body.append_lines(lines)
        body.set_partial("")
        # Frozen output equals a one-shot render of the same content.
        assert body.render().plain == render_content("\n".join(lines)).plain

    def test_empty_body_renders_empty(self):
        assert _ProseBody().render().plain == ""
