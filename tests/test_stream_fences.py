"""Streaming code-fence edge cases (reviewer-flagged correctness bugs)."""

import pytest
from textual.app import App, ComposeResult

from cascade.widgets.stream_message import StreamMessage, _ProseBody
from cascade.widgets.code_block import CodeBlock


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield StreamMessage("claude")


async def _stream(chunks):
    app = _Harness()
    async with app.run_test() as pilot:
        sm = app.query_one(StreamMessage)
        await pilot.pause()
        for c in chunks:
            sm.feed(c)
            await pilot.pause()
        sm.finish()
        await pilot.pause()
        prose = " ".join(b.render().plain for b in sm.query(_ProseBody))
        code = " ".join(getattr(cb, "_code", "") for cb in sm.query(CodeBlock))
        return prose, code


@pytest.mark.asyncio
async def test_prose_before_fence_in_same_batch_survives():
    prose, code = await _stream(["before code\n```python\nx = 1\n```\nafter"])
    assert "before code" in prose
    assert "x = 1" in code
    assert "after" in prose


@pytest.mark.asyncio
async def test_unterminated_code_block_keeps_last_line():
    prose, code = await _stream(["```python\nline1\nline2 no newline"])
    assert "line1" in code
    assert "line2 no newline" in code


@pytest.mark.asyncio
async def test_closing_fence_without_newline_not_leaked_as_prose():
    prose, code = await _stream(["```\ncode body\n```"])
    assert "```" not in prose
    assert "code body" in code


@pytest.mark.asyncio
async def test_char_by_char_streaming_still_works():
    # Feed one character at a time -- the pathological case.
    text = "intro\n```js\nconsole.log(1)\n```\ndone"
    prose, code = await _stream(list(text))
    assert "intro" in prose and "done" in prose
    assert "console.log(1)" in code
