"""The tool-turn final answer renders code blocks (regression fix)."""

import pytest
from textual.app import App, ComposeResult

from cascade.widgets.message import ChatHistory
from cascade.widgets.stream_message import StreamMessage
from cascade.widgets.code_block import CodeBlock


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield ChatHistory()


@pytest.mark.asyncio
async def test_stream_message_renders_fenced_code():
    # Proves the widget the tool path now uses handles fences (a plain
    # MessageWidget would render the ``` as literal text).
    app = _Harness()
    async with app.run_test() as pilot:
        chat = app.query_one(ChatHistory)
        await pilot.pause()
        msg = StreamMessage("claude")
        chat.mount(msg)
        await pilot.pause()
        msg.feed("Here is the fix:\n```python\nx = 42\n```\nDone.")
        msg.finish()
        await pilot.pause()
        code = " ".join(getattr(cb, "_code", "") for cb in msg.query(CodeBlock))
        assert "x = 42" in code
