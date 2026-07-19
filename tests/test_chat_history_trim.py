"""ChatHistory trims ALL content-widget types, not just messages (review)."""

import pytest

from cascade.widgets.message import ChatHistory, MessageWidget
from cascade.widgets.tool_call import ToolCallWidget


class _FakeChild:
    """Stand-in for a mounted content widget."""

    def __init__(self, name):
        self._name = name
        self.removed = False

    async def remove(self):
        self.removed = True


@pytest.mark.asyncio
async def test_trims_tool_widgets_not_only_messages(monkeypatch):
    chat = ChatHistory(max_widgets=3)
    # Simulate 6 mounted content widgets: mix of messages and tool rows.
    kids = [
        MessageWidget("you", "q1"),
        ToolCallWidget("read_file", {"path": "a"}, "..."),
        ToolCallWidget("write_file", {"path": "b"}, "..."),
        MessageWidget("claude", "a1"),
        ToolCallWidget("run_command", {"command": "ls"}, "..."),
        MessageWidget("you", "q2"),
    ]
    removed = []
    for k in kids:
        orig = k.remove

        async def _rm(_k=k, _orig=orig):
            removed.append(_k)
            # don't actually touch Textual internals

        k.remove = _rm

    monkeypatch.setattr(type(chat), "children", property(lambda self: kids))
    # No real mount; guard the indicator mount.
    async def _noop_mount(*a, **k):
        pass
    monkeypatch.setattr(chat, "mount", _noop_mount)

    await chat.trim_overflow()

    # 6 content widgets, cap 3 -> trim the 3 oldest, regardless of type.
    assert len(removed) == 3
    assert removed == kids[:3]
    # A trimmed tool widget is NOT stored as a recoverable message.
    assert all(role in ("you", "claude") for role, _ in chat._overflow)
    # Only the one trimmed MessageWidget (kids[0]) is recoverable.
    assert chat._overflow == [("you", "q1")]


@pytest.mark.asyncio
async def test_no_indicator_when_only_tools_trimmed(monkeypatch):
    chat = ChatHistory(max_widgets=1)
    kids = [
        ToolCallWidget("read_file", {"path": "a"}, "x"),
        ToolCallWidget("read_file", {"path": "b"}, "y"),
    ]
    for k in kids:
        async def _rm():
            pass
        k.remove = _rm
    monkeypatch.setattr(type(chat), "children", property(lambda self: kids))
    mounted = []
    async def _mount(w, **k):
        mounted.append(w)
    monkeypatch.setattr(chat, "mount", _mount)

    await chat.trim_overflow()
    # Trimmed a tool widget but no messages -> no "N earlier messages" banner.
    assert chat._overflow == []
    assert mounted == []
