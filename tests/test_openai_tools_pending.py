"""on_pending_message injects a queued follow-up at the next round boundary.

The codex-style behavior: a prompt the user types mid-turn lands before the
next model call instead of only after the turn. The OpenAI-family loop (Eve's
bulk path: openrouter/deepseek/local) consults the callback each round.
"""

import copy
import json

from cascade.providers._openai_tools import openai_ask_with_tools
from cascade.tools.schema import callable_to_tool_def


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []

    def post(self, url, json=None, headers=None):
        # Deep-copy: the loop mutates one api_messages list in place, so a
        # by-reference capture would show every round the FINAL state.
        self.payloads.append(copy.deepcopy(json))
        return _FakeResponse(self._responses.pop(0))


def _tool_call(call_id, name, args):
    return {"choices": [{"message": {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]},
            "finish_reason": "tool_calls"}]}


def _final(text):
    return {"choices": [{"message": {"content": text, "tool_calls": []},
            "finish_reason": "stop"}]}


def _tools(calls):
    def read_file(path: str) -> str:
        """Read a file."""
        calls.append(path)
        return "contents"
    return {"read_file": callable_to_tool_def("read_file", read_file, "read", read_only=True)}


def test_pending_message_injected_at_a_later_round_boundary():
    calls = []
    # Round 0 -> tool call; round 1 -> final answer.
    client = _FakeClient([_tool_call("c1", "read_file", {"path": "a.py"}), _final("done")])
    # The queue is empty at round 0 (the turn just started) and the user types a
    # follow-up during round 0, so it is ready at the round-1 boundary.
    rounds = {"n": 0}

    def on_pending():
        rounds["n"] += 1
        return "also handle errors" if rounds["n"] == 2 else None

    text, _log = openai_ask_with_tools(
        client=client, url="http://x/v1/chat/completions", headers={},
        model="deepseek", temperature=0.0, max_tokens=None,
        messages=[{"role": "user", "content": "read a.py"}],
        tools=_tools(calls), system="SYS", max_rounds=6,
        on_pending_message=on_pending,
    )

    assert text == "done"
    # Round 0's request must NOT contain the follow-up yet.
    round0 = [m["content"] for m in client.payloads[0]["messages"] if m["role"] == "user"]
    assert "also handle errors" not in round0
    # Round 1's request carries it, sitting after the tool result (valid order).
    r1 = client.payloads[1]["messages"]
    assert any(m["role"] == "user" and m["content"] == "also handle errors" for m in r1)
    roles = [m["role"] for m in r1]
    last_tool = len(roles) - 1 - roles[::-1].index("tool")
    injected_at = next(i for i, m in enumerate(r1)
                       if m["role"] == "user" and m["content"] == "also handle errors")
    assert injected_at > last_tool, "injected prompt must follow the tool result"


def test_no_pending_message_is_a_noop():
    calls = []
    client = _FakeClient([_final("hi")])
    text, _log = openai_ask_with_tools(
        client=client, url="http://x/v1/chat/completions", headers={},
        model="deepseek", temperature=0.0, max_tokens=None,
        messages=[{"role": "user", "content": "hi"}],
        tools=_tools(calls), system="SYS", max_rounds=3,
        on_pending_message=lambda: None,
    )
    assert text == "hi"
    # No phantom user messages beyond the original.
    users = [m for m in client.payloads[0]["messages"] if m["role"] == "user"]
    assert len(users) == 1
