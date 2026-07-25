"""Read-dedup staleness hardening across all three tool loops.

The read de-duplication that serves a repeat read of an already-read path from a
short stub must not go stale. Two ways it can:

1. **Eviction** -- budget compaction elides the cached read's result, but the
   dedup entry still points at it, so a repeat read is told "[already read
   above]" while the bytes it named are gone. Fixed for the shared OpenAI loop
   in 2bc3e3e; these tests extend the guarantee to the Claude and Gemini native
   loops, which kept dedup as a plain set that eviction never pruned.
2. **Edits** -- a write/append/replace to a path makes the cached read stale, so
   a later read of that path must re-fetch fresh content, not the pre-edit stub.

For each loop we pin: an evicted read is re-readable, and a read after an edit to
the same path re-fetches.
"""

import json
from unittest.mock import MagicMock, patch

from cascade.providers.base import Message, ProviderConfig
from cascade.providers.claude import ClaudeProvider
from cascade.providers.gemini import GeminiProvider
from cascade.tools.schema import callable_to_tool_def


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #


def _msgs(prompt: str) -> list[Message]:
    return [{"role": "user", "content": prompt}]


def _rw_tools(reads: list[str], writes: list[str]) -> dict:
    """read_file (big result, records paths) + write_file (records paths)."""

    def read_file(path: str) -> str:
        """Read a file."""
        reads.append(path)
        return "FILECONTENT " + "X" * 20000

    def write_file(path: str, content: str) -> str:
        """Write a file."""
        writes.append(path)
        return "wrote " + path

    return {
        "read_file": callable_to_tool_def("read_file", read_file, "read", read_only=True),
        "write_file": callable_to_tool_def("write_file", write_file, "write"),
    }


# --------------------------------------------------------------------------- #
# Shared OpenAI-compatible loop
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.payloads: list[dict] = []

    def post(self, url, json=None, headers=None):
        self.payloads.append(json)
        return _FakeResponse(self._responses.pop(0))


def _oai_call(call_id: str, name: str, args: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def _oai_final(text: str) -> dict:
    return {
        "choices": [
            {"message": {"content": text, "tool_calls": []}, "finish_reason": "stop"}
        ]
    }


def _run_openai(client, tools, *, context_window):
    from cascade.providers._openai_tools import openai_ask_with_tools

    return openai_ask_with_tools(
        client=client,
        url="http://x/v1/chat/completions",
        headers={},
        model="qwen",
        temperature=0.0,
        max_tokens=None,
        messages=_msgs("work the files"),
        tools=tools,
        system="SYS",
        max_rounds=12,
        context_window=context_window,
    )


def test_openai_evicted_read_is_re_readable():
    reads: list[str] = []
    tools = _rw_tools(reads, [])
    responses = [
        _oai_call("c1", "read_file", {"path": "a.py"}),
        _oai_call("c2", "read_file", {"path": "b.py"}),
        _oai_call("c3", "read_file", {"path": "c.py"}),
        _oai_call("c4", "read_file", {"path": "d.py"}),
        _oai_call("c5", "read_file", {"path": "e.py"}),  # a now old -> evicted
        _oai_call("c6", "read_file", {"path": "a.py"}),  # re-read a
        _oai_final("done"),
    ]
    text, log = _run_openai(_FakeClient(responses), tools, context_window=8000)

    assert text == "done"
    assert reads.count("a.py") == 2, reads
    a_outputs = [e["output"] for e in log if e["input"].get("path") == "a.py"]
    assert all("already read above" not in o for o in a_outputs), a_outputs


def test_openai_read_after_edit_refetches():
    reads: list[str] = []
    writes: list[str] = []
    tools = _rw_tools(reads, writes)
    responses = [
        _oai_call("c1", "read_file", {"path": "a.py"}),
        _oai_call("c2", "write_file", {"path": "a.py", "content": "new"}),
        _oai_call("c3", "read_file", {"path": "a.py"}),  # must re-fetch post-edit
        _oai_final("done"),
    ]
    text, log = _run_openai(_FakeClient(responses), tools, context_window=200000)

    assert text == "done"
    assert reads == ["a.py", "a.py"], reads  # read both before and after the edit
    assert writes == ["a.py"]
    read_outputs = [e["output"] for e in log if e["tool"] == "read_file"]
    assert all("already read above" not in o for o in read_outputs), read_outputs


def test_openai_edit_of_other_path_keeps_dedup():
    # Editing b.py must NOT invalidate a.py's cached read.
    reads: list[str] = []
    writes: list[str] = []
    tools = _rw_tools(reads, writes)
    responses = [
        _oai_call("c1", "read_file", {"path": "a.py"}),
        _oai_call("c2", "write_file", {"path": "b.py", "content": "new"}),
        _oai_call("c3", "read_file", {"path": "a.py"}),  # still deduped
        _oai_final("done"),
    ]
    _text, log = _run_openai(_FakeClient(responses), tools, context_window=200000)

    assert reads == ["a.py"]  # a.py read only once; b.py edit did not invalidate it
    assert log[-1]["output"] == "[already read above: a.py]"


# --------------------------------------------------------------------------- #
# Claude native loop
# --------------------------------------------------------------------------- #


def _resp(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _claude_provider(context_window=None) -> ClaudeProvider:
    with patch("cascade.providers.claude.shutil.which", return_value=None):
        return ClaudeProvider(
            ProviderConfig(
                api_key="sk-ant-api03-key",
                model="claude-opus-4-8",
                context_window=context_window,
            )
        )


def _cl_use(tid: str, name: str, inp: dict) -> dict:
    return {
        "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 5},
    }


def _cl_final(text: str) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def test_claude_evicted_read_is_re_readable():
    reads: list[str] = []
    tools = _rw_tools(reads, [])
    prov = _claude_provider(context_window=8000)
    responses = [
        _resp(_cl_use("t1", "read_file", {"path": "a.py"})),
        _resp(_cl_use("t2", "read_file", {"path": "b.py"})),
        _resp(_cl_use("t3", "read_file", {"path": "c.py"})),
        _resp(_cl_use("t4", "read_file", {"path": "d.py"})),
        _resp(_cl_use("t5", "read_file", {"path": "e.py"})),  # a now old -> evicted
        _resp(_cl_use("t6", "read_file", {"path": "a.py"})),  # re-read a
        _resp(_cl_final("done")),
    ]
    with patch.object(prov.client, "post", side_effect=responses):
        out, log = prov.ask_with_tools(_msgs("work the files"), tools, max_rounds=12)

    assert out == "done"
    assert reads.count("a.py") == 2, reads
    a_outputs = [e["output"] for e in log if e["input"].get("path") == "a.py"]
    assert all("already read above" not in o for o in a_outputs), a_outputs


def test_claude_read_after_edit_refetches():
    reads: list[str] = []
    writes: list[str] = []
    tools = _rw_tools(reads, writes)
    prov = _claude_provider()  # default 200k window: no eviction in play
    responses = [
        _resp(_cl_use("t1", "read_file", {"path": "a.py"})),
        _resp(_cl_use("t2", "write_file", {"path": "a.py", "content": "new"})),
        _resp(_cl_use("t3", "read_file", {"path": "a.py"})),  # re-fetch post-edit
        _resp(_cl_final("done")),
    ]
    with patch.object(prov.client, "post", side_effect=responses):
        out, log = prov.ask_with_tools(_msgs("edit then re-read"), tools, max_rounds=12)

    assert out == "done"
    assert reads == ["a.py", "a.py"], reads
    assert writes == ["a.py"]
    read_outputs = [e["output"] for e in log if e["tool"] == "read_file"]
    assert all("already read above" not in o for o in read_outputs), read_outputs


# --------------------------------------------------------------------------- #
# Gemini native loop
# --------------------------------------------------------------------------- #


def _gemini_provider(context_window=None) -> GeminiProvider:
    with patch("cascade.providers.gemini.shutil.which", return_value=None):
        return GeminiProvider(
            ProviderConfig(
                api_key="AIza-key",
                model="gemini-2.5-flash",
                context_window=context_window,
            )
        )


def _g_call(name: str, args: dict) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"functionCall": {"name": name, "args": args}}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5},
    }


def _g_text(text: str) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 3},
    }


def test_gemini_evicted_read_is_re_readable():
    reads: list[str] = []
    tools = _rw_tools(reads, [])
    prov = _gemini_provider(context_window=8000)
    responses = [
        _resp(_g_call("read_file", {"path": "a.py"})),
        _resp(_g_call("read_file", {"path": "b.py"})),
        _resp(_g_call("read_file", {"path": "c.py"})),
        _resp(_g_call("read_file", {"path": "d.py"})),
        _resp(_g_call("read_file", {"path": "e.py"})),  # a now old -> evicted
        _resp(_g_call("read_file", {"path": "a.py"})),  # re-read a
        _resp(_g_text("done")),
    ]
    with patch.object(prov.client, "post", side_effect=responses):
        out, log = prov.ask_with_tools(_msgs("work the files"), tools, max_rounds=12)

    assert out == "done"
    assert reads.count("a.py") == 2, reads
    a_outputs = [e["output"] for e in log if e["input"].get("path") == "a.py"]
    assert all("already read above" not in o for o in a_outputs), a_outputs


def test_gemini_read_after_edit_refetches():
    reads: list[str] = []
    writes: list[str] = []
    tools = _rw_tools(reads, writes)
    prov = _gemini_provider()  # default 1M window: no eviction in play
    responses = [
        _resp(_g_call("read_file", {"path": "a.py"})),
        _resp(_g_call("write_file", {"path": "a.py", "content": "new"})),
        _resp(_g_call("read_file", {"path": "a.py"})),  # re-fetch post-edit
        _resp(_g_text("done")),
    ]
    with patch.object(prov.client, "post", side_effect=responses):
        out, log = prov.ask_with_tools(_msgs("edit then re-read"), tools, max_rounds=12)

    assert out == "done"
    assert reads == ["a.py", "a.py"], reads
    assert writes == ["a.py"]
    read_outputs = [e["output"] for e in log if e["tool"] == "read_file"]
    assert all("already read above" not in o for o in read_outputs), read_outputs
