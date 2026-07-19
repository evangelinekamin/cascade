"""Context-window management for the shared OpenAI-compatible tool loop.

Small-context local models (e.g. a 32K Qwen) overflow when every file read is
appended to the running conversation and never evicted. These tests pin the
three defences that keep the loop inside a model's window:

1. ``_compact_messages_to_budget`` -- pure oldest-first eviction of large tool
   results, protecting the system prompt, the original task, and recent rounds.
2. Read de-duplication -- a second read of an already-read path returns a stub.
3. ``context_window`` threading -- providers forward their real window into the
   loop so eviction fires at the right size.
"""

import json
from unittest.mock import patch

import pytest

from cascade.providers._openai_tools import (
    _compact_messages_to_budget,
    _estimate_tokens,
    _read_dedup_key,
    openai_ask_with_tools,
)
from cascade.providers.base import ProviderConfig
from cascade.tools.schema import callable_to_tool_def


# --------------------------------------------------------------------------- #
# Token estimation
# --------------------------------------------------------------------------- #


def test_estimate_tokens_counts_content_and_tool_call_arguments():
    messages = [
        {"role": "user", "content": "a" * 40},  # 10 tokens
        {
            "role": "assistant",
            "content": None,  # robust to the None assistant tool-call carries
            "tool_calls": [
                {"id": "c1", "function": {"name": "write_file", "arguments": "b" * 40}}
            ],
        },
    ]
    # 40 content chars + (10 name + 40 args) tool-call chars = 90 chars -> 22 tokens
    assert _estimate_tokens(messages) == (40 + len("write_file") + 40) // 4


def test_estimate_tokens_ignores_non_string_content():
    assert _estimate_tokens([{"role": "assistant", "content": None}]) == 0


# --------------------------------------------------------------------------- #
# _compact_messages_to_budget
# --------------------------------------------------------------------------- #


def _tool_msg(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _assistant_call(call_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": "{}"}}],
    }


def test_compact_is_noop_when_under_budget():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _tool_msg("c1", "small result"),
    ]
    out = _compact_messages_to_budget(messages, budget=10_000, keep_recent=1)
    assert out == messages


def test_compact_evicts_oldest_tool_result_first():
    big = "X" * 8000  # ~2000 tokens each
    messages = [
        {"role": "user", "content": "task"},
        _assistant_call("c1", "read_file"),
        _tool_msg("c1", big),  # oldest large result -> should be elided
        _assistant_call("c2", "read_file"),
        _tool_msg("c2", big),  # newer, but outside keep_recent here
        {"role": "user", "content": "keep going"},
    ]
    # keep_recent=1 protects only the final user message; budget forces one evict.
    out = _compact_messages_to_budget(messages, budget=2500, keep_recent=1)

    assert out[2]["content"].startswith("[elided to fit context: read_file result,")
    assert out[4]["content"] == big  # the newer result survives


def test_compact_preserves_system_task_and_recent_window():
    big = "X" * 8000
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT " + "s" * 8000},
        {"role": "user", "content": "ORIGINAL TASK " + "t" * 8000},
        _assistant_call("c1", "read_file"),
        _tool_msg("c1", big),  # old -> elidable
        _assistant_call("c2", "read_file"),
        _tool_msg("c2", "RECENT " + big),  # inside keep_recent -> protected
    ]
    out = _compact_messages_to_budget(messages, budget=100, keep_recent=2)

    assert out[0]["content"].startswith("SYSTEM PROMPT")  # system never elided
    assert out[1]["content"].startswith("ORIGINAL TASK")  # first task never elided
    assert out[3]["content"].startswith("[elided")  # old tool result elided
    assert out[5]["content"].startswith("RECENT")  # recent tool result untouched


def test_compact_stub_names_the_tool_and_original_size():
    content = "Y" * 5000
    messages = [
        {"role": "user", "content": "task"},
        _assistant_call("c9", "read_file"),
        _tool_msg("c9", content),
        {"role": "user", "content": "next"},
    ]
    out = _compact_messages_to_budget(messages, budget=1, keep_recent=1)
    assert out[2]["content"] == f"[elided to fit context: read_file result, {len(content)} chars]"


def test_compact_leaves_small_results_alone_even_when_over_budget():
    # Nothing is beneficially elidable: each stub would be longer than the result.
    messages = [
        {"role": "user", "content": "task"},
        _assistant_call("c1", "read_file"),
        _tool_msg("c1", "tiny"),
        _assistant_call("c2", "read_file"),
        _tool_msg("c2", "also small"),
    ]
    out = _compact_messages_to_budget(messages, budget=0, keep_recent=0)
    assert [m["content"] for m in out] == [
        "task",
        None,
        "tiny",
        None,
        "also small",
    ]


def test_compact_one_huge_recent_result_is_protected():
    huge = "Z" * 40000
    messages = [
        {"role": "user", "content": "task"},
        _assistant_call("c1", "read_file"),
        _tool_msg("c1", huge),  # the only large payload, but it is the recent tail
    ]
    out = _compact_messages_to_budget(messages, budget=1, keep_recent=2)
    assert out[2]["content"] == huge  # protected -> stays full despite over budget


def test_compact_is_idempotent():
    big = "X" * 8000
    messages = [
        {"role": "user", "content": "task"},
        _assistant_call("c1", "read_file"),
        _tool_msg("c1", big),
        _assistant_call("c2", "read_file"),
        _tool_msg("c2", big),
        {"role": "user", "content": "next"},
    ]
    once = _compact_messages_to_budget(messages, budget=2500, keep_recent=1)
    twice = _compact_messages_to_budget(once, budget=2500, keep_recent=1)
    assert twice == once


def test_compact_does_not_mutate_input():
    big = "X" * 8000
    original_content = big
    messages = [
        {"role": "user", "content": "task"},
        _assistant_call("c1", "read_file"),
        _tool_msg("c1", big),
        {"role": "user", "content": "next"},
    ]
    _compact_messages_to_budget(messages, budget=1, keep_recent=1)
    assert messages[2]["content"] == original_content  # caller's dict untouched


# --------------------------------------------------------------------------- #
# _read_dedup_key
# --------------------------------------------------------------------------- #


def test_read_dedup_key_for_read_tools_with_a_path():
    assert _read_dedup_key("read_file", {"path": "a.py"}) == ("read_file", "a.py")
    assert _read_dedup_key("read", {"file": "b.py"}) == ("read", "b.py")


def test_read_dedup_key_none_for_non_read_tools():
    assert _read_dedup_key("write_file", {"path": "a.py"}) is None
    assert _read_dedup_key("list_files", {"path": "."}) is None


def test_read_dedup_key_none_without_a_path():
    assert _read_dedup_key("read_file", {}) is None
    assert _read_dedup_key("read_file", {"path": ""}) is None


# --------------------------------------------------------------------------- #
# Loop integration: a mock HTTP client drives the tool rounds
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _FakeClient:
    """Return queued chat-completions payloads and record every request body."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.payloads: list[dict] = []

    def post(self, url, json=None, headers=None):
        self.payloads.append(json)
        return _FakeResponse(self._responses.pop(0))


def _tool_call_response(call_id: str, name: str, args: dict) -> dict:
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


def _final_response(text: str) -> dict:
    return {
        "choices": [
            {"message": {"content": text, "tool_calls": []}, "finish_reason": "stop"}
        ]
    }


def _big_read_tools(calls: list[str]) -> dict:
    def read_file(path: str) -> str:
        """Read a file from the workspace."""
        calls.append(path)
        return "FILECONTENT " + "X" * 20000

    return {"read_file": callable_to_tool_def("read_file", read_file, "read", read_only=True)}


def _run_loop(client, tools, *, context_window, max_rounds=12):
    return openai_ask_with_tools(
        client=client,
        url="http://x/v1/chat/completions",
        headers={},
        model="qwen",
        temperature=0.0,
        max_tokens=None,
        messages=[{"role": "user", "content": "read the files then summarize"}],
        tools=tools,
        system="SYS",
        max_rounds=max_rounds,
        context_window=context_window,
    )


def test_loop_evicts_old_tool_results_on_a_small_window():
    calls: list[str] = []
    tools = _big_read_tools(calls)
    responses = [
        _tool_call_response("c1", "read_file", {"path": "a.py"}),
        _tool_call_response("c2", "read_file", {"path": "b.py"}),
        _tool_call_response("c3", "read_file", {"path": "c.py"}),
        _tool_call_response("c4", "read_file", {"path": "d.py"}),
        _tool_call_response("c5", "read_file", {"path": "e.py"}),
        _final_response("done"),
    ]
    client = _FakeClient(responses)

    text, log = _run_loop(client, tools, context_window=8000)

    assert text == "done"
    final_payload = client.payloads[-1]
    contents = [m.get("content") or "" for m in final_payload["messages"]]
    # The system prompt and the original task survive in full...
    assert any(c == "SYS" for c in contents)
    assert any("read the files then summarize" in c for c in contents)
    # ...the oldest reads are elided...
    assert any(c.startswith("[elided to fit context: read_file result,") for c in contents)
    # ...and the most recent read is still present verbatim.
    assert any("FILECONTENT" in c for c in contents)


def test_loop_does_not_evict_when_window_is_large():
    calls: list[str] = []
    tools = _big_read_tools(calls)
    responses = [
        _tool_call_response("c1", "read_file", {"path": "a.py"}),
        _tool_call_response("c2", "read_file", {"path": "b.py"}),
        _final_response("done"),
    ]
    client = _FakeClient(responses)

    text, _log = _run_loop(client, tools, context_window=200000)

    assert text == "done"
    final_payload = client.payloads[-1]
    contents = [m.get("content") or "" for m in final_payload["messages"]]
    assert not any(c.startswith("[elided") for c in contents)


def test_loop_deduplicates_repeat_reads_of_same_path():
    calls: list[str] = []
    tools = _big_read_tools(calls)
    responses = [
        _tool_call_response("c1", "read_file", {"path": "a.py"}),
        _tool_call_response("c2", "read_file", {"path": "a.py"}),  # same path again
        _final_response("done"),
    ]
    client = _FakeClient(responses)

    text, log = _run_loop(client, tools, context_window=200000)

    assert text == "done"
    assert calls == ["a.py"]  # the file was actually read only once
    # The second tool result the model sees is the dedup stub, not the content.
    second_result = log[1]["output"]
    assert second_result == "[already read above: a.py]"


def test_loop_does_not_deduplicate_distinct_paths():
    calls: list[str] = []
    tools = _big_read_tools(calls)
    responses = [
        _tool_call_response("c1", "read_file", {"path": "a.py"}),
        _tool_call_response("c2", "read_file", {"path": "b.py"}),
        _final_response("done"),
    ]
    client = _FakeClient(responses)

    _run_loop(client, tools, context_window=200000)
    assert calls == ["a.py", "b.py"]  # both distinct reads executed


# --------------------------------------------------------------------------- #
# context_window threading through the providers
# --------------------------------------------------------------------------- #


def test_provider_config_context_window_defaults_to_none():
    """None delegates window resolution to cascade.context.budget.window_for."""
    cfg = ProviderConfig(api_key="k", model="m")
    assert cfg.context_window is None


def _capture_context_window(module_path: str):
    """Return a fake loop that records the context_window it was called with."""
    captured: dict = {}

    def fake_loop(**kwargs):
        captured.update(kwargs)
        return ("ok", [])

    return captured, patch(f"{module_path}.openai_ask_with_tools", fake_loop)


def test_openai_compatible_forwards_context_window():
    from cascade.providers.openai_compatible import OpenAICompatibleProvider

    prov = OpenAICompatibleProvider(
        ProviderConfig(api_key="", model="qwen36", context_window=32768)
    )
    captured, patcher = _capture_context_window("cascade.providers.openai_compatible")
    with patcher:
        prov.ask_with_tools([{"role": "user", "content": "hi"}], tools={})
    assert captured["context_window"] == 32768


def test_openrouter_forwards_context_window():
    from cascade.providers.openrouter import OpenRouterProvider

    prov = OpenRouterProvider(
        ProviderConfig(api_key="k", model="qwen/qwen3.5-9b", context_window=64000)
    )
    captured, patcher = _capture_context_window("cascade.providers.openrouter")
    with patcher:
        prov.ask_with_tools([{"role": "user", "content": "hi"}], tools={})
    assert captured["context_window"] == 64000


def test_openai_provider_forwards_context_window():
    from cascade.providers.openai_provider import OpenAIProvider

    prov = OpenAIProvider(ProviderConfig(api_key="sk-test", model="gpt", context_window=200000))
    captured, patcher = _capture_context_window("cascade.providers.openai_provider")
    with patcher:
        prov.ask_with_tools([{"role": "user", "content": "hi"}], tools={})
    assert captured["context_window"] == 200000


# --------------------------------------------------------------------------- #
# provider_preferences threading (OpenRouter upstream-host pinning)
# --------------------------------------------------------------------------- #


def test_loop_pins_provider_field_when_preferences_set():
    prefs = {"order": ["Baidu", "Fireworks", "Alibaba"], "allow_fallbacks": True}
    client = _FakeClient([_final_response("done")])
    text, _log = openai_ask_with_tools(
        client=client,
        url="http://x/v1/chat/completions",
        headers={},
        model="qwen",
        temperature=0.0,
        max_tokens=None,
        messages=[{"role": "user", "content": "hi"}],
        tools={},
        system="SYS",
        max_rounds=3,
        context_window=200000,
        provider_preferences=prefs,
    )
    assert text == "done"
    assert client.payloads[0]["provider"] == prefs


def test_loop_omits_provider_field_when_preferences_none():
    client = _FakeClient([_final_response("done")])
    openai_ask_with_tools(
        client=client,
        url="http://x/v1/chat/completions",
        headers={},
        model="qwen",
        temperature=0.0,
        max_tokens=None,
        messages=[{"role": "user", "content": "hi"}],
        tools={},
        system="SYS",
        max_rounds=3,
        context_window=200000,
    )
    assert "provider" not in client.payloads[0]


def test_loop_raises_on_top_level_in_band_provider_error():
    client = _FakeClient([{
        "error": {
            "code": 429,
            "message": "provider overloaded",
            "metadata": {"error_type": "provider_overloaded"},
        },
        "choices": [],
    }])

    with pytest.raises(RuntimeError, match="provider_overloaded"):
        _run_loop(client, {}, context_window=200000)


def test_loop_raises_on_choice_error_after_generation_started():
    client = _FakeClient([{
        "choices": [{
            "message": {"content": "partial"},
            "finish_reason": "error",
            "error": {"code": 502, "message": "provider disconnected"},
        }],
    }])

    with pytest.raises(RuntimeError, match="provider disconnected"):
        _run_loop(client, {}, context_window=200000)
