"""Token-economy behaviour on the claude/gemini direct-API + CLI-proxy loops.

Covers three fixes:

(a) claude -p (CLI proxy) undercounts a multi-round solve because the base
    event handler keeps only the final internal turn's usage. Accumulated spend
    must sum every turn while the context anchor stays the final turn.
(b) The hand-rolled claude/gemini tool loops gain the two protections the shared
    OpenAI loop already has: a budget compaction that stubs old oversized tool
    results, and a read-dedup that serves a repeat read from a short stub.
(c) The claude direct-API tool loop sets cache_control breakpoints, and its token
    accounting stays correct once most input is reported under the cache fields.
"""

from unittest.mock import MagicMock, patch

import cascade.providers.claude as claude_mod
import cascade.providers.gemini as gemini_mod
from cascade.providers.base import Message, ProviderConfig
from cascade.providers.claude import ClaudeProvider, _compact_anthropic_tool_results
from cascade.providers.gemini import GeminiProvider, _compact_gemini_tool_results
from cascade.providers.usage import Usage
from cascade.tools.schema import callable_to_tool_def


def _msgs(prompt: str) -> list[Message]:
    return [{"role": "user", "content": prompt}]


def _resp(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _fake_popen(lines: list[str]):
    class _FakePopen:
        def __init__(self, *_a, **_k):
            self.stdout = iter(lines)
            self.returncode = 0

        def wait(self):
            return 0

    return _FakePopen


# --------------------------------------------------------------------------- #
# (a) CLI-proxy output accumulation across claude -p's internal turns
# --------------------------------------------------------------------------- #


def _cli_provider() -> ClaudeProvider:
    with patch("cascade.providers.claude.shutil.which", return_value="/usr/bin/claude"):
        with patch.dict("os.environ", {"CASCADE_CLAUDE_ACTIVITY": "0"}, clear=False):
            return ClaudeProvider(
                ProviderConfig(api_key="sk-ant-oat01-token", model="claude-opus-4-8")
            )


def test_cli_proxy_accumulates_output_across_internal_turns():
    lines = [
        '{"type":"system","subtype":"init","model":"claude-opus-4-8"}\n',
        '{"type":"stream_event","event":{"type":"message_start","message":{"usage":{"input_tokens":100,"output_tokens":1}}}}\n',
        '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"text":"aaa"}}}\n',
        '{"type":"stream_event","event":{"type":"message_delta","usage":{"output_tokens":50}}}\n',
        '{"type":"stream_event","event":{"type":"message_start","message":{"usage":{"input_tokens":120,"output_tokens":1}}}}\n',
        '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"text":"bbb"}}}\n',
        '{"type":"stream_event","event":{"type":"message_delta","usage":{"output_tokens":80}}}\n',
        '{"type":"result","subtype":"success","usage":{"input_tokens":120,"output_tokens":80}}\n',
    ]
    prov = _cli_provider()
    with patch("cascade.providers._cli_proxy.subprocess.Popen", _fake_popen(lines)):
        chunks = list(prov.stream_single("go"))

    assert chunks == ["aaa", "bbb"]
    # Accumulated spend sums both internal turns, not just the final one.
    assert prov.last_usage == Usage(input=220, output=130)
    # The context anchor stays the final turn only (the deliberate invariant).
    assert prov.last_round_usage == Usage(input=120, output=80)
    assert prov.last_usage.output > prov.last_round_usage.output


def test_cli_proxy_accumulation_is_cache_aware():
    # message_start reports cache_read_input_tokens; accumulation must retain it,
    # not collapse the prompt to bare input_tokens.
    lines = [
        '{"type":"stream_event","event":{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":5000}}}}\n',
        '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"text":"a"}}}\n',
        '{"type":"stream_event","event":{"type":"message_delta","usage":{"output_tokens":40}}}\n',
        '{"type":"stream_event","event":{"type":"message_start","message":{"usage":{"input_tokens":12,"output_tokens":1,"cache_read_input_tokens":5200}}}}\n',
        '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"text":"b"}}}\n',
        '{"type":"stream_event","event":{"type":"message_delta","usage":{"output_tokens":60}}}\n',
        '{"type":"result","subtype":"success","usage":{"input_tokens":12,"output_tokens":60,"cache_read_input_tokens":5200}}\n',
    ]
    prov = _cli_provider()
    with patch("cascade.providers._cli_proxy.subprocess.Popen", _fake_popen(lines)):
        list(prov.stream_single("go"))

    assert prov.last_usage == Usage(input=22, output=100, cache_read=10200)
    assert prov.last_usage.prompt_total == 10222


def test_cli_proxy_single_turn_still_uses_result_usage():
    # No message_start/message_delta pair -> fall back to the final result usage
    # (the pre-existing single-turn behaviour must not regress).
    lines = [
        '{"type":"system","subtype":"init","model":"claude-opus-4-8"}\n',
        '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"text":"Hi"}}}\n',
        '{"type":"result","subtype":"success","usage":{"input_tokens":11,"output_tokens":3}}\n',
    ]
    prov = _cli_provider()
    with patch("cascade.providers._cli_proxy.subprocess.Popen", _fake_popen(lines)):
        chunks = list(prov.stream_single("hi"))

    assert chunks == ["Hi"]
    assert prov.last_usage == Usage(input=11, output=3)
    assert prov.last_round_usage == Usage(input=11, output=3)


# --------------------------------------------------------------------------- #
# (c) Anthropic prompt caching + cache-aware accounting on the direct-API loop
# --------------------------------------------------------------------------- #


def _api_provider() -> ClaudeProvider:
    with patch("cascade.providers.claude.shutil.which", return_value=None):
        return ClaudeProvider(
            ProviderConfig(
                api_key="sk-ant-api03-key",
                model="claude-opus-4-8",
                temperature=0.7,
                max_tokens=1024,
            )
        )


def _echo_tools() -> dict:
    def echo(message: str) -> str:
        """Echo a message back."""
        return message

    return {"echo": callable_to_tool_def("echo", echo, "Echo tool")}


def _tool_use(tid: str, name: str, inp: dict) -> dict:
    return {
        "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 5},
    }


def _final(text: str, usage: dict | None = None) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": usage or {"input_tokens": 1, "output_tokens": 1},
    }


def test_claude_caching_marks_last_tool_and_system_block():
    def a(x: str) -> str:
        """Tool a."""
        return x

    def b(x: str) -> str:
        """Tool b."""
        return x

    tools = {
        "a": callable_to_tool_def("a", a, "a"),
        "b": callable_to_tool_def("b", b, "b"),
    }
    prov = _api_provider()
    with patch.object(prov.client, "post", return_value=_resp(_final("done"))) as post:
        out, _log = prov.ask_with_tools(_msgs("hi"), tools, system="SYS")

    payload = post.call_args.kwargs["json"]
    # One breakpoint on the last tool only.
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in payload["tools"][0]
    # System is promoted to a cached text block.
    assert payload["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    assert out == "done"


def test_claude_caching_omitted_when_no_system():
    prov = _api_provider()
    with patch.object(prov.client, "post", return_value=_resp(_final("done"))) as post:
        prov.ask_with_tools(_msgs("hi"), _echo_tools())
    assert "system" not in post.call_args.kwargs["json"]


def test_claude_accounting_stays_correct_for_a_cached_response():
    # With caching enabled Anthropic reports most input under the cache fields;
    # accounting that read only input_tokens would collapse to ~zero prompt.
    cached = _final(
        "ok",
        usage={
            "input_tokens": 12,
            "output_tokens": 40,
            "cache_read_input_tokens": 18000,
            "cache_creation_input_tokens": 700,
        },
    )
    prov = _api_provider()
    with patch.object(prov.client, "post", return_value=_resp(cached)):
        prov.ask_with_tools(_msgs("hi"), _echo_tools(), system="SYS")

    assert prov.last_usage == Usage(input=12, output=40, cache_read=18000, cache_write=700)
    assert prov.last_usage.prompt_total == 18712  # not collapsed to 12
    assert prov.last_round_usage == prov.last_usage


def test_claude_multi_round_cached_spend_accumulates_and_anchors_final_round():
    # Round 1 writes cache; round 2 reads it. Spend accumulates; the anchor is
    # the final round only -- the two must stay distinct even under caching.
    r1 = _tool_use("t1", "echo", {"message": "x"})
    r1["usage"] = {
        "input_tokens": 500,
        "output_tokens": 10,
        "cache_creation_input_tokens": 2000,
    }
    r2 = _final("done", usage={"input_tokens": 5, "output_tokens": 8, "cache_read_input_tokens": 2400})
    prov = _api_provider()
    with patch.object(prov.client, "post", side_effect=[_resp(r1), _resp(r2)]):
        out, _log = prov.ask_with_tools(_msgs("go"), _echo_tools())

    assert out == "done"
    assert prov.last_usage == Usage(input=505, output=18, cache_read=2400, cache_write=2000)
    assert prov.last_round_usage == Usage(input=5, output=8, cache_read=2400)
    assert prov.last_round_usage.total < prov.last_usage.total


# --------------------------------------------------------------------------- #
# (b) read-dedup + budget compaction on the claude direct-API loop
# --------------------------------------------------------------------------- #


def _read_tools(calls: list[str]) -> dict:
    def read_file(path: str) -> str:
        """Read a file."""
        calls.append(path)
        return "CONTENT " + "X" * 200

    return {"read_file": callable_to_tool_def("read_file", read_file, "read", read_only=True)}


def test_claude_dedup_repeat_read_returns_stub():
    calls: list[str] = []
    tools = _read_tools(calls)
    prov = _api_provider()
    responses = [
        _resp(_tool_use("t1", "read_file", {"path": "a.py"})),
        _resp(_tool_use("t2", "read_file", {"path": "a.py"})),
        _resp(_final("done")),
    ]
    with patch.object(prov.client, "post", side_effect=responses):
        out, log = prov.ask_with_tools(_msgs("read a"), tools, max_rounds=5)

    assert calls == ["a.py"]  # actually read only once
    assert log[1]["output"] == "[already read above: a.py]"
    assert out == "done"


def test_claude_dedup_distinct_paths_both_execute():
    calls: list[str] = []
    tools = _read_tools(calls)
    prov = _api_provider()
    responses = [
        _resp(_tool_use("t1", "read_file", {"path": "a.py"})),
        _resp(_tool_use("t2", "read_file", {"path": "b.py"})),
        _resp(_final("done")),
    ]
    with patch.object(prov.client, "post", side_effect=responses):
        prov.ask_with_tools(_msgs("read"), tools, max_rounds=5)

    assert calls == ["a.py", "b.py"]


def test_compact_anthropic_stubs_old_tool_result_preserving_pairing():
    big = "X" * 8000
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "b"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "RECENT " + big}]},
    ]
    out = _compact_anthropic_tool_results(messages, budget=2500, keep_recent=2)

    stubbed = out[2]["content"][0]
    assert stubbed["content"].startswith("[elided to fit context: tool result,")
    assert stubbed["tool_use_id"] == "t1"  # pairing intact
    assert out[4]["content"][0]["content"].startswith("RECENT")  # recent untouched
    assert out[0]["content"] == "task"  # first task never elided
    assert messages[2]["content"][0]["content"] == big  # input not mutated


def test_compact_anthropic_noop_under_budget():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "small"}]},
    ]
    assert _compact_anthropic_tool_results(messages, budget=10_000, keep_recent=1) == messages


def test_claude_ask_with_tools_compacts_each_round_using_claude_window():
    prov = _api_provider()
    responses = [_resp(_tool_use("t1", "echo", {"message": "x"})), _resp(_final("done"))]
    budgets: list[int] = []
    real = claude_mod._compact_anthropic_tool_results

    def spy(msgs, budget, keep):
        budgets.append(budget)
        return real(msgs, budget, keep)

    with patch("cascade.providers.claude._compact_anthropic_tool_results", spy):
        with patch.object(prov.client, "post", side_effect=responses):
            prov.ask_with_tools(_msgs("go"), _echo_tools(), max_rounds=3)

    assert len(budgets) == 2  # once per round
    assert budgets[0] == int(1_000_000 * 0.7)  # window_for("claude","claude-opus-4-8") = 1M


# --------------------------------------------------------------------------- #
# (b) read-dedup + budget compaction on the gemini direct-API loop
# --------------------------------------------------------------------------- #


def _gemini_provider() -> GeminiProvider:
    with patch("cascade.providers.gemini.shutil.which", return_value=None):
        return GeminiProvider(ProviderConfig(api_key="AIza-key", model="gemini-2.5-flash"))


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


def test_gemini_dedup_repeat_read_returns_stub():
    calls: list[str] = []
    tools = _read_tools(calls)
    prov = _gemini_provider()
    responses = [
        _resp(_g_call("read_file", {"path": "a.py"})),
        _resp(_g_call("read_file", {"path": "a.py"})),
        _resp(_g_text("done")),
    ]
    with patch.object(prov.client, "post", side_effect=responses):
        out, log = prov.ask_with_tools(_msgs("read a"), tools, max_rounds=5)

    assert calls == ["a.py"]
    assert log[1]["output"] == "[already read above: a.py]"
    assert out == "done"


def test_compact_gemini_stubs_old_function_response_preserving_name():
    big = "Y" * 8000
    contents = [
        {"role": "user", "parts": [{"text": "task"}]},
        {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {"path": "a"}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {"result": big}}}]},
        {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {"path": "b"}}}]},
        {"role": "user", "parts": [{"functionResponse": {"name": "read_file", "response": {"result": "RECENT " + big}}}]},
    ]
    out = _compact_gemini_tool_results(contents, budget=2500, keep_recent=2)

    stubbed = out[2]["parts"][0]["functionResponse"]
    assert stubbed["response"]["result"].startswith("[elided to fit context: tool result,")
    assert stubbed["name"] == "read_file"  # which call it answers is preserved
    assert out[4]["parts"][0]["functionResponse"]["response"]["result"].startswith("RECENT")
    assert contents[2]["parts"][0]["functionResponse"]["response"]["result"] == big  # not mutated


def test_gemini_ask_with_tools_compacts_each_round_using_gemini_window():
    prov = _gemini_provider()
    responses = [_resp(_g_call("echo", {"message": "x"})), _resp(_g_text("done"))]
    budgets: list[int] = []
    real = gemini_mod._compact_gemini_tool_results

    def spy(contents, budget, keep):
        budgets.append(budget)
        return real(contents, budget, keep)

    tools = {"echo": callable_to_tool_def("echo", lambda message: message, "echo")}
    with patch("cascade.providers.gemini._compact_gemini_tool_results", spy):
        with patch.object(prov.client, "post", side_effect=responses):
            prov.ask_with_tools(_msgs("go"), tools, max_rounds=3)

    assert len(budgets) == 2
    assert budgets[0] == int(1_000_000 * 0.7)  # window_for("gemini") default * fraction
