"""Tests for provider tool-calling integration.

These tests mock the HTTP layer to verify that providers correctly format
tool definitions and handle tool_use/tool_result round trips.
"""

import itertools
import json
from unittest.mock import patch, MagicMock

import httpx

from cascade.providers.base import BaseProvider, ProviderConfig, Message
from cascade.tools.schema import callable_to_tool_def
from cascade.providers.usage import Usage


def _make_tools():
    """Build a small tool registry for testing."""
    def echo(message: str) -> str:
        """Echo a message back."""
        return message

    return {
        "echo": callable_to_tool_def("echo", echo, "Echo tool"),
    }


def _make_config():
    return ProviderConfig(
        api_key="test-key",
        model="test-model",
        temperature=0.7,
        max_tokens=1024,
    )


def _msgs(prompt: str) -> list[Message]:
    """Build a single-message list from a prompt string."""
    return [{"role": "user", "content": prompt}]


def _make_edit_run_tools():
    """Registry with a file-writing tool and a shell-running tool."""
    def write_file(path: str, content: str) -> str:
        """Write content to a file."""
        return f"wrote {path}"

    def run_command(command: str) -> str:
        """Run a shell command."""
        return f"ran {command}"

    return {
        "write_file": callable_to_tool_def("write_file", write_file, "Write a file"),
        "run_command": callable_to_tool_def("run_command", run_command, "Run a command"),
    }


def _tool_call_response(tool_name: str, arguments: dict, call_id: str) -> MagicMock:
    """Mock chat-completions response carrying a single tool call."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "choices": [{
            "message": {"content": "", "tool_calls": [{
                "id": call_id,
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }]},
            "finish_reason": "tool_calls",
        }],
    }
    resp.raise_for_status = MagicMock()
    return resp


def _text_response(text: str) -> MagicMock:
    """Mock final (no tool call) chat-completions response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "choices": [{
            "message": {"content": text, "tool_calls": []},
            "finish_reason": "stop",
        }],
    }
    resp.raise_for_status = MagicMock()
    return resp


class TestBaseProviderToolCalling:
    """Test the default ask_with_tools fallback."""

    def test_default_falls_back_to_ask(self):
        """BaseProvider.ask_with_tools should fall back to ask()."""
        class StubProvider(BaseProvider):
            def ask(self, messages, system=None):
                return f"echo: {messages[-1]['content']}"
            def stream(self, messages, system=None):
                yield self.ask(messages, system)
            def compare(self, prompt, system=None):
                return {}

        prov = StubProvider(_make_config())
        result, log = prov.ask_with_tools(_msgs("hello"), _make_tools())
        assert result == "echo: hello"
        assert log == []


class TestClaudeToolCalling:
    """Test Claude provider tool-calling format."""

    def test_tool_definitions_format(self):
        """Verify Claude tool defs use input_schema."""
        from cascade.providers.claude import ClaudeProvider

        prov = ClaudeProvider(_make_config())
        tools = _make_tools()

        # Mock a simple text response (no tool calls)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "No tools needed."}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(prov.client, "post", return_value=mock_response) as mock_post:
            result, log = prov.ask_with_tools(_msgs("test"), tools)

            # Verify the payload
            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "tools" in payload
            assert payload["tools"][0]["name"] == "echo"
            assert "input_schema" in payload["tools"][0]

        assert result == "No tools needed."
        assert log == []

    def test_tool_call_round_trip(self):
        """Verify Claude tool_use -> execute -> tool_result flow."""
        from cascade.providers.claude import ClaudeProvider

        prov = ClaudeProvider(_make_config())
        tools = _make_tools()

        # First response: tool_use
        tool_use_response = MagicMock()
        tool_use_response.status_code = 200
        tool_use_response.json.return_value = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "echo",
                    "input": {"message": "hello"},
                }
            ],
            "stop_reason": "tool_use",
        }
        tool_use_response.raise_for_status = MagicMock()

        # Second response: final text
        final_response = MagicMock()
        final_response.status_code = 200
        final_response.json.return_value = {
            "content": [{"type": "text", "text": "The echo returned: hello"}],
            "stop_reason": "end_turn",
        }
        final_response.raise_for_status = MagicMock()

        with patch.object(
            prov.client, "post",
            side_effect=[tool_use_response, final_response],
        ):
            result, log = prov.ask_with_tools(_msgs("echo hello"), tools)

        assert result == "The echo returned: hello"
        assert len(log) == 1
        assert log[0]["tool"] == "echo"
        assert log[0]["input"] == {"message": "hello"}


class TestGeminiToolCalling:
    """Test Gemini provider tool-calling format."""

    def test_function_declarations_format(self):
        """Verify Gemini uses function_declarations."""
        from cascade.providers.gemini import GeminiProvider

        prov = GeminiProvider(_make_config())
        tools = _make_tools()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Done."}],
                },
            }],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(prov.client, "post", return_value=mock_response) as mock_post:
            result, log = prov.ask_with_tools(_msgs("test"), tools)

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "tools" in payload
            assert "function_declarations" in payload["tools"][0]

        assert result == "Done."

    def test_function_call_round_trip(self):
        """Verify Gemini functionCall -> execute -> functionResponse flow."""
        from cascade.providers.gemini import GeminiProvider

        prov = GeminiProvider(_make_config())
        tools = _make_tools()

        fc_response = MagicMock()
        fc_response.status_code = 200
        fc_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "echo",
                            "args": {"message": "ping"},
                        }
                    }],
                },
            }],
        }
        fc_response.raise_for_status = MagicMock()

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Echo said: ping"}],
                },
            }],
        }
        final_response.raise_for_status = MagicMock()

        with patch.object(
            prov.client, "post",
            side_effect=[fc_response, final_response],
        ):
            result, log = prov.ask_with_tools(_msgs("echo ping"), tools)

        assert result == "Echo said: ping"
        assert len(log) == 1
        assert log[0]["tool"] == "echo"


class TestOpenAIToolCalling:
    """Test OpenAI provider tool-calling format."""

    def test_openai_tool_format(self):
        """Verify OpenAI uses type:function wrapper."""
        from cascade.providers.openai_provider import OpenAIProvider

        prov = OpenAIProvider(_make_config())
        tools = _make_tools()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            "choices": [{
                "message": {"content": "OK", "tool_calls": []},
                "finish_reason": "stop",
            }],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(prov.client, "post", return_value=mock_response) as mock_post:
            result, log = prov.ask_with_tools(_msgs("test"), tools)

            call_kwargs = mock_post.call_args
            payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert "tools" in payload
            tool_def = payload["tools"][0]
            assert tool_def["type"] == "function"
            assert "function" in tool_def

        assert result == "OK"
        assert prov.last_usage == Usage(input=7, output=2)


class TestToolLoopExhaustion:
    """The tool loop must never hand back a silent empty string on round exhaustion."""

    def test_exhaustion_returns_guidance_not_empty(self):
        """When the model tool-calls until the budget is spent without finishing,
        surface a clear message (pointing at /solve), not ''."""
        from cascade.providers.openai_provider import OpenAIProvider

        prov = OpenAIProvider(_make_config())
        tools = _make_tools()

        # Every round is a tool call with empty content -- the model never finishes.
        tool_round = MagicMock()
        tool_round.status_code = 200
        tool_round.json.return_value = {
            "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "echo", "arguments": '{"message": "x"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        tool_round.raise_for_status = MagicMock()

        with patch.object(prov.client, "post", return_value=tool_round):
            result, log = prov.ask_with_tools(
                _msgs("do a big multi-step task"), tools, max_rounds=2
            )

        assert result.strip()      # not the silent empty that looked like "stopped"
        assert "/solve" in result  # points at the right tool
        assert len(log) == 2       # ran the tool each round until the budget was spent


class TestDoomLoopGuard:
    """A model spinning on the same tool call is stalled -- nudge, then bail."""

    @staticmethod
    def _repeating_response():
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "choices": [{
                "message": {"content": "", "tool_calls": [{
                    "id": "c1",
                    "function": {"name": "echo", "arguments": '{"message": "same"}'},
                }]},
                "finish_reason": "tool_calls",
            }],
        }
        resp.raise_for_status = MagicMock()
        return resp

    def test_doom_loop_bails_before_burning_the_round_budget(self):
        from cascade.providers.openai_provider import OpenAIProvider

        prov = OpenAIProvider(_make_config())
        with patch.object(prov.client, "post", return_value=self._repeating_response()) as post:
            result, _log = prov.ask_with_tools(_msgs("go"), _make_tools(), max_rounds=20)

        # Bailed near the 4th identical call, not the full 20-round budget.
        assert post.call_count <= 5
        assert "stall" in result.lower() or "handing off" in result.lower()

    def test_doom_loop_nudges_on_the_third_call_before_bailing(self):
        from cascade.providers.openai_provider import OpenAIProvider

        prov = OpenAIProvider(_make_config())
        with patch.object(prov.client, "post", return_value=self._repeating_response()):
            _result, log = prov.ask_with_tools(_msgs("go"), _make_tools(), max_rounds=20)

        # The 3rd identical call is intercepted with a corrective nudge, not run again.
        outputs = [entry["output"] for entry in log]
        assert any("different approach" in out for out in outputs)


class TestEditRunCycleGuard:
    """Alternating edits and command runs make no progress -- bail before the budget."""

    def test_edit_run_cycle_bails_before_max_rounds(self):
        """write_file(same path) alternating with run_command past the threshold
        trips the edit-run guard (which the consecutive-identical doom guard misses,
        since the alternation resets its streak)."""
        from cascade.providers.openai_provider import OpenAIProvider

        prov = OpenAIProvider(_make_config())
        edit = _tool_call_response("write_file", {"path": "app.py", "content": "x"}, "e1")
        run_x = _tool_call_response("run_command", {"command": "pytest -x"}, "r1")
        run_v = _tool_call_response("run_command", {"command": "pytest -v"}, "r2")
        # The two pytest invocations differ only in flags, proving they share one
        # normalized-prefix bucket ("pytest") rather than counting separately.
        cycle = itertools.cycle([edit, run_x, edit, run_v])

        with patch.object(prov.client, "post", side_effect=cycle) as post:
            result, _log = prov.ask_with_tools(
                _msgs("fix the failing test"), _make_edit_run_tools(), max_rounds=30
            )

        assert "edit-run cycle" in result.lower()
        assert "app.py" in result
        # Bailed once one file (6 edits) and one command (6 runs) both crossed the
        # threshold -- at the 12th call, not the full 30-round budget.
        assert post.call_count == 12

    def test_short_edit_run_burst_then_finish_does_not_bail(self):
        """A handful of edit/run cycles (below the threshold) that then finish must
        return the model's answer, not a stall note."""
        from cascade.providers.openai_provider import OpenAIProvider

        prov = OpenAIProvider(_make_config())
        edit = _tool_call_response("write_file", {"path": "app.py", "content": "x"}, "e1")
        run = _tool_call_response("run_command", {"command": "pytest"}, "r1")
        done = _text_response("All tests pass.")
        responses = [edit, run, edit, run, edit, run, done]  # 3 cycles, then finish

        with patch.object(prov.client, "post", side_effect=responses) as post:
            result, log = prov.ask_with_tools(
                _msgs("fix the failing test"), _make_edit_run_tools(), max_rounds=30
            )

        assert result == "All tests pass."
        assert "edit-run cycle" not in result.lower()
        assert post.call_count == 7
        assert len(log) == 6

    def test_identical_edit_repeats_still_trip_the_doom_guard(self):
        """The consecutive-identical doom guard must still fire for edit tools:
        repeating write_file with identical args bails on the doom path (4th call),
        well before the edit-run threshold -- edit-run counting must not suppress it."""
        from cascade.providers.openai_provider import OpenAIProvider

        prov = OpenAIProvider(_make_config())
        edit = _tool_call_response("write_file", {"path": "app.py", "content": "x"}, "e1")

        with patch.object(prov.client, "post", return_value=edit) as post:
            result, _log = prov.ask_with_tools(
                _msgs("write the file"), _make_edit_run_tools(), max_rounds=30
            )

        assert "identical arguments" in result
        assert "edit-run cycle" not in result.lower()
        assert post.call_count <= 5  # doom bailed near the 4th identical call


class TestOpenRouterToolCalling:
    """Test OpenRouter provider uses same format as OpenAI."""

    def test_openrouter_tool_format(self):
        from cascade.providers.openrouter import OpenRouterProvider

        prov = OpenRouterProvider(_make_config())
        tools = _make_tools()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            "choices": [{
                "message": {"content": "OK", "tool_calls": []},
                "finish_reason": "stop",
            }],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(prov.client, "post", return_value=mock_response):
            result, log = prov.ask_with_tools(_msgs("test"), tools)
            assert result == "OK"
            assert prov.last_usage == Usage(input=9, output=3)

    def test_openrouter_tool_calling_falls_back_on_503(self):
        from cascade.providers.openrouter import OpenRouterProvider

        prov = OpenRouterProvider(
            ProviderConfig(
                api_key="test-key",
                model="qwen/qwen3.5-9b",
                fallback_model="minimax/minimax-m2.5",
            )
        )
        tools = _make_tools()

        first_request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        first_response = httpx.Response(503, request=first_request)

        fallback_response = MagicMock()
        fallback_response.status_code = 200
        fallback_response.json.return_value = {
            "choices": [{
                "message": {"content": "OK", "tool_calls": []},
                "finish_reason": "stop",
            }],
        }
        fallback_response.raise_for_status = MagicMock()

        with patch.object(prov.client, "post", side_effect=[httpx.HTTPStatusError("503", request=first_request, response=first_response), fallback_response]) as mock_post:
            result, log = prov.ask_with_tools(_msgs("test"), tools)

        assert result == "OK"
        assert log == []
        first_payload = mock_post.call_args_list[0].kwargs["json"]
        second_payload = mock_post.call_args_list[1].kwargs["json"]
        assert first_payload["model"] == "qwen/qwen3.5-9b"
        assert second_payload["model"] == "minimax/minimax-m2.5"


class TestHookGating:
    """TOOL_CALL hooks must actually gate tool execution inside provider loops.

    Regression guard: every live ToolExecutor used to be constructed
    without a hook_runner, so hooks could never block a real tool call.
    """

    class _BlockingRunner:
        """Minimal HookRunner stand-in blocking every TOOL_CALL."""

        def __init__(self):
            self.tool_call_events = 0

        def emit(self, event, ctx=None):
            from types import SimpleNamespace

            from cascade.hooks import HookEvent

            if event == HookEvent.TOOL_CALL:
                self.tool_call_events += 1
                return SimpleNamespace(
                    block=True, reason="denied by test", transformed_value=None,
                )
            return None

    def test_blocking_hook_prevents_tool_execution_in_openai_loop(self):
        from cascade.providers.openai_provider import OpenAIProvider

        executed = []

        def echo(message: str) -> str:
            """Echo a message back."""
            executed.append(message)
            return message

        tools = {"echo": callable_to_tool_def("echo", echo, "Echo tool")}
        prov = OpenAIProvider(_make_config())
        runner = self._BlockingRunner()
        prov.hook_runner = runner

        responses = [
            _tool_call_response("echo", {"message": "ping"}, "call_1"),
            _text_response("done"),
        ]
        with patch.object(prov.client, "post", side_effect=responses) as mock_post:
            result, log = prov.ask_with_tools(_msgs("echo ping"), tools)

            # The blocked result is what the model sees in round 2
            second_payload = mock_post.call_args_list[1].kwargs.get("json") \
                or mock_post.call_args_list[1][1].get("json")
            tool_messages = [
                m for m in second_payload["messages"] if m.get("role") == "tool"
            ]
            assert len(tool_messages) == 1
            assert "blocked by hook" in tool_messages[0]["content"]
            assert "denied by test" in tool_messages[0]["content"]

        assert result == "done"
        assert executed == []
        assert runner.tool_call_events == 1

    def test_provider_without_hook_runner_still_executes_tools(self):
        from cascade.providers.openai_provider import OpenAIProvider

        executed = []

        def echo(message: str) -> str:
            """Echo a message back."""
            executed.append(message)
            return message

        tools = {"echo": callable_to_tool_def("echo", echo, "Echo tool")}
        prov = OpenAIProvider(_make_config())
        assert prov.hook_runner is None

        responses = [
            _tool_call_response("echo", {"message": "ping"}, "call_1"),
            _text_response("done"),
        ]
        with patch.object(prov.client, "post", side_effect=responses):
            result, _log = prov.ask_with_tools(_msgs("echo ping"), tools)

        assert result == "done"
        assert executed == ["ping"]


class TestRoundUsageAnchor:
    """last_usage accumulates spend across rounds; last_round_usage is the
    final round only -- the context anchor must never be the multi-round sum."""

    def test_two_round_loop_splits_spend_from_anchor(self):
        from cascade.providers.openai_provider import OpenAIProvider

        def echo(message: str) -> str:
            """Echo a message back."""
            return message

        tools = {"echo": callable_to_tool_def("echo", echo, "Echo tool")}
        prov = OpenAIProvider(_make_config())

        round1 = _tool_call_response("echo", {"message": "ping"}, "call_1")
        round1.json.return_value["usage"] = {"prompt_tokens": 100, "completion_tokens": 10}
        round2 = _text_response("done")
        round2.json.return_value["usage"] = {"prompt_tokens": 150, "completion_tokens": 5}

        with patch.object(prov.client, "post", side_effect=[round1, round2]):
            result, _log = prov.ask_with_tools(_msgs("echo ping"), tools)

        assert result == "done"
        assert prov.last_usage == Usage(input=250, output=15)
        assert prov.last_round_usage == Usage(input=150, output=5)
        assert prov.last_round_usage.total < prov.last_usage.total
