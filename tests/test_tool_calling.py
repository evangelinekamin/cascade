"""Tests for provider tool-calling integration.

These tests mock the HTTP layer to verify that providers correctly format
tool definitions and handle tool_use/tool_result round trips.
"""

from unittest.mock import patch, MagicMock

import httpx

from cascade.providers.base import BaseProvider, ProviderConfig, Message
from cascade.tools.schema import callable_to_tool_def


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
        assert prov.last_usage == (7, 2)


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
            assert prov.last_usage == (9, 3)

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
