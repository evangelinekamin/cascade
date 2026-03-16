"""Tests for AgentRunner -- context manager safety, tool filtering."""

import pytest
from unittest.mock import MagicMock

from cascade.agents.schema import AgentDef
from cascade.agents.runner import AgentRunner
from cascade.providers.base import ProviderConfig


def _make_app(
    provider_name="gemini",
    model="gemini-pro",
    temperature=0.7,
    tools=None,
):
    """Build a minimal mock CascadeApp with one provider."""
    config = ProviderConfig(api_key="test", model=model, temperature=temperature)

    provider = MagicMock()
    provider.config = config
    provider.ask.return_value = "response"
    provider.ask_single.return_value = "response"
    provider.ask_with_tools.return_value = ("tool response", [{"tool": "x"}])
    provider.stream.return_value = iter(["chunk1", "chunk2"])
    provider.stream_single.return_value = iter(["chunk1", "chunk2"])

    app = MagicMock()
    app.providers = {provider_name: provider}
    app.config.get_default_provider.return_value = provider_name
    app.tool_registry = tools or {}

    # Real PromptPipeline for system prompt tests
    from cascade.prompts.layers import PromptPipeline
    app.prompt_pipeline = PromptPipeline()

    return app, provider


class TestAgentRunner:
    def test_run_simple(self):
        app, prov = _make_app()
        runner = AgentRunner(app)
        agent = AgentDef(name="test")

        result = runner.run(agent, "hello")
        assert result == "response"
        prov.ask_single.assert_called_once()

    def test_run_with_tools(self):
        tools = {"read_file": MagicMock(), "write_file": MagicMock()}
        app, prov = _make_app(tools=tools)
        runner = AgentRunner(app)
        agent = AgentDef(name="test")  # allowed_tools=None -> unrestricted

        result = runner.run(agent, "hello")
        assert result == "tool response"
        prov.ask_with_tools.assert_called_once()

    def test_run_with_empty_allowed_tools_skips_tools(self):
        tools = {"read_file": MagicMock()}
        app, prov = _make_app(tools=tools)
        runner = AgentRunner(app)
        agent = AgentDef(name="test", allowed_tools=())

        result = runner.run(agent, "hello")
        assert result == "response"
        prov.ask_single.assert_called_once()
        prov.ask_with_tools.assert_not_called()

    def test_run_with_filtered_tools(self):
        tools = {"read_file": MagicMock(), "write_file": MagicMock(), "delete": MagicMock()}
        app, prov = _make_app(tools=tools)
        runner = AgentRunner(app)
        agent = AgentDef(name="test", allowed_tools=("read_file",))

        runner.run(agent, "hello")
        call_args = prov.ask_with_tools.call_args
        passed_tools = call_args[0][1]
        assert "read_file" in passed_tools
        assert "write_file" not in passed_tools
        assert "delete" not in passed_tools

    def test_model_override_restored(self):
        app, prov = _make_app(model="original-model")
        runner = AgentRunner(app)
        agent = AgentDef(name="test", model="override-model")

        runner.run(agent, "hello")
        # After run, model should be restored
        assert prov.config.model == "original-model"

    def test_temperature_override_restored(self):
        app, prov = _make_app(temperature=0.7)
        runner = AgentRunner(app)
        agent = AgentDef(name="test", temperature=1.5)

        runner.run(agent, "hello")
        assert prov.config.temperature == 0.7

    def test_model_restored_on_exception(self):
        app, prov = _make_app(model="original")
        prov.ask_single.side_effect = RuntimeError("boom")
        runner = AgentRunner(app)
        agent = AgentDef(name="test", model="override")

        with pytest.raises(RuntimeError, match="boom"):
            runner.run(agent, "hello")

        assert prov.config.model == "original"

    def test_temperature_restored_on_exception(self):
        app, prov = _make_app(temperature=0.5)
        prov.ask_single.side_effect = RuntimeError("boom")
        runner = AgentRunner(app)
        agent = AgentDef(name="test", temperature=2.0)

        with pytest.raises(RuntimeError, match="boom"):
            runner.run(agent, "hello")

        assert prov.config.temperature == 0.5

    def test_provider_override(self):
        app, _ = _make_app(provider_name="gemini")
        claude_config = ProviderConfig(api_key="k", model="claude-3")
        claude_prov = MagicMock()
        claude_prov.config = claude_config
        claude_prov.ask_single.return_value = "claude says"
        app.providers["claude"] = claude_prov

        runner = AgentRunner(app)
        agent = AgentDef(name="test", provider="claude")

        result = runner.run(agent, "hello")
        assert result == "claude says"
        claude_prov.ask_single.assert_called_once()

    def test_missing_provider_raises(self):
        app, _ = _make_app(provider_name="gemini")
        runner = AgentRunner(app)
        agent = AgentDef(name="test", provider="nonexistent")

        with pytest.raises(RuntimeError, match="not available"):
            runner.run(agent, "hello")

    def test_stream(self):
        app, prov = _make_app()
        prov.stream_single.return_value = iter(["chunk1", "chunk2"])
        runner = AgentRunner(app)
        agent = AgentDef(name="test")

        chunks = list(runner.stream(agent, "hello"))
        assert chunks == ["chunk1", "chunk2"]

    def test_stream_restores_model(self):
        app, prov = _make_app(model="original")
        prov.stream_single.return_value = iter(["chunk1", "chunk2"])
        runner = AgentRunner(app)
        agent = AgentDef(name="test", model="override")

        list(runner.stream(agent, "hello"))
        assert prov.config.model == "original"

    def test_system_prompt_injection(self):
        app, prov = _make_app()
        runner = AgentRunner(app)
        agent = AgentDef(name="test", system_prompt="You are a helpful agent.")

        runner.run(agent, "hello")
        call_args = prov.ask_single.call_args
        system = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("system")
        assert system is not None
        assert "You are a helpful agent." in system
