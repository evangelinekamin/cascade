"""Tests for AgentRunner -- context manager safety, tool filtering."""

import pytest
from unittest.mock import MagicMock

from cascade.agents.schema import AgentDef
from cascade.agents.runner import AgentRunner
from cascade.hooks import HookDefinition, HookEvent, HookResult, HookRunner
from cascade.providers.base import ProviderConfig


class _FakeProvider:
    """A real (cloneable) provider stand-in that records its config.

    type(inst)(config) works, so AgentRunner's clone path exercises real
    behavior instead of a MagicMock that swallows overrides.
    """

    instances: list = []

    def __init__(self, config):
        self.config = config
        self.hook_runner = None
        self.permission_engine = None
        self.ask_single_error = None
        _FakeProvider.instances.append(self)

    def ask_single(self, prompt, system=None):
        if self.ask_single_error:
            raise self.ask_single_error
        return f"resp:{self.config.model}:{self.config.temperature}"

    def ask_with_tools(self, messages, tools, system=None):
        return "tool response", []

    def stream_single(self, prompt, system=None):
        yield f"chunk:{self.config.model}"


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
    app.hook_runner = None

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

    def _fake_app(self, model="original-model", temperature=0.7):
        _FakeProvider.instances = []
        prov = _FakeProvider(
            ProviderConfig(api_key="k", model=model, temperature=temperature)
        )
        app = MagicMock()
        app.providers = {"gemini": prov}
        app.config.get_default_provider.return_value = "gemini"
        app.tool_registry = {}
        app.hook_runner = None
        from cascade.prompts.layers import PromptPipeline
        app.prompt_pipeline = PromptPipeline()
        return app, prov

    def test_override_never_mutates_shared_provider(self):
        app, prov = self._fake_app(model="original-model", temperature=0.7)
        runner = AgentRunner(app)
        agent = AgentDef(name="test", model="override-model", temperature=1.5)

        result = runner.run(agent, "hello")
        # The clone ran with the override; the shared provider is untouched.
        assert result == "resp:override-model:1.5"
        assert prov.config.model == "original-model"
        assert prov.config.temperature == 0.7
        # A distinct clone instance was created and carried the gates.
        assert len(_FakeProvider.instances) == 2
        clone = _FakeProvider.instances[-1]
        assert clone is not prov

    def test_no_override_reuses_shared_provider(self):
        app, prov = self._fake_app()
        runner = AgentRunner(app)
        agent = AgentDef(name="test")  # no model/temperature override
        runner.run(agent, "hello")
        # No clone created -- the shared instance is used directly.
        assert len(_FakeProvider.instances) == 1

    def test_override_clone_carries_gates(self):
        app, prov = self._fake_app()
        sentinel_hooks, sentinel_perms = object(), object()
        prov.hook_runner = sentinel_hooks
        prov.permission_engine = sentinel_perms
        runner = AgentRunner(app)
        agent = AgentDef(name="test", model="x")
        runner.run(agent, "hello")
        clone = _FakeProvider.instances[-1]
        assert clone.hook_runner is sentinel_hooks
        assert clone.permission_engine is sentinel_perms

    def test_max_tokens_override_uses_clone(self):
        app, prov = self._fake_app()
        runner = AgentRunner(app)

        runner.run(AgentDef(name="test", max_tokens=8192), "hello")

        clone = _FakeProvider.instances[-1]
        assert clone is not prov
        assert clone.config.max_tokens == 8192
        assert prov.config.max_tokens is None

    def test_provider_exception_propagates(self):
        app, prov = self._fake_app(model="original")
        # Error configured on the shared provider is inherited by the clone
        # config, so make the clone raise by erroring on any instance.
        _FakeProvider.instances = []
        prov.ask_single_error = RuntimeError("boom")
        # Without an override there is no clone: the shared provider raises.
        runner = AgentRunner(app)
        with pytest.raises(RuntimeError, match="boom"):
            runner.run(AgentDef(name="test"), "hello")

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

    def test_stream_override_uses_clone_not_shared(self):
        _FakeProvider.instances = []
        prov = _FakeProvider(ProviderConfig(api_key="k", model="original"))
        app = MagicMock()
        app.providers = {"gemini": prov}
        app.config.get_default_provider.return_value = "gemini"
        app.tool_registry = {}
        app.hook_runner = None
        from cascade.prompts.layers import PromptPipeline
        app.prompt_pipeline = PromptPipeline()

        runner = AgentRunner(app)
        chunks = list(runner.stream(AgentDef(name="test", model="override"), "hi"))
        assert chunks == ["chunk:override"]
        assert prov.config.model == "original"  # shared untouched

    def test_system_prompt_injection(self):
        app, prov = _make_app()
        runner = AgentRunner(app)
        agent = AgentDef(name="test", system_prompt="You are a helpful agent.")

        runner.run(agent, "hello")
        call_args = prov.ask_single.call_args
        system = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("system")
        assert system is not None
        assert "You are a helpful agent." in system

    def test_named_agent_lifecycle_hooks_transform_and_observe(self):
        app, prov = _make_app()
        seen = []

        def start(ctx):
            seen.append((ctx.event, ctx.agent_name, ctx.prompt))
            return HookResult(transformed_value=ctx.prompt + " rewritten")

        def after(ctx):
            seen.append((ctx.event, ctx.agent_name, ctx.response))

        def end(ctx):
            seen.append((ctx.event, ctx.agent_name, ctx.response))
            return HookResult(transformed_value=ctx.response + " polished")

        app.hook_runner = HookRunner(hooks=(
            HookDefinition(
                name="start",
                event=HookEvent.AGENT_START,
                handler=start,
            ),
            HookDefinition(
                name="after",
                event=HookEvent.AFTER_RESPONSE,
                handler=after,
            ),
            HookDefinition(
                name="end",
                event=HookEvent.AGENT_END,
                handler=end,
            ),
        ))
        prov.ask_single.side_effect = lambda prompt, _system=None: f"response:{prompt}"

        result = AgentRunner(app).run(AgentDef(name="reviewer"), "inspect")

        assert result == "response:inspect rewritten polished"
        assert [event for event, _agent, _value in seen] == [
            "agent_start",
            "after_response",
            "agent_end",
        ]
        assert all(agent == "reviewer" for _event, agent, _value in seen)

    def test_named_agent_errors_emit_on_error(self):
        app, prov = _make_app()
        errors = []
        app.hook_runner = HookRunner(hooks=(
            HookDefinition(
                name="errors",
                event=HookEvent.ON_ERROR,
                handler=lambda ctx: errors.append(
                    (ctx.agent_name, ctx.workflow, ctx.error)
                ),
            ),
        ))
        prov.ask_single.side_effect = RuntimeError("agent exploded")

        with pytest.raises(RuntimeError, match="agent exploded"):
            AgentRunner(app).run(AgentDef(name="reviewer"), "inspect")

        assert errors == [("reviewer", "agent", "agent exploded")]
