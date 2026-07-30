"""AgentRunner -- executes an AgentDef against a CascadeCore's providers."""

from dataclasses import replace
from typing import Any, Iterator, Optional, TYPE_CHECKING

from ..hooks import HookContext, HookEvent
from ..prompts.layers import PRIORITY_USER_OVERRIDE

if TYPE_CHECKING:
    from ..cli import CascadeCore
    from .schema import AgentDef


class AgentRunner:
    """Borrows a CascadeCore for a single agent interaction.

    Model/temperature/system-prompt/tool overrides are applied to a
    per-run provider clone, never by mutating the shared provider config --
    so a concurrent chat turn or a parallel agent on the same provider is
    never affected.
    """

    def __init__(self, app: "CascadeCore") -> None:
        self._app = app

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        agent: "AgentDef",
        prompt: str,
        extra_context: Optional[str] = None,
    ) -> str:
        """Blocking call -- returns the complete response text."""
        prov = self._provider_for(agent)
        system = self._build_system(agent, extra_context)
        prompt, system, messages, hook_runner = self._prepare_request(
            agent, prov, prompt, system,
        )
        tools = self._filter_tools(agent)

        tool_log = []
        try:
            if tools:
                response, tool_log = prov.ask_with_tools(messages, tools, system=system)
            elif messages == [{"role": "user", "content": prompt}]:
                response = prov.ask_single(prompt, system)
            else:
                response = prov.ask(messages, system)
        except Exception as exc:
            self._emit_error(hook_runner, agent, prov, prompt, exc)
            raise
        return self._finish_request(
            hook_runner, agent, prov, prompt, response, tool_log,
        )

    def stream(
        self,
        agent: "AgentDef",
        prompt: str,
        extra_context: Optional[str] = None,
    ) -> Iterator[str]:
        """Yields text chunks from the provider."""
        prov = self._provider_for(agent)
        system = self._build_system(agent, extra_context)
        prompt, system, messages, hook_runner = self._prepare_request(
            agent, prov, prompt, system,
        )
        chunks: list[str] = []
        try:
            source = (
                prov.stream_single(prompt, system)
                if messages == [{"role": "user", "content": prompt}]
                else prov.stream(messages, system)
            )
            for chunk in source:
                chunks.append(chunk)
                yield chunk
        except Exception as exc:
            self._emit_error(hook_runner, agent, prov, prompt, exc)
            raise
        self._finish_request(
            hook_runner, agent, prov, prompt, "".join(chunks), (),
            allow_transform=False,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prepare_request(self, agent, prov, prompt: str, system: Optional[str]):
        """Run named-agent preflight hooks and return the effective request."""
        provider_name = self._provider_label(agent, prov)
        runner = getattr(self._app, "hook_runner", None)
        if runner is None:
            return prompt, system, [{"role": "user", "content": prompt}], None

        start = runner.emit(
            HookEvent.AGENT_START,
            HookContext(
                event=HookEvent.AGENT_START.value,
                provider=provider_name,
                agent_name=agent.name,
                workflow="agent",
                prompt=prompt,
            ),
        )
        if start is not None:
            if start.block:
                raise RuntimeError(f"Agent blocked by hook: {start.reason}")
            if start.transformed_value is not None:
                if not isinstance(start.transformed_value, str):
                    raise RuntimeError("Agent hook returned an invalid prompt (expected text)")
                prompt = start.transformed_value

        context_result = runner.emit(
            HookEvent.CONTEXT_BUILD,
            HookContext(
                event=HookEvent.CONTEXT_BUILD.value,
                provider=provider_name,
                agent_name=agent.name,
                workflow="agent",
                prompt=prompt,
                system_prompt=system or "",
            ),
        )
        if context_result is not None:
            if context_result.block:
                raise RuntimeError(f"Agent context blocked by hook: {context_result.reason}")
            if context_result.transformed_value is not None:
                system = str(context_result.transformed_value)

        messages = [{"role": "user", "content": prompt}]
        request_result = runner.emit(
            HookEvent.BEFORE_PROVIDER_REQUEST,
            HookContext(
                event=HookEvent.BEFORE_PROVIDER_REQUEST.value,
                provider=provider_name,
                agent_name=agent.name,
                workflow="agent",
                prompt=prompt,
                messages=tuple(messages),
                system_prompt=system or "",
            ),
        )
        if request_result is not None:
            if request_result.block:
                raise RuntimeError(f"Agent request blocked by hook: {request_result.reason}")
            if request_result.transformed_value is not None:
                messages = self._validated_messages(request_result.transformed_value)
        return prompt, system, messages, runner

    @staticmethod
    def _provider_label(agent, prov) -> str:
        name = getattr(prov, "name", None)
        if isinstance(name, str) and name:
            return name
        return agent.provider or type(prov).__name__

    @staticmethod
    def _validated_messages(value: Any) -> list[dict]:
        if not (
            isinstance(value, (list, tuple))
            and all(isinstance(message, dict) for message in value)
        ):
            raise RuntimeError(
                "Agent request hook returned invalid messages "
                "(expected a list of objects)"
            )
        return [dict(message) for message in value]

    @staticmethod
    def _emit_error(runner, agent, prov, prompt: str, exc: Exception) -> None:
        if runner is None:
            return
        runner.emit(
            HookEvent.ON_ERROR,
            HookContext(
                event=HookEvent.ON_ERROR.value,
                provider=AgentRunner._provider_label(agent, prov),
                agent_name=agent.name,
                workflow="agent",
                prompt=prompt,
                error=str(exc),
            ),
        )

    @staticmethod
    def _finish_request(
        runner,
        agent,
        prov,
        prompt: str,
        response: str,
        tool_log,
        *,
        allow_transform: bool = True,
    ) -> str:
        if runner is None:
            return response
        context = HookContext(
            event=HookEvent.AFTER_RESPONSE.value,
            provider=AgentRunner._provider_label(agent, prov),
            agent_name=agent.name,
            workflow="agent",
            prompt=prompt,
            response=response,
            tool_log=tuple(dict(entry) for entry in tool_log),
        )
        runner.emit(HookEvent.AFTER_RESPONSE, context)
        end = runner.emit(
            HookEvent.AGENT_END,
            replace(context, event=HookEvent.AGENT_END.value),
        )
        if end is not None:
            if end.block:
                raise RuntimeError(f"Agent result blocked by hook: {end.reason}")
            if allow_transform and end.transformed_value is not None:
                if not isinstance(end.transformed_value, str):
                    raise RuntimeError(
                        "Agent end hook returned an invalid response (expected text)"
                    )
                response = end.transformed_value
        return response

    def _resolve_provider(self, agent: "AgentDef"):
        """Return the provider instance to use for this agent."""
        name = agent.provider or self._app.config.get_default_provider()
        prov = self._app.providers.get(name)
        if prov is None:
            available = list(self._app.providers.keys())
            raise RuntimeError(
                f"Agent '{agent.name}' requires provider '{name}' "
                f"but it is not available. Have: {available}"
            )
        return prov

    def _build_system(
        self, agent: "AgentDef", extra_context: Optional[str] = None,
    ) -> Optional[str]:
        """Build the system prompt, injecting the agent's prompt layer."""
        pipeline = self._app.prompt_pipeline
        if agent.system_prompt:
            pipeline = pipeline.add_layer(
                f"agent:{agent.name}", agent.system_prompt, PRIORITY_USER_OVERRIDE,
            )
        if extra_context:
            pipeline = pipeline.add_layer(
                "agent_context", extra_context, PRIORITY_USER_OVERRIDE + 1,
            )
        return pipeline.build() or None

    def _filter_tools(self, agent: "AgentDef") -> dict:
        """Return the tool registry filtered by agent.allowed_tools."""
        if agent.allowed_tools is None:
            return dict(self._app.tool_registry)
        if not agent.allowed_tools:
            return {}
        return {
            name: td
            for name, td in self._app.tool_registry.items()
            if name in agent.allowed_tools
        }

    def _provider_for(self, agent: "AgentDef"):
        """Resolve the provider, cloning it when the agent overrides config.

        No override -> the shared instance (cheap). An override -> a fresh
        instance with a replaced config, carrying the hook + permission
        gates, so nothing on the shared provider is mutated.
        """
        prov = self._resolve_provider(agent)
        overrides: dict[str, object] = {}
        if agent.model:
            overrides["model"] = agent.model
        if agent.temperature is not None:
            overrides["temperature"] = agent.temperature
        if agent.max_tokens is not None:
            overrides["max_tokens"] = agent.max_tokens
        if not overrides:
            return prov
        try:
            clone = type(prov)(replace(prov.config, **overrides))
        except Exception:
            # A provider whose config is not a dataclass (e.g. a test double)
            # cannot be cloned safely; fall back to the shared instance.
            return prov
        clone.hook_runner = getattr(prov, "hook_runner", None)
        clone.permission_engine = getattr(prov, "permission_engine", None)
        return clone
