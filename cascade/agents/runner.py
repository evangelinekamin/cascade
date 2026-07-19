"""AgentRunner -- executes an AgentDef against a CascadeCore's providers."""

from dataclasses import replace
from typing import Iterator, Optional, TYPE_CHECKING

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
        tools = self._filter_tools(agent)

        if tools:
            messages = [{"role": "user", "content": prompt}]
            response, _log = prov.ask_with_tools(messages, tools, system=system)
        else:
            response = prov.ask_single(prompt, system)
        return response

    def stream(
        self,
        agent: "AgentDef",
        prompt: str,
        extra_context: Optional[str] = None,
    ) -> Iterator[str]:
        """Yields text chunks from the provider."""
        prov = self._provider_for(agent)
        system = self._build_system(agent, extra_context)
        yield from prov.stream_single(prompt, system)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
        overrides = {}
        if agent.model:
            overrides["model"] = agent.model
        if agent.temperature is not None:
            overrides["temperature"] = agent.temperature
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
