"""AgentRunner -- executes an AgentDef against a CascadeCore's providers."""

from contextlib import contextmanager
from typing import Iterator, Optional, TYPE_CHECKING

from ..prompts.layers import PRIORITY_USER_OVERRIDE

if TYPE_CHECKING:
    from ..cli import CascadeCore
    from .schema import AgentDef


class AgentRunner:
    """Borrows a CascadeCore for a single agent interaction.

    Temporarily overrides model, temperature, system prompt, and tool set
    for the duration of the call, then restores everything -- even on
    exception.
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
        with self._agent_env(agent):
            prov = self._resolve_provider(agent)
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
        """Yields text chunks from the provider.

        The context manager cleanup runs after the iterator is exhausted
        or when the caller breaks out / an exception is raised.
        """
        cm = self._agent_env(agent)
        cm.__enter__()
        try:
            prov = self._resolve_provider(agent)
            system = self._build_system(agent, extra_context)
            for chunk in prov.stream_single(prompt, system):
                yield chunk
        finally:
            cm.__exit__(None, None, None)

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

    @contextmanager
    def _agent_env(self, agent: "AgentDef"):
        """Temporarily swap model + temperature on the provider config."""
        prov = self._resolve_provider(agent)
        orig_model = prov.config.model
        orig_temp = prov.config.temperature

        try:
            if agent.model:
                prov.config.model = agent.model
            if agent.temperature is not None:
                prov.config.temperature = agent.temperature
            yield
        finally:
            prov.config.model = orig_model
            prov.config.temperature = orig_temp
