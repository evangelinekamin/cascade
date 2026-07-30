"""Hook context API for Python module hooks.

Provides a frozen context object that hooks receive, allowing them
to inspect the current state of the interaction and return results
to signal blocking or transformation.

Each hook receives its own shallow copy, so mutations on one hook's
context do not affect subsequent hooks.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class HookResult:
    """Result returned by a hook to control flow.

    Hooks return this to signal blocking, transformation, or passthrough.
    """

    block: bool = False
    reason: str = ""
    transformed_value: Optional[Any] = None


@dataclass(frozen=True)
class HookContext:
    """Frozen context passed to Python module hooks.

    Contains all relevant state for the current lifecycle event.
    Immutable to prevent hooks from accidentally corrupting shared state.
    """

    event: str
    provider: str = ""
    mode: str = ""
    agent_name: str = ""
    workflow: str = ""
    prompt: str = ""
    response: str = ""
    error: str = ""
    messages: tuple[dict, ...] = ()
    system_prompt: str = ""
    tool_name: str = ""
    tool_input: tuple = ()  # tuple of (key, value) pairs
    tool_output: str = ""
    tool_log: tuple[dict, ...] = ()
    episode_id: str = ""
    session_id: str = ""
    metadata: tuple = ()  # tuple of (key, value) pairs

    def to_payload(self) -> dict[str, Any]:
        """Return the structured JSON payload supplied to subprocess hooks.

        Unlike environment variables, stdin can safely carry full prompt, tool,
        and response content without shell interpolation.  The payload mirrors
        the Python hook context so either hook type can make the same decision.
        """
        return {
            "event": self.event,
            "provider": self.provider,
            "mode": self.mode,
            "agent_name": self.agent_name,
            "workflow": self.workflow,
            "prompt": self.prompt,
            "response": self.response,
            "error": self.error,
            "messages": [dict(message) for message in self.messages],
            "system_prompt": self.system_prompt,
            "tool_name": self.tool_name,
            "tool_input": dict(self.tool_input),
            "tool_output": self.tool_output,
            "tool_log": [dict(entry) for entry in self.tool_log],
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "metadata": dict(self.metadata),
        }

    def to_env_dict(self) -> dict[str, str]:
        """Convert to CASCADE_* environment variables for shell hooks.

        Security: only passes safe scalar values as env vars.
        Prompt content is NOT passed (only length) to avoid shell injection.
        """
        env = {"CASCADE_EVENT": self.event}
        if self.provider:
            env["CASCADE_PROVIDER"] = self.provider
        if self.mode:
            env["CASCADE_MODE"] = self.mode
        if self.agent_name:
            env["CASCADE_AGENT"] = self.agent_name
        if self.workflow:
            env["CASCADE_WORKFLOW"] = self.workflow
        if self.prompt:
            # Only pass length, not content (avoids shell injection surface)
            env["CASCADE_PROMPT_LENGTH"] = str(len(self.prompt))
        if self.response:
            env["CASCADE_RESPONSE_LENGTH"] = str(len(self.response))
        if self.error:
            env["CASCADE_ERROR_LENGTH"] = str(len(self.error))
        if self.tool_name:
            env["CASCADE_TOOL_NAME"] = self.tool_name
        if self.tool_log:
            env["CASCADE_TOOL_CALLS"] = str(len(self.tool_log))
        if self.session_id:
            env["CASCADE_SESSION_ID"] = self.session_id
        # Metadata: only pass string values, skip anything complex
        for key, value in self.metadata:
            sanitized_key = "".join(c for c in str(key).upper() if c.isalnum() or c == "_")
            if sanitized_key:
                env[f"CASCADE_{sanitized_key}"] = str(value)[:200]
        return env
