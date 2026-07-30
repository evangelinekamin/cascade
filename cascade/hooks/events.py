"""Rich lifecycle events for the Cascade hook system.

Inspired by Pi's extension lifecycle: ~15 lifecycle events that Python
module hooks can intercept, transform, and block at every stage.

This replaces the limited 4-event shell-command-only system with a
full lifecycle that supports both shell hooks and Python module hooks.
"""

from enum import Enum


class HookEvent(str, Enum):
    """Lifecycle events that can trigger hooks.

    Events follow this flow:

        session_start
        user sends prompt
          -> input_received (can transform prompt)
          -> agent_start (named agent begins)
          -> workflow_start (orchestration lane begins)
          -> before_ask (legacy compat)
          -> context_build (can inject/modify context)
          -> before_provider_request (inspect/modify messages)
          -> provider responds, may call tools:
              -> tool_call (can block tool)
              -> tool_result (can modify result)
          -> after_response (legacy compat)
          -> agent_end (named agent finishes)
          -> workflow_end (orchestration lane finishes)
          -> episode_generated (new episode created)
        provider/mode switches:
          -> provider_switch (model changed)
        session ends:
          -> on_exit (legacy compat)
        errors:
          -> on_error (legacy compat)
    """

    # --- Session lifecycle ---
    SESSION_START = "session_start"
    SESSION_RESUME = "session_resume"

    # --- Input processing ---
    INPUT_RECEIVED = "input_received"

    # --- Agent/orchestration lifecycle ---
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_END = "workflow_end"

    # --- Provider request lifecycle ---
    BEFORE_ASK = "before_ask"
    CONTEXT_BUILD = "context_build"
    BEFORE_PROVIDER_REQUEST = "before_provider_request"
    AFTER_RESPONSE = "after_response"

    # --- Tool lifecycle ---
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"

    # --- Episode lifecycle ---
    EPISODE_GENERATED = "episode_generated"

    # --- Provider/mode changes ---
    PROVIDER_SWITCH = "provider_switch"

    # --- Session end ---
    ON_EXIT = "on_exit"

    # --- Errors ---
    ON_ERROR = "on_error"


# Map string names to events (including legacy names)
EVENT_MAP: dict[str, HookEvent] = {e.value: e for e in HookEvent}
