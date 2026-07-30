"""Hook runner for lifecycle events.

Executes both shell commands and Python module hooks at defined lifecycle
points. Python hooks receive a HookContext and can return HookResult to
block or transform behavior.

Shell hooks receive CASCADE_* environment variables (backward compat).
"""

import importlib
import importlib.util
import json
import logging
import os
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass, fields, replace
from typing import Any, Callable, Optional

from .events import HookEvent
from .context import HookContext, HookResult
from .matchers import ToolMatcher


logger = logging.getLogger("cascade.hooks")

# Re-export HookEvent for backward compatibility
__all__ = ["HookEvent", "HookDefinition", "HookRunner"]


@dataclass(frozen=True)
class HookDefinition:
    """A single hook configuration.

    Supports two handler types:
    - Shell: command is a shell string, handler is None
    - Python: handler is a callable, command is empty

    Optional ``tool_filter`` restricts the hook to matching tool calls
    using Claude Code-style patterns like ``Bash(git:*)``.
    """

    name: str
    event: HookEvent
    command: str = ""
    handler: Optional[Callable[[HookContext], Optional[HookResult]]] = None
    timeout: int = 30
    enabled: bool = True
    priority: int = 100  # lower = runs first
    tool_filter: Optional[ToolMatcher] = None  # CC-style tool matcher
    # None selects the safe event default: TOOL_CALL fails closed while
    # observational hooks fail open. Set explicitly to override that policy.
    fail_closed: Optional[bool] = None

    @property
    def is_python_hook(self) -> bool:
        return self.handler is not None

    @property
    def effective_fail_closed(self) -> bool:
        if self.fail_closed is not None:
            return self.fail_closed
        return self.event == HookEvent.TOOL_CALL

    def matches_tool(self, tool_name: str, arguments: dict | None = None) -> bool:
        """Check if this hook applies to the given tool call.

        Returns True if no filter is set (matches everything) or if
        the filter pattern matches.
        """
        if self.tool_filter is None:
            return True
        return self.tool_filter.matches(tool_name, arguments)


class HookRunner:
    """Execute hooks at lifecycle events.

    Supports both shell command hooks (legacy) and Python module hooks.
    Python hooks can inspect, transform, and block at each lifecycle point.
    """

    def __init__(self, hooks: tuple[HookDefinition, ...] = (), *, enabled: bool = True):
        self._hooks = hooks
        self.enabled = enabled
        self._recent_results: deque[dict] = deque(maxlen=100)

    @property
    def hook_count(self) -> int:
        return len(self._hooks)

    @property
    def recent_results(self) -> tuple[dict, ...]:
        """Recent hook outcomes for diagnostics and UI surfaces."""
        return tuple(dict(result) for result in self._recent_results)

    def add_hook(self, hook: HookDefinition) -> "HookRunner":
        """Return a new runner with the hook added. Immutable."""
        return HookRunner(self._hooks + (hook,), enabled=self.enabled)

    def hooks_for_event(
        self,
        event: HookEvent,
        tool_name: str = "",
        tool_args: dict | None = None,
    ) -> tuple[HookDefinition, ...]:
        """Return all enabled hooks for a given event, sorted by priority.

        For tool-related events, filters by tool_filter pattern. Returns an
        empty tuple while the runner is disabled (runtime master switch).
        """
        if not self.enabled:
            return ()
        hooks = []
        for h in self._hooks:
            if h.event != event or not h.enabled:
                continue
            # Apply tool filter for tool lifecycle events
            if tool_name and not h.matches_tool(tool_name, tool_args):
                continue
            hooks.append(h)
        hooks.sort(key=lambda h: h.priority)
        return tuple(hooks)

    def run_hooks(
        self,
        event: HookEvent,
        context: Optional[dict[str, Any]] = None,
        hook_context: Optional[HookContext] = None,
    ) -> list[dict]:
        """Execute all enabled hooks for an event.

        Args:
            event: The lifecycle event that triggered.
            context: Key-value pairs for CASCADE_* env vars (shell hooks).
            hook_context: Rich context for Python module hooks.

        Returns:
            List of result dicts with keys: name, success, output, duration,
            and optionally 'blocked' and 'transformed_value'.
        """
        # Build hook context if not provided
        if hook_context is None:
            hook_context = self._context_from_legacy(event, context or {})

        tool_args = dict(hook_context.tool_input) if hook_context.tool_input else None
        hooks = self.hooks_for_event(
            event,
            tool_name=hook_context.tool_name,
            tool_args=tool_args,
        )
        if not hooks:
            return []

        results = []
        current_context = hook_context
        for hook in hooks:
            env = dict(os.environ)
            env.update(current_context.to_env_dict())
            # Also merge legacy context dict.
            if context:
                for key, value in context.items():
                    env_key = f"CASCADE_{key.upper()}"
                    env[env_key] = str(value)
            if hook.is_python_hook:
                result = self._run_python_hook(hook, self._copy_context(current_context))
            else:
                result = self._run_shell_hook(hook, env, current_context)

            failed = not result.get("success")
            if failed and hook.effective_fail_closed:
                result["blocked"] = True
            results.append(result)
            self._record_result(event, result)

            # If a hook blocked, stop processing remaining hooks
            if result.get("blocked"):
                break
            if failed:
                continue
            if result.get("transformed_value") is not None:
                current_context = self._context_after_transform(
                    event, current_context, result["transformed_value"],
                )

        return results

    def emit(
        self,
        event: HookEvent,
        hook_context: HookContext,
    ) -> Optional[HookResult]:
        """Emit an event, stopping on a block and chaining transformations.

        This is the preferred API for new code. Returns the final transform,
        the first block, or None when every hook passes through.
        """
        # Extract tool info from context for pattern matching
        tool_name = hook_context.tool_name
        tool_args = dict(hook_context.tool_input) if hook_context.tool_input else None
        hooks = self.hooks_for_event(event, tool_name=tool_name, tool_args=tool_args)
        if not hooks:
            return None

        current_context = hook_context
        transformed = False
        transformed_value: Any = None
        for hook in hooks:
            env = dict(os.environ)
            env.update(current_context.to_env_dict())
            if hook.is_python_hook:
                result_dict = self._run_python_hook(
                    hook, self._copy_context(current_context),
                )
            else:
                result_dict = self._run_shell_hook(hook, env, current_context)

            failed = not result_dict.get("success")
            if failed and hook.effective_fail_closed:
                result_dict["blocked"] = True
            self._record_result(event, result_dict)

            if result_dict.get("blocked"):
                return HookResult(
                    block=True,
                    reason=(
                        result_dict.get("output")
                        or f"Hook '{hook.name}' failed"
                    ),
                )
            if failed:
                continue
            if result_dict.get("transformed_value") is not None:
                transformed = True
                transformed_value = result_dict["transformed_value"]
                current_context = self._context_after_transform(
                    event, current_context, transformed_value,
                )

        return (
            HookResult(transformed_value=transformed_value)
            if transformed
            else None
        )

    def _run_python_hook(self, hook: HookDefinition, ctx: HookContext) -> dict:
        """Execute a Python module hook."""
        start = time.monotonic()
        try:
            handler = hook.handler
            if handler is None:
                raise RuntimeError("Python hook has no handler")
            result = handler(ctx)
            duration = time.monotonic() - start

            output: dict[str, Any] = {
                "name": hook.name,
                "success": True,
                "output": "",
                "duration": round(duration, 3),
                "type": "python",
            }

            if isinstance(result, HookResult):
                if result.block:
                    output["blocked"] = True
                    output["output"] = result.reason
                if result.transformed_value is not None:
                    output["transformed_value"] = result.transformed_value

            return output

        except Exception as e:
            duration = time.monotonic() - start
            logger.warning("Python hook %s failed: %s", hook.name, e)
            return {
                "name": hook.name,
                "success": False,
                "output": f"Python hook failed: {e}",
                "duration": round(duration, 3),
                "type": "python",
            }

    def _run_shell_hook(
        self,
        hook: HookDefinition,
        env: dict,
        ctx: HookContext,
    ) -> dict:
        """Execute a shell hook with structured context on stdin.

        Environment variables remain for compatibility. New hooks should read
        the JSON payload from stdin and may return a JSON control object:
        ``{"block": true, "reason": "..."}`` or
        ``{"transformed_value": ...}``.
        """
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                hook.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=os.name == "posix",
            )
            payload = json.dumps(ctx.to_payload(), default=str)
            try:
                stdout_raw, stderr_raw = proc.communicate(
                    input=payload,
                    timeout=hook.timeout,
                )
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                proc.communicate()
                raise
            duration = time.monotonic() - start
            stdout = stdout_raw.strip()
            stderr = stderr_raw.strip()
            control = self._parse_shell_control(stdout)
            output = stdout or stderr
            result = {
                "name": hook.name,
                "success": proc.returncode == 0,
                "output": output,
                "return_code": proc.returncode,
                "duration": round(duration, 3),
                "type": "shell",
            }
            if control is not None:
                if control.get("blocked"):
                    result["blocked"] = True
                    result["output"] = control.get("reason") or output
                if "transformed_value" in control:
                    result["transformed_value"] = control["transformed_value"]
            # Exit 2 is the conventional blocking status used by coding-agent
            # hook systems. Preserve stderr/stdout as the reason.
            if proc.returncode == 2:
                result["blocked"] = True
                result["output"] = stderr or stdout or f"Hook '{hook.name}' blocked the event"
            if proc.returncode != 0:
                logger.warning(
                    "Shell hook %s exited %s: %s",
                    hook.name,
                    proc.returncode,
                    stderr or stdout,
                )
            return result
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            logger.warning("Shell hook %s timed out after %ss", hook.name, hook.timeout)
            return {
                "name": hook.name,
                "success": False,
                "output": f"Hook timed out after {hook.timeout}s",
                "return_code": -1,
                "duration": round(duration, 3),
                "type": "shell",
            }
        except Exception as e:
            duration = time.monotonic() - start
            logger.warning("Shell hook %s failed: %s", hook.name, e)
            return {
                "name": hook.name,
                "success": False,
                "output": f"Hook failed: {e}",
                "return_code": -1,
                "duration": round(duration, 3),
                "type": "shell",
            }

    @staticmethod
    def _parse_shell_control(output: str) -> Optional[dict[str, Any]]:
        """Normalize supported subprocess-hook JSON control shapes."""
        if not output:
            return None
        try:
            value = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(value, dict):
            return None

        decision = str(value.get("decision") or value.get("action") or "").lower()
        blocked = (
            value.get("block") is True
            or value.get("cancel") is True
            or decision in {"block", "deny", "cancel"}
        )
        result: dict[str, Any] = {}
        if blocked:
            result["blocked"] = True
            result["reason"] = str(
                value.get("reason")
                or value.get("message")
                or value.get("errorMessage")
                or ""
            )

        if "transformed_value" in value:
            result["transformed_value"] = value["transformed_value"]
        elif "overrideInput" in value:
            result["transformed_value"] = value["overrideInput"]
        elif "contextModification" in value:
            result["transformed_value"] = value["contextModification"]
        elif "context" in value:
            result["transformed_value"] = value["context"]
        return result or None

    @staticmethod
    def _copy_context(ctx: HookContext) -> HookContext:
        """Give each Python hook fresh containers, matching the public contract."""
        return replace(
            ctx,
            messages=tuple(dict(message) for message in ctx.messages),
            tool_input=tuple(ctx.tool_input),
            tool_log=tuple(dict(entry) for entry in ctx.tool_log),
            metadata=tuple(ctx.metadata),
        )

    @staticmethod
    def _context_after_transform(
        event: HookEvent,
        ctx: HookContext,
        value: Any,
    ) -> HookContext:
        """Make the next transformer observe the previous transform."""
        if event in {
            HookEvent.INPUT_RECEIVED,
            HookEvent.AGENT_START,
            HookEvent.WORKFLOW_START,
        }:
            return replace(ctx, prompt=value if isinstance(value, str) else str(value))
        if event == HookEvent.CONTEXT_BUILD:
            return replace(ctx, system_prompt=value if isinstance(value, str) else str(value))
        if event == HookEvent.BEFORE_PROVIDER_REQUEST and isinstance(value, (list, tuple)):
            messages = tuple(dict(message) for message in value if isinstance(message, dict))
            return replace(ctx, messages=messages)
        if event == HookEvent.TOOL_CALL and isinstance(value, dict):
            return replace(ctx, tool_input=tuple(value.items()))
        if event == HookEvent.TOOL_RESULT:
            return replace(ctx, tool_output=value if isinstance(value, str) else str(value))
        if event in {HookEvent.AGENT_END, HookEvent.WORKFLOW_END}:
            return replace(ctx, response=value if isinstance(value, str) else str(value))
        return ctx

    @staticmethod
    def _context_from_legacy(event: HookEvent, context: dict[str, Any]) -> HookContext:
        """Convert the old env-var context without passing unknown kwargs."""
        field_names = {field.name for field in fields(HookContext)}
        tuple_fields = {"messages", "tool_input", "tool_log", "metadata"}
        known: dict[str, Any] = {}
        metadata_value = context.get("metadata")
        if isinstance(metadata_value, dict):
            metadata = list(metadata_value.items())
        elif isinstance(metadata_value, (list, tuple)):
            metadata = list(metadata_value)
        else:
            metadata = []
        for key, value in context.items():
            if key in {"event", "metadata"}:
                continue
            if key not in field_names:
                continue
            if key in tuple_fields:
                if isinstance(value, (list, tuple)):
                    known[key] = tuple(value)
            elif isinstance(value, str):
                known[key] = value
        metadata.extend(
            (key, value)
            for key, value in context.items()
            if key not in known and key not in {"event", "metadata"}
        )
        return HookContext(event=event.value, metadata=tuple(metadata), **known)

    def _record_result(self, event: HookEvent, result: dict) -> None:
        # Diagnostics should stay bounded and must not retain transformed
        # prompts, messages, tool arguments, or results. Shell-hook stdout may
        # itself be the JSON transform, so replace it with a marker.
        output = result.get("output", "")
        if "transformed_value" in result and not result.get("blocked"):
            output = "[transform returned]"
        recorded = {
            "event": event.value,
            "name": result.get("name", ""),
            "success": bool(result.get("success")),
            "blocked": bool(result.get("blocked")),
            "output": str(output)[:500],
            "return_code": result.get("return_code"),
            "duration": result.get("duration", 0),
            "type": result.get("type", ""),
        }
        self._recent_results.append(recorded)

    def describe(self) -> list[dict]:
        """Return a summary of all hooks for display."""
        return [
            {
                "name": h.name,
                "event": h.event.value,
                "command": h.command,
                "enabled": h.enabled,
                "timeout": h.timeout,
                "type": "python" if h.is_python_hook else "shell",
                "priority": h.priority,
                "fail_closed": h.effective_fail_closed,
                "if": h.tool_filter.raw if h.tool_filter is not None else "",
            }
            for h in self._hooks
        ]


def _allowed_hook_dirs() -> list:
    """Return directories from which Python hooks may be loaded."""
    from pathlib import Path
    return [
        Path.home() / ".cascade" / "hooks",
        Path.home() / ".config" / "cascade" / "hooks",
        Path.cwd() / ".cascade" / "hooks",
    ]


def load_python_hook(module_path: str) -> Optional[Callable]:
    """Load a Python hook handler from a module path.

    The module should define a `hook(ctx: HookContext) -> Optional[HookResult]`
    function.

    Security: file paths must be under an allowed hooks directory
    (~/.cascade/hooks/, ~/.config/cascade/hooks/, or .cascade/hooks/).

    Args:
        module_path: Dotted module path (e.g. "cascade.hooks.my_hook")
                     or file path (e.g. "~/.cascade/hooks/my_hook.py").

    Returns:
        The hook callable, or None if loading fails or path is disallowed.
    """
    from pathlib import Path

    try:
        if module_path.endswith(".py"):
            # Validate path is under an allowed directory
            resolved = Path(module_path).expanduser().resolve()
            allowed = _allowed_hook_dirs()
            if not any(
                resolved == d or resolved.is_relative_to(d)
                for d in allowed
            ):
                return None

            spec = importlib.util.spec_from_file_location("_cascade_hook", str(resolved))
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Dotted module path -- only allow cascade.* namespace
            if not module_path.startswith("cascade."):
                return None
            module = importlib.import_module(module_path)

        handler = getattr(module, "hook", None)
        if callable(handler):
            return handler
        return None
    except Exception:
        return None
