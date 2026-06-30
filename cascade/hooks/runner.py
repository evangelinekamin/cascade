"""Hook runner for lifecycle events.

Executes both shell commands and Python module hooks at defined lifecycle
points. Python hooks receive a HookContext and can return HookResult to
block or transform behavior.

Shell hooks receive CASCADE_* environment variables (backward compat).
"""

import importlib
import importlib.util
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .events import HookEvent
from .context import HookContext, HookResult
from .matchers import ToolMatcher


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

    @property
    def is_python_hook(self) -> bool:
        return self.handler is not None

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

    @property
    def hook_count(self) -> int:
        return len(self._hooks)

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
        hooks = self.hooks_for_event(event)
        if not hooks:
            return []

        # Build hook context if not provided
        if hook_context is None:
            hook_context = HookContext(
                event=event.value,
                **(context or {}),
            )

        # Build env dict for shell hooks
        env = dict(os.environ)
        env.update(hook_context.to_env_dict())
        # Also merge legacy context dict
        if context:
            for key, value in context.items():
                env_key = f"CASCADE_{key.upper()}"
                env[env_key] = str(value)

        results = []
        for hook in hooks:
            if hook.is_python_hook:
                result = self._run_python_hook(hook, hook_context)
            else:
                result = self._run_shell_hook(hook, env)

            results.append(result)

            # If a hook blocked, stop processing remaining hooks
            if result.get("blocked"):
                break

        return results

    def emit(
        self,
        event: HookEvent,
        hook_context: HookContext,
    ) -> Optional[HookResult]:
        """Emit an event and return the first blocking/transforming result.

        This is the preferred API for new code. Returns None if no hook
        blocked or transformed.
        """
        # Extract tool info from context for pattern matching
        tool_name = hook_context.tool_name
        tool_args = dict(hook_context.tool_input) if hook_context.tool_input else None
        hooks = self.hooks_for_event(event, tool_name=tool_name, tool_args=tool_args)
        if not hooks:
            return None

        env = dict(os.environ)
        env.update(hook_context.to_env_dict())

        for hook in hooks:
            if hook.is_python_hook:
                result_dict = self._run_python_hook(hook, hook_context)
            else:
                result_dict = self._run_shell_hook(hook, env)

            if result_dict.get("blocked"):
                return HookResult(
                    block=True,
                    reason=result_dict.get("output", ""),
                )
            if result_dict.get("transformed_value") is not None:
                return HookResult(
                    transformed_value=result_dict["transformed_value"],
                )

        return None

    def _run_python_hook(self, hook: HookDefinition, ctx: HookContext) -> dict:
        """Execute a Python module hook."""
        start = time.monotonic()
        try:
            result = hook.handler(ctx)
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
            return {
                "name": hook.name,
                "success": False,
                "output": f"Python hook failed: {e}",
                "duration": round(duration, 3),
                "type": "python",
            }

    def _run_shell_hook(self, hook: HookDefinition, env: dict) -> dict:
        """Execute a shell command hook."""
        start = time.monotonic()
        try:
            proc = subprocess.run(
                hook.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=hook.timeout,
                env=env,
            )
            duration = time.monotonic() - start
            return {
                "name": hook.name,
                "success": proc.returncode == 0,
                "output": proc.stdout.strip() or proc.stderr.strip(),
                "return_code": proc.returncode,
                "duration": round(duration, 3),
                "type": "shell",
            }
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
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
            return {
                "name": hook.name,
                "success": False,
                "output": f"Hook failed: {e}",
                "return_code": -1,
                "duration": round(duration, 3),
                "type": "shell",
            }

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
            resolved = Path(module_path).resolve()
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
