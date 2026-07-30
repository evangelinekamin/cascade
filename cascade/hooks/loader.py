"""Parse hook definitions from configuration data.

Supports both legacy shell-command hooks and new Python module hooks,
with optional CC-style tool pattern matching.

Legacy format (shell):
    - name: my_hook
      event: before_ask
      command: "echo hello"

New format (Python module):
    - name: my_hook
      event: tool_call
      module: path/to/hook.py
      priority: 50

CC-style format (settings.json):
    - name: block_rm
      event: tool_call
      if: "Bash(rm:*)"
      command: "echo 'blocked dangerous command'"

    - name: git_audit
      event: tool_call
      if: "Bash(git:*)"
      command: "logger 'git operation detected'"
"""

from typing import Any

from .events import EVENT_MAP
from .matchers import compile_matcher
from .runner import HookDefinition, load_python_hook


def load_hooks_from_config(hooks_data: list[dict[str, Any]]) -> tuple[HookDefinition, ...]:
    """Parse a list of hook config dicts into HookDefinition instances.

    Each dict should have:
        name: str (required)
        event: str (required) - one of the HookEvent values
        command: str (required for shell hooks)
        module: str (optional) - Python module path for module hooks
        if: str (optional) - CC-style tool matcher pattern
        timeout: int (optional, default 30)
        enabled: bool (optional, default True)
        priority: int (optional, default 100, lower = runs first)

    Invalid entries are silently skipped.
    """
    hooks = []

    for entry in hooks_data:
        if not isinstance(entry, dict):
            continue

        name = entry.get("name")
        event_str = entry.get("event")

        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(event_str, str) or not event_str:
            continue

        event = EVENT_MAP.get(event_str)
        if event is None:
            continue

        command = entry.get("command", "")
        module_path = entry.get("module", "")
        if not isinstance(command, str) or not isinstance(module_path, str):
            continue
        handler = None

        # Python module hook
        if module_path:
            handler = load_python_hook(module_path)
            if handler is None:
                continue  # Skip if module can't be loaded

        # Shell hook requires a command
        if not module_path and not command:
            continue

        # CC-style tool pattern filter
        tool_filter = None
        if_pattern = entry.get("if", "")
        if not isinstance(if_pattern, str):
            continue
        if if_pattern:
            tool_filter = compile_matcher(if_pattern)

        timeout = entry.get("timeout", 30)
        priority = entry.get("priority", 100)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            timeout = 30
        if not isinstance(priority, int) or isinstance(priority, bool):
            priority = 100
        fail_closed = entry.get("fail_closed")
        if fail_closed is not None and not isinstance(fail_closed, bool):
            fail_closed = None
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            enabled = True

        hooks.append(HookDefinition(
            name=name.strip(),
            event=event,
            command=command,
            handler=handler,
            timeout=timeout,
            enabled=enabled,
            priority=priority,
            tool_filter=tool_filter,
            fail_closed=fail_closed,
        ))

    return tuple(hooks)
