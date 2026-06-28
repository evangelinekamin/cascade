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

        if not all((name, event_str)):
            continue

        event = EVENT_MAP.get(event_str)
        if event is None:
            continue

        command = entry.get("command", "")
        module_path = entry.get("module", "")
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
        if if_pattern:
            tool_filter = compile_matcher(if_pattern)

        hooks.append(HookDefinition(
            name=name,
            event=event,
            command=command,
            handler=handler,
            timeout=entry.get("timeout", 30),
            enabled=entry.get("enabled", True),
            priority=entry.get("priority", 100),
            tool_filter=tool_filter,
        ))

    return tuple(hooks)
