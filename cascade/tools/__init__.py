"""Tool system for Cascade function calling.

Builds a unified tool registry from all registered plugins.
"""

from .schema import ToolDef, callable_to_tool_def
from .executor import ToolExecutor, ConcurrentToolExecutor


# Safety flags per tool. These are security-load-bearing: the permission
# engine auto-approves is_read_only tools, so a mutating tool must never
# appear here as read-only.
_READ_ONLY_TOOLS = frozenset({"read_file", "list_files", "reflect"})
_DESTRUCTIVE_TOOLS = frozenset({"run_command"})


def build_tool_registry() -> dict[str, ToolDef]:
    """Collect all plugin tools and convert to flagged ToolDefs."""
    from ..plugins.registry import get_plugin_registry

    tools = {}
    for _name, plugin_cls in get_plugin_registry().items():
        plugin = plugin_cls()
        for tool_name, fn in plugin.get_tools().items():
            tools[tool_name] = callable_to_tool_def(
                tool_name, fn, description=plugin.description,
                read_only=tool_name in _READ_ONLY_TOOLS,
                destructive=tool_name in _DESTRUCTIVE_TOOLS,
            )
    return tools


__all__ = [
    "ToolDef",
    "callable_to_tool_def",
    "ToolExecutor",
    "ConcurrentToolExecutor",
    "build_tool_registry",
]
