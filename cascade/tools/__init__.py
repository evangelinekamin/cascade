"""Tool system for Cascade function calling.

Builds a unified tool registry from all registered plugins.
"""

from .schema import ToolDef, callable_to_tool_def
from .executor import ToolExecutor, ConcurrentToolExecutor


def build_tool_registry() -> dict[str, ToolDef]:
    """Collect all plugin tools and convert to ToolDefs."""
    from ..plugins.registry import get_plugin_registry

    tools = {}
    for _name, plugin_cls in get_plugin_registry().items():
        plugin = plugin_cls()
        for tool_name, fn in plugin.get_tools().items():
            tools[tool_name] = callable_to_tool_def(
                tool_name, fn, description=plugin.description,
            )
    return tools


__all__ = [
    "ToolDef",
    "callable_to_tool_def",
    "ToolExecutor",
    "ConcurrentToolExecutor",
    "build_tool_registry",
]
