"""Plugin system for extending Cascade."""

from .base import BasePlugin
from .file_ops import FileOpsPlugin
from .execution import ExecPlugin
from .registry import register_plugin, get_plugin_registry

# Import reflection plugin to trigger registration
from ..tools.reflection import ReflectionPlugin

__all__ = [
    "BasePlugin",
    "FileOpsPlugin",
    "ExecPlugin",
    "ReflectionPlugin",
    "register_plugin",
    "get_plugin_registry",
]
