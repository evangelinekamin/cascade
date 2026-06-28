"""Lifecycle hook system for Cascade.

Supports both shell command hooks (legacy) and Python module hooks
with rich lifecycle events for intercepting, transforming, and
blocking at every stage of the interaction pipeline.
"""

from .events import HookEvent
from .context import HookContext, HookResult
from .runner import HookDefinition, HookRunner, load_python_hook
from .loader import load_hooks_from_config
from .matchers import ToolMatcher, compile_matcher, matches_any

__all__ = [
    "HookEvent",
    "HookContext",
    "HookResult",
    "HookDefinition",
    "HookRunner",
    "load_hooks_from_config",
    "load_python_hook",
    "ToolMatcher",
    "compile_matcher",
    "matches_any",
]
