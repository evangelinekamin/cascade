"""Pattern matching for hook tool filters.

Supports Claude Code-style tool matchers:
    "Bash"              - matches tool name exactly
    "Bash(git:*)"       - matches tool name + argument content pattern
    "Bash(*rm*)"        - wildcard anywhere in arguments
    "*"                 - matches any tool
    "mcp__server__tool" - exact MCP tool name

Patterns are compiled once and cached for fast matching.
"""

import fnmatch
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolMatcher:
    """Compiled matcher for a single tool pattern."""

    tool_pattern: str   # fnmatch pattern for tool name
    arg_pattern: str    # fnmatch pattern for argument content (empty = any)
    raw: str            # original pattern string

    def matches(self, tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
        """Check if a tool call matches this pattern."""
        if not fnmatch.fnmatch(tool_name, self.tool_pattern):
            return False

        if not self.arg_pattern:
            return True

        if arguments is None:
            return False

        # Match the arg pattern against each value individually and
        # against the full flattened string. This handles both
        # "Bash(git:*)" matching a command value and broader patterns.
        for value in arguments.values():
            val_str = str(value)
            if fnmatch.fnmatch(val_str, self.arg_pattern):
                return True
            # Also try with key prefix for "key:value" patterns
            for key, val in arguments.items():
                if fnmatch.fnmatch(f"{key}:{val}", self.arg_pattern):
                    return True

        return False


# Regex for parsing "ToolName(content:pattern)" syntax
_MATCHER_RE = re.compile(r"^([^(]+?)(?:\(([^)]*)\))?$")


def compile_matcher(pattern: str) -> ToolMatcher:
    """Compile a pattern string into a ToolMatcher.

    Examples:
        "Bash"           -> tool_pattern="Bash", arg_pattern=""
        "Bash(git:*)"    -> tool_pattern="Bash", arg_pattern="git:*"
        "Bash(*rm*)"     -> tool_pattern="Bash", arg_pattern="*rm*"
        "*"              -> tool_pattern="*",    arg_pattern=""
    """
    m = _MATCHER_RE.match(pattern.strip())
    if m is None:
        return ToolMatcher(tool_pattern=pattern, arg_pattern="", raw=pattern)

    tool_pat = m.group(1).strip()
    arg_pat = (m.group(2) or "").strip()

    return ToolMatcher(tool_pattern=tool_pat, arg_pattern=arg_pat, raw=pattern)


def matches_any(
    matchers: tuple[ToolMatcher, ...],
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> bool:
    """Check if a tool call matches any of the given matchers."""
    return any(m.matches(tool_name, arguments) for m in matchers)


def _flatten_args(arguments: dict[str, Any]) -> str:
    """Flatten arguments dict into a colon-separated searchable string.

    Example: {"command": "git commit -m 'fix'"} -> "command:git commit -m 'fix'"
    """
    parts = []
    for key, value in arguments.items():
        parts.append(f"{key}:{value}")
    return " ".join(parts)
