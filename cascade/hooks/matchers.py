"""Pattern matching for hook tool filters.

Supports Claude Code-style tool matchers:
    "Bash"              - matches tool name exactly
    "Bash(git:*)"       - tool name + argument value starting with "git"
    "Bash(*rm*)"        - wildcard match anywhere in an argument value
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
    arg_pattern: str    # argument-content pattern, "prefix:glob" or glob (empty = any)
    raw: str            # original pattern string

    def matches(self, tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
        """Check if a tool call matches this pattern."""
        if not fnmatch.fnmatch(tool_name, self.tool_pattern):
            return False

        if not self.arg_pattern:
            return True

        if arguments is None:
            return False

        # The arg pattern matches the call if it matches any argument value.
        return any(
            _arg_matches(str(value), self.arg_pattern)
            for value in arguments.values()
        )


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


def _arg_matches(value: str, pattern: str) -> bool:
    """Match a single argument value against an arg-content pattern.

    Two forms are supported:

        "prefix:glob"  Claude Code convention -- the value must start with
                       ``prefix`` and the remainder must match ``glob``.
                       ``git:*`` matches any value beginning with "git"
                       (e.g. "git status") but not "npm install".
        "glob"         a plain fnmatch pattern applied to the whole value,
                       so ``*rm*`` matches "rm" appearing anywhere.
    """
    if ":" in pattern:
        prefix, glob = pattern.split(":", 1)
        return value.startswith(prefix) and fnmatch.fnmatch(value[len(prefix):], glob)
    return fnmatch.fnmatch(value, pattern)
