"""Run a provider as a tool-using agent rooted in an isolated worktree.

The one place that knows how to give a provider edit-capability scoped to a
single worktree path -- shared by the code competition and the verification
loop. CLI-proxy providers (claude/gemini/codex) drive their own native agent
inside the worktree via ``working_directory``; API providers get sandboxed
file tools rooted at the worktree.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from ..tools.schema import ToolDef, callable_to_tool_def


class WorkspaceTools:
    """Restricted file tools rooted at a single worktree path."""

    def __init__(self, root: str):
        self._root = Path(root).resolve()

    def build(self) -> dict[str, ToolDef]:
        description = "Read, write, append, and list files inside the isolated coding worktree"
        return {
            "read_file": callable_to_tool_def("read_file", self.read_file, description=description),
            "write_file": callable_to_tool_def("write_file", self.write_file, description=description),
            "append_file": callable_to_tool_def("append_file", self.append_file, description=description),
            "list_files": callable_to_tool_def("list_files", self.list_files, description=description),
        }

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self._root / candidate
        resolved = candidate.resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise ValueError(f"Path escapes workspace: {path}")
        return resolved

    def read_file(self, path: str) -> str:
        """Read file contents from the worktree."""
        try:
            return self._resolve(path).read_text()
        except Exception as exc:
            return f"Error reading file: {exc}"

    def write_file(self, path: str, content: str) -> bool:
        """Write file contents inside the worktree."""
        try:
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            return True
        except Exception:
            return False

    def append_file(self, path: str, content: str) -> bool:
        """Append file contents inside the worktree."""
        try:
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as handle:
                handle.write(content)
            return True
        except Exception:
            return False

    def list_files(self, path: str = ".") -> list[str]:
        """List immediate children under a worktree path."""
        try:
            target = self._resolve(path)
            return sorted(str(item) for item in target.iterdir())
        except Exception as exc:
            return [f"Error: {exc}"]


def run_agent_in_worktree(
    provider,
    prompt: str,
    worktree_path: str,
    system: Optional[str] = None,
) -> str:
    """Run *provider* as a tool-using agent rooted at *worktree_path*.

    Returns the provider's final response text. CLI-proxy providers edit files
    directly through their native agent (driven into the worktree via
    ``working_directory``); API providers receive sandboxed ``WorkspaceTools``.
    """
    workdir = getattr(provider, "working_directory", None)
    ctx = provider.working_directory(worktree_path) if callable(workdir) else nullcontext()
    with ctx:
        if getattr(provider, "_use_cli_proxy", False):
            return provider.ask_single(prompt, system=system)
        response, _tool_log = provider.ask_with_tools(
            [{"role": "user", "content": prompt}],
            WorkspaceTools(worktree_path).build(),
            system=system,
        )
        return response
