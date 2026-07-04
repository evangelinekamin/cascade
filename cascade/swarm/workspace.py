"""Run a provider as a tool-using agent rooted in an isolated worktree.

The one place that knows how to give a provider edit-capability scoped to a
single worktree path -- shared by the code competition and the verification
loop. CLI-proxy providers (claude/gemini/codex) drive their own native agent
inside the worktree via ``working_directory``; API providers get sandboxed
file tools rooted at the worktree.
"""

from __future__ import annotations

import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from ..tools.schema import ToolDef, callable_to_tool_def


# Command output is capped to its tail: errors and test summaries live at the
# end, and an uncapped dump would blow a small local model's context window.
_OUTPUT_CAP = 4000
_TRUNCATION_MARKER = "[...truncated...]"


def _truncate_tail(text: str, cap: int = _OUTPUT_CAP) -> str:
    """Cap *text* to its last *cap* chars, marking that the head was dropped."""
    if len(text) <= cap:
        return text
    return f"{_TRUNCATION_MARKER}\n{text[-cap:]}"


class WorkspaceTools:
    """Restricted file tools rooted at a single worktree path."""

    def __init__(self, root: str, command_timeout: float = 120.0):
        self._root = Path(root).resolve()
        self._command_timeout = command_timeout

    def build(self) -> dict[str, ToolDef]:
        description = "Read, write, append, and list files inside the isolated coding worktree"
        return {
            "read_file": callable_to_tool_def(
                "read_file", self.read_file, description=description, read_only=True,
            ),
            "write_file": callable_to_tool_def("write_file", self.write_file, description=description),
            "append_file": callable_to_tool_def("append_file", self.append_file, description=description),
            "list_files": callable_to_tool_def(
                "list_files", self.list_files, description=description, read_only=True,
            ),
            "run_command": callable_to_tool_def(
                "run_command", self.run_command,
                description="Run a shell command in the workspace root",
            ),
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

    def run_command(self, command: str) -> str:
        """Run a shell command in the workspace root and return its output.

        Runs *command* through the shell with the worktree as the working
        directory and returns the exit code followed by the combined
        stdout+stderr. Use it to run tests (e.g. ``uv run pytest tests/test_x.py
        -q``), grep the tree, or run throwaway Python -- then read the output and
        iterate instead of editing blind. Long output is truncated to its tail.

        Args:
            command: The shell command to run in the workspace root.
        """
        # cwd is the isolated worktree (a throwaway copy under
        # ~/.cache/cascade/worktrees), so this cannot touch the user's real tree
        # -- the same trust model as the CLI-proxy providers, which already drive
        # full bash. v1 deliberately does no further sandboxing.
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self._root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self._command_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self._command_timeout:g}s"
        except Exception as exc:
            return f"Error running command: {exc}"

        combined = (completed.stdout or "") + (completed.stderr or "")
        return f"Exit code: {completed.returncode}\n{_truncate_tail(combined)}"


def run_agent_in_worktree(
    provider,
    prompt: str,
    worktree_path: str,
    system: Optional[str] = None,
    max_rounds: int = 15,
) -> str:
    """Run *provider* as a tool-using agent rooted at *worktree_path*.

    Returns the provider's final response text. CLI-proxy providers edit files
    directly through their native agent (driven into the worktree via
    ``working_directory``); API providers receive sandboxed ``WorkspaceTools``
    and up to ``max_rounds`` tool-calling round trips -- higher than the plain
    ``ask_with_tools`` default so a model can read several files before it writes.
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
            max_rounds=max_rounds,
        )
        return response
