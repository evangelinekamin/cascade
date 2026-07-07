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


def _post_edit_check(path: Path) -> str:
    """Fast, dependency-free validity check for a just-written file.

    Returns "" when the file looks valid, else a one-line finding. Today this is a
    Python syntax parse only (milliseconds, in-process, no subprocess) -- the cheap
    first tier of "verify before moving on" that catches a broken edit immediately
    instead of letting it surface at the far more expensive full-suite gate. Other
    languages pass through silently.
    """
    if path.suffix != ".py":
        return ""
    try:
        import ast

        ast.parse(path.read_text(), filename=str(path))
        return ""
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        return f"{exc.msg}{where}"
    except Exception:
        return ""


def _replace_in_text(content: str, old: str, new: str) -> "tuple[Optional[str], str]":
    """Replace *old* with *new* in *content* via a ladder of match strategies.

    Returns ``(new_content, note)``; ``new_content`` is None when *old* cannot be
    located uniquely. Cheap models rarely reproduce whitespace/indentation exactly,
    so an exact-match-only edit tool wastes /solve iterations -- this first tries an
    exact unique match, then falls back to a per-line whitespace-tolerant match.
    """
    if not old:
        return None, "old_string is empty"

    exact = content.count(old)
    if exact == 1:
        return content.replace(old, new, 1), "replaced (exact match)"
    if exact > 1:
        return None, f"old_string matches {exact} places -- add surrounding context to disambiguate"

    # Whitespace-tolerant: compare lines with leading/trailing whitespace stripped,
    # so the model need not reproduce indentation exactly.
    target = [ln.strip() for ln in old.split("\n")]
    lines = content.split("\n")
    stripped = [ln.strip() for ln in lines]
    span = len(target)
    hits = [
        i for i in range(len(stripped) - span + 1)
        if stripped[i:i + span] == target
    ]
    if len(hits) == 1:
        i = hits[0]
        merged = lines[:i] + new.split("\n") + lines[i + span:]
        return "\n".join(merged), "replaced (whitespace-tolerant match)"
    if len(hits) > 1:
        return None, f"old_string matches {len(hits)} places (whitespace-tolerant) -- add more context"
    return None, "old_string not found in the file"


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
            "replace_in_file": callable_to_tool_def(
                "replace_in_file", self.replace_in_file,
                description="Replace a snippet in a file (whitespace-tolerant) -- a surgical edit that avoids rewriting the whole file",
            ),
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

    def write_file(self, path: str, content: str) -> str:
        """Write file contents inside the worktree, then fast-check the result.

        Returns a short status. If the file is Python and now has a syntax error,
        the message names it so the model fixes the edit on its next turn -- rather
        than the break only surfacing at the (expensive) full-suite gate.
        """
        try:
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except Exception as exc:
            return f"Error writing {path}: {exc}"
        issue = _post_edit_check(target)
        if issue:
            return f"Wrote {path}, but a syntax check failed: {issue}. Fix it."
        return f"Wrote {path}"

    def append_file(self, path: str, content: str) -> str:
        """Append file contents inside the worktree, then fast-check the result."""
        try:
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as handle:
                handle.write(content)
        except Exception as exc:
            return f"Error appending {path}: {exc}"
        issue = _post_edit_check(target)
        if issue:
            return f"Appended {path}, but a syntax check failed: {issue}. Fix it."
        return f"Appended {path}"

    def replace_in_file(self, path: str, old_string: str, new_string: str) -> str:
        """Replace old_string with new_string in a file -- a surgical edit that
        avoids rewriting the whole file. Whitespace-tolerant per line, so exact
        indentation need not be reproduced; old_string must match exactly one place
        (add surrounding context to disambiguate). Cheaper and less error-prone than
        rewriting a large file with write_file.
        """
        try:
            target = self._resolve(path)
            content = target.read_text()
        except Exception as exc:
            return f"Error reading {path}: {exc}"
        updated, note = _replace_in_text(content, old_string, new_string)
        if updated is None:
            return f"No change to {path}: {note}"
        try:
            target.write_text(updated)
        except Exception as exc:
            return f"Error writing {path}: {exc}"
        issue = _post_edit_check(target)
        if issue:
            return f"{note} in {path}, but a syntax check failed: {issue}. Fix it."
        return f"{note} in {path}"

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
    on_tool_event=None,
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
            on_tool_event=on_tool_event,
        )
        return response
