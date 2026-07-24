"""Run a provider as a tool-using agent rooted in an isolated worktree.

The one place that knows how to give a provider edit-capability scoped to a
single worktree path -- shared by the code competition and the verification
loop. CLI-proxy providers (claude/gemini/codex) drive their own native agent
inside the worktree via ``working_directory``; API providers get sandboxed
file tools rooted at the worktree.
"""

from __future__ import annotations

from contextlib import nullcontext
from fnmatch import fnmatch
from pathlib import Path
from pathlib import PurePosixPath
from typing import Optional

from ..tools.schema import ToolDef, callable_to_tool_def
from .lifecycle import CancellationToken, RunCancelled, run_cancellable_shell


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

    def __init__(
        self,
        root: str,
        command_timeout: float = 120.0,
        cancel_token: Optional[CancellationToken] = None,
    ):
        self._root = Path(root).resolve()
        self._command_timeout = command_timeout
        self._cancel_token = cancel_token

    def _checkpoint(self) -> None:
        if self._cancel_token is not None:
            self._cancel_token.checkpoint()

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

    def build_read_only(self) -> dict[str, ToolDef]:
        """Return repository-inspection tools with no mutation or shell access.

        This is the capability set used by the cheap reconnaissance lane.  Keeping
        it separate from :meth:`build` makes the coordinator's promise of
        "read-only" executable rather than a prompt instruction a model can ignore.
        """
        return {
            "read_file": callable_to_tool_def(
                "read_file",
                self.read_file,
                description="Read a UTF-8 text file inside the repository",
                read_only=True,
            ),
            "list_files": callable_to_tool_def(
                "list_files",
                self.list_files,
                description="List immediate children inside the repository",
                read_only=True,
            ),
            "search_files": callable_to_tool_def(
                "search_files",
                self.search_files,
                description=(
                    "Search repository text files for a literal string; returns "
                    "path:line matches without invoking a shell"
                ),
                read_only=True,
            ),
        }

    def build_verify(self) -> dict[str, ToolDef]:
        """Read-only inspection tools PLUS run_command, but no write tools.

        The test-mode recon lane: it must actually run the project's checks
        (tests/build/type-check) to verify it works -- which a pure read-only
        set cannot -- while still being unable to edit source. run_command is
        permission-gated at the executor, so transparent test/build commands
        auto-approve and destructive ones are refused.
        """
        tools = self.build_read_only()
        tools["run_command"] = callable_to_tool_def(
            "run_command", self.run_command,
            description="Run a shell command in the workspace root",
        )
        return tools

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
        self._checkpoint()
        try:
            content = self._resolve(path).read_text()
        except Exception as exc:
            return f"Error reading file: {exc}"
        self._checkpoint()
        return content

    def write_file(self, path: str, content: str) -> str:
        """Write file contents inside the worktree, then fast-check the result.

        Returns a short status. If the file is Python and now has a syntax error,
        the message names it so the model fixes the edit on its next turn -- rather
        than the break only surfacing at the (expensive) full-suite gate.
        """
        self._checkpoint()
        try:
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except Exception as exc:
            return f"Error writing {path}: {exc}"
        self._checkpoint()
        issue = _post_edit_check(target)
        if issue:
            return f"Wrote {path}, but a syntax check failed: {issue}. Fix it."
        return f"Wrote {path}"

    def append_file(self, path: str, content: str) -> str:
        """Append file contents inside the worktree, then fast-check the result."""
        self._checkpoint()
        try:
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as handle:
                handle.write(content)
        except Exception as exc:
            return f"Error appending {path}: {exc}"
        self._checkpoint()
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
        self._checkpoint()
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
        self._checkpoint()
        issue = _post_edit_check(target)
        if issue:
            return f"{note} in {path}, but a syntax check failed: {issue}. Fix it."
        return f"{note} in {path}"

    def list_files(self, path: str = ".") -> list[str]:
        """List immediate children under a worktree path."""
        self._checkpoint()
        try:
            target = self._resolve(path)
            result = sorted(str(item) for item in target.iterdir())
        except Exception as exc:
            return [f"Error: {exc}"]
        self._checkpoint()
        return result

    def search_files(
        self,
        query: str,
        path: str = ".",
        file_glob: str = "*",
        max_results: int = 100,
    ) -> list[str]:
        """Search text files under *path* for a case-insensitive literal string.

        The implementation deliberately avoids a shell and regular expressions:
        inputs come from a model, and reconnaissance should have no command
        execution side door. Results and file sizes are bounded to protect the
        model context from generated/vendor trees and accidental huge files.
        """
        self._checkpoint()
        if not query:
            return ["Error: query is empty"]
        glob_path = PurePosixPath(file_glob.replace("\\", "/"))
        if glob_path.is_absolute() or ".." in glob_path.parts:
            return [f"Error: unsafe file_glob: {file_glob}"]
        try:
            target = self._resolve(path)
            limit = max(1, min(int(max_results), 200))
        except Exception as exc:
            return [f"Error: {exc}"]

        needle = query.casefold()
        matches: list[str] = []
        try:
            candidates = target.rglob("*")
            for candidate in candidates:
                self._checkpoint()
                if len(matches) >= limit:
                    break
                if not candidate.is_file():
                    continue
                try:
                    resolved = candidate.resolve()
                    if resolved != self._root and self._root not in resolved.parents:
                        continue
                    relative = resolved.relative_to(self._root)
                    relative_text = relative.as_posix()
                    if ".git" in relative.parts:
                        continue
                    if not (
                        fnmatch(relative_text, file_glob)
                        or fnmatch(relative.name, file_glob)
                    ):
                        continue
                    if resolved.stat().st_size > 1_000_000:
                        continue
                    lines = resolved.read_text(errors="replace").splitlines()
                except (OSError, UnicodeError, ValueError):
                    continue
                for line_number, line in enumerate(lines, start=1):
                    if needle in line.casefold():
                        excerpt = line.strip()
                        if len(excerpt) > 240:
                            excerpt = excerpt[:237] + "..."
                        matches.append(f"{relative_text}:{line_number}: {excerpt}")
                        if len(matches) >= limit:
                            break
        except RunCancelled:
            raise
        except Exception as exc:
            return [f"Error: {exc}"]
        self._checkpoint()
        return matches or ["No matches"]

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
            combined, returncode, timed_out = run_cancellable_shell(
                command,
                str(self._root),
                self._command_timeout,
                self._cancel_token,
            )
        except RunCancelled:
            raise
        except Exception as exc:
            return f"Error running command: {exc}"
        if timed_out:
            return f"Command timed out after {self._command_timeout:g}s"
        return f"Exit code: {returncode}\n{_truncate_tail(combined)}"


def run_agent_in_worktree(
    provider,
    prompt: str,
    worktree_path: str,
    system: Optional[str] = None,
    max_rounds: int = 15,
    on_tool_event=None,
    cancel_token: Optional[CancellationToken] = None,
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
    cancellation_scope = getattr(provider, "cancellation_scope", None)
    cancel_ctx = (
        cancellation_scope(cancel_token)
        if callable(cancellation_scope) and cancel_token is not None
        else nullcontext()
    )
    # Scope the permission gate to the worktree: WorkspaceTools already
    # confine writes here, so in-worktree edits should auto-approve while
    # the sacred/dangerous-shell floors still catch escapes (curl|sh,
    # rm -rf ~). Without this, the launch-cwd-rooted engine would treat
    # every legitimate worktree write as an out-of-workspace ask and, in a
    # headless lane, escalate to a hard stop. Restored in the finally.
    engine = getattr(provider, "permission_engine", None)
    scoped = engine.for_workspace(worktree_path) if engine is not None else None

    with ctx, cancel_ctx:
        if cancel_token is not None:
            cancel_token.checkpoint()
        if getattr(provider, "_use_cli_proxy", False):
            response = provider.ask_single(prompt, system=system)
            if cancel_token is not None:
                cancel_token.checkpoint()
            return response
        if scoped is not None:
            provider.permission_engine = scoped
        try:
            response, _tool_log = provider.ask_with_tools(
                [{"role": "user", "content": prompt}],
                WorkspaceTools(worktree_path, cancel_token=cancel_token).build(),
                system=system,
                max_rounds=max_rounds,
                on_tool_event=on_tool_event,
            )
        finally:
            if scoped is not None:
                provider.permission_engine = engine
        if cancel_token is not None:
            cancel_token.checkpoint()
        return response
