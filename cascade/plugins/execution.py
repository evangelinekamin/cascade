"""Shell execution plugin for Cascade chat tools.

Gives tool-enabled (direct-API) providers a single ``run_command`` tool so a
chatting model -- e.g. GPT in test mode -- can run the project's tests and
inspect the tree, then report findings. CLI-proxy providers (Claude via the
subscription) already have this natively through the CLI's own bash tool.

Opt-in: gated behind ``tools.exec`` in config (default off) because, unlike
/solve's runner, this executes real shell commands in the launch directory, not
a throwaway worktree.
"""

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import BasePlugin
from .registry import register_plugin
from ..tools.filters import apply_output_filter

_MAX_OUTPUT_CHARS = 4000
_COMMAND_TIMEOUT = 120.0

# tee: raw command output spills here so a filtered/truncated result
# always has a recoverable full copy (rtk's bound-and-spill principle).
_ARTIFACT_DIR = Path.home() / ".cache" / "cascade" / "command-output"


def _spill_raw(command: str, raw: str) -> str:
    """Write full output to the artifact dir; return a recovery hint or ''."""
    try:
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(f"{time.time_ns()}:{command}".encode()).hexdigest()[:10]
        path = _ARTIFACT_DIR / f"{digest}.txt"
        path.write_text(f"$ {command}\n\n{raw}", errors="replace")
        return f"read {path}"
    except Exception:
        return ""


def _truncate_tail(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Keep the tail of long output -- the exit status and final errors matter most."""
    if len(text) <= limit:
        return text
    return "...(truncated)...\n" + text[-limit:]


@register_plugin("exec")
class ExecPlugin(BasePlugin):
    """Run shell commands in the working directory (opt-in via tools.exec)."""

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Run a shell command in the working directory and return its output"

    def get_tools(self) -> dict[str, Any]:
        return {"run_command": self.run_command}

    @staticmethod
    def run_command(command: str) -> str:
        """Run a shell command in the current working directory and return output.

        Returns the exit code followed by the combined stdout+stderr (long output
        truncated to its tail). Use it to run the project's tests (e.g.
        ``uv run pytest tests/ -q``), grep the tree, or run throwaway scripts, then
        read the result and report findings instead of guessing.

        This runs in the real launch directory, which is why it is gated behind
        ``tools.exec`` -- the same trust model as the CLI-proxy providers, which
        already drive full bash there.

        Args:
            command: The shell command to run.
        """
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=_COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {_COMMAND_TIMEOUT:g}s"
        except Exception as exc:
            return f"Error running command: {exc}"

        combined = (completed.stdout or "") + (completed.stderr or "")

        # Filter first (failures-only, grouped grep, etc.), spilling the full
        # raw output to disk so nothing is unrecoverable; then tail-truncate
        # whatever survives. never_worse inside the filter guarantees the
        # filtered form is only used when it is genuinely smaller.
        hint = _spill_raw(command, combined) if len(combined) > _MAX_OUTPUT_CHARS else ""
        filtered = apply_output_filter(command, combined, recovery_hint=hint)
        return f"Exit code: {completed.returncode}\n{_truncate_tail(filtered)}"
