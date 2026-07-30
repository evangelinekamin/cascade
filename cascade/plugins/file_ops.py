"""File operations plugin for Cascade."""

import hashlib
import time
from pathlib import Path
from typing import Any

from .base import BasePlugin
from .registry import register_plugin
from ..runtime import user_cache_path

# A single read must not flood the context. Bound the default read the way the
# exec/web tools bound theirs, spilling the full file so nothing is lost.
_MAX_READ_LINES = 2000
_MAX_READ_CHARS = 50_000
def _spill_read(path: str, text: str) -> str:
    """Write the full file to the artifact dir; return its path or ''."""
    try:
        artifact_dir = user_cache_path("file-reads")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(f"{time.time_ns()}:{path}".encode()).hexdigest()[:10]
        out = artifact_dir / f"{digest}.txt"
        out.write_text(text, errors="replace")
        return str(out)
    except Exception:
        return ""


@register_plugin("file_ops")
class FileOpsPlugin(BasePlugin):
    """Handle file reading and writing operations."""

    @property
    def name(self) -> str:
        return "file_ops"

    @property
    def description(self) -> str:
        return "Read, write, list, and append files"

    def get_tools(self) -> dict[str, Any]:
        return {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "append_file": self.append_file,
        }

    @staticmethod
    def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
        """Read file contents, bounded so one read can't flood the context.

        By default returns up to the first 2000 lines (50k chars); if the
        file is larger, the full text is spilled to disk and a recovery hint
        is appended. Pass offset (0-based line) and limit to page a window.

        Args:
            path: File to read.
            offset: First line to include (0-based).
            limit: Max lines to return (0 = default cap).
        """
        try:
            with open(Path(path).expanduser(), "r", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file: {e}"

        lines = content.split("\n")
        total = len(lines)
        offset = max(0, offset)
        span = limit if limit > 0 else _MAX_READ_LINES
        window = lines[offset:offset + span]
        body = "\n".join(window)

        char_capped = len(body) > _MAX_READ_CHARS
        if char_capped:
            body = body[:_MAX_READ_CHARS]
        shown_to = offset + len(window)
        truncated = char_capped or shown_to < total or offset > 0

        # A small whole-file read returns verbatim, no annotation.
        if not truncated:
            return content

        spill = _spill_read(path, content)
        hint = (
            f"read {spill} for the full file" if spill
            else f"call read_file with offset={shown_to}"
        )
        note = f"[lines {offset + 1}-{shown_to} of {total}"
        if char_capped:
            note += f"; truncated at {_MAX_READ_CHARS} chars"
        note += f" — {hint}, or pass offset/limit to page]"
        return f"{body}\n\n{note}"

    @staticmethod
    def write_file(path: str, content: str) -> bool:
        """Write content to file."""
        try:
            file_path = Path(path).expanduser()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing file: {e}")
            return False

    @staticmethod
    def list_files(path: str = ".") -> list[str]:
        """List files in a directory."""
        try:
            return [str(f) for f in Path(path).expanduser().iterdir()]
        except Exception as e:
            return [f"Error: {e}"]

    @staticmethod
    def append_file(path: str, content: str) -> bool:
        """Append content to file."""
        try:
            with open(Path(path).expanduser(), "a") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error appending to file: {e}")
            return False
