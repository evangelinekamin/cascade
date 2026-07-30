"""Small runtime portability helpers shared by execution lanes."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from pathlib import Path

_PYTHON_MODULE_RE = re.compile(r"(?<!\S)python(?=\s+-m\s+)")


def user_cache_path(*parts: str) -> Path:
    """Return an XDG-aware path inside Cascade's user cache directory."""
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", "").strip()
    cache_home = (
        Path(xdg_cache_home).expanduser()
        if xdg_cache_home
        else Path.home() / ".cache"
    )
    return cache_home / "cascade" / Path(*parts)


def portable_python_command(command: str) -> str:
    """Use Cascade's interpreter when a command assumes a missing ``python`` alias."""
    if not command or shutil.which("python") is not None:
        return command
    # Cascade is often installed with pipx: sys.executable then points at
    # Cascade's minimal private environment, which intentionally lacks a
    # project's dev dependencies. Prefer the system/active ``python3`` command
    # when present; fall back to our interpreter only where no such alias exists.
    executable = shlex.quote(shutil.which("python3") or sys.executable)
    return _PYTHON_MODULE_RE.sub(executable, command)
