"""Bottom status line -- dock bottom, height 1.

Left:  ~/path . branch*
Right: colored dot per provider + token count
"""

import os
import subprocess
from pathlib import Path

from rich.text import Text
from textual.widgets import Static

from ..theme import PALETTE, PROVIDERS


def _git_info() -> tuple[str, bool]:
    """Return (branch_name, dirty) or ("", False) if not a repo."""
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        if not branch:
            return "", False
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
        )
        return branch, dirty
    except Exception:
        return "", False


def _shorten_path(path: str) -> str:
    home = str(Path.home())
    if path.startswith(home):
        return "~" + path[len(home):]
    return path


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


class StatusBar(Static):
    """Single-line status bar docked to the bottom of the screen."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        width: 100%;
        background: #0d1117;
        padding: 0 2;
    }
    """

    def __init__(
        self,
        cwd: str = "",
        branch: str = "",
        provider_tokens: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cwd = cwd or _shorten_path(os.getcwd())
        self._branch = branch
        self._dirty = False
        self._provider_tokens: dict[str, int] = provider_tokens or {}
        self._override_text: str = ""

        if not branch:
            b, d = _git_info()
            self._branch = b
            self._dirty = d

    def render(self) -> Text:
        t = Text()

        if self._override_text:
            t.append(f" {self._override_text}", style=f"dim {PALETTE.text_dim}")
            return t

        # Left: path . branch
        t.append(f" {self._cwd}", style=f"dim {PALETTE.text_dim}")
        if self._branch:
            suffix = "*" if self._dirty else ""
            t.append(f" . {self._branch}{suffix}", style=f"dim {PALETTE.text_dim}")

        # Build right side
        right = Text()
        for name, ptheme in PROVIDERS.items():
            count = self._provider_tokens.get(name, 0)
            right.append(" \u25cf", style=ptheme.accent)
            right.append(f" {_fmt(count)}", style=f"dim {PALETTE.text_dim}")

        # Pad between left and right
        width = self.size.width if self.size.width > 0 else 80
        available = width - len(t.plain) - len(right.plain) - 2
        t.append(" " * max(available, 2))
        t.append_text(right)

        return t

    def update_tokens(self, provider_tokens: dict[str, int]) -> None:
        """Update token counts and refresh."""
        self._provider_tokens = dict(provider_tokens)
        self.refresh()

    def set_override(self, text: str) -> None:
        """Replace entire status bar with a single message."""
        self._override_text = text
        self.refresh()
