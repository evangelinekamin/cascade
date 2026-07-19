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
        # Context occupancy: (tokens or None, compact threshold, warn
        # threshold, compaction count). None until the first real response.
        self._ctx: tuple[int | None, int, int, int] | None = None
        self._override_text: str = ""
        self._flash: str = ""
        self._flash_timer = None
        self._git_timer = None

        if not branch:
            b, d = _git_info()
            self._branch = b
            self._dirty = d

    def on_mount(self) -> None:
        # The branch/dirty state was a one-time snapshot, so it went stale
        # after any commit or checkout mid-session. Refresh on a slow
        # interval, but run the git subprocesses in a THREAD -- doing them
        # on the UI thread every few seconds visibly freezes the TUI.
        self._git_timer = self.set_interval(5.0, self._schedule_git_refresh)

    def _schedule_git_refresh(self) -> None:
        self.run_worker(self._refresh_git_worker, thread=True, exclusive=True, group="git")

    def _refresh_git_worker(self) -> None:
        branch, dirty = _git_info()
        if (branch, dirty) != (self._branch, self._dirty):
            self.app.call_from_thread(self._apply_git, branch, dirty)

    def _apply_git(self, branch: str, dirty: bool) -> None:
        self._branch = branch
        self._dirty = dirty
        self.refresh()

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

        # Build right side. A transient flash (copy confirmation, exit hint)
        # takes the corner briefly; otherwise surface a provider's dot only once
        # it has been used, so a fresh session isn't a row of "\u25cf 0 \u25cf 0".
        right = Text()
        if self._flash:
            right.append(self._flash, style=PALETTE.text_dim)
        else:
            if self._ctx is not None:
                tokens, threshold, warn, compactions = self._ctx
                if tokens is None:
                    right.append("ctx ?", style=f"dim {PALETTE.text_dim}")
                else:
                    pct = min(tokens * 100 // threshold, 999) if threshold > 0 else 0
                    if tokens >= threshold:
                        style = PALETTE.error
                    elif tokens >= warn:
                        style = PALETTE.amber
                    else:
                        style = f"dim {PALETTE.text_dim}"
                    right.append(f"ctx {pct}%", style=style)
                if compactions:
                    right.append(f" \u27f3{compactions}", style=f"dim {PALETTE.text_dim}")
                right.append("  ")
            for name, ptheme in PROVIDERS.items():
                count = self._provider_tokens.get(name, 0)
                if count <= 0:
                    continue
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

    def update_context(
        self,
        tokens: int | None,
        threshold: int,
        warn: int,
        compactions: int,
    ) -> None:
        """Update the context-occupancy display (None tokens = unknown)."""
        self._ctx = (tokens, threshold, warn, compactions)
        self.refresh()

    def set_override(self, text: str) -> None:
        """Replace entire status bar with a single message."""
        self._override_text = text
        self.refresh()

    def flash(self, message: str, timeout: float = 1.5) -> None:
        """Briefly show a right-aligned note, then revert to the token counts."""
        self._flash = message
        if self._flash_timer is not None:
            self._flash_timer.stop()
        self._flash_timer = self.set_timer(timeout, self._clear_flash)
        self.refresh()

    def _clear_flash(self) -> None:
        self._flash = ""
        self._flash_timer = None
        self.refresh()
