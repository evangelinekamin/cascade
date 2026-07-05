"""Split-flap odometer counter for the exit summary.

10-digit zero-padded, right-to-left digit settling at ~30fps.
Any keypress skips to the final value.
"""

import time

from rich.text import Text
from textual.widgets import Static

from ..theme import PALETTE


class OdometerCounter(Static):
    """Animated 10-digit token counter."""

    DEFAULT_CSS = """
    OdometerCounter {
        height: 1;
        width: 100%;
        text-align: center;
    }
    """

    def __init__(self, target_value: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._target = target_value
        self._display_value = 0
        self._animating = True
        self._timer = None
        self._started_at = 0.0
        self._duration = 1.3 if target_value < 100_000 else 1.7

    def on_mount(self) -> None:
        if self._target > 0:
            self._started_at = time.monotonic()
            self._timer = self.set_interval(1 / 24, self._step)

    def _step(self) -> None:
        if not self._animating:
            if self._timer:
                self._timer.stop()
            return

        elapsed = max(0.0, time.monotonic() - self._started_at)
        progress = min(1.0, elapsed / self._duration)
        eased = 1 - ((1 - progress) ** 3)
        self._display_value = int(round(self._target * eased))

        if progress >= 1.0:
            self._display_value = self._target
            self._animating = False

        self.refresh()

    def skip_animation(self) -> None:
        """Jump to final value immediately."""
        self._display_value = self._target
        self._animating = False
        if self._timer:
            self._timer.stop()
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append("  total: ", style=f"dim {PALETTE.text_dim}")
        for d in f"{self._display_value:010d}":
            t.append(str(d), style=f"bold {PALETTE.text_bright}")
        t.append(" tokens", style=f"dim {PALETTE.text_dim}")
        return t
