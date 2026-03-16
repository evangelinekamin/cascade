"""Split-flap odometer counter for the exit summary.

10-digit zero-padded, right-to-left digit settling at ~30fps.
Any keypress skips to the final value.
"""

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
        self._target_digits = [int(d) for d in f"{target_value:010d}"]
        self._current = [0] * 10
        self._animating = True
        # Settle from right (pos 9) to left (pos 0)
        self._settle_pos = 9
        self._timer = None

    def on_mount(self) -> None:
        if self._target > 0:
            # ~8fps for a satisfying mechanical tick-up effect
            self._timer = self.set_interval(1 / 8, self._step)

    def _step(self) -> None:
        if not self._animating:
            if self._timer:
                self._timer.stop()
            return

        # Advance the digit at _settle_pos toward its target
        pos = self._settle_pos
        if self._current[pos] != self._target_digits[pos]:
            self._current[pos] = (self._current[pos] + 1) % 10
        else:
            # This digit settled; move to the next one left
            if self._settle_pos > 0:
                self._settle_pos -= 1
            else:
                self._animating = False

        self.refresh()

    def skip_animation(self) -> None:
        """Jump to final value immediately."""
        self._current = list(self._target_digits)
        self._animating = False
        if self._timer:
            self._timer.stop()
        self.refresh()

    def render(self) -> Text:
        t = Text()
        t.append("  total: ", style=f"dim {PALETTE.text_dim}")
        for d in self._current:
            t.append(str(d), style=f"bold {PALETTE.text_bright}")
        t.append(" tokens", style=f"dim {PALETTE.text_dim}")
        return t
