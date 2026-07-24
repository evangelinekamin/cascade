"""Permission ask dialog: the rare interruption in an auto-posture world.

Only sacred paths, dangerous shell shapes, explicit ask rules, and
out-of-workspace writes reach this screen; everything else auto-resolves.
Dismisses with "allow" | "always" | "deny".
"""

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ..theme import PALETTE


class PermissionScreen(ModalScreen[str]):
    """Modal asking the user to resolve a permission verdict."""

    DEFAULT_CSS = """
    PermissionScreen {
        align: center middle;
    }
    PermissionScreen > Vertical {
        width: 72;
        max-width: 90%;
        padding: 1 2;
        border: solid #e5c747;
        background: #0d1117;
    }
    PermissionScreen .perm-buttons {
        height: 3;
        align-horizontal: center;
    }
    PermissionScreen Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("y", "answer('allow')", "Allow", show=False),
        Binding("a", "answer('always')", "Always (session)", show=False),
        Binding("n", "answer('deny')", "Deny", show=False),
        Binding("escape", "answer('deny')", "Deny", show=False),
    ]

    def __init__(self, tool_name: str, arguments: dict, reason: str) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._arguments = arguments
        self._reason = reason

    def compose(self) -> ComposeResult:
        header = Text()
        header.append("permission ", style=f"bold {PALETTE.amber}")
        header.append(f"· {self._tool_name}", style=PALETTE.text_bright)

        detail = Text()
        for key in ("command", "path", "file_path", "url"):
            value = self._arguments.get(key)
            if isinstance(value, str) and value:
                shown = value if len(value) <= 200 else value[:200] + "…"
                detail.append(f"{key}: ", style=f"dim {PALETTE.text_dim}")
                detail.append(shown, style=PALETTE.text_primary)
                break

        reason = Text(self._reason, style=f"dim {PALETTE.text_dim}")

        with Vertical():
            yield Static(header)
            yield Static(detail)
            yield Static(reason)
            with Horizontal(classes="perm-buttons"):
                yield Button("allow (y)", id="perm-allow", variant="success")
                yield Button("always (a)", id="perm-always")
                yield Button("deny (n)", id="perm-deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        answers = {
            "perm-allow": "allow",
            "perm-always": "always",
            "perm-deny": "deny",
        }
        self.dismiss(answers.get(event.button.id or "", "deny"))

    def action_answer(self, answer: str) -> None:
        self.dismiss(answer)
