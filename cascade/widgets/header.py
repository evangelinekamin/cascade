"""Welcome header: 3D shadow banner + version line + ghost table.

Hidden after the first user message is sent.
"""

from rich.text import Text
from textual.widget import Widget
from textual.app import ComposeResult
from textual.widgets import Static

from ..theme import PALETTE, PROVIDERS, get_provider_theme
from ..ui.banner import render_banner


class WelcomeHeader(Widget):
    """Top region with logo banner, version info, and provider roster."""

    DEFAULT_CSS = """
    WelcomeHeader {
        height: auto;
        width: 100%;
        padding: 1 2;
    }
    """

    def __init__(
        self,
        active_provider: str = "gemini",
        providers: dict | None = None,
        version: str = "0.3.0",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._active_provider = active_provider
        self._providers = providers or {}
        self._version = version

    def compose(self) -> ComposeResult:
        yield Static(render_banner(), id="banner")
        count = len(self._providers) if self._providers else len(PROVIDERS)
        yield Static(
            f"v{self._version} . {count} providers active",
            id="version_line",
        )
        yield ProviderGhostTable(
            providers=self._providers,
            active_provider=self._active_provider,
            id="ghost_table",
        )


class ProviderGhostTable(Static):
    """Borderless 3-column provider roster.

    Active row in bold accent; inactive rows barely visible.
    Reads from live provider dict rather than hardcoded values.
    """

    DEFAULT_CSS = """
    ProviderGhostTable {
        width: auto;
        height: auto;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        providers: dict | None = None,
        active_provider: str = "gemini",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._providers = providers or {}
        self._active_provider = active_provider

    def set_active(self, provider: str) -> None:
        """Update which provider row is highlighted and refresh."""
        self._active_provider = provider
        self.refresh()

    def render(self) -> Text:
        col_name = 14
        col_model = 26

        result = Text()

        if not self._providers:
            for name in sorted(PROVIDERS.keys()):
                pt = get_provider_theme(name)
                is_active = name == self._active_provider
                if is_active:
                    result.append("  \u25cf ", style=f"bold {pt.accent}")
                    result.append(name.ljust(col_name), style=f"bold {pt.accent}")
                    result.append("-".ljust(col_model), style=PALETTE.text_primary)
                    result.append("active", style=f"dim {pt.accent}")
                else:
                    result.append("    ")
                    result.append(name.ljust(col_name), style=f"dim {PALETTE.text_muted}")
                    result.append("\u00b7", style=f"dim {PALETTE.text_muted}")
                result.append("\n")
            return result

        for name in sorted(self._providers.keys()):
            prov = self._providers[name]
            model = str(getattr(getattr(prov, "config", None), "model", "?") or "?")
            is_active = name == self._active_provider
            pt = get_provider_theme(name)

            if is_active:
                result.append("  \u25cf ", style=f"bold {pt.accent}")
                result.append(name.ljust(col_name), style=f"bold {pt.accent}")
                result.append(model.ljust(col_model), style=PALETTE.text_primary)
                result.append("active", style=f"dim {pt.accent}")
            else:
                result.append("    ")
                result.append(name.ljust(col_name), style=f"dim {PALETTE.text_muted}")
                result.append("\u00b7", style=f"dim {PALETTE.text_muted}")
            result.append("\n")

        return result
