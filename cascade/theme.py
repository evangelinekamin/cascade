"""Cascade TUI theme: canonical color system and provider identities.

All hex values live here. Widgets never hardcode colors.
"""

from dataclasses import dataclass
from typing import Collection, Dict


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Palette:
    """Full surface / text / functional color palette."""

    # Surfaces
    bg: str = "#0d1117"
    surface: str = "#121218"
    code_bg: str = "#161b22"
    border: str = "#30363d"
    border_subtle: str = "#1a1a28"
    user_msg_bg: str = "#111820"

    # Text hierarchy
    text_bright: str = "#e8e8f0"
    text_primary: str = "#c9d1d9"
    text_dim: str = "#6e7681"
    text_muted: str = "#363648"

    # Logo wordmark (vertical gradient: light aqua -> deep blue),
    # from the "Cascade Logo (Blue)" design.
    logo_top: str = "#a5e8f0"
    logo_bottom: str = "#1e4fa8"

    # Functional
    cyan: str = "#00d4e5"
    green: str = "#34d399"
    yellow: str = "#e5c747"
    amber: str = "#e5c747"
    red: str = "#e55a6e"
    blue: str = "#5a9cf0"
    pink: str = "#e55a9b"
    purple: str = "#7c3aed"

    # Semantic aliases
    inline_code: str = "#00d4e5"
    file_ops: str = "#e5c747"
    diff_add: str = "#34d399"
    diff_del: str = "#e55a6e"
    error: str = "#e55a6e"
    spinner: str = "#e5c747"

    # Syntax tokens
    syntax_keyword: str = "#5a9cf0"
    syntax_string: str = "#34d399"
    syntax_builtin: str = "#00d4e5"
    syntax_self: str = "#e55a6e"
    syntax_class: str = "#e5c747"


PALETTE = Palette()


# ---------------------------------------------------------------------------
# Provider themes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderTheme:
    """Visual identity for a single provider."""

    name: str
    accent: str
    accent_shimmer: str  # lighter variant for animation oscillation
    dim: str             # accent at ~12% opacity on dark bg
    abbreviation: str    # 3-char label for gutter
    default_mode: str
    label: str


PROVIDERS: Dict[str, ProviderTheme] = {
    "gemini": ProviderTheme(
        name="gemini",
        accent="#b44dff",
        accent_shimmer="#c97aff",
        dim="#2d1a3d",
        abbreviation="gem",
        default_mode="design",
        label="design mode",
    ),
    "claude": ProviderTheme(
        name="claude",
        accent="#f0956c",
        accent_shimmer="#f5b896",
        dim="#3d2a1d",
        abbreviation="cla",
        default_mode="plan",
        label="plan mode",
    ),
    "openai": ProviderTheme(
        name="openai",
        accent="#34d399",
        accent_shimmer="#6ee7b7",
        dim="#1d3d2a",
        abbreviation="oai",
        default_mode="build",
        label="build mode",
    ),
    "openrouter": ProviderTheme(
        name="openrouter",
        accent="#d94060",
        accent_shimmer="#e87088",
        dim="#3d1d25",
        abbreviation="ort",
        default_mode="test",
        label="test mode",
    ),
}

_NEUTRAL = ProviderTheme(
    name="unknown",
    accent=PALETTE.text_dim,
    accent_shimmer=PALETTE.text_primary,
    dim=PALETTE.border_subtle,
    abbreviation="???",
    default_mode="chat",
    label="chat mode",
)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

MODES: Dict[str, Dict[str, str]] = {
    "design": {"provider": "gemini", "color": "#b44dff"},
    "plan": {"provider": "claude", "color": "#f0956c"},
    "build": {"provider": "openai", "color": "#34d399"},
    "test": {"provider": "openrouter", "color": "#d94060"},
}

MODE_CYCLE = ("design", "plan", "build", "test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_provider_theme(name: str) -> ProviderTheme:
    """Look up a provider theme by name, with a neutral fallback."""
    return PROVIDERS.get(name, _NEUTRAL)


def get_available_modes(available_providers: Collection[str] | None = None) -> tuple[str, ...]:
    """Return mode cycle filtered to the providers available in this session."""
    if not available_providers:
        return MODE_CYCLE
    available = set(available_providers)
    return tuple(
        mode_name
        for mode_name in MODE_CYCLE
        if MODES.get(mode_name, {}).get("provider") in available
    )


def get_accent(provider: str) -> str:
    """Get the accent hex for a provider."""
    return get_provider_theme(provider).accent


def get_shimmer(provider: str) -> str:
    """Get the shimmer hex for a provider."""
    return get_provider_theme(provider).accent_shimmer


def get_dim(provider: str) -> str:
    """Get the dim tint hex for a provider."""
    return get_provider_theme(provider).dim


def get_abbreviation(provider: str) -> str:
    """Get the 3-char gutter abbreviation for a provider."""
    return get_provider_theme(provider).abbreviation
