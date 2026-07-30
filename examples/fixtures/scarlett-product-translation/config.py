"""Minimal config stub for the sandboxed product_translation test.

The real scarlett config module loads settings from the environment via
pydantic-settings and pulls in the whole backend. The product_translation tests
never exercise it -- they patch ``product_translation.get_settings`` with their
own SimpleNamespace -- so the sandbox only needs ``get_settings`` to exist and be
importable. This stub keeps the sandbox self-contained with no external deps.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """The handful of fields product_translation reads off settings."""

    openrouter_api_key: str = ""
    extraction_model: str = "extraction-model"
    creative_model: str = "creative-model"


def get_settings() -> Settings:
    """Return a Settings singleton (patched out in tests)."""
    return Settings()
