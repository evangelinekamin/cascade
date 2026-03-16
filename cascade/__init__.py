"""Cascade - Multi-model AI assistant CLI."""

__version__ = "0.3.0"
__author__ = "Eve"

from .cli import cli, get_app, CascadeCore
from .config import ConfigManager
from .providers.base import BaseProvider, ProviderConfig

__all__ = [
    "cli",
    "get_app",
    "CascadeCore",
    "BaseProvider",
    "ProviderConfig",
    "ConfigManager",
]
