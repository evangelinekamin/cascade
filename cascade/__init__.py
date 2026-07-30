"""Cascade - Multi-model AI assistant CLI."""

from importlib import import_module

__version__ = "0.3.0"
__author__ = "Eve"

__all__ = [
    "get_app",
    "CascadeCore",
    "BaseProvider",
    "ProviderConfig",
    "ConfigManager",
]

_LAZY_EXPORTS = {
    "get_app": ("cascade.cli", "get_app"),
    "CascadeCore": ("cascade.cli", "CascadeCore"),
    "ConfigManager": ("cascade.config", "ConfigManager"),
    "BaseProvider": ("cascade.providers.base", "BaseProvider"),
    "ProviderConfig": ("cascade.providers.base", "ProviderConfig"),
}


def __getattr__(name: str):
    """Resolve convenience class/function exports without eager CLI startup."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
