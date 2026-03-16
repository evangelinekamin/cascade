"""External tool integrations for Cascade.

Integrations are loaded lazily to avoid adding import weight to the core.
External packages can register integrations via the ``cascade.integrations``
entry point group.
"""

from importlib.metadata import entry_points
from typing import Any


def _load_entry_points() -> dict[str, Any]:
    """Discover integrations registered via entry points."""
    eps = entry_points()
    group = eps.get("cascade.integrations", []) if isinstance(eps, dict) else eps
    if not isinstance(group, list):
        # Python 3.12+ returns SelectableGroups
        try:
            group = eps.select(group="cascade.integrations")
        except (AttributeError, TypeError):
            group = []
    result = {}
    for ep in group:
        try:
            result[ep.name] = ep.load()
        except Exception:
            pass
    return result


def get_integration(name: str) -> Any:
    """Get an integration class by name via entry points."""
    externals = _load_entry_points()
    return externals.get(name)


__all__ = ["get_integration"]
