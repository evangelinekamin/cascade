"""Every provider's ask_with_tools accepts the kwargs main.py passes.

Regression guard: main._tool_worker calls prov.ask_with_tools(..., on_tool_event=,
on_pending_message=) generically, so EVERY provider that overrides ask_with_tools
must accept those kwargs. OpenRouterProvider missed on_pending_message and every
build turn on openrouter (Eve's bulk provider) died with a TypeError. This
introspects the signatures so a new provider (or a new kwarg) fails here, loudly,
instead of at runtime mid-turn.
"""

import inspect
import pkgutil
import importlib

import cascade.providers as providers_pkg
from cascade.providers.base import BaseProvider

# The kwargs main.py hands to prov.ask_with_tools on the live tool path.
_REQUIRED_KWARGS = {"system", "max_rounds", "on_tool_event", "on_pending_message"}


def _all_provider_classes():
    classes = []
    for mod in pkgutil.iter_modules(providers_pkg.__path__):
        module = importlib.import_module(f"cascade.providers.{mod.name}")
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseProvider)
                and obj is not BaseProvider
                and "ask_with_tools" in obj.__dict__  # overrides it
            ):
                classes.append(obj)
    return classes


def test_every_provider_ask_with_tools_accepts_the_live_kwargs():
    offenders = {}
    for cls in _all_provider_classes():
        params = set(inspect.signature(cls.ask_with_tools).parameters)
        missing = _REQUIRED_KWARGS - params
        if missing:
            offenders[cls.__name__] = sorted(missing)
    assert not offenders, (
        "these providers' ask_with_tools reject kwargs main.py passes on every "
        f"tool turn (runtime TypeError mid-turn): {offenders}"
    )


def test_base_and_openrouter_are_actually_discovered():
    # Guard the guard: if discovery silently found nothing, the test above is vacuous.
    names = {c.__name__ for c in _all_provider_classes()}
    assert "OpenRouterProvider" in names
    assert len(names) >= 4
