"""Shared pytest fixtures."""

import pytest

import cascade.providers.openrouter as _openrouter


@pytest.fixture(autouse=True)
def _no_openrouter_meta_fetch(request):
    """Keep the suite hermetic and fast.

    OpenRouterProvider._apply_meta makes a live network call to fetch endpoint
    metadata on the first request. Disable it via the module flag for every test
    except the module that tests it explicitly -- a flag read at call time,
    rather than patching a method reached through a generator (which the mock
    does not reliably intercept). Nothing else reaches the network or has its
    provider_preferences mutated out from under an assertion.
    """
    is_meta_test = request.module.__name__.rsplit(".", 1)[-1] == "test_openrouter_meta"
    previous = _openrouter._META_ENABLED
    _openrouter._META_ENABLED = is_meta_test
    try:
        yield
    finally:
        _openrouter._META_ENABLED = previous
