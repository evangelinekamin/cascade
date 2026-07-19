"""Token accounting: the single authority for windows, thresholds, and fill.

Every consumer of "how big is the context window", "how full are we", or
"should we compact" resolves through this module, so the number the user
sees, the number the compactor fires on, and the number the ledger
records are the same function. (Claude Code's statusline famously
disagrees with its own compactor; do not reintroduce that bug by
computing occupancy anywhere else.)

Anchoring rule: the last provider response's ``Usage.total`` already
accounts for the entire conversation the model saw plus its reply —
never sum usage across turns. Only the messages appended since that
response are estimated, at ~4 chars/token.
"""

from __future__ import annotations

import os
import re

from ..providers.usage import Usage

# Environment override: hard cap on any resolved window (tokens).
ENV_WINDOW_CAP = "CASCADE_MAX_CONTEXT_TOKENS"

DEFAULT_WINDOW = 128_000

# Per-provider fallback windows, keyed by config-level provider name.
PROVIDER_WINDOWS: dict[str, int] = {
    "claude": 200_000,
    "gemini": 1_000_000,
    "openai": 400_000,
    "openrouter": 128_000,
}

# Exact (provider, model) overrides. Grows as models with known windows
# are adopted; checked before the provider fallback.
MODEL_WINDOWS: dict[tuple[str, str], int] = {}

_MILLION_SUFFIX = re.compile(r"\[1m\]", re.IGNORECASE)

_CHARS_PER_TOKEN = 4

# Reserved for the model's reply (and a compaction summary, which must fit
# in the same request). Scaled down for small local windows.
_MAX_OUTPUT_RESERVE = 16_000

# Safety buffer below the effective window at which compaction fires.
_COMPACT_BUFFER = 13_000

# Warning band above the compaction threshold for UI coloring.
_WARN_BAND = 20_000


def window_for(
    provider: str,
    model: str = "",
    configured: int | None = None,
) -> int:
    """Resolve the context window for a (provider, model) pair.

    Precedence: explicit configuration > [1m] model suffix > exact
    (provider, model) entry > provider fallback > DEFAULT_WINDOW. The
    CASCADE_MAX_CONTEXT_TOKENS environment variable caps the result.
    """
    provider = (provider or "").lower()
    if configured:
        window = configured
    elif model and _MILLION_SUFFIX.search(model):
        window = 1_000_000
    elif (provider, model) in MODEL_WINDOWS:
        window = MODEL_WINDOWS[(provider, model)]
    else:
        window = PROVIDER_WINDOWS.get(provider, DEFAULT_WINDOW)

    cap = os.environ.get(ENV_WINDOW_CAP)
    if cap:
        try:
            window = min(window, int(cap))
        except ValueError:
            pass
    return window


def output_reserve(window: int, max_output: int | None = None) -> int:
    """Tokens held back for the model's reply (never more than 1/4 window)."""
    return min(max_output or _MAX_OUTPUT_RESERVE, _MAX_OUTPUT_RESERVE, window // 4)


def effective_window(window: int, max_output: int | None = None) -> int:
    """Usable context after the output reserve."""
    return window - output_reserve(window, max_output)


def compact_threshold(window: int, max_output: int | None = None) -> int:
    """Token count at which compaction fires (buffer below effective)."""
    return effective_window(window, max_output) - min(_COMPACT_BUFFER, window // 8)


def warn_threshold(window: int, max_output: int | None = None) -> int:
    """Token count at which the UI shifts to its warning band."""
    return compact_threshold(window, max_output) - min(_WARN_BAND, window // 8)


def estimate_tokens_from_chars(chars: int) -> int:
    """~4 chars/token heuristic — for the un-sent tail only, never the whole."""
    return max(chars, 0) // _CHARS_PER_TOKEN


def current_tokens(anchor: Usage | None, trailing_chars: int = 0) -> int:
    """Context occupancy: last real usage total + estimate of the tail.

    ``anchor`` is the most recent provider response's Usage (its input
    already contains all prior context). ``trailing_chars`` counts only
    content appended after that response.
    """
    anchored = anchor.total if anchor is not None else 0
    return anchored + estimate_tokens_from_chars(trailing_chars)
