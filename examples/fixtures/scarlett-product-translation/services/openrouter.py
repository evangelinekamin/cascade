"""Minimal OpenRouter client stub for the sandboxed product_translation test.

The real client talks to the OpenRouter HTTP API and tracks per-user spend
through the database. The product_translation tests never call the network --
they patch ``product_translation.call_completion`` with an AsyncMock -- so the
sandbox only needs an importable ``call_completion`` with a matching signature.
This stub keeps the sandbox self-contained with no httpx / sqlalchemy / DB deps.
"""

from __future__ import annotations

from typing import Any


async def call_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0,
    tools: list[dict[str, Any]] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Stand-in for the real completion call (patched out in tests)."""
    raise RuntimeError("call_completion stub should be patched in tests")
