"""Normalized per-response token usage, uniform across all providers.

Field contract (pi-style normalization):

- ``input``: NON-cached input tokens. Adapters whose APIs report cached
  tokens as a subset of the prompt count (OpenAI ``cached_tokens``,
  Gemini ``cachedContentTokenCount``) must subtract the cached portion.
  Anthropic already reports ``input_tokens`` exclusive of cache fields.
- ``cache_read`` / ``cache_write``: cached-prompt tokens read/written.
- ``output``: completion tokens.
- ``cost``: provider-reported USD for the request(s), when available
  (OpenRouter); ``None`` when the provider does not report cost.

``prompt_total`` is therefore the context the model actually saw, and
``total`` is what the request(s) consumed end to end — the same formula
Claude Code anchors its context accounting on (input + cache_read +
cache_write + output).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float | None = None

    @property
    def prompt_total(self) -> int:
        """Total prompt-side tokens: non-cached + cache reads + cache writes."""
        return self.input + self.cache_read + self.cache_write

    @property
    def total(self) -> int:
        """Everything the request consumed: prompt side plus output."""
        return self.prompt_total + self.output

    def add(self, other: "Usage") -> "Usage":
        """Accumulate another round's usage (tool loops sum per-round usage)."""
        if other.cost is None:
            cost = self.cost
        elif self.cost is None:
            cost = other.cost
        else:
            cost = self.cost + other.cost
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            cost=cost,
        )

    @classmethod
    def from_anthropic(cls, usage: dict) -> "Usage":
        """From an Anthropic usage dict (API SSE, tool loop, or claude CLI).

        Anthropic reports ``input_tokens`` exclusive of the cache fields,
        so no subtraction is needed.
        """
        return cls(
            input=int(usage.get("input_tokens") or 0),
            output=int(usage.get("output_tokens") or 0),
            cache_read=int(usage.get("cache_read_input_tokens") or 0),
            cache_write=int(usage.get("cache_creation_input_tokens") or 0),
        )

    @classmethod
    def from_openai(cls, usage: dict) -> "Usage":
        """From a chat-completions usage dict (OpenAI, OpenRouter, compatibles).

        ``prompt_tokens`` includes the cached portion, which is reported
        separately in ``prompt_tokens_details.cached_tokens`` — subtract it
        so ``input`` is non-cached. OpenRouter additionally reports ``cost``.
        """
        prompt = int(usage.get("prompt_tokens", usage.get("input_tokens")) or 0)
        completion = int(
            usage.get("completion_tokens", usage.get("output_tokens")) or 0
        )
        details = usage.get("prompt_tokens_details")
        cached = 0
        if isinstance(details, dict):
            cached = int(details.get("cached_tokens") or 0)
        raw_cost = usage.get("cost")
        cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else None
        return cls(
            input=max(prompt - cached, 0),
            output=completion,
            cache_read=cached,
            cost=cost,
        )

    @classmethod
    def from_gemini(cls, usage: dict) -> "Usage":
        """From a Gemini ``usageMetadata`` dict.

        ``promptTokenCount`` includes ``cachedContentTokenCount`` — subtract
        it. Reasoning tokens (``thoughtsTokenCount``) are billed as output.
        """
        prompt = int(usage.get("promptTokenCount") or 0)
        cached = int(usage.get("cachedContentTokenCount") or 0)
        output = int(usage.get("candidatesTokenCount") or 0) + int(
            usage.get("thoughtsTokenCount") or 0
        )
        return cls(
            input=max(prompt - cached, 0),
            output=output,
            cache_read=cached,
        )
