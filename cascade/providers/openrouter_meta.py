"""Live OpenRouter endpoint metadata -- context window, quant, price, speed.

The same model is served by several endpoints with DIFFERENT context lengths
(64k / 128k / 163k), quantization levels, prices, and throughput. Hardcoding a
window is therefore wrong in both directions; this fetches the truth from
``/models/{id}/endpoints`` once per model (cached) and derives:

  * the effective (conservative) context window -- the smallest among the
    endpoints a request could actually route to, so compaction never lets the
    payload exceed the serving host and 400;
  * a recommended provider order that prefers UNQUANTIZED endpoints with good
    throughput and price, so routing favors faithful, fast, cheap hosts.

All best-effort: any failure returns None and the caller keeps its defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

# Quantization tags that degrade tool-call fidelity / quality; excluded when an
# unquantized endpoint is available.
_QUANTIZED = {"fp8", "fp6", "fp4", "int8", "int4", "int3", "int2", "gptq", "awq", "bnb"}

_DEFAULT_BASE = "https://openrouter.ai/api/v1"
_cache: dict[str, "ModelMeta | None"] = {}


@dataclass(frozen=True)
class Endpoint:
    provider_name: str
    context_length: int
    quantization: str
    prompt_price: float
    completion_price: float
    supports_tools: bool
    throughput: float | None

    @property
    def unquantized(self) -> bool:
        return self.quantization not in _QUANTIZED


@dataclass(frozen=True)
class ModelMeta:
    effective_context: int | None
    recommended_order: tuple[str, ...]
    endpoints: tuple[Endpoint, ...]


def _parse_endpoints(data: dict) -> list[Endpoint]:
    raw = ((data or {}).get("data") or {}).get("endpoints") or []
    out: list[Endpoint] = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        pricing = e.get("pricing") or {}
        try:
            context = int(e.get("context_length") or 0)
        except (TypeError, ValueError):
            context = 0
        try:
            throughput = e.get("throughput_last_30m")
            throughput = float(throughput) if throughput is not None else None
        except (TypeError, ValueError):
            throughput = None
        out.append(
            Endpoint(
                provider_name=str(e.get("provider_name") or ""),
                context_length=context,
                quantization=str(e.get("quantization") or "").lower(),
                prompt_price=_as_float(pricing.get("prompt")),
                completion_price=_as_float(pricing.get("completion")),
                supports_tools="tools" in (e.get("supported_parameters") or []),
                throughput=throughput,
            )
        )
    return out


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _analyze(endpoints: list[Endpoint], *, prefer_unquantized: bool = True) -> ModelMeta:
    """Derive the conservative window + a quality-ranked provider order."""
    usable = [e for e in endpoints if e.context_length > 0 and e.supports_tools]
    pool = usable or [e for e in endpoints if e.context_length > 0]
    unquant = [e for e in pool if e.unquantized]
    eligible = unquant if (prefer_unquantized and unquant) else pool
    effective = min((e.context_length for e in eligible), default=None)
    # Prefer faster (throughput desc, unknown last) then cheaper input.
    ranked = sorted(eligible, key=lambda e: (-(e.throughput or 0.0), e.prompt_price))
    order = tuple(dict.fromkeys(e.provider_name for e in ranked if e.provider_name))
    return ModelMeta(effective, order, tuple(endpoints))


def model_meta(
    model_id: str,
    *,
    base_url: str = _DEFAULT_BASE,
    timeout: float = 10.0,
    prefer_unquantized: bool = True,
) -> ModelMeta | None:
    """Cached, best-effort endpoint metadata for an OpenRouter *model_id*.

    Returns ``None`` on any network/parse failure so callers fall back to their
    configured defaults. ``ModelMeta.effective_context`` may be ``None`` if the
    model has no listed endpoints.
    """
    if not model_id:
        return None
    if model_id in _cache:
        return _cache[model_id]
    try:
        base = (base_url or _DEFAULT_BASE).rstrip("/")
        resp = httpx.get(f"{base}/models/{model_id}/endpoints", timeout=timeout)
        resp.raise_for_status()
        endpoints = _parse_endpoints(resp.json())
    except Exception:
        return None  # not cached -- a transient failure can be retried next time
    meta = _analyze(endpoints, prefer_unquantized=prefer_unquantized)
    _cache[model_id] = meta
    return meta
