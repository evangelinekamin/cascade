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

# Quantization tags that degrade tool-call fidelity / quality.
_QUANTIZED = {"fp8", "fp6", "fp4", "int8", "int4", "int3", "int2", "gptq", "awq", "bnb"}

# How far below a native model each reported quantization is assumed to sit, as a
# fraction subtracted from the faithfulness axis (0 = indistinguishable). Anything
# not listed here -- bf16, fp16, "unknown", unreported -- carries no penalty, so a
# host is only docked for degradation it actually declares.
_QUANT_SEVERITY = {
    "fp8": 0.30,
    "int8": 0.30,
    "fp6": 0.45,
    "bnb": 0.45,
    "gptq": 0.40,
    "awq": 0.40,
    "fp4": 0.60,
    "int4": 0.60,
    "int3": 0.75,
    "int2": 0.85,
}

# Eve's observed monthly token mix (see reference-cheap-model-digs): cache reads
# are ~95% of volume, so an endpoint's cache-read price is the price that governs
# her bill -- ~27x more decisive than fresh input and ~56x more than output.
# Blending the three by this real mix collapses them into one honest $/token.
_MIX_CACHE_READ = 0.948
_MIX_FRESH_INPUT = 0.035
_MIX_OUTPUT = 0.017

# Relative pull of the three ranking axes. Cost leads because at ~11B tokens/mo it
# dwarfs a speed tier; faithfulness (quantization) is a real but bounded penalty,
# so a markedly cheaper, faster quantized host can still win on balance.
_COST_WEIGHT = 0.55
_SPEED_WEIGHT = 0.20
_QUANT_WEIGHT = 0.25

# Neutral speed score when an endpoint reports no throughput -- it neither gains
# nor loses on an axis we can't measure for it.
_UNKNOWN_SPEED = 0.5

# Guards ratio math against free ($0) endpoints without distorting real prices.
_EPS = 1e-12

# How many top-ranked endpoints we pin as the provider order (and, correspondingly,
# the set the conservative context window is taken across).
_ORDER_LIMIT = 6

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
    cache_read_price: float | None = None

    @property
    def unquantized(self) -> bool:
        return self.quantization not in _QUANTIZED

    @property
    def effective_input_price(self) -> float:
        """Input $/token weighted for a cache-heavy workload.

        Cache reads dominate real coding-agent traffic, so an endpoint's
        cache-read price -- when it publishes one -- is the cost that matters;
        otherwise fall back to the (higher) fresh-prompt price, which correctly
        deprioritizes endpoints that don't discount cached reads.
        """
        if self.cache_read_price is not None:
            return self.cache_read_price
        return self.prompt_price


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
        cache_read = pricing.get("input_cache_read")
        out.append(
            Endpoint(
                provider_name=str(e.get("provider_name") or ""),
                context_length=context,
                quantization=str(e.get("quantization") or "").lower(),
                prompt_price=_as_float(pricing.get("prompt")),
                completion_price=_as_float(pricing.get("completion")),
                supports_tools="tools" in (e.get("supported_parameters") or []),
                throughput=throughput,
                cache_read_price=_as_float(cache_read) if cache_read is not None else None,
            )
        )
    return out


def _as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _blended_price(e: Endpoint) -> float:
    """Collapse an endpoint's three prices into one $/token for Eve's mix.

    Cache-heavy traffic means the cache-read price (or the fresh-prompt price when
    the host publishes no cache discount) carries almost all the weight; fresh
    input and output are minor corrections.
    """
    return (
        _MIX_CACHE_READ * e.effective_input_price
        + _MIX_FRESH_INPUT * e.prompt_price
        + _MIX_OUTPUT * e.completion_price
    )


def _score(e: Endpoint, *, min_price: float, max_tput: float, weigh_quant: bool) -> float:
    """Composite desirability in [0, 1]: cost, speed, and faithfulness combined.

    Each axis is normalized to the candidate pool so the three weights compose on a
    common scale. Cost is ratio-to-cheapest (robust to a single ultra-cheap
    outlier), speed is ratio-to-fastest (unknown throughput scores neutral), and
    faithfulness is 1 minus the declared quantization penalty.
    """
    cost = (min_price + _EPS) / (_blended_price(e) + _EPS)
    speed = (e.throughput / max_tput) if (e.throughput and max_tput) else _UNKNOWN_SPEED
    penalty = _QUANT_SEVERITY.get(e.quantization, 0.0) if weigh_quant else 0.0
    faithfulness = 1.0 - penalty
    return _COST_WEIGHT * cost + _SPEED_WEIGHT * speed + _QUANT_WEIGHT * faithfulness


def _analyze(endpoints: list[Endpoint], *, prefer_unquantized: bool = True) -> ModelMeta:
    """Derive the conservative window + a composite-ranked provider order.

    ``prefer_unquantized`` now weighs quantization into the score rather than
    hard-filtering it; passing ``False`` drops the faithfulness penalty entirely
    (rank on cost and speed alone). The context window is the smallest among the
    pinned top endpoints -- the set a request could actually route to -- so
    compaction never overfills the serving host.
    """
    usable = [e for e in endpoints if e.context_length > 0 and e.supports_tools]
    pool = usable or [e for e in endpoints if e.context_length > 0]
    if not pool:
        return ModelMeta(None, (), tuple(endpoints))
    min_price = min(_blended_price(e) for e in pool)
    known_tput = [e.throughput for e in pool if e.throughput]
    max_tput = max(known_tput) if known_tput else 0.0
    ranked = sorted(
        pool,
        key=lambda e: _score(
            e, min_price=min_price, max_tput=max_tput, weigh_quant=prefer_unquantized
        ),
        reverse=True,
    )
    top = ranked[:_ORDER_LIMIT]
    effective = min(e.context_length for e in top)
    order = tuple(dict.fromkeys(e.provider_name for e in top if e.provider_name))
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
