"""Live OpenRouter endpoint metadata: conservative window + quality-ranked order."""

from unittest.mock import MagicMock, patch

from cascade.providers import openrouter_meta as orm
from cascade.providers.openrouter_meta import Endpoint, ModelMeta, _analyze, model_meta
from cascade.providers.base import ProviderConfig
from cascade.providers.openrouter import OpenRouterProvider


def _ep(name, ctx, quant, price=0.0, tput=None, tools=True, cache_read=None, out=0.0):
    return Endpoint(name, ctx, quant, price, out, tools, tput, cache_read)


def test_analyze_ranks_unquantized_cheapest_first_and_keeps_quantized():
    # Unquantized StreamLake is cheapest per cache-weighted token, so it leads
    # despite being slowest. fp4/fp8 hosts are penalized, not excluded, so they
    # remain in the order -- and the window is the conservative min across the
    # pinned set (Novita's 64k), since a request could route there on fallback.
    eps = [
        _ep("StreamLake", 128000, "unknown", price=0.0002, tput=50),
        _ep("DeepInfra", 163840, "fp4", price=0.00032, tput=100),
        _ep("Novita", 64000, "fp8", price=0.0004, tput=200),
    ]
    meta = _analyze(eps)
    assert meta.recommended_order[0] == "StreamLake"
    assert set(meta.recommended_order) == {"StreamLake", "DeepInfra", "Novita"}
    assert meta.effective_context == 64000


def test_analyze_ranks_all_unquantized_by_composite_cost_and_speed():
    # All faithful, so the score reduces to cost (cache-weighted) and speed: C is
    # both cheapest and fastest, B next, A (priciest, slowest) last.
    eps = [
        _ep("A", 128000, "unknown", price=0.0003, tput=50),
        _ep("B", 200000, "bf16", price=0.0002, tput=100),
        _ep("C", 150000, "fp16", price=0.0001, tput=100),
    ]
    meta = _analyze(eps)
    assert meta.effective_context == 128000  # conservative min across pinned set
    assert meta.recommended_order == ("C", "B", "A")


def test_cache_read_price_dominates_ranking():
    # Groq lists a higher sticker (prompt) price but a deep cache-read discount;
    # the other host is cheaper on fresh prompts but never discounts cache reads.
    # For a 95%-cache workload Groq is far cheaper overall, so it must rank first.
    eps = [
        _ep("Groq", 131072, "unknown", price=0.0000005, cache_read=0.00000005, tput=500),
        _ep("Other", 131072, "unknown", price=0.0000002, cache_read=None, tput=500),
    ]
    meta = _analyze(eps)
    assert meta.recommended_order[0] == "Groq"


def test_quantization_is_a_soft_penalty_that_a_cheaper_host_can_overcome():
    # A much cheaper fp4 host outranks a pricier unquantized one -- the penalty is
    # weighed, not absolute -- but both are retained.
    eps = [
        _ep("Unq", 128000, "bf16", price=0.0003, tput=100),
        _ep("Quant", 128000, "fp4", price=0.0002, tput=100),
    ]
    meta = _analyze(eps)
    assert meta.recommended_order == ("Quant", "Unq")


def test_prefer_unquantized_flag_toggles_the_faithfulness_penalty():
    # Identical cost and speed: with the penalty the unquantized host wins; with it
    # disabled the two tie and input order stands -- proving the flag gates quant.
    eps = [
        _ep("Quant", 128000, "fp8", price=0.0002, tput=100),
        _ep("Unq", 128000, "bf16", price=0.0002, tput=100),
    ]
    assert _analyze(eps, prefer_unquantized=True).recommended_order == ("Unq", "Quant")
    assert _analyze(eps, prefer_unquantized=False).recommended_order == ("Quant", "Unq")


def test_parse_endpoints_reads_cache_read_price():
    eps = orm._parse_endpoints(
        {
            "data": {
                "endpoints": [
                    {
                        "provider_name": "Groq",
                        "context_length": 131072,
                        "quantization": "unknown",
                        "pricing": {
                            "prompt": "0.0000005",
                            "completion": "0.0000008",
                            "input_cache_read": "0.00000005",
                        },
                        "supported_parameters": ["tools"],
                    }
                ]
            }
        }
    )
    assert eps[0].cache_read_price == 5e-8
    assert eps[0].effective_input_price == 5e-8  # cache price wins when present


def test_analyze_falls_back_to_quantized_when_no_unquantized_exists():
    meta = _analyze([_ep("X", 64000, "fp8", tput=200)])
    assert meta.effective_context == 64000
    assert meta.recommended_order == ("X",)


def test_model_meta_is_best_effort_on_failure_and_caches_success():
    orm._cache.clear()
    with patch.object(orm.httpx, "get", side_effect=Exception("no net")):
        assert model_meta("deepseek/x") is None
    assert "deepseek/x" not in orm._cache  # transient failure is retryable

    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "data": {
            "endpoints": [
                {
                    "provider_name": "StreamLake",
                    "context_length": 128000,
                    "quantization": "unknown",
                    "pricing": {"prompt": "0.0000002", "completion": "0.0000008"},
                    "supported_parameters": ["tools", "tool_choice"],
                    "throughput_last_30m": 50,
                }
            ]
        }
    }
    with patch.object(orm.httpx, "get", return_value=resp):
        meta = model_meta("deepseek/y")
    assert meta.effective_context == 128000
    assert "deepseek/y" in orm._cache


def test_apply_meta_sets_window_and_order_when_unset():
    prov = OpenRouterProvider(ProviderConfig(api_key="k", model="deepseek/x"))
    fake = ModelMeta(128000, ("StreamLake",), ())
    with patch("cascade.providers.openrouter_meta.model_meta", return_value=fake):
        prov._apply_meta()
    assert prov.config.context_window == 128000
    prefs = prov.config.provider_preferences
    assert prefs["order"] == ["StreamLake"]
    assert prefs["require_parameters"] is True


def test_apply_meta_respects_explicit_config():
    prov = OpenRouterProvider(
        ProviderConfig(
            api_key="k",
            model="deepseek/x",
            context_window=999,
            provider_preferences={"order": ["Baidu"]},
        )
    )
    fake = ModelMeta(128000, ("StreamLake",), ())
    with patch("cascade.providers.openrouter_meta.model_meta", return_value=fake):
        prov._apply_meta()
    assert prov.config.context_window == 999  # explicit window not overridden
    assert prov.config.provider_preferences["order"] == ["Baidu"]  # explicit order kept


def test_apply_meta_runs_once():
    prov = OpenRouterProvider(ProviderConfig(api_key="k", model="deepseek/x"))
    with patch("cascade.providers.openrouter_meta.model_meta", return_value=None) as mm:
        prov._apply_meta()
        prov._apply_meta()
    assert mm.call_count == 1
