"""Live OpenRouter endpoint metadata: conservative window + quality-ranked order."""

from unittest.mock import MagicMock, patch

from cascade.providers import openrouter_meta as orm
from cascade.providers.openrouter_meta import Endpoint, ModelMeta, _analyze, model_meta
from cascade.providers.base import ProviderConfig
from cascade.providers.openrouter import OpenRouterProvider


def _ep(name, ctx, quant, price=0.0, tput=None, tools=True):
    return Endpoint(name, ctx, quant, price, 0.0, tools, tput)


def test_analyze_prefers_unquantized_and_uses_conservative_window():
    # StreamLake (unknown quant) is kept; fp4/fp8 are dropped, so the window is
    # StreamLake's 128k, not DeepInfra's larger-but-quantized 163k.
    eps = [
        _ep("StreamLake", 128000, "unknown", price=0.0002, tput=50),
        _ep("DeepInfra", 163840, "fp4", price=0.00032, tput=100),
        _ep("Novita", 64000, "fp8", price=0.0004, tput=200),
    ]
    meta = _analyze(eps)
    assert meta.effective_context == 128000
    assert meta.recommended_order == ("StreamLake",)


def test_analyze_ranks_unquantized_by_throughput_then_price():
    eps = [
        _ep("A", 128000, "unknown", price=0.0003, tput=50),
        _ep("B", 200000, "bf16", price=0.0002, tput=100),
        _ep("C", 150000, "fp16", price=0.0001, tput=100),
    ]
    meta = _analyze(eps)
    assert meta.effective_context == 128000  # conservative min across eligible
    assert meta.recommended_order == ("C", "B", "A")  # tput desc, then price asc


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
