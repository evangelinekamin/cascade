"""Tests for English<->Chinese product translation (services.product_translation)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from services import product_translation as pt


@pytest.fixture(autouse=True)
def _clear_cache():
    pt._cache.clear()
    yield
    pt._cache.clear()


def _settings(key="test-key"):
    return SimpleNamespace(
        openrouter_api_key=key,
        extraction_model="test-model",
        creative_model="test-creative-model",
    )


async def test_translate_to_chinese_returns_translation():
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock(return_value={"text": "工装裤"})
    ) as mock_call:
        result = await pt.translate_to_chinese("cargo pants")
    assert result == "工装裤"
    mock_call.assert_awaited_once()


async def test_translate_to_chinese_caches_second_call():
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock(return_value={"text": "连衣裙"})
    ) as mock_call:
        first = await pt.translate_to_chinese("dress")
        second = await pt.translate_to_chinese("dress")
    assert first == second == "连衣裙"
    mock_call.assert_awaited_once()  # second served from cache


async def test_translate_to_chinese_skips_when_already_chinese():
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock()
    ) as mock_call:
        result = await pt.translate_to_chinese("连衣裙")
    assert result == "连衣裙"
    mock_call.assert_not_awaited()


async def test_translate_to_chinese_without_key_returns_input():
    with patch.object(pt, "get_settings", return_value=_settings(key="")), patch.object(
        pt, "call_completion", new=AsyncMock()
    ) as mock_call:
        result = await pt.translate_to_chinese("skirt")
    assert result == "skirt"
    mock_call.assert_not_awaited()


async def test_translate_to_chinese_falls_back_on_error():
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        result = await pt.translate_to_chinese("blouse")
    assert result == "blouse"


async def test_translate_to_chinese_keeps_only_first_phrase():
    # Model over-produces space-separated alternatives; Taobao AND-matches them
    # and returns nothing — we keep just the primary phrase.
    response = {"text": "田园风碎花连衣裙 森系 碎花长裙 复古"}
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock(return_value=response)
    ):
        result = await pt.translate_to_chinese("cottagecore prairie dress")
    assert result == "田园风碎花连衣裙"


async def test_translate_titles_batch_maps_in_order():
    titles = ["工装裤", "连衣裙"]
    response = {"text": json.dumps(["Cargo Pants", "Dress"], ensure_ascii=False)}
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock(return_value=response)
    ) as mock_call:
        result = await pt.translate_titles_to_english(titles)
    assert result == ["Cargo Pants", "Dress"]
    mock_call.assert_awaited_once()  # one batched call for the whole list


async def test_translate_titles_empty_list():
    result = await pt.translate_titles_to_english([])
    assert result == []


async def test_translate_titles_size_mismatch_falls_back_to_originals():
    titles = ["工装裤", "连衣裙"]
    response = {"text": json.dumps(["only one"])}
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock(return_value=response)
    ):
        result = await pt.translate_titles_to_english(titles)
    assert result == titles


async def test_translate_titles_tolerates_code_fence():
    titles = ["工装裤"]
    response = {"text": '```json\n["Cargo Pants"]\n```'}
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock(return_value=response)
    ):
        result = await pt.translate_titles_to_english(titles)
    assert result == ["Cargo Pants"]


async def test_translate_titles_uses_cache_on_repeat():
    titles = ["工装裤"]
    response = {"text": json.dumps(["Cargo Pants"], ensure_ascii=False)}
    with patch.object(pt, "get_settings", return_value=_settings()), patch.object(
        pt, "call_completion", new=AsyncMock(return_value=response)
    ) as mock_call:
        await pt.translate_titles_to_english(titles)
        await pt.translate_titles_to_english(titles)
    mock_call.assert_awaited_once()
