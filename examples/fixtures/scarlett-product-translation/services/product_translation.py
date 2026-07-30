"""English<->Chinese translation for the "English Taobao" experience.

Taobao's catalog and keyword search are Chinese-first, so a usable English
experience needs translation in both directions:
  - the user's English query -> Chinese, so search returns good matches
  - Chinese product titles -> English, so results are readable

Translations route through the cheap extraction model and are cached in-memory
(queries and titles repeat heavily across searches). Every failure path falls
back to the original text, so translation can never break search.
"""

import json
import logging
import re
from collections import OrderedDict
from typing import Optional

from config import get_settings
from services.openrouter import call_completion

logger = logging.getLogger(__name__)

_CACHE_MAX = 4096
# Keyed by (source_text, target_lang) -> translated_text
_cache: "OrderedDict[tuple[str, str], str]" = OrderedDict()

_QUERY_SYSTEM = (
    "You convert an English fashion shopping query into ONE short Simplified "
    "Chinese search phrase, the way a shopper types it on Taobao: a single noun "
    "phrase with at most one style descriptor (e.g. 美式复古工装裤, 碎花连衣裙, "
    "厚底马丁靴). Use idiomatic Chinese fashion vocabulary, not a literal "
    "word-for-word translation. Do NOT output multiple alternatives or a synonym "
    "list, and do NOT separate terms with spaces or commas. Reply with ONLY the "
    "single phrase — no punctuation, no quotes, no explanation."
)
_TITLES_SYSTEM = (
    "You translate Chinese e-commerce product titles into concise, natural "
    "English. Keep brand and model names intact. Reply with ONLY a JSON array "
    "of translated strings, same order and same length as the input array."
)


def _cache_get(key: tuple[str, str]) -> Optional[str]:
    raise NotImplementedError


def _cache_put(key: tuple[str, str], value: str) -> None:
    raise NotImplementedError


def _has_cjk(text: str) -> bool:
    raise NotImplementedError


async def translate_to_chinese(text: str) -> str:
    """Translate an English search query to Chinese for Taobao search.

    Returns the input unchanged if it's empty, already contains Chinese, the
    API key is missing, or translation fails.
    """
    raise NotImplementedError


async def translate_titles_to_english(titles: list[str]) -> list[str]:
    """Translate a batch of Chinese product titles to English.

    Returns a list the same length as ``titles``. Cached titles are served from
    memory; only cache misses hit the model. Any title that can't be translated
    falls back to its original.
    """
    raise NotImplementedError


async def _translate_batch_en(texts: list[str]) -> list[Optional[str]]:
    """Translate Chinese strings to English; returns None per item on failure."""
    raise NotImplementedError


def _parse_json_array(text: str):
    """Parse a JSON array, tolerating ``` fences and surrounding prose."""
    raise NotImplementedError
