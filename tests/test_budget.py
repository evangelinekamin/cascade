"""Golden tests for the token-accounting authority (context/budget.py)."""

import pytest

from cascade.context.budget import (
    ENV_WINDOW_CAP,
    compact_threshold,
    current_tokens,
    effective_window,
    estimate_tokens_from_chars,
    output_reserve,
    warn_threshold,
    window_for,
)
from cascade.providers.usage import Usage


class TestWindowFor:
    def test_provider_fallbacks(self):
        assert window_for("claude") == 200_000
        assert window_for("gemini") == 1_000_000
        assert window_for("openai") == 400_000
        assert window_for("openrouter") == 128_000

    def test_unknown_provider_uses_default(self):
        assert window_for("local") == 128_000
        assert window_for("") == 128_000

    def test_million_suffix_detection(self):
        assert window_for("claude", "claude-opus-4-8[1m]") == 1_000_000
        assert window_for("claude", "claude-opus-4-8[1M]") == 1_000_000
        assert window_for("claude", "claude-opus-4-8") == 200_000

    def test_explicit_configuration_wins(self):
        assert window_for("openai", "gpt-5.2", configured=32_768) == 32_768
        assert window_for("claude", "x[1m]", configured=200_000) == 200_000

    def test_env_cap_bounds_the_result(self, monkeypatch):
        monkeypatch.setenv(ENV_WINDOW_CAP, "50000")
        assert window_for("gemini") == 50_000
        assert window_for("local", configured=32_768) == 32_768

    def test_env_cap_ignores_garbage(self, monkeypatch):
        monkeypatch.setenv(ENV_WINDOW_CAP, "not-a-number")
        assert window_for("claude") == 200_000

    def test_provider_name_is_case_insensitive(self):
        assert window_for("Claude") == 200_000


class TestThresholds:
    def test_large_window_math(self):
        # 200k: reserve 16k -> effective 184k; buffer 13k -> threshold 171k;
        # warn band 20k -> 151k.
        assert output_reserve(200_000) == 16_000
        assert effective_window(200_000) == 184_000
        assert compact_threshold(200_000) == 171_000
        assert warn_threshold(200_000) == 151_000

    def test_small_window_scales_reserve_and_buffers(self):
        # 32k: reserve min(16k, 8k) = 8k -> effective 24k; buffer
        # min(13k, 4k) = 4k -> threshold 20k; warn min(20k, 4k) -> 16k.
        assert output_reserve(32_768) == 8_192
        assert effective_window(32_768) == 24_576
        assert compact_threshold(32_768) == 20_480
        assert warn_threshold(32_768) == 16_384

    def test_configured_max_output_lowers_reserve_only(self):
        assert output_reserve(200_000, max_output=4_096) == 4_096
        assert output_reserve(200_000, max_output=64_000) == 16_000

    def test_threshold_is_strictly_inside_the_window(self):
        for window in (32_768, 128_000, 200_000, 1_000_000):
            assert 0 < warn_threshold(window) < compact_threshold(window) < window


class TestCurrentTokens:
    def test_anchor_total_plus_tail_estimate(self):
        anchor = Usage(input=1_000, output=200, cache_read=50_000)
        assert current_tokens(anchor, trailing_chars=400) == 51_200 + 100

    def test_no_anchor_estimates_everything(self):
        assert current_tokens(None, trailing_chars=4_000) == 1_000

    def test_never_sums_usage_across_turns(self):
        # The anchor already contains all prior context: two consecutive
        # calls with the same anchor must not double-count.
        anchor = Usage(input=10_000, output=500)
        assert current_tokens(anchor) == current_tokens(anchor) == 10_500

    def test_negative_chars_clamped(self):
        assert estimate_tokens_from_chars(-100) == 0
