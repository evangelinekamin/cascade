"""Tests for the normalized Usage type and its provider constructors."""

from cascade.providers.usage import Usage


class TestTotals:
    def test_prompt_total_sums_input_and_cache_fields(self):
        u = Usage(input=100, output=20, cache_read=500, cache_write=30)
        assert u.prompt_total == 630
        assert u.total == 650

    def test_defaults_are_zero_and_costless(self):
        u = Usage()
        assert u.total == 0
        assert u.cost is None

    def test_add_accumulates_fields_and_cost(self):
        a = Usage(input=10, output=5, cache_read=100, cost=0.001)
        b = Usage(input=20, output=8, cache_write=50, cost=0.002)
        c = a.add(b)
        assert c == Usage(
            input=30, output=13, cache_read=100, cache_write=50, cost=0.003
        )

    def test_add_keeps_cost_none_when_neither_side_reports(self):
        assert Usage(input=1).add(Usage(output=2)).cost is None

    def test_add_adopts_the_only_reported_cost(self):
        assert Usage(cost=0.5).add(Usage()).cost == 0.5
        assert Usage().add(Usage(cost=0.5)).cost == 0.5


class TestFromAnthropic:
    def test_reads_cache_fields_without_subtraction(self):
        u = Usage.from_anthropic(
            {
                "input_tokens": 12,
                "output_tokens": 40,
                "cache_read_input_tokens": 18000,
                "cache_creation_input_tokens": 700,
            }
        )
        assert u == Usage(input=12, output=40, cache_read=18000, cache_write=700)
        assert u.prompt_total == 18712

    def test_missing_and_null_fields_default_to_zero(self):
        assert Usage.from_anthropic({"input_tokens": None}) == Usage()
        assert Usage.from_anthropic({}) == Usage()


class TestFromOpenAI:
    def test_subtracts_cached_subset_from_prompt(self):
        u = Usage.from_openai(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 800},
            }
        )
        assert u == Usage(input=200, output=50, cache_read=800)
        assert u.prompt_total == 1000

    def test_accepts_input_output_token_key_spelling(self):
        u = Usage.from_openai({"input_tokens": 7, "output_tokens": 3})
        assert u == Usage(input=7, output=3)

    def test_reads_openrouter_cost(self):
        u = Usage.from_openai(
            {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0004}
        )
        assert u.cost == 0.0004

    def test_ignores_non_numeric_cost_and_bad_details(self):
        u = Usage.from_openai(
            {"prompt_tokens": 10, "completion_tokens": 2, "cost": "n/a",
             "prompt_tokens_details": None}
        )
        assert u == Usage(input=10, output=2)


class TestFromGemini:
    def test_subtracts_cached_content_and_bills_thoughts_as_output(self):
        u = Usage.from_gemini(
            {
                "promptTokenCount": 900,
                "candidatesTokenCount": 30,
                "cachedContentTokenCount": 600,
                "thoughtsTokenCount": 120,
            }
        )
        assert u == Usage(input=300, output=150, cache_read=600)

    def test_minimal_metadata(self):
        u = Usage.from_gemini({"promptTokenCount": 10, "candidatesTokenCount": 4})
        assert u == Usage(input=10, output=4)
