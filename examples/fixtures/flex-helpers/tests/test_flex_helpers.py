import pytest

from src.flex_helpers import first_attr, is_transient_flex_error, to_float


@pytest.mark.parametrize(
    "message",
    [
        "Statement generation is in progress",
        "Too many requests; please try again later",
        "Report is not ready, try again",
        "temporarily unavailable",
    ],
)
def test_transient_errors(message):
    assert is_transient_flex_error(message)


@pytest.mark.parametrize(
    "message",
    ["Invalid token", "Query does not exist", "Account is not authorized"],
)
def test_permanent_errors(message):
    assert not is_transient_flex_error(message)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 0.0), ("", 0.0), ("bad", 0.0), (" 12.50 ", 12.5), ("-7", -7.0)],
)
def test_to_float(value, expected):
    assert to_float(value) == expected


def test_first_attr_skips_missing_and_empty_values():
    attrs = {"symbol": "", "ticker": "NVDA", "description": "NVIDIA"}
    assert first_attr(attrs, "symbol", "ticker", "description") == "NVDA"
    assert first_attr(attrs, "missing", "symbol") == ""
