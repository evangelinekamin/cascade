import pytest

from src.retry import retry_delays


def test_retries_rate_limits_timeouts_and_server_errors_exponentially():
    assert retry_delays([429, 408, 503, 200]) == [0.5, 1.0, 2.0]


def test_stops_on_first_permanent_error_or_success():
    assert retry_delays([500, 400, 500]) == [0.5]
    assert retry_delays([200, 500]) == []


def test_caps_backoff_and_accepts_generators():
    statuses = (status for status in [500, 500, 500, 500, 500, 200])
    assert retry_delays(statuses, base_seconds=1.0, cap_seconds=4.0) == [
        1.0,
        2.0,
        4.0,
        4.0,
        4.0,
    ]


@pytest.mark.parametrize(
    ("base", "cap"),
    [(0.0, 8.0), (-1.0, 8.0), (2.0, 1.0)],
)
def test_rejects_invalid_backoff_configuration(base, cap):
    with pytest.raises(ValueError):
        retry_delays([500], base_seconds=base, cap_seconds=cap)
