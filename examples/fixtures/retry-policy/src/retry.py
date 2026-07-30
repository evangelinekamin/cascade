from typing import Iterable


def retry_delays(
    statuses: Iterable[int],
    *,
    base_seconds: float = 0.5,
    cap_seconds: float = 8.0,
) -> list[float]:
    """Return delays for retryable failures until success/permanent failure."""
    raise NotImplementedError
