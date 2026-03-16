"""Retry logic with exponential backoff for transient API errors."""

import time
import random
from functools import wraps

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


class RetryableError(Exception):
    """Raised when an API call fails with a retryable status code."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


def retry_on_transient_error(max_retries=MAX_RETRIES, base_delay=BASE_DELAY):
    """Decorator that retries on transient HTTP errors with jittered backoff."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except RetryableError as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(delay)
                    continue
            raise last_error
        return wrapper
    return decorator
