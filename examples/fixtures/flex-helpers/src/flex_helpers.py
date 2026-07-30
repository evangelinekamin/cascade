from typing import Mapping, Optional


def is_transient_flex_error(message: str) -> bool:
    """Return whether an IBKR Flex error should be retried."""
    raise NotImplementedError


def to_float(value: Optional[str]) -> float:
    """Parse an IBKR numeric attribute, returning 0.0 for missing/bad values."""
    raise NotImplementedError


def first_attr(attrs: Mapping[str, str], *names: str) -> str:
    """Return the first present, non-empty attribute from *names*."""
    raise NotImplementedError
