from typing import Iterable


def rollup_bom(components: Iterable[dict]) -> list[dict]:
    """Aggregate populated components into deterministic JLCPCB BOM rows."""
    raise NotImplementedError
