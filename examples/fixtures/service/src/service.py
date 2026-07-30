"""Small service fixture used by Cascade's end-to-end evaluation example."""


def dispatch(path: str) -> tuple[int, dict]:
    return 404, {"error": "not found"}
