from src.service import dispatch


def test_unknown_endpoint():
    assert dispatch("/missing")[0] == 404


def test_health_endpoint():
    assert dispatch("/health") == (200, {"status": "ok"})


def test_version_endpoint():
    assert dispatch("/version") == (200, {"version": "1.0.0"})
