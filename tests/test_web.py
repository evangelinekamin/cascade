"""Tests for the web file uploader through HTTPX's direct ASGI transport."""

import asyncio

import httpx
import pytest

from cascade.context.memory import ContextBuilder

try:
    from cascade.web.server import FileUploaderServer

    _HAS_STARLETTE = True
except ImportError:
    _HAS_STARLETTE = False


pytestmark = pytest.mark.skipif(not _HAS_STARLETTE, reason="starlette not installed")


@pytest.fixture
def client():
    cb = ContextBuilder()
    server = FileUploaderServer(cb)
    return server.app, cb


def _request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://cascade.test",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health(client):
    app, _ = client
    resp = _request(app, "GET", "/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_returns_html(client):
    app, _ = client
    resp = _request(app, "GET", "/")
    assert resp.status_code == 200
    assert "CASCADE" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_upload_text_file(client):
    app, cb = client
    resp = _request(
        app,
        "POST",
        "/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["filename"] == "test.txt"
    assert cb.source_count == 1


def test_upload_binary_file(client):
    app, cb = client
    resp = _request(
        app,
        "POST",
        "/upload",
        files={"file": ("img.png", b"\x89PNG\r\n\x1a\n" + b"\xff" * 100, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert cb.source_count == 1


def test_upload_no_file(client):
    app, _ = client
    resp = _request(app, "POST", "/upload")
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_context_endpoint(client):
    app, cb = client
    cb.add_text("data", label="test")
    resp = _request(app, "GET", "/context")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_count"] == 1
    assert data["token_estimate"] >= 0
    assert len(data["sources"]) == 1


def test_multiple_uploads(client):
    app, cb = client
    for i in range(3):
        _request(
            app,
            "POST",
            "/upload",
            files={"file": (f"file{i}.txt", f"content {i}".encode(), "text/plain")},
        )
    assert cb.source_count == 3


def test_upload_file_too_large():
    cb = ContextBuilder()
    server = FileUploaderServer(cb, max_upload_bytes=10)

    resp = _request(
        server.app,
        "POST",
        "/upload",
        files={"file": ("big.txt", b"0123456789ABCDEF", "text/plain")},
    )

    assert resp.status_code == 413
    data = resp.json()
    assert data["ok"] is False
    assert "exceeds upload limit" in data["error"]
    assert cb.source_count == 0
