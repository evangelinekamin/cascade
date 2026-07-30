"""SSRF module + fetch pipeline + fetch permission gating."""

import httpx
import pytest

from cascade.tools.permissions import PermissionEngine, ReviewDecision
from cascade.tools.schema import callable_to_tool_def
from cascade.web import url_safety
from cascade.web.fetch import fetch_url


def _fetch_tool():
    def web_fetch(url: str) -> str:
        """Fetch."""
        return "ok"

    return callable_to_tool_def("web_fetch", web_fetch, "fetch")


class TestSSRF:
    def test_blocks_metadata_endpoint(self):
        assert not url_safety.is_safe_url("http://169.254.169.254/latest/meta-data/")
        assert "metadata" in url_safety.url_safety_error(
            "http://169.254.169.254/"
        )

    def test_blocks_metadata_hostname(self):
        assert not url_safety.is_safe_url("http://metadata.google.internal/")

    def test_blocks_loopback_and_private(self):
        assert not url_safety.is_safe_url("http://127.0.0.1:8080/")
        assert not url_safety.is_safe_url("http://10.0.0.5/")
        assert not url_safety.is_safe_url("http://192.168.1.1/")

    def test_rejects_non_http_scheme(self):
        assert not url_safety.is_safe_url("file:///etc/passwd")
        assert not url_safety.is_safe_url("ftp://example.com/")

    def test_rejects_userinfo(self):
        assert "userinfo" in url_safety.url_safety_error("https://user:pw@example.com/")

    def test_rejects_overlong_url(self):
        assert url_safety.url_safety_error("https://x.com/" + "a" * 3000)

    def test_normalize_upgrades_http_to_https(self):
        assert url_safety.normalize_url("http://example.com/x").startswith("https://")

    def test_secret_detection(self):
        assert url_safety.contains_secret("https://x.com/?token=sk-abc123")
        assert url_safety.contains_secret("https://x.com/?api_key=hunter2")
        assert url_safety.contains_secret("https://x.com/page") is None

    def test_same_origin(self):
        assert url_safety.same_origin("https://a.com/x", "https://a.com/y")
        assert not url_safety.same_origin("https://a.com/", "https://b.com/")
        assert not url_safety.same_origin("https://a.com/", "http://a.com/")


class TestFetchPipeline:
    @pytest.fixture(autouse=True)
    def _public_dns(self, monkeypatch):
        monkeypatch.setattr(
            url_safety.socket,
            "getaddrinfo",
            lambda *_args, **_kwargs: [
                (
                    url_safety.socket.AF_INET,
                    url_safety.socket.SOCK_STREAM,
                    0,
                    "",
                    ("93.184.216.34", 0),
                )
            ],
        )

    def test_blocks_ssrf_before_network(self):
        result = fetch_url("http://169.254.169.254/")
        assert not result.ok
        assert "blocked" in result.error

    def test_blocks_secret_url(self):
        result = fetch_url("https://example.com/?access_token=sk-live-xyz")
        assert not result.ok
        assert "secret" in result.error

    def test_html_extracted_to_text(self):
        html = b"<html><head><title>T</title><style>x{}</style></head>" \
               b"<body><script>evil()</script><p>Hello world</p>" \
               b"<p>Second paragraph</p></body></html>"

        def handler(request):
            return httpx.Response(200, headers={"content-type": "text/html"}, content=html)

        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        result = fetch_url("https://example.com/page", client=client)
        assert result.ok
        assert "Hello world" in result.content
        assert "evil()" not in result.content

    def test_cross_origin_redirect_blocked(self):
        def handler(request):
            return httpx.Response(302, headers={"location": "https://evil.com/"})

        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        result = fetch_url("https://example.com/", client=client)
        assert not result.ok
        assert "redirect" in result.error

    def test_size_cap_enforced(self):
        big = b"<p>" + b"x" * (11 * 1024 * 1024) + b"</p>"

        def handler(request):
            return httpx.Response(200, headers={"content-type": "text/html"}, content=big)

        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        result = fetch_url("https://example.com/big", client=client)
        assert not result.ok
        assert "10MB" in result.error

    def test_non_html_content_type_rejected(self):
        def handler(request):
            return httpx.Response(200, headers={"content-type": "image/png"}, content=b"\x89PNG")

        client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
        result = fetch_url("https://example.com/img.png", client=client)
        assert not result.ok
        assert "content-type" in result.error


class TestFetchPermissions:
    def _engine(self, **kw):
        return PermissionEngine(**kw)

    def test_preapproved_docs_domain_auto_allowed(self):
        eng = self._engine()
        v = eng.evaluate(_fetch_tool(), "web_fetch", {"url": "https://docs.python.org/3/x"})
        assert v.decision == "allow"
        assert v.rule == "docs-allowlist"

    def test_unknown_host_requires_review(self):
        eng = self._engine()
        v = eng.evaluate(_fetch_tool(), "web_fetch", {"url": "https://random.example/x"})
        assert v.decision == "review"
        assert v.rule == "network"

    def test_unknown_host_is_reviewed_each_time(self):
        eng = self._engine()
        reviews = []
        eng.review_handler = lambda review: (
            reviews.append(review),
            ReviewDecision(True, "public documentation"),
        )[1]
        v1 = eng.resolve(_fetch_tool(), "web_fetch", {"url": "https://blog.example/post-1"})
        assert v1.decision == "allow"
        v2 = eng.resolve(_fetch_tool(), "web_fetch", {"url": "https://blog.example/post-2"})
        assert v2.decision == "allow"
        assert len(reviews) == 2

    def test_domain_allow_rule(self):
        eng = self._engine(allow=("web_fetch(internal.corp)",))
        v = eng.evaluate(_fetch_tool(), "web_fetch", {"url": "https://internal.corp/wiki"})
        assert v.decision == "allow"

    def test_readonly_posture_blocks_egress(self):
        eng = self._engine(posture="readonly")
        v = eng.evaluate(_fetch_tool(), "web_fetch", {"url": "https://random.example/"})
        assert v.decision == "deny"
