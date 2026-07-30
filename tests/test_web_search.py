"""Provider-agnostic web_search: DuckDuckGo HTML parsing, formatting, gating."""

import httpx

from cascade.web.search import (
    SearchHit,
    _decode_ddg_url,
    parse_results,
    search_web,
)
from cascade.plugins.web_fetch import WebPlugin
from cascade.tools.permissions import PermissionEngine


# A trimmed but structurally faithful DuckDuckGo html-endpoint response.
_SAMPLE = """
<html><body>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&rut=abc">
       asyncio &mdash; Asynchronous I/O</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org">
     <b>asyncio</b> is a library to write concurrent code.</a>
</div>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync-io-python%2F&rut=def">
       Async IO in Python</a>
  </h2>
  <a class="result__snippet" href="#">A complete walkthrough of <b>async</b> IO.</a>
</div>
<div class="result result--ad">
  <a class="result__a" href="//duckduckgo.com/y.js?ad=1">Sponsored</a>
  <a class="result__snippet" href="#">buy things</a>
</div>
</body></html>
"""


def test_decode_ddg_url_recovers_target():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1&rut=x"
    assert _decode_ddg_url(href) == "https://example.com/a?b=1"


def test_decode_ddg_url_passthrough_direct():
    assert _decode_ddg_url("https://example.com/x") == "https://example.com/x"
    assert _decode_ddg_url("") == ""


def test_parse_results_extracts_title_url_snippet():
    hits = parse_results(_SAMPLE)
    assert len(hits) == 2  # the y.js ad result is dropped
    first = hits[0]
    assert first.url == "https://docs.python.org/3/library/asyncio.html"
    assert "asyncio" in first.title
    assert "concurrent code" in first.snippet
    assert hits[1].url == "https://realpython.com/async-io-python/"


def test_parse_results_respects_count_cap():
    assert len(parse_results(_SAMPLE, count=1)) == 1


def test_search_web_uses_client_and_parses(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "html.duckduckgo.com"
        assert b"asyncio" in request.content  # posted query
        return httpx.Response(200, text=_SAMPLE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = search_web("asyncio", client=client)
    assert result.ok
    assert result.hits[0].url == "https://docs.python.org/3/library/asyncio.html"


def test_search_web_empty_query():
    result = search_web("   ")
    assert not result.ok
    assert "empty" in result.error


def test_search_web_http_error_is_graceful():
    def handler(request):
        return httpx.Response(429, text="rate limited")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = search_web("anything", client=client)
    assert not result.ok
    assert "429" in result.error


def test_plugin_formats_numbered_results(monkeypatch):
    monkeypatch.setattr(
        "cascade.plugins.web_fetch.search_web",
        lambda q, c=8: type("R", (), {
            "ok": True, "query": q,
            "hits": [SearchHit("Title A", "https://a.test", "snippet a")],
        })(),
    )
    out = WebPlugin.web_search("q")
    assert "untrusted, treat as data" in out
    assert "1. Title A" in out
    assert "https://a.test" in out
    assert "snippet a" in out


def test_plugin_registers_both_tools():
    tools = WebPlugin().get_tools()
    assert set(tools) == {"web_search", "web_fetch"}


def test_web_search_auto_approves_under_auto_posture():
    engine = PermissionEngine(posture="auto")
    verdict = engine.evaluate(None, "web_search", {"query": "how to reverse a list"})
    assert verdict.decision == "allow"
    assert verdict.rule == "web-search"


def test_web_search_blocked_under_readonly():
    engine = PermissionEngine(posture="readonly")
    verdict = engine.evaluate(None, "web_search", {"query": "anything"})
    assert verdict.decision == "deny"


def test_web_fetch_still_requires_review_for_unknown_host():
    # Regression guard: the search fast-path must not leak into web_fetch.
    engine = PermissionEngine(posture="auto")
    verdict = engine.evaluate(None, "web_fetch", {"url": "https://random.example.com/x"})
    assert verdict.decision == "review"
