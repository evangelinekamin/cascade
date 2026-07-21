"""Local, provider-agnostic web search via DuckDuckGo's keyless HTML endpoint.

Complements ``web_fetch``: search returns ranked result links + snippets and
the model then fetches the ones it wants. Works for every provider -- none of
the cheap direct-API models (deepseek, mercury, local kimi) expose a native
server-side search tool -- and needs no API key. Best-effort: DuckDuckGo may
rate-limit or change markup, in which case an empty result set is returned
with a clear message rather than raising.

The request always goes to a single fixed host; the only thing leaving the
machine is the query text (the permission engine gates it as such). Result
URLs are returned as data, never auto-fetched.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

_ENDPOINT = "https://html.duckduckgo.com/html/"
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 30.0
_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_COUNT = 8
_MAX_COUNT = 10
# A realistic UA materially reduces DuckDuckGo blocking of the HTML endpoint.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


def _decode_ddg_url(href: str) -> str:
    """Recover the real target from a DuckDuckGo redirect href.

    Results link through ``//duckduckgo.com/l/?uddg=<encoded>&rut=...``; the
    real URL is the ``uddg`` query parameter. Direct hrefs pass through.
    """
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    try:
        parts = urlsplit(href)
        query = parse_qs(parts.query)
        if "uddg" in query and query["uddg"]:
            return unquote(query["uddg"][0])
    except Exception:
        pass
    return href


class _ResultParser(HTMLParser):
    """Extract (title, url, snippet) triples from DuckDuckGo HTML results."""

    def __init__(self) -> None:
        super().__init__()
        self.hits: list[SearchHit] = []
        self._in_title = False
        self._in_snippet = False
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._url = ""
        self._pending: Optional[tuple[str, str]] = None

    def _flush_pending(self, snippet: str = "") -> None:
        if self._pending is None:
            return
        title, url = self._pending
        self._pending = None
        # Skip DuckDuckGo ad/redirect noise and empty targets.
        if not url or "duckduckgo.com/y.js" in url:
            return
        self.hits.append(SearchHit(title=title, url=url, snippet=snippet))

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        cls = dict(attrs).get("class", "") or ""
        if "result__a" in cls:
            # A new result begins; flush any prior result that had no snippet.
            self._flush_pending()
            self._in_title = True
            self._title = []
            self._url = _decode_ddg_url(dict(attrs).get("href", ""))
        elif "result__snippet" in cls:
            self._in_snippet = True
            self._snippet = []

    def handle_endtag(self, tag):
        if tag != "a":
            return
        if self._in_title:
            self._in_title = False
            self._pending = ("".join(self._title).strip(), self._url)
        elif self._in_snippet:
            self._in_snippet = False
            self._flush_pending("".join(self._snippet).strip())

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        elif self._in_snippet:
            self._snippet.append(data)

    def close(self):
        super().close()
        self._flush_pending()


def parse_results(html: str, count: int = _DEFAULT_COUNT) -> list[SearchHit]:
    """Parse DuckDuckGo HTML into at most ``count`` search hits."""
    parser = _ResultParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    return parser.hits[:count]


class SearchResult:
    def __init__(self, ok: bool, hits: list[SearchHit], query: str, error: str = "") -> None:
        self.ok = ok
        self.hits = hits
        self.query = query
        self.error = error


def search_web(
    query: str, count: int = _DEFAULT_COUNT, client: Optional[httpx.Client] = None,
) -> SearchResult:
    """Search the web and return ranked hits, or an error result."""
    query = (query or "").strip()
    if not query:
        return SearchResult(False, [], query, "empty query")
    count = max(1, min(count, _MAX_COUNT))

    owns_client = client is None
    client = client or httpx.Client(
        timeout=httpx.Timeout(_READ_TIMEOUT, connect=_CONNECT_TIMEOUT),
        follow_redirects=True,
    )
    try:
        response = client.post(
            _ENDPOINT,
            data={"q": query},
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
        body = response.content[:_MAX_BYTES].decode("utf-8", errors="replace")
    except httpx.HTTPStatusError as exc:
        return SearchResult(False, [], query, f"HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        return SearchResult(False, [], query, f"request failed: {exc}")
    finally:
        if owns_client:
            client.close()

    hits = parse_results(body, count)
    if not hits:
        return SearchResult(False, [], query, "no results (search may be rate-limited)")
    return SearchResult(True, hits, query)
