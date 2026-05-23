"""
search/fetcher.py
Page content fetcher — httpx + trafilatura, BeautifulSoup fallback.
Supports single fetch and concurrent batch fetch via asyncio.
"""

import asyncio
from datetime import datetime
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_TIMEOUT = 12
_MAX_BATCH = 6


def _extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def _extract_title(html: str, url: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else _extract_domain(url)
    except Exception:
        return _extract_domain(url)


def _extract_text(html: str) -> str | None:
    """Try trafilatura first; BeautifulSoup fallback.
    Returns None only when trafilatura yields nothing AND BS yields < 50 chars.
    """
    trafilatura_text = trafilatura.extract(html)
    if trafilatura_text and trafilatura_text.strip():
        return trafilatura_text.strip()
    # BeautifulSoup fallback
    try:
        soup = BeautifulSoup(html, "html.parser")
        bs_text = soup.get_text(separator=" ", strip=True)
        return bs_text if len(bs_text) >= 50 else None
    except Exception:
        return None


class PageFetcher:
    # ------------------------------------------------------------------
    # Sync single fetch
    # ------------------------------------------------------------------
    def fetch(self, url: str) -> dict | None:
        try:
            resp = httpx.get(url, timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
            content = _extract_text(html)
            if not content:
                print(f"[fetcher] Skipping {url} — content too short or unextractable")
                return None
            title = _extract_title(html, url)
            domain = _extract_domain(url)
            return {
                "url":             url,
                "title":          title,
                "domain":         domain,
                "content":        content,
                "retrieved_at":   datetime.now().isoformat(),
                "word_count":     len(content.split()),
                "snippet_preview": content[:200],
            }
        except Exception as e:
            print(f"[fetcher] Error fetching {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------
    async def _async_fetch_one(self, client: httpx.AsyncClient, url: str) -> dict | None:
        try:
            resp = await client.get(url, timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
            content = _extract_text(html)
            if not content:
                return None
            title = _extract_title(html, url)
            domain = _extract_domain(url)
            return {
                "url":             url,
                "title":          title,
                "domain":         domain,
                "content":        content,
                "retrieved_at":   datetime.now().isoformat(),
                "word_count":     len(content.split()),
                "snippet_preview": content[:200],
            }
        except Exception as e:
            print(f"[fetcher] Async error for {url}: {e}")
            return None

    async def _batch_async(self, urls: list[str]) -> list[dict]:
        async with httpx.AsyncClient() as client:
            tasks = [self._async_fetch_one(client, url) for url in urls]
            results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Public batch interface
    # ------------------------------------------------------------------
    def batch_fetch(self, urls: list[str]) -> list[dict]:
        """Fetch up to 6 URLs concurrently. Returns list of successful page dicts."""
        limited = urls[:_MAX_BATCH]
        return asyncio.run(self._batch_async(limited))


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    fetcher = PageFetcher()
    page = fetcher.fetch("https://en.wikipedia.org/wiki/Large_language_model")
    if page:
        print(f"Title:      {page['title']}")
        print(f"Domain:     {page['domain']}")
        print(f"Words:      {page['word_count']}")
        print(f"Retrieved:  {page['retrieved_at']}")
        print(f"\nContent (first 300 chars):\n{page['content'][:300]}")
    else:
        print("Fetch returned None — check network.")
