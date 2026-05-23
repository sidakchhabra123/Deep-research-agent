"""
search/search_client.py
Web search module — Tavily primary, Serper fallback.
"""

import os
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def _extract_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        # Strip leading www. for cleaner display
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


class SearchClient:
    def __init__(self):
        self.tavily_key = os.getenv("TAVILY_API_KEY", "")
        self.serper_key = os.getenv("SERPER_API_KEY", "")

    # ------------------------------------------------------------------
    # Schema normalizer — all results pass through this
    # ------------------------------------------------------------------
    def _normalize_result(self, raw: dict, provider: str) -> dict | None:
        """
        Enforce the canonical result schema:
          title, url, snippet, score (float), domain (www-stripped)
        Returns None if title, url, or snippet is missing/empty.
        """
        if provider == "tavily":
            url = raw.get("url", "") or ""
            title = raw.get("title", "") or ""
            # Tavily uses 'content' for the snippet text
            snippet = raw.get("snippet") or raw.get("content", "") or ""
            try:
                score = float(raw.get("score") or 0.5)
            except (TypeError, ValueError):
                score = 0.5
        else:  # serper
            url = raw.get("link", "") or ""
            title = raw.get("title", "") or ""
            snippet = raw.get("snippet", "") or ""
            score = 0.5

        domain = _extract_domain(url)

        # Reject incomplete results
        if not title.strip() or not url.strip() or not snippet.strip():
            return None

        return {
            "title":   title.strip(),
            "url":     url.strip(),
            "snippet": snippet.strip(),
            "score":   score,
            "domain":  domain,
        }

    # ------------------------------------------------------------------
    # Primary: Tavily
    # ------------------------------------------------------------------
    def _tavily_search(self, query: str, num_results: int) -> list[dict]:
        client = TavilyClient(api_key=self.tavily_key)
        resp = client.search(query, max_results=num_results)
        results = []
        for item in resp.get("results", []):
            normalized = self._normalize_result(item, "tavily")
            if normalized:
                results.append(normalized)
        return results

    # ------------------------------------------------------------------
    # Fallback: Serper
    # ------------------------------------------------------------------
    def _serper_search(self, query: str, num_results: int) -> list[dict]:
        resp = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self.serper_key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", []):
            normalized = self._normalize_result(item, "serper")
            if normalized:
                results.append(normalized)
        return results

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def search(self, query: str, num_results: int = 8) -> list[dict]:
        """Search the web. Returns normalized, deduplicated results. Filters empty fields."""
        results = []
        try:
            results = self._tavily_search(query, num_results)
            print(f"[search] provider: tavily | query: {query!r}")
        except Exception as e:
            print(f"[search] Tavily failed ({e}), falling back to Serper")
            try:
                results = self._serper_search(query, num_results)
                print(f"[search] provider: serper | query: {query!r}")
            except Exception as e2:
                print(f"[search] Serper also failed ({e2}). Returning empty results.")
                return []

        # Deduplicate by URL (preserve insertion order)
        seen: set[str] = set()
        deduped = []
        for r in results:
            if r["url"] not in seen:
                seen.add(r["url"])
                deduped.append(r)
        return deduped


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    client = SearchClient()
    results = client.search("open source LLMs 2024")
    print(f"\nReturned {len(results)} results. First 3:")
    for r in results[:3]:
        print(f"  [{r['score']:.2f}] {r['title']}")
        print(f"         {r['url']}")
        print(f"         {r['snippet'][:100]}...")
        print()
