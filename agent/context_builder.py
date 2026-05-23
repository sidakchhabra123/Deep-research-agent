"""
agent/context_builder.py
Selects, ranks, and packages fetched page content into a token-budgeted LLM context block.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

try:
    import tiktoken
except ImportError:
    tiktoken = None

_CONFLICT_PAIRS = [
    ("increased", "decreased"), ("confirmed", "denied"),
    ("safe", "unsafe"), ("yes", "no"), ("true", "false"),
]

_STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "both", "but",
    "can", "compare", "does", "exact", "for", "from", "has", "have", "how",
    "into", "its", "latest", "more", "most", "much", "not", "that", "the",
    "their", "this", "was", "what", "when", "where", "which", "who", "why",
    "with", "would", "your",
}


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _keyword_set(text: str) -> set[str]:
    return {w for w in _word_set(text) if len(w) > 2 and w not in _STOPWORDS}


def _recency_score(retrieved_at: str) -> float:
    try:
        dt = datetime.fromisoformat(retrieved_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        return 1.0 if age_hours <= 24 else 0.5
    except Exception:
        return 0.5


class ContextBuilder:
    def __init__(self, llm_client=None):
        self.enc = tiktoken.get_encoding("cl100k_base") if tiktoken else None
        self.llm = llm_client

    def count_tokens(self, text: str) -> int:
        text = text or ""
        if self.enc:
            return len(self.enc.encode(text))
        return max(1, len(text.split()))

    def score_page(self, page: dict, query: str, search_score: float = 0.5) -> float:
        q_words = _word_set(query)
        content_words = _word_set(page.get("content", ""))
        relevance = len(q_words & content_words) / max(len(q_words), 1)
        recency = _recency_score(page.get("retrieved_at", ""))
        return 0.6 * relevance + 0.3 * search_score + 0.1 * recency

    def extract_snippet(self, content: str, query: str, max_chars: int = 1200) -> str:
        q_words = _word_set(query)
        paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 50]
        if not paragraphs:
            return content[:max_chars]
        scored = [(len(q_words & _word_set(p)), p) for p in paragraphs]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = "\n\n".join(p for _, p in scored[:2])
        return top[:max_chars]

    def snippet_density(self, snippet: str, keyword_text: str) -> float:
        keywords = _keyword_set(keyword_text)
        if not keywords:
            return 0.0
        snippet_words = _word_set(snippet)
        return len(keywords & snippet_words) / max(len(keywords), 1)

    def _detect_conflicts(self, snippets: list[str]) -> str | None:
        if not snippets or len(snippets) < 2:
            return None

        if self.llm:
            joined = "\n---\n".join(snippets)[:2000]
            prompt = (
                "Do any of these source snippets directly contradict each other on factual claims? "
                "If there is a factual contradiction, you MUST start your response with the exact phrase: "
                "'Sources disagree:' and then state which sources contradict and on what claim. "
                "If there is no factual contradiction, answer ONLY: NO.\n"
                f"Snippets:\n{joined}"
            )
            try:
                resp = self.llm.generate(prompt, stream=False)
                if hasattr(resp, "__iter__") and not isinstance(resp, str):
                    resp = "".join(resp)
                resp = resp.strip()
                if resp.lower().startswith("sources disagree"):
                    return resp.splitlines()[0].strip()[:300]
                if resp.upper().startswith("YES"):
                    for line in resp.splitlines():
                        stripped = line.strip()
                        if stripped.upper().startswith("YES"):
                            return stripped[:300]
                    return resp.splitlines()[0].strip()[:300]
            except Exception as e:
                print(f"[context] LLM conflict detection failed: {e}, using heuristic")

        # Keyword heuristic fallback
        combined = " ".join(snippets).lower()
        for word_a, word_b in _CONFLICT_PAIRS:
            if word_a in combined and word_b in combined:
                return f"Possible conflicting information detected ('{word_a}' vs '{word_b}' found across sources). Verify claims carefully."
        return None

    def build(
        self,
        query: str,
        fetched_pages: list[dict],
        search_scores: dict[str, float] | None = None,
        search_queries: list[str] | None = None,
        max_tokens: int = 6000,
        history_tokens: int = 0,
    ) -> dict:
        if search_scores is None:
            search_scores = {}

        query_tokens = self.count_tokens(query)
        _RELEVANCE_THRESHOLD = 0.10  # pages below this score are skipped before budget check
        _DENSITY_THRESHOLD = 0.18
        keyword_text = " ".join([query] + (search_queries or []))

        # Score and sort all pages
        valid_pages = [p for p in fetched_pages if isinstance(p, dict) and p.get("url")]
        scored = [
            (self.score_page(p, query, search_scores.get(p.get("url", ""), 0.5)), p)
            for p in valid_pages
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        selected_snippets: list[str] = []
        citation_map: dict[int, dict] = {}
        context_blocks: list[str] = []
        tokens_used = 0
        pages_skipped_reason: list[str] = []
        seen_domains: set[str] = set()
        unique_domain_count = len({p.get("domain", "unknown") for _, p in scored})
        min_unique_domains = min(3, unique_domain_count)
        duplicate_domain_penalty = 0.25
        candidates: list[dict] = []

        for score, page in scored:
            domain = page.get("domain", "unknown")
            is_duplicate_domain = domain in seen_domains
            adjusted_score = score

            if is_duplicate_domain:
                if len(seen_domains) < min_unique_domains:
                    pages_skipped_reason.append(
                        f"{domain} — duplicate domain skipped to preserve source diversity"
                    )
                    continue
                adjusted_score *= duplicate_domain_penalty

            # Relevance gate — skip clearly irrelevant pages before consuming budget
            if adjusted_score < _RELEVANCE_THRESHOLD:
                pages_skipped_reason.append(
                    f"{domain} — relevance score {adjusted_score:.2f} below threshold {_RELEVANCE_THRESHOLD}"
                )
                continue

            snippet = self.extract_snippet(page.get("content", ""), query)
            snippet_tokens = self.count_tokens(snippet)
            density = self.snippet_density(snippet, keyword_text)
            if density < _DENSITY_THRESHOLD:
                pages_skipped_reason.append(
                    f"{domain} — keyword density {density:.2f} below threshold {_DENSITY_THRESHOLD}"
                )
                continue

            seen_domains.add(domain)
            candidates.append({
                "score": adjusted_score + density,
                "density": density,
                "tokens": snippet_tokens,
                "snippet": snippet,
                "page": page,
                "domain": domain,
                "rescued": False,
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        selected_candidates: list[dict] = []
        for candidate in candidates:
            selected_candidates.append(candidate)
            tokens_used = sum(item["tokens"] for item in selected_candidates)
            if tokens_used > max_tokens:
                selected_candidates.sort(key=lambda item: item["score"])
                dropped = selected_candidates.pop(0)
                pages_skipped_reason.append(
                    f"{dropped['domain']} — dropped during dense context pruning"
                )
                selected_candidates.sort(key=lambda item: item["score"], reverse=True)
                tokens_used = sum(item["tokens"] for item in selected_candidates)

        selected_domains = {item["domain"] for item in selected_candidates}
        if len(selected_domains) < 2 and unique_domain_count >= 2:
            selected_urls = {
                item["page"].get("url")
                for item in selected_candidates
                if item["page"].get("url")
            }
            rescue_candidates: list[dict] = []
            for score, page in scored:
                domain = page.get("domain", "unknown")
                if domain in selected_domains or page.get("url") in selected_urls:
                    continue
                snippet = self.extract_snippet(page.get("content", ""), query)
                density = self.snippet_density(snippet, keyword_text)
                snippet_tokens = self.count_tokens(snippet)
                rescue_candidates.append({
                    "score": score + density,
                    "density": density,
                    "tokens": snippet_tokens,
                    "snippet": snippet,
                    "page": page,
                    "domain": domain,
                    "rescued": True,
                })

            rescue_candidates.sort(key=lambda item: item["score"], reverse=True)
            for rescue in rescue_candidates:
                while selected_candidates and tokens_used + rescue["tokens"] > max_tokens:
                    selected_candidates.sort(key=lambda item: item["score"])
                    dropped = selected_candidates.pop(0)
                    tokens_used -= dropped["tokens"]
                    pages_skipped_reason.append(
                        f"{dropped['domain']} — dropped to force second source domain"
                    )
                if tokens_used + rescue["tokens"] > max_tokens:
                    pages_skipped_reason.append(
                        f"{rescue['domain']} — second-domain rescue skipped because snippet exceeds budget"
                    )
                    continue
                selected_candidates.append(rescue)
                tokens_used += rescue["tokens"]
                pages_skipped_reason.append(
                    f"{rescue['domain']} — added by source-diversity rescue"
                )
                break

        selected_candidates.sort(key=lambda item: item["score"], reverse=True)
        for cite_idx, candidate in enumerate(selected_candidates, start=1):
            page = candidate["page"]
            domain = candidate["domain"]
            snippet = candidate["snippet"]
            selected_snippets.append(snippet)
            citation_map[cite_idx] = {
                "title":          page.get("title", ""),
                "url":            page.get("url", ""),
                "domain":         domain,
                "snippet_preview": page.get("snippet_preview", snippet[:200]),
            }
            block = f"[{cite_idx}] {page.get('title', 'Untitled')} ({domain})\n{snippet}"
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)
        conflict_warning = self._detect_conflicts(selected_snippets)

        total_tokens = query_tokens + history_tokens + tokens_used
        web_budget_pct = min(100.0, round(tokens_used / max(max_tokens, 1) * 100, 1))
        breakdown = {
            "current_query_tokens": query_tokens,
            "history_tokens":       history_tokens,
            "web_context_tokens":   tokens_used,
            "total_tokens":         total_tokens,
            "budget_used_percent":  web_budget_pct,
            "pages_considered":     len(scored),
            "pages_selected":       len(citation_map),
            "pages_skipped_reason": pages_skipped_reason,
        }

        return {
            "context_text":      context_text,
            "citation_map":      citation_map,
            "conflict_warning":  conflict_warning,
            "pages_used":        len(citation_map),
            "context_breakdown": breakdown,
        }

    @classmethod
    def format_context_breakdown(cls, breakdown: dict) -> str:
        """Return a human-readable context window usage string for the UI."""
        q   = breakdown.get("current_query_tokens", 0)
        h   = breakdown.get("history_tokens", 0)
        w   = breakdown.get("web_context_tokens", 0)
        tot = breakdown.get("total_tokens", 0)
        pct = breakdown.get("budget_used_percent", 0.0)
        considered = breakdown.get("pages_considered", 0)
        selected   = breakdown.get("pages_selected", 0)
        skipped    = breakdown.get("pages_skipped_reason", [])

        lines = [
            "--- Context Window Usage ---",
            f"Query:        {q:>6} tokens",
            f"Conversation: {h:>6} tokens",
            f"Web sources:  {w:>6} tokens",
            f"Total:        {tot:>6} tokens  (web context {pct}% of budget)",
            "",
            f"Sources considered: {considered}",
            f"Sources selected:   {selected}",
            f"Sources skipped:    {len(skipped)}",
        ]
        for reason in skipped:
            lines.append(f"  - {reason}")
        lines.append("----------------------------")
        return "\n".join(lines)


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    mock_pages = [
        {
            "url": "https://example.com/llm-survey",
            "title": "LLM Survey 2024",
            "domain": "example.com",
            "content": "Large language models have increased in capability significantly. "
                       "GPT-4 and Llama 2 represent the state of the art. "
                       "Open source models decreased the cost of deployment.\n\n"
                       "Researchers confirmed that fine-tuning improves performance on specific tasks.",
            "retrieved_at": datetime.now().isoformat(),
            "word_count": 45,
        },
        {
            "url": "https://arxiv.org/fake/2024",
            "title": "Open Source LLMs Review",
            "domain": "arxiv.org",
            "content": "Open source models like Mistral 7B and Llama 2 have shown impressive results. "
                       "The performance gap with proprietary models has decreased. "
                       "Some researchers denied that open source models are safe for production use.",
            "retrieved_at": datetime.now().isoformat(),
            "word_count": 40,
        },
    ]
    cb = ContextBuilder()
    result = cb.build("best open source LLMs 2024", mock_pages)
    print("Citation map:", result["citation_map"])
    print("Pages used:", result["pages_used"])
    print("Conflict warning:", result["conflict_warning"])
    print("\nContext (first 200 chars):", result["context_text"][:200])
    print(ContextBuilder.format_context_breakdown(result["context_breakdown"]))
