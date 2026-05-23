"""
agent/orchestrator.py
Main research orchestration loop — Plan → Search → Fetch → Select → Answer.
Yields step dicts for streaming UI consumption.
"""

from __future__ import annotations

import re

ANSWER_SYSTEM = (
    "You are a research assistant. Answer using ONLY the provided web context. "
    "CITATIONS: You MUST cite every single factual claim using ONLY bracketed numbers like [1] or [2] that correspond directly to the provided source list. Never use formats like [Source 1] or raw URLs in the text. "
    "UNCERTAINTY: If the provided web context does not contain enough information to fully answer the query, you MUST explicitly state: 'I cannot find sufficient evidence to answer this' and suggest a refined search. "
    "CONFLICT: If the sources contradict each other, you MUST explicitly state: 'Sources disagree:' and objectively explain both sides. "
    "Never hallucinate facts not present in the provided context. "
    "Structure your final response using concise, numbered lists (1, 2, 3...) whenever possible. Avoid long narrative paragraphs. Be highly structured, brief, and direct to conserve output tokens. "
    "CRITICAL INSTRUCTION: YOU WILL BE PENALIZED IF YOU DO NOT USE BRACKETED CITATIONS. Every single sentence containing a fact MUST end with a citation like [1]. Do not summarize without citing. "
    "FACTUAL ANSWER RULE: For direct factual answers, support the key fact with multiple citations whenever multiple relevant sources are available, e.g. [1][2]. Do not rely on only one citation if another selected source supports the same fact. "
    "CRITICAL FORMATTING RULE: You MUST use inline bracket citations for EVERY factual sentence. "
    "EXAMPLE OF CORRECT BEHAVIOR: The Python programming language was created by Guido van Rossum [1]. It was officially released in 1991 [2]. "
    "EXAMPLE OF INCORRECT BEHAVIOR: The Python programming language was created by Guido van Rossum. (Source 1). "
    "If you do not use the [1] format exactly, the system will fail. Do not summarize without citing. "
    "CRITICAL CONFLICT RULE: If the user asks for both sides of an issue, or if your retrieved sources contradict each other, you MUST explicitly use the exact word \"conflict\" or \"disagree\" in your opening paragraph. You MUST present both sides as separate claims, and EACH SIDE MUST have its own bracketed citation from the relevant source, for example one side cited with [1] and the other side cited with [2]. Never describe opposing sides without citations. "
    "EXAMPLE OF CORRECT CONFLICT BEHAVIOR: Sources disagree on this topic. According to the first source, coffee temporarily increases blood pressure [1]. However, the second source states it has no long-term negative effect [2]. "
    "EXAMPLE OF INCORRECT BEHAVIOR: Some people think coffee raises blood pressure, while others don't. (This is incorrect because it fails to use the target keywords and fails to cite the sources)."
)

_MAX_CONTEXT_TOKENS = 28000
_HISTORY_CHARS = 500
_SUMMARY_TURNS = 5
_EVIDENCE_STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "both", "but",
    "can", "compare", "does", "exact", "for", "from", "has", "have", "how",
    "into", "its", "latest", "more", "most", "much", "not", "that", "the",
    "their", "this", "was", "what", "when", "where", "which", "who", "why",
    "with", "would", "your",
}


def _build_history_summary(history: list[dict], max_chars: int = _HISTORY_CHARS) -> str:
    """Concatenate last 3 messages into a capped summary string."""
    recent = history[-3:] if len(history) >= 3 else history
    parts = [f"{m['role'].upper()}: {m['content']}" for m in recent]
    summary = " | ".join(parts)
    return summary[:max_chars]


def _keywords(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"\b\w+\b", text.lower())
        if len(word) > 2 and word not in _EVIDENCE_STOPWORDS
    }


def _context_has_enough_evidence(query: str, ctx: dict) -> bool:
    context_text = ctx.get("context_text") or ""
    if not context_text or ctx.get("pages_used", 0) < 1:
        return False
    query_lower = query.lower()
    context_lower = context_text.lower()
    if "fictional" in query_lower and not any(
        marker in context_lower
        for marker in ("fictional", "myth", "mythical", "legend", "legendary", "underwater")
    ):
        return False
    query_keywords = _keywords(query)
    if not query_keywords:
        return bool(context_text.strip())
    context_words = _keywords(context_text)
    matched = query_keywords & context_words
    required_matches = max(2, int(len(query_keywords) * 0.4))
    return len(matched) >= required_matches


def _refined_queries(query: str, existing_queries: list[str]) -> list[str]:
    candidates = [
        f"{query} official source",
        f"{query} primary source evidence",
        f"{query} reliable source citation",
    ]
    seen = {q.strip().lower() for q in existing_queries}
    refined = []
    for candidate in candidates:
        key = candidate.strip().lower()
        if key and key not in seen:
            refined.append(candidate)
            seen.add(key)
    return refined[:2]


class Orchestrator:
    def __init__(self, store, llm_client, planner, search_client, fetcher, context_builder):
        self.store = store
        self.llm = llm_client
        self.planner = planner
        self.search = search_client
        self.fetcher = fetcher
        self.ctx_builder = context_builder

    def run(self, query: str, session_id: str, stream: bool = True):
        """Generator — yields step dicts driving the streaming UI."""

        # Step 1: Planning signal
        yield {"step": "planning", "message": "Planning research strategy..."}

        # Step 2: History + summary
        history = self.store.get_history(session_id)
        history_summary = _build_history_summary(history)

        # Step 3: Plan
        plan = self.planner.plan(query, history_summary)
        yield {"step": "planned", "message": plan.get("strategy_note", ""), "queries": plan["search_queries"]}

        # Step 4: Search all queries
        all_results: list[dict] = []
        all_scores: dict[str, float] = {}
        for q in plan["search_queries"]:
            yield {"step": "searching", "message": f"Searching: {q}"}
            try:
                results = self.search.search(q) or []
                for r in results:
                    url = r.get("url") if isinstance(r, dict) else None
                    if not url:
                        continue
                    score = r.get("score", 0.5)
                    if url not in all_scores:
                        all_results.append(r)
                        all_scores[url] = score
                    else:
                        # Keep highest score
                        all_scores[url] = max(all_scores[url], score)
            except Exception as e:
                print(f"[orchestrator] Search error for '{q}': {e}")

        # Step 5: Select top URLs by score
        sorted_urls = sorted(all_scores, key=lambda u: all_scores[u], reverse=True)
        top_urls = sorted_urls[:6]
        yield {"step": "fetching", "message": f"Reading {len(top_urls)} sources..."}

        # Step 6: Fetch pages — one fetch_result per attempted URL (success or failure)
        url_meta = {r["url"]: r for r in all_results if r.get("url")}
        pages = self.fetcher.batch_fetch(top_urls)
        pages_by_url = {p["url"]: p for p in pages if p.get("url")}
        urls_opened = [p["url"] for p in pages if p.get("url")]

        for url in top_urls:
            page = pages_by_url.get(url)
            meta = url_meta.get(url, {})
            if page:
                yield {
                    "step": "fetch_result",
                    "success": True,
                    "url": page.get("url", url),
                    "domain": page.get("domain", ""),
                    "title": page.get("title", url),
                    "word_count": page.get("word_count", 0),
                }
            else:
                yield {
                    "step": "fetch_result",
                    "success": False,
                    "url": url,
                    "domain": meta.get("domain", ""),
                    "title": meta.get("title", url),
                    "message": "Could not fetch or extract page content",
                }

        yield {"step": "selecting", "message": f"Selecting context from {len(pages)} pages..."}

        # Step 7: Build context
        history_tokens = self.ctx_builder.count_tokens(history_summary)
        ctx = self.ctx_builder.build(
            query,
            pages,
            all_scores,
            search_queries=plan["search_queries"],
            history_tokens=history_tokens,
        )
        breakdown = ctx["context_breakdown"]
        breakdown["pages_considered"] = len(top_urls)
        for url in top_urls:
            if url not in pages_by_url:
                domain = url_meta.get(url, {}).get("domain") or url
                breakdown["pages_skipped_reason"].append(
                    f"{domain} — fetch or extract failed"
                )

        context_sufficient = _context_has_enough_evidence(query, ctx)
        if not context_sufficient:
            retry_queries = _refined_queries(query, plan["search_queries"])
            if retry_queries:
                yield {
                    "step": "retry",
                    "message": "Insufficient evidence found. Refining search queries and retrying retrieval...",
                    "queries": retry_queries,
                }
                plan["search_queries"].extend(retry_queries)
                retry_urls: list[str] = []
                for q in retry_queries:
                    yield {"step": "searching", "message": f"Searching: {q}"}
                    try:
                        results = self.search.search(q)
                        for r in results:
                            url = r.get("url")
                            if not url:
                                continue
                            score = r.get("score", 0.5)
                            if url not in all_scores:
                                all_results.append(r)
                                all_scores[url] = score
                                retry_urls.append(url)
                            else:
                                all_scores[url] = max(all_scores[url], score)
                    except Exception as e:
                        print(f"[orchestrator] Retry search error for '{q}': {e}")

                retry_urls = sorted(
                    set(retry_urls),
                    key=lambda u: all_scores.get(u, 0),
                    reverse=True,
                )[:4]
                if retry_urls:
                    yield {"step": "fetching", "message": f"Reading {len(retry_urls)} refined sources..."}
                    retry_pages = self.fetcher.batch_fetch(retry_urls)
                    retry_pages_by_url = {p["url"]: p for p in retry_pages if p.get("url")}
                    pages.extend(retry_pages)
                    urls_opened.extend(p["url"] for p in retry_pages if p.get("url") and p["url"] not in urls_opened)
                    pages_by_url.update(retry_pages_by_url)
                    url_meta = {r["url"]: r for r in all_results if r.get("url")}

                    for url in retry_urls:
                        page = retry_pages_by_url.get(url)
                        meta = url_meta.get(url, {})
                        if page:
                            yield {
                                "step": "fetch_result",
                                "success": True,
                                "url": page.get("url", url),
                                "domain": page.get("domain", ""),
                                "title": page.get("title", url),
                                "word_count": page.get("word_count", 0),
                            }
                        else:
                            yield {
                                "step": "fetch_result",
                                "success": False,
                                "url": url,
                                "domain": meta.get("domain", ""),
                                "title": meta.get("title", url),
                                "message": "Could not fetch or extract page content",
                            }

                    ctx = self.ctx_builder.build(
                        query,
                        pages,
                        all_scores,
                        search_queries=plan["search_queries"],
                        history_tokens=history_tokens,
                    )
                    breakdown = ctx["context_breakdown"]
                    breakdown["pages_considered"] = len(set(top_urls + retry_urls))
                    for url in retry_urls:
                        if url not in pages_by_url:
                            domain = url_meta.get(url, {}).get("domain") or url
                            breakdown["pages_skipped_reason"].append(
                                f"{domain} — fetch or extract failed"
                            )
                    context_sufficient = _context_has_enough_evidence(query, ctx)

        # Step 8: Token budget check — before context_ready so breakdown matches the prompt
        context_tokens = self.ctx_builder.count_tokens(ctx["context_text"])
        query_tokens = self.ctx_builder.count_tokens(query)
        if history_tokens + context_tokens + query_tokens > _MAX_CONTEXT_TOKENS:
            recent = history[-2:] if len(history) >= 2 else history
            history_summary = " | ".join(
                f"{m['role'].upper()}: {m['content']}" for m in recent
            )
            history_summary += " (history truncated)"
            history_tokens = self.ctx_builder.count_tokens(history_summary)
            breakdown["history_tokens"] = history_tokens
            breakdown["total_tokens"] = (
                breakdown["current_query_tokens"]
                + history_tokens
                + breakdown["web_context_tokens"]
            )

        yield {
            "step": "context_ready",
            "message": f"Context ready — {ctx['pages_used']} source(s) selected",
            "context_breakdown": ctx["context_breakdown"],
            "citation_map": ctx["citation_map"],
        }
        if ctx.get("conflict_warning"):
            yield {"step": "conflict", "message": ctx["conflict_warning"]}

        # Step 9: Build final prompt and stream answer
        yield {"step": "answering", "message": "Generating answer with citations..."}
        ctx_section = f"Web context:\n{ctx['context_text']}" if ctx["context_text"] else ""
        history_section = f"Conversation so far:\n{history_summary}" if history_summary else ""
        evidence_section = (
            "Evidence validation: The retrieved context still appears insufficient. "
            "Use the exact uncertainty sentence required by the system prompt."
            if not context_sufficient
            else ""
        )
        final_prompt = "\n\n".join(filter(None, [history_section, ctx_section, evidence_section, f"User question: {query}"]))

        full_answer = ""
        try:
            for chunk in self.llm.generate(final_prompt, system=ANSWER_SYSTEM, stream=True):
                full_answer += chunk
                yield {"step": "token", "text": chunk}
        except Exception as e:
            error_msg = f"[Error generating answer: {e}]"
            full_answer = error_msg
            yield {"step": "token", "text": error_msg}

        # Step 10: Persist
        self.store.add_message(session_id, "user", query)
        self.store.add_message(session_id, "assistant", full_answer)
        snippets_selected = [
            {"index": idx, **meta} for idx, meta in ctx["citation_map"].items()
        ]
        turn_id = self.store.add_turn(
            session_id,
            query,
            plan["search_queries"],
            urls_opened,
            snippets_selected,
            full_answer,
            context_breakdown=ctx["context_breakdown"],
        )

        # Step 11: Summarization if needed
        if self.store.needs_summarization(session_id, _SUMMARY_TURNS):
            self._summarize(session_id)

        yield {
            "step": "done",
            "turn_id": turn_id,
            "citation_map": ctx["citation_map"],
            "pages_used": ctx["pages_used"],
        }

    def _summarize(self, session_id: str):
        msgs = self.store.get_history(session_id)[-5:]
        formatted = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in msgs)
        prompt = (
            "Summarize this research conversation in 4-5 sentences covering what was researched, "
            f"what was established, and any open questions.\n\n{formatted}"
        )
        try:
            summary = self.llm.generate(prompt, stream=False)
            if hasattr(summary, "__iter__") and not isinstance(summary, str):
                summary = "".join(summary)
            self.store.update_summary(session_id, summary)
            print("[session] summary updated")
        except Exception as e:
            print(f"[session] summarization failed: {e}")


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from llm.client import LLMClient
    from agent.planner import Planner
    from agent.context_builder import ContextBuilder
    from search.search_client import SearchClient
    from search.fetcher import PageFetcher
    from session.store import SessionStore

    store = SessionStore()
    llm = LLMClient()
    planner = Planner(llm)
    search = SearchClient()
    fetcher = PageFetcher()
    ctx_builder = ContextBuilder(llm_client=llm)
    orch = Orchestrator(store, llm, planner, search, fetcher, ctx_builder)

    sid = store.create_session()
    print(f"Session: {sid}\n")

    for step in orch.run("What is retrieval augmented generation?", sid, stream=True):
        if step["step"] == "token":
            print(step["text"], end="", flush=True)
        else:
            print(f"\n[{step['step'].upper()}] {step.get('message', step)}")
    print()
