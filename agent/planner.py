"""
agent/planner.py
Research planner — uses LLM to decompose a user query into targeted search queries.
"""

from __future__ import annotations

_DIVERSITY_TRIGGERS = (
    "compare", "vs", "versus", "disagree", "better", "conflict",
    "increase", "decrease", "both sides", "different",
)

PLANNER_SYSTEM = (
    "You are a research planner. Output ONLY a valid JSON object. "
    "No preamble, no markdown, no explanation. "
    'Schema: {"search_queries": ["q1", "q2", "q3"], "strategy_note": "brief plan"}. '
    "Maximum 3 search queries. Make them specific, diverse, and targeted. "
    "If the user's query contains multiple entities, requires a comparison, or implies a multi-step logical deduction, you MUST break it down. You must generate at least 2 highly distinct search queries that target different aspects of the question. Never issue multiple queries that mean the exact same thing."
)

_FALLBACK = {"search_queries": [], "strategy_note": "direct search"}


def _needs_diverse_queries(user_query: str) -> bool:
    q = user_query.lower()
    return any(trigger in q for trigger in _DIVERSITY_TRIGGERS)


def _diverse_queries(user_query: str) -> list[str]:
    return [
        f"{user_query} evidence supporting one side",
        f"{user_query} evidence supporting the opposite side",
        f"{user_query} conflicting findings systematic review",
    ]


class Planner:
    def __init__(self, llm_client):
        self.llm = llm_client

    def plan(self, user_query: str, history_summary: str = "") -> dict:
        """
        Produce a research plan with up to 3 search queries.
        Returns dict with 'search_queries' (list) and 'strategy_note' (str).
        """
        prompt = user_query
        if history_summary:
            prompt = f"Previous context: {history_summary}\n\nNew question: {user_query}"

        result = self.llm.generate_json(prompt, system=PLANNER_SYSTEM)

        if "search_queries" not in result:
            print(f"[planner] Invalid response — using direct search. Raw: {result}")
            return {"search_queries": [user_query], "strategy_note": "direct search"}

        # Cap at 3 queries
        result["search_queries"] = result["search_queries"][:3]
        if not result["search_queries"]:
            result["search_queries"] = [user_query]
        if _needs_diverse_queries(user_query):
            existing = []
            seen = set()
            for q in result["search_queries"] + _diverse_queries(user_query):
                normalized = q.strip().lower()
                if normalized and normalized not in seen:
                    existing.append(q)
                    seen.add(normalized)
                if len(existing) == 3:
                    break
            result["search_queries"] = existing
            result["strategy_note"] = (
                result.get("strategy_note")
                or "diverse search plan for comparison or conflict"
            )

        return result


# ------------------------------------------------------------------
# Smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from llm.client import LLMClient

    llm = LLMClient()
    planner = Planner(llm)
    result = planner.plan("What are the best open source LLMs in 2024?")
    print(f"Strategy: {result.get('strategy_note')}")
    print(f"Queries ({len(result['search_queries'])}):")
    for i, q in enumerate(result["search_queries"], 1):
        print(f"  {i}. {q}")
