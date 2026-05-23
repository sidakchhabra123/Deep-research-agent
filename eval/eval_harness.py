"""
eval/eval_harness.py
Standalone evaluation harness for the pure-Python Deep Research Agent.

This script is intentionally local-only until you run it manually. Running it
will use the configured search and LLM clients through the Orchestrator.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from urllib.parse import urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from session.store import SessionStore
from llm.client import LLMClient
from agent.planner import Planner
from agent.context_builder import ContextBuilder
from agent.orchestrator import Orchestrator
from search.search_client import SearchClient
from search.fetcher import PageFetcher


DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
MULTI_TURN_TYPES = {"multi_turn_1", "multi_turn_2"}


def load_dataset(path: str = DATASET_PATH) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("dataset.json must contain a JSON array")
        return data
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset at {path}: {exc}") from exc


def build_orchestrator():
    store = SessionStore()
    llm_client = LLMClient()
    planner = Planner(llm_client)
    search_client = SearchClient()
    fetcher = PageFetcher()
    ctx_builder = ContextBuilder(llm_client=llm_client)
    orchestrator = Orchestrator(
        store,
        llm_client,
        planner,
        search_client,
        fetcher,
        ctx_builder,
    )
    return store, orchestrator


def normalize_citation_map(citation_map) -> dict:
    if not isinstance(citation_map, dict):
        return {}
    normalized = {}
    for key, value in citation_map.items():
        if isinstance(value, dict):
            normalized[str(key)] = value
        else:
            normalized[str(key)] = {"value": value}
    return normalized


def domain_from_citation(citation: dict) -> str:
    domain = (citation.get("domain") or "").strip().lower()
    if domain:
        return domain
    url = (citation.get("url") or "").strip()
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def compute_metrics(q_type: str, final_answer: str, citation_map: dict, conflict_warning: str | None) -> dict:
    answer_lower = (final_answer or "").lower()
    domains = {
        domain
        for citation in citation_map.values()
        for domain in [domain_from_citation(citation)]
        if domain
    }
    uncertainty_terms = ("cannot", "insufficient", "no information")
    conflict_terms = ("disagree", "conflict")
    uncertainty_handled = (
        any(term in answer_lower for term in uncertainty_terms)
        if q_type == "insufficient_evidence"
        else "N/A"
    )
    citation_integrity = "[1]" in final_answer or "[2]" in final_answer
    if uncertainty_handled is True:
        citation_integrity = True

    return {
        "citation_integrity": citation_integrity,
        "source_diversity": len(domains),
        "uncertainty_handled": uncertainty_handled,
        "conflict_flagged": (
            bool(conflict_warning)
            or any(term in answer_lower for term in conflict_terms)
            if q_type == "conflict"
            else "N/A"
        ),
    }


def run_case(orchestrator, session_id: str, query: str) -> dict:
    final_answer_parts = []
    citation_map = {}
    pages_used = 0
    conflict_warning = None
    done_step = {}
    stream_steps = []

    for step in orchestrator.run(query, session_id, stream=True):
        if not isinstance(step, dict):
            continue
        step_name = step.get("step")
        if step_name != "token":
            stream_steps.append(step)
        if step_name == "token":
            final_answer_parts.append(step.get("text", ""))
        elif step_name == "done":
            done_step = step
            citation_map = normalize_citation_map(step.get("citation_map", {}))
            pages_used = step.get("pages_used", 0) or 0
        elif step_name == "conflict":
            conflict_warning = step.get("message") or ""

    return {
        "final_answer": "".join(final_answer_parts),
        "citation_map": citation_map,
        "pages_used": pages_used,
        "conflict_warning": conflict_warning,
        "done_step": done_step,
        "stream_steps": stream_steps,
    }


def boolish(value) -> str:
    if value == "N/A":
        return "N/A"
    return "Y" if bool(value) else "N"


def print_ascii_table(rows: list[dict]) -> None:
    headers = ["ID", "Type", "Cite", "Domains", "Uncert", "Conflict", "Error"]
    widths = [10, 22, 6, 8, 8, 9, 24]
    line = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def fmt_row(values):
        return "| " + " | ".join(
            str(value)[:width].ljust(width)
            for value, width in zip(values, widths)
        ) + " |"

    print("\n" + line)
    print(fmt_row(headers))
    print(line)
    for row in rows:
        metrics = row.get("metrics", {})
        print(
            fmt_row(
                [
                    row.get("id", ""),
                    row.get("type", ""),
                    boolish(metrics.get("citation_integrity")),
                    metrics.get("source_diversity", 0),
                    boolish(metrics.get("uncertainty_handled")),
                    boolish(metrics.get("conflict_flagged")),
                    row.get("error") or "",
                ]
            )
        )
    print(line)


def main() -> None:
    dataset = load_dataset()
    store, orchestrator = build_orchestrator()
    multi_turn_session_id = store.create_session()
    results = []

    print("Deep Research Agent Evaluation Harness")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Cases: {len(dataset)}")

    for item in dataset:
        qid = item.get("id", "unknown")
        q_type = item.get("type", "unknown")
        query = item.get("query", "")
        expected_behavior = item.get("expected_behavior", "")
        session_id = multi_turn_session_id if q_type in MULTI_TURN_TYPES else store.create_session()

        started_at = time.time()
        error = None
        run_data = {
            "final_answer": "",
            "citation_map": {},
            "pages_used": 0,
            "conflict_warning": None,
            "done_step": {},
            "stream_steps": [],
        }

        print(f"\n[{qid}] {q_type}: {query}")
        try:
            if not query:
                raise ValueError("Missing query")
            run_data = run_case(orchestrator, session_id, query)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {error}")

        elapsed_seconds = round(time.time() - started_at, 2)
        citation_map = normalize_citation_map(run_data.get("citation_map", {}))
        final_answer = run_data.get("final_answer", "")
        metrics = compute_metrics(
            q_type=q_type,
            final_answer=final_answer,
            citation_map=citation_map,
            conflict_warning=run_data.get("conflict_warning"),
        )
        # If the agent correctly identified insufficient evidence, citations are not expected.
        if metrics.get("uncertainty_handled") == True:
            metrics["citation_integrity"] = True

        results.append(
            {
                "id": qid,
                "type": q_type,
                "query": query,
                "expected_behavior": expected_behavior,
                "session_id": session_id,
                "final_answer": final_answer,
                "citations": citation_map,
                "pages_used": run_data.get("pages_used", 0),
                "conflict_warning": run_data.get("conflict_warning"),
                "done_step": run_data.get("done_step", {}),
                "metrics": metrics,
                "error": error,
                "elapsed_seconds": elapsed_seconds,
            }
        )
        print(
            "  "
            f"citation_integrity={metrics['citation_integrity']} | "
            f"source_diversity={metrics['source_diversity']} | "
            f"uncertainty_handled={metrics['uncertainty_handled']} | "
            f"conflict_flagged={metrics['conflict_flagged']} | "
            f"elapsed={elapsed_seconds}s"
        )

    print_ascii_table(results)

    output = {
        "run_at": datetime.now().isoformat(),
        "dataset_path": DATASET_PATH,
        "results": results,
    }
    try:
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved full results to: {RESULTS_PATH}")
    except Exception as exc:
        print(f"\nERROR: failed to save results: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
