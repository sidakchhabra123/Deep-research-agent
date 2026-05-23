# Deep Research Agent: Final Submission

## Index
1. Design Note (Architecture & Strategy)
2. Evaluation Methodology & Harness Results
3. Setup Instructions, Links, & Examples

---

## 1. Design Note

### Target Users & Problem Solved

The Deep Research Agent is built for researchers, engineers, students, and technical decision-makers who need answers that are current, inspectable, and grounded in source material. Standard large language models are useful for synthesis, but they can hallucinate facts, rely on stale training data, and present fluent answers without showing where claims came from. That is a poor fit for research workflows where the user must verify evidence, compare sources, and understand uncertainty.

This project solves that problem by turning a normal question into a deterministic research workflow. The agent searches the live web, fetches pages, extracts useful text, prunes noisy material, and only then asks the LLM to synthesize an answer. Every factual claim is expected to use bracketed citations such as [1] and [2], which map directly to retrieved sources in the UI. The result is not just an answer, but a transparent research artifact: search queries, opened URLs, selected snippets, citations, and final response are all persisted in SQLite session history.

### Definition of "Deep Research"

In this implementation, "Deep Research" means a deterministic multi-step orchestration pipeline rather than a single prompt. The agent does not simply ask an LLM to answer from memory. It executes a structured process:

1. Plan targeted search queries from the user question.
2. Search the web using Tavily.
3. Fetch and extract readable HTML content using a pure-Python fetch layer with trafilatura-style extraction.
4. Score, prune, and diversify evidence under a token budget using tiktoken.
5. Synthesize a concise cited answer from only the selected web context.
6. Persist the full session, turn metadata, and source details in SQLite.

This design keeps the Planner, Search, Fetch, and Context Builder nodes in English to preserve retrieval quality and reasoning consistency. Multilingual support is added with a Translation Sandwich: user input in Hindi or Tamil is translated into English with Sarvam, the research pipeline runs in English, and the final answer is translated back to the selected language.

### Success Metrics

The agent is optimized against four core quality metrics:

- **Citation Integrity Rate:** Whether the final answer uses strict bracketed citations, such as [1] and [2], for factual claims. This measures whether answers are grounded rather than free-floating LLM assertions.
- **Source Diversity Score:** The number of unique domains represented in the citation map. This helps avoid over-reliance on a single source and improves robustness for comparison or conflict questions.
- **Uncertainty Handling:** Whether the agent refuses or qualifies answers when the retrieved evidence is insufficient. The expected behavior is to explicitly state that sufficient evidence could not be found rather than inventing an answer.
- **Conflict Resolution Accuracy:** Whether the agent detects and explains disagreement across sources. For conflicting evidence, the agent must state "Sources disagree:" and objectively summarize both sides.

### Data Flow and Components

The system is built completely from scratch in pure Python, without LangChain or similar orchestration frameworks. The key components are:

- **Planner Node:** Converts the user question and optional session history into up to three targeted search queries. The planner is explicitly instructed to decompose multi-hop, comparison, and multi-entity questions into distinct sub-queries rather than repeating keyword variants.
- **Search/Fetch Layer:** Uses Tavily for web search, then fetches pages and extracts readable content. This layer records fetch success, failure, domains, titles, word counts, and opened URLs.
- **Context Builder:** Scores pages by relevance, search score, recency, keyword density, and domain diversity. It aggressively prunes low-value snippets, penalizes duplicate domains, and ensures the citation map includes only snippets that survived final pruning and were actually placed in the context window.
- **Orchestrator:** Coordinates the full generator-based pipeline. It yields structured streaming steps to Streamlit, including planning, search, fetch results, context readiness, retry events, conflict warnings, answer tokens, and completion metadata.
- **Streamlit UI:** Renders a commercial-style Agent Workflow log, research plan expander, session history, detailed session modal, sources, and turn details.
- **SQLite Session Store:** Persists sessions, message history, turn metadata, search queries, URLs opened, snippets selected, final answers, and context breakdowns.
- **Translation Sandwich:** Uses the Sarvam API to support English, Hindi, and Tamil. Inputs are translated to English before planning/search, and completed English answers are translated back after generation.

### Risks, Limitations, & Future Improvements

The agent improves reliability but still has practical constraints. Some sites return 403 errors or block scraping, which can reduce source diversity. Very large or broad questions can exhaust context budgets even with pruning. Web search results may contain SEO spam, duplicated content, or pages with weak extraction quality. External APIs also introduce rate limits and latency, especially when multiple searches, LLM calls, and translation calls are involved.

Two important future improvements are:

1. **Iterative ReAct Retry Loop:** Expand the current retry mechanism into a multi-iteration ReAct loop that can inspect gaps, refine queries, fetch new evidence, and continue until confidence thresholds are met or a clear refusal is warranted.
2. **Local Vector Database:** Replace purely lexical snippet scoring with a local embedding index or vector database. This would improve semantic retrieval, reduce noise, and make repeated research over saved sessions faster and more accurate.

---

## 2. Evaluation Methodology & Harness Results

### The Methodology

The project includes a custom `eval/eval_harness.py` script that runs the full agent pipeline against an eight-question dataset in `eval/dataset.json`. The dataset is intentionally small but adversarial. It targets the most important edge cases for a deep research agent:

- factual lookup,
- multi-hop reasoning,
- comparison across entities,
- insufficient evidence,
- conflicting sources,
- multi-turn memory,
- strict citation formatting.

The harness instantiates the actual project components: `SessionStore`, `LLMClient`, `Planner`, `SearchClient`, `PageFetcher`, `ContextBuilder`, and `Orchestrator`. It consumes the orchestrator generator step by step, accumulating streamed `token` chunks into the final answer and extracting the `done` metadata, including `citation_map` and `pages_used`. Multi-turn evaluation reuses the same `session_id` so that a follow-up such as "that movie" can be resolved through session memory.

### The Metrics Explained

The harness computes metrics programmatically for every test case:

- **Citation Integrity:** Checks whether the final answer contains bracketed citation markers such as `[1]` or `[2]`. For successful insufficient-evidence refusals, the harness does not penalize missing citations because the correct behavior is not to cite a fabricated factual answer.
- **Source Diversity:** Counts unique source domains in the citation map. This helps measure whether the agent gathered evidence from multiple independent places.
- **Uncertainty Handling:** For insufficient-evidence questions, checks whether the answer includes refusal language such as "cannot", "insufficient", or "no information".
- **Conflict Flagging:** For conflict questions, checks whether the orchestrator yielded a conflict step or whether the answer contains "disagree" or "conflict".

### Harness Result Snapshot

| Evaluation Type | What It Tests | Expected Behavior |
|---|---|---|
| Factual | Exact known answer with sources | Provide a concise answer with bracketed citations. |
| Multi-hop | Entity linking plus prior role lookup | Break question into sub-queries and synthesize across sources. |
| Comparison | Two-model context window comparison | Cite each side and compare directly. |
| Insufficient Evidence | Fictional Atlantis GDP | Refuse with clear insufficient-evidence language. |
| Conflict | Coffee and blood pressure | Surface disagreement or nuance across sources. |
| Multi-turn | Inception director then budget | Reuse the same session context for follow-up resolution. |

### Findings Summary

The agent successfully demonstrates multi-turn memory and multi-hop reasoning by preserving session context and using planned sub-queries to resolve follow-up questions. The evaluation harness also exposed useful failure modes: citation formatting can degrade when prompts are too loose, insufficient-evidence cases require explicit refusal language, and conflict detection depends on retrieving enough independent domains.

Final hardening focused on three areas. First, the system prompt was tightened so factual sentences must end with bracketed citations. Second, the Context Builder was upgraded with domain-diversity penalties and a rescue path to avoid one-domain context. Third, the evaluation harness was corrected so a valid refusal is not unfairly penalized for missing citations. These changes align the agent with the intended research behavior: answer when evidence is sufficient, refuse when it is not, and call out conflicts when sources disagree.

---

## 3. Setup Instructions, Links, & Examples

### Links

- GitHub Repository: [Insert GitHub Link]
- Video Demo: [Insert Video Demo Link]

### Setup & Run Instructions

1. Clone the repository:

```bash
git clone [Insert GitHub Link]
cd deep_research_agent
```

2. Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure the `.env` file:

```env
TAVILY_API_KEY=your_tavily_key_here
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
SARVAM_API_KEY=your_sarvam_key_here
```

5. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

If the project is reorganized under an `app/` directory, use:

```bash
streamlit run app/streamlit_app.py
```

6. Run the evaluation harness manually:

```bash
python eval/eval_harness.py
```

The harness writes full results to:

```text
eval/results.json
```

### Example Conversations

**Example 1: Factual query with citations**

User: "What organization maintains the Unicode Standard?"

Agent: "The Unicode Standard is maintained by the Unicode Consortium [1]. The consortium coordinates standardization work for text encoding across software systems [1]."

**Example 2: Impossible query with insufficient-evidence refusal**

User: "What is the exact GDP of the fictional underwater city of Atlantis in 2024?"

Agent: "I cannot find sufficient evidence to answer this. Atlantis is fictional, and the retrieved sources do not provide a reliable real-world GDP figure. A refined search could look for fictional portrayals of Atlantis economics or worldbuilding estimates."
