# Deep Research Agent: Final Submission
Pure-Python research orchestration with Streamlit, SQLite, Tavily, tiktoken pruning, strict citations, evaluation harness, and Sarvam multilingual support.
## Index
1. Design Note (Architecture & Strategy) ( page 1 and 2 )
1. Evaluation Methodology & Harness Results ( page 3,  4 and 5 )
1. Setup Instructions, Git Hub and Video demo Links & Examples ( page 5 and 6 )
# 1. Design Note
## Target Users & Problem Solved
The Deep Research Agent is built for researchers, engineers, students, and technical decision-makers who need answers that are current, inspectable, and grounded in source material. Standard large language models are useful for synthesis, but they can hallucinate facts, rely on stale training data, and present fluent answers without showing where claims came from. That is a poor fit for research workflows where the user must verify evidence, compare sources, and understand uncertainty.
This project solves that problem by turning a normal question into a deterministic research workflow. The agent searches the live web, fetches pages, extracts useful text, prunes noisy material, and only then asks the LLM to synthesize an answer. Every factual claim is expected to use bracketed citations such as [1] and [2], which map directly to retrieved sources in the UI. The result is not just an answer, but a transparent research artifact: search queries, opened URLs, selected snippets, citations, and final response are all persisted in SQLite session history.
## Definition of "Deep Research"
"Deep Research" means a deterministic multi-step orchestration pipeline rather than a single prompt. The agent does not simply ask an LLM to answer from memory. It executes a structured process:
1. Plan targeted search queries from the user question.
1. Search the web using Tavily.
1. Fetch and extract readable HTML content using a pure-Python fetch layer with trafilatura-style extraction.
1. Score, prune, and diversify evidence under a token budget using tiktoken.
1. Synthesize a concise cited answer from only the selected web context.
1. Persist the full session, turn metadata, and source details in SQLite.
This design keeps the Planner, Search, Fetch, and Context Builder nodes in English to preserve retrieval quality and reasoning consistency. Multilingual support is added with a Translation Sandwich: user input in Hindi or Tamil is translated into English with Sarvam, the research pipeline runs in English, and the final answer is translated back to the selected language.
## Success Metrics
- Citation Integrity Rate: Whether the final answer uses strict bracketed citations, such as [1] and [2], for factual claims.
- Source Diversity Score: The number of unique domains represented in the citation map, reducing over-reliance on a single source.
- Uncertainty Handling: Whether the agent refuses or qualifies answers when retrieved evidence is insufficient.
- Conflict Resolution Accuracy: Whether the agent detects and explains disagreement across sources with objective summaries of both sides.
## Data Flow and Components
The system is built completely from scratch in pure Python, without LangChain or similar orchestration frameworks. The key components are:
- Planner Node: Converts the user question and optional session history into up to three targeted search queries, with explicit decomposition for multi-hop and comparison tasks.
- Search/Fetch Layer: Uses Tavily for web search, then fetches pages and extracts readable content while recording fetch success, failure, domains, titles, word counts, and opened URLs.
- Context Builder: Scores pages by relevance, search score, recency, keyword density, and domain diversity, then prunes low-value snippets under a token budget.
- Orchestrator: Coordinates the generator-based pipeline and yields structured streaming steps to the Streamlit UI.
- Streamlit UI: Renders the Agent Workflow log, research plan expander, session history, detailed session modal, sources, and turn details.
- SQLite Session Store: Persists sessions, message history, turn metadata, search queries, URLs opened, snippets selected, final answers, and context breakdowns.
- Translation Sandwich: Uses Sarvam API support for English, Hindi, and Tamil while keeping core retrieval and reasoning in English.
## Risks, Limitations, & Future Improvements
The agent improves reliability but still has practical constraints. Some sites return 403 errors or block scraping, which can reduce source diversity. Very large or broad questions can exhaust context budgets even with pruning. Web search results may contain SEO spam, duplicated content, or pages with weak extraction quality. External APIs also introduce rate limits and latency, especially when multiple searches, LLM calls, and translation calls are involved.

Some improvements for future
1. Multi-Iteration ReAct Loop : The current system includes a single-pass retry mechanism when evidence is weak. A future improvement would expand this into a true multi-iteration ReAct loop that repeatedly inspects evidence gaps, generates targeted follow-up queries, fetches new sources, and stops only when confidence thresholds are met or a clear refusal is warranted.
1. Local Vector Database: Replace purely lexical snippet scoring with a local embedding index or vector database for better semantic retrieval and faster repeated research over saved sessions.
# 2. Evaluation Methodology & Harness Results
## The Methodology
The project includes a custom eval/eval_harness.py script that runs the full agent pipeline against an eight-question dataset in eval/dataset.json. The dataset is intentionally small but adversarial. It targets factual lookup, multi-hop reasoning, comparison across entities, insufficient evidence, conflicting sources, multi-turn memory, and strict citation formatting.
The harness instantiates the actual project components: SessionStore, LLMClient, Planner, SearchClient, PageFetcher, ContextBuilder, and Orchestrator. It consumes the orchestrator generator step by step, accumulating streamed token chunks into the final answer and extracting done metadata, including citation_map and pages_used. Multi-turn evaluation reuses the same session_id so that a follow-up such as 'that movie' can be resolved through session memory.
## The Metrics Explained
### The Evaluation Metrics
The harness evaluates the agent’s output based on four key performance indicators (KPIs):
Citation Integrity: A regex-based validator ensures the LLM provides bracketed citations (e.g., [1]) for every factual claim. If the agent correctly refuses an impossible question, it is exempted from this check.
Source Diversity (Domains): The harness counts the number of unique top-level domains cited. A score of >= 2 indicates the agent successfully cross-referenced information, preventing single-source bias.
Uncertainty Handling: A keyword-density check ensures that for impossible prompts (e.g., "GDP of Atlantis"), the agent triggers a refusal phrase rather than hallucinating.
Conflict Resolution: For comparative or debated queries, the harness verifies that the agent retrieved sufficient data to identify and present opposing viewpoints.

## Harness Result Snapshot

## Architectural Hardening
The initial testing phase revealed that LLMs are prone to "laziness"—often skipping citations or aggregating from only a single domain. To rectify this, I implemented:
Few-Shot Citation Forcing: Added explicit "correct vs. incorrect" examples to the system prompt to force the [X] format.
Structural Domain Forcing: Updated the context_builder.py to programmatically reject any context that does not contain at least 2 distinct source domains, ensuring balanced research.
Evaluation Bypass Logic: Added a hardcoded conditional in the harness to prevent false negatives on impossible queries (recognizing that refusal is a successful "pass" for uncertainty handling).
These architectural patches transformed the agent from a basic search wrapper into a highly reliable, deterministic research engine.

## Findings Summary
The agent successfully demonstrates multi-turn memory and multi-hop reasoning by preserving session context and using planned sub-queries to resolve follow-up questions. The evaluation harness also exposed useful failure modes: citation formatting can degrade when prompts are too loose, insufficient-evidence cases require explicit refusal language, and conflict detection depends on retrieving enough independent domains.
Final hardening focused on three areas. First, the system prompt was tightened so factual sentences must end with bracketed citations. Second, the Context Builder was upgraded with domain-diversity penalties and a rescue path to avoid one-domain context. Third, the evaluation harness was corrected so a valid refusal is not unfairly penalized for missing citations. These changes align the agent with the intended research behavior: answer when evidence is sufficient, refuse when it is not, and call out conflicts when sources disagree.
# 3. Setup Instructions, Links, & Examples
## Links
- GitHub Repository: [Link]
- Video Demo: [Link]
## Setup & Run Instructions
1. Clone the repository.
1. Create and activate a Python environment.
1. Install dependencies.
1. Configure the .env file.
1. Run the Streamlit app.
1. Run the evaluation harness manually.
git clone [Link]
cd deep_research_agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

TAVILY_API_KEY=your_tavily_key_here
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
SARVAM_API_KEY=your_sarvam_key_here

streamlit run streamlit_app.py
# If reorganized under app/:
streamlit run app/streamlit_app.py

python eval/eval_harness.py
# Results: eval/results.json
## Example Conversations
### Example 1: Factual query with citations
User: "What organization maintains the Unicode Standard?"
Agent: "The Unicode Standard is maintained by the Unicode Consortium [1]. The consortium coordinates standardization work for text encoding across software systems [1]."
### Example 2: Impossible query with insufficient-evidence refusal
User: "What is the exact GDP of the fictional underwater city of Atlantis in 2024?"
Agent: "I cannot find sufficient evidence to answer this. Atlantis is fictional, and the retrieved sources do not provide a reliable real-world GDP figure. A refined search could look for fictional portrayals of Atlantis economics or worldbuilding estimates."
