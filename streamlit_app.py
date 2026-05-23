"""
streamlit_app.py
Deep Research Agent — Streamlit UI with streaming progress updates.
"""

import re
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.dialog("Detailed Session History", width="large")
def show_session_details(session_id: str, store):
    try:
        turns = store.get_turns(session_id)
    except Exception:
        turns = []
    if not turns:
        st.info("No detailed turns found for this session.")
        return
    for i, turn in enumerate(turns, start=1):
        query = turn.get("query") or "Unknown Query"
        st.markdown(f"### Turn {i}: {query}")
        st.markdown("**🔍 Search Queries Issued:**")
        st.json(turn.get("search_queries") or [])
        st.markdown("**🌐 URLs Opened:**")
        st.json(turn.get("urls_opened") or [])
        st.markdown("**✂️ Context Snippets Selected:**")
        st.json(turn.get("snippets_selected") or [])
        st.markdown("**✅ Final Answer:**")
        st.info(turn.get("final_answer") or "No answer recorded.")
        st.divider()


# ------------------------------------------------------------------
# Component initialization (cached for app lifetime)
# ------------------------------------------------------------------
@st.cache_resource
def init_components():
    from session.store import SessionStore
    from llm.client import LLMClient
    from llm.translator import SarvamTranslator
    from agent.planner import Planner
    from agent.context_builder import ContextBuilder
    from agent.orchestrator import Orchestrator
    from search.search_client import SearchClient
    from search.fetcher import PageFetcher

    store = SessionStore()
    llm_client = LLMClient()
    translator = SarvamTranslator()
    planner = Planner(llm_client)
    search_client = SearchClient()
    fetcher = PageFetcher()
    ctx_builder = ContextBuilder(llm_client=llm_client)
    orchestrator = Orchestrator(store, llm_client, planner, search_client, fetcher, ctx_builder)
    return store, orchestrator, ctx_builder, translator


store, orchestrator, ctx_builder, translator = init_components()


def _citation_source_markdown(citation_map: dict, selected_language: str) -> str:
    citation_map = _normalize_citation_map(citation_map)
    if not citation_map:
        return ""
    headings = {
        "Hindi": "#### स्रोत वेबसाइटें",
        "Tamil": "#### மூல இணையதளங்கள்",
    }
    heading = headings.get(selected_language, "#### Source Websites")
    lines = ["", heading]
    for idx, cite in citation_map.items():
        try:
            title = cite.get("title") or "Untitled"
            domain = cite.get("domain") or "source"
            url = cite.get("url") or ""
            if url:
                lines.append(f"- **[{idx}]** [{title} - {domain}]({url})")
            else:
                lines.append(f"- **[{idx}]** {title} - {domain}")
        except Exception:
            lines.append(f"- **[{idx}]** Source unavailable")
    return "\n".join(lines)


def _normalize_citation_map(citation_map: dict | None) -> dict[str, dict]:
    normalized = {}
    if not isinstance(citation_map, dict):
        return normalized

    for idx, cite in citation_map.items():
        if not isinstance(cite, dict):
            continue
        key = str(idx)
        url = cite.get("url") or ""
        normalized[key] = {
            "title": cite.get("title") or f"Source {key}",
            "domain": cite.get("domain") or (urlparse(url).netloc if url else "source"),
            "url": url,
        }
    return normalized


def _citation_map_from_turn(turn: dict) -> dict:
    citations = {}
    try:
        snippets = turn.get("snippets_selected") or []
        for pos, snip in enumerate(snippets, start=1):
            if not isinstance(snip, dict):
                continue
            idx = str(snip.get("index") or pos)
            url = snip.get("url") or ""
            domain = snip.get("domain") or (urlparse(url).netloc if url else "source")
            citations[idx] = {
                "title": snip.get("title") or f"Source {idx}",
                "domain": domain or "source",
                "url": url,
            }

        if citations:
            return citations

        for pos, url in enumerate(turn.get("urls_opened") or [], start=1):
            if not isinstance(url, str) or not url.strip():
                continue
            citations[str(pos)] = {
                "title": f"Source {pos}",
                "domain": urlparse(url).netloc or "source",
                "url": url,
            }
    except Exception:
        return {}
    return citations


def _linkify_inline_citations(answer: str, citation_map: dict) -> str:
    citation_map = _normalize_citation_map(citation_map)
    if not answer or not citation_map:
        return answer

    def repl(match):
        idx = match.group(1)
        cite = citation_map.get(idx)
        url = cite.get("url") if isinstance(cite, dict) else None
        return f"[[{idx}]]({url})" if url else match.group(0)

    try:
        return re.sub(r"\[(\d+)\]", repl, answer)
    except Exception:
        return answer


def _render_answer_with_sources(answer: str, citation_map: dict, selected_language: str) -> str:
    rendered = _linkify_inline_citations(answer, citation_map)
    rendered += _citation_source_markdown(citation_map, selected_language)
    return rendered


def _has_source_urls(citation_map: dict) -> bool:
    return any(bool(cite.get("url")) for cite in _normalize_citation_map(citation_map).values())

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stMarkdownContainer"],
[data-testid="stChatMessageContent"],
textarea,
input {
    font-family: "Nirmala UI", "Mangal", "Latha", "Arial Unicode MS", "Segoe UI", sans-serif;
}
[data-testid="stExpander"] summary p {
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.25;
}
.source-block { background: #1e293b; border-radius: 8px; padding: 12px; margin-top: 8px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar — session management
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Deep Research")
    lang_codes = {"English": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN"}
    selected_language = st.selectbox(
        "🌐 Response Language",
        ["English", "Hindi", "Tamil"],
        index=0,
    )
    target_code = lang_codes.get(selected_language, "en-IN")
    st.markdown("---")

    if st.button("➕ New Session", use_container_width=True, type="primary"):
        new_sid = store.create_session()
        st.session_state["session_id"] = new_sid
        st.rerun()

    st.markdown("### 📜 Past Research History")
    sessions = store.list_sessions()
    if sessions:
        for session in sessions:
            session_id_value = session.get("session_id")
            if not session_id_value:
                continue
            display_id = session.get("display_id") or session.get("friendly_name") or session_id_value[:8]
            first_query = (session.get("first_query") or "New Session").strip() or "New Session"
            last_query = (session.get("last_query") or "No questions asked yet").strip() or "No questions asked yet"
            date_formatted = (session.get("updated_at") or "").replace("T", " ")
            date_formatted = date_formatted[:16] if date_formatted else "Unknown"
            short_display_id = display_id[:18]
            label_query = f"{first_query[:18]}..." if len(first_query) > 18 else first_query

            with st.expander(f"👉 {short_display_id} | {date_formatted} | {label_query}", expanded=False):
                if st.button(
                    "🔎 View Full Details",
                    key=f"details_{session_id_value}",
                    use_container_width=True,
                ):
                    show_session_details(session_id_value, store)
                if st.button(
                    "📂 Load this Research Session",
                    key=f"load_{session_id_value}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["session_id"] = session_id_value
                    st.rerun()
    else:
        st.caption("No sessions yet. Create one above.")

    if "session_id" in st.session_state:
        st.markdown("---")
        active = next((s for s in sessions if s.get("session_id") == st.session_state["session_id"]), {})
        st.caption(f"Active: `{active.get('display_id') or active.get('friendly_name') or 'Current Session'}`")

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
session_id = st.session_state.get("session_id")

if not session_id:
    st.markdown("## 🔬 Deep Research Agent")
    st.info("👈 Create or select a session from the sidebar to begin researching.")
    st.markdown("""
    **This agent will:**
    - 🎯 Plan targeted search queries from your question
    - 🔍 Search the web via Tavily
    - 📄 Read full source pages
    - 🧠 Select the most relevant content
    - 📝 Generate a cited answer with source links
    """)
    st.stop()

st.markdown(f"### 🔬 Research Session `{session_id[:8]}...`")

# Display conversation history
history = store.get_history(session_id)
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------
query = st.chat_input("Ask anything to research...")

if query:
    english_query = query
    if selected_language != "English":
        english_query = translator.translate(query, source_lang=target_code, target_lang="en-IN")

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        answer_buffer = ""
        displayed_answer = ""
        answer_placeholder = st.empty()
        plan_placeholder = st.empty()
        citation_map = {}
        fetched_source_map = {}
        last_turn_id = None

        try:
            with st.status("⚙️ Agent Workflow", expanded=True) as status:
                if selected_language != "English" and english_query != query:
                    status.write("🌐 **[Translate]** Translated your question to English for research quality.")

                for step in orchestrator.run(english_query, session_id, stream=True):
                    s = step["step"]

                    if s == "planning":
                        status.write("🧠 **[Planner]** Analyzing question and planning research strategy...")

                    elif s == "planned":
                        with plan_placeholder.container():
                            with st.expander("📋 Research Plan", expanded=False):
                                st.markdown("**Thought Process & Strategy:**")
                                st.info(step.get("message", ""))
                                st.markdown("**Planned Search Queries:**")
                                for q in step.get("queries", []):
                                    st.markdown(f"- `{q}`")

                    elif s == "searching":
                        status.write(f"🔍 **[Search]** Searching the web for: *\"{step.get('message', '')}\"*")

                    elif s == "retry":
                        status.write(f"🔁 **[ReAct]** {step.get('message', 'Refining search and retrying retrieval...')}")

                    elif s == "fetching":
                        status.write(f"📥 **[Fetcher]** Fetching content from {step.get('n', 'multiple')} web pages...")

                    elif s == "fetch_result":
                        source_url = step.get("url") or ""
                        if source_url and step.get("success"):
                            source_idx = str(len(fetched_source_map) + 1)
                            fetched_source_map[source_idx] = {
                                "title": step.get("title") or f"Source {source_idx}",
                                "domain": step.get("domain") or urlparse(source_url).netloc or "source",
                                "url": source_url,
                            }
                        icon = "✅" if step.get("success") else "⚠️"
                        fetch_message = step.get("message") or (
                            f"{step.get('domain', '')} — {step.get('title', '')} "
                            f"({step.get('word_count', 0)} words)"
                        )
                        status.write(f"{icon} **[Fetcher]** Read {fetch_message}")

                    elif s == "selecting":
                        status.write(f"🎯 **[Selector]** {step.get('message', 'Selecting relevant context...')}")

                    elif s == "context_ready":
                        context_citations = step.get("citation_map") or {}
                        if _has_source_urls(context_citations):
                            citation_map = context_citations
                        breakdown = step.get("context_breakdown", {})
                        summary = ctx_builder.format_context_breakdown(breakdown)
                        with status.expander("📊 Context window usage", expanded=False):
                            st.code(summary, language=None)
                        status.write("🧩 **[Context]** Semantic chunks generated and context window built.")

                    elif s == "conflict":
                        status.write(f"🛡️ **[Analysis]** Conflicting claims detected: {step.get('message', '')}")

                    elif s == "answering":
                        status.update(label="⚙️ Agent Workflow (Complete)", state="complete")
                        status.write("💬 **[LLM]** Synthesizing grounded answer from retrieved evidence...")

                    elif s == "token":
                        answer_buffer += step["text"]
                        answer_placeholder.markdown(answer_buffer + "▌")

                    elif s == "done":
                        status.update(label="⚙️ Agent Workflow (Complete)", state="complete")
                        done_citations = step.get("citation_map") or {}
                        if _has_source_urls(done_citations):
                            citation_map = done_citations
                        elif not _has_source_urls(citation_map):
                            citation_map = fetched_source_map
                        last_turn_id = step.get("turn_id")
                        if selected_language != "English":
                            translated_answer = translator.translate(
                                answer_buffer,
                                source_lang="en-IN",
                                target_lang=target_code,
                            )
                            if translated_answer == answer_buffer:
                                st.warning(
                                    "Translation could not be rendered safely, so the English answer is shown."
                                )
                            displayed_answer = translated_answer
                        else:
                            displayed_answer = answer_buffer
                        answer_placeholder.markdown(
                            _render_answer_with_sources(displayed_answer, citation_map, selected_language)
                        )

            if last_turn_id:
                turn = store.get_turn_detail(last_turn_id)
                if turn:
                    if not _has_source_urls(citation_map):
                        fallback_citation_map = _citation_map_from_turn(turn)
                        if fallback_citation_map:
                            citation_map = fallback_citation_map
                        answer_placeholder.markdown(
                            _render_answer_with_sources(displayed_answer, citation_map, selected_language)
                        )
                    with st.expander("🔎 Turn details", expanded=False):
                        st.markdown(f"**Query:** {turn['query']}")
                        st.markdown(f"**Timestamp:** {turn['timestamp']}")
                        st.markdown("**Search queries:**")
                        for sq in turn.get("search_queries") or []:
                            st.markdown(f"- {sq}")
                        st.markdown(f"**URLs opened ({len(turn.get('urls_opened') or [])}):**")
                        for url in turn.get("urls_opened") or []:
                            st.markdown(f"- {url}")
                        st.markdown("**Snippets selected:**")
                        for snip in turn.get("snippets_selected") or []:
                            if isinstance(snip, dict):
                                st.markdown(
                                    f"- [{snip.get('index', '?')}] {snip.get('title', '')} "
                                    f"({snip.get('domain', '')})"
                                )
                            else:
                                st.markdown(f"- {snip}")
                        if turn.get("context_breakdown"):
                            st.markdown("**Context breakdown:**")
                            st.code(
                                ctx_builder.format_context_breakdown(turn["context_breakdown"]),
                                language=None,
                            )

            # Insufficient evidence detection
            low_evidence_signals = [
                "cannot find", "insufficient", "unable to find",
                "no information available", "not enough information",
            ]
            if any(signal in answer_buffer.lower() for signal in low_evidence_signals):
                st.info(
                    "💡 The agent could not find strong evidence. "
                    "Try rephrasing your question or asking about a more specific aspect."
                )

        except Exception as e:
            st.error(f"❌ Research failed: {e}")
            raise
