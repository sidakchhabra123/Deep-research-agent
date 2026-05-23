"""
session/store.py
SQLite-backed session store for the Deep Research Agent.
Handles sessions, conversation messages, and per-turn research metadata.
"""

import sqlite3
import json
import threading
import uuid
from datetime import datetime


DB_PATH = "research.db"


class SessionStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row  # allows dict-like access
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _create_tables(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id  TEXT PRIMARY KEY,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    summary     TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id          TEXT NOT NULL,
                    query               TEXT NOT NULL,
                    search_queries      TEXT NOT NULL,
                    urls_opened         TEXT NOT NULL,
                    snippets_selected   TEXT NOT NULL,
                    final_answer        TEXT NOT NULL,
                    timestamp           TEXT NOT NULL,
                    context_breakdown   TEXT
                );
            """)
            self._conn.commit()
        self._migrate_turns_schema()

    def _migrate_turns_schema(self):
        """Upgrade legacy turns columns without losing existing rows."""
        with self._lock:
            cur = self._conn.execute("PRAGMA table_info(turns)")
            cols = {row[1] for row in cur.fetchall()}
            if not cols:
                return
            if "urls_fetched" in cols and "urls_opened" not in cols:
                self._conn.execute(
                    "ALTER TABLE turns RENAME COLUMN urls_fetched TO urls_opened"
                )
            if "snippets_used" in cols and "snippets_selected" not in cols:
                self._conn.execute(
                    "ALTER TABLE turns RENAME COLUMN snippets_used TO snippets_selected"
                )
            cur = self._conn.execute("PRAGMA table_info(turns)")
            cols = {row[1] for row in cur.fetchall()}
            if "context_breakdown" not in cols:
                self._conn.execute(
                    "ALTER TABLE turns ADD COLUMN context_breakdown TEXT"
                )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def create_session(self) -> str:
        """Generate a new UUID session, persist it, and return session_id."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, summary) VALUES (?, ?, ?, ?)",
                (session_id, now, now, None),
            )
            self._conn.commit()
        return session_id

    def list_sessions(self) -> list[dict]:
        """Return all sessions with their first and last research questions."""
        try:
            cur = self._conn.execute(
                """
                WITH turn_bounds AS (
                    SELECT
                        session_id,
                        MIN(id) AS first_turn_id,
                        MAX(id) AS last_turn_id
                    FROM turns
                    GROUP BY session_id
                )
                SELECT
                    s.session_id,
                    s.created_at,
                    s.updated_at,
                    first_turn.query AS first_query,
                    last_turn.query AS last_query
                FROM sessions s
                LEFT JOIN turn_bounds tb ON tb.session_id = s.session_id
                LEFT JOIN turns first_turn ON first_turn.id = tb.first_turn_id
                LEFT JOIN turns last_turn ON last_turn.id = tb.last_turn_id
                """
            )
            sessions = []
            rows = sorted(
                [dict(row) for row in cur.fetchall()],
                key=lambda item: item.get("created_at") or "",
            )
            for index, session in enumerate(rows, start=1):
                session["friendly_name"] = f"Session_{index}"
                session["display_id"] = f"Session_{index}-{session.get('session_id', '')[:8]}"
                session["first_query"] = (
                    session.get("first_query") or "Empty Session"
                ).strip() or "Empty Session"
                session["last_query"] = (
                    session.get("last_query") or "No questions asked yet"
                ).strip() or "No questions asked yet"
                sessions.append(session)
            sessions.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
            return sessions
        except sqlite3.Error:
            return []

    def update_summary(self, session_id: str, summary: str) -> None:
        """Persist a rolling summary string for the session."""
        now = datetime.now().isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, now, session_id),
            )
            self._conn.commit()

    def get_summary(self, session_id: str) -> str | None:
        cur = self._conn.execute(
            "SELECT summary FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = cur.fetchone()
        return row["summary"] if row else None

    # ------------------------------------------------------------------
    # Message history
    # ------------------------------------------------------------------
    def add_message(self, session_id: str, role: str, content: str) -> None:
        now = datetime.now().isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            # bump updated_at on the parent session
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            self._conn.commit()

    def get_history(self, session_id: str) -> list[dict]:
        """Return all messages for a session ordered chronologically."""
        cur = self._conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Turn history
    # ------------------------------------------------------------------
    def add_turn(
        self,
        session_id: str,
        query: str,
        search_queries: list,
        urls_opened: list,
        snippets_selected: list,
        final_answer: str,
        context_breakdown: dict | None = None,
    ) -> int:
        """Persist a single research turn with all intermediate artifacts. Returns turn id."""
        now = datetime.now().isoformat()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO turns
                   (session_id, query, search_queries, urls_opened, snippets_selected,
                    final_answer, timestamp, context_breakdown)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    query,
                    json.dumps(search_queries),
                    json.dumps(urls_opened),
                    json.dumps(snippets_selected),
                    final_answer,
                    now,
                    json.dumps(context_breakdown) if context_breakdown is not None else None,
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            self._conn.commit()
            return cur.lastrowid

    @staticmethod
    def _parse_turn_row(row: sqlite3.Row) -> dict:
        turn = dict(row)
        for field in ("search_queries", "urls_opened", "snippets_selected", "context_breakdown"):
            raw = turn.get(field)
            if raw is None:
                turn[field] = None if field == "context_breakdown" else []
            else:
                try:
                    turn[field] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    turn[field] = None if field == "context_breakdown" else []
        return turn

    def get_turns(self, session_id: str) -> list[dict]:
        """Return all turns for a session ordered chronologically."""
        try:
            cur = self._conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            )
            return [self._parse_turn_row(row) for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    def get_turn_detail(self, turn_id: int) -> dict | None:
        """Return a single turn with JSON fields parsed."""
        try:
            cur = self._conn.execute("SELECT * FROM turns WHERE id = ?", (turn_id,))
            row = cur.fetchone()
            return self._parse_turn_row(row) if row else None
        except sqlite3.Error:
            return None

    def needs_summarization(self, session_id: str, max_turns: int = 5) -> bool:
        """Return True when the session has accumulated >= max_turns turns."""
        cur = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM turns WHERE session_id = ?", (session_id,)
        )
        row = cur.fetchone()
        return row["cnt"] >= max_turns

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------
    def close(self):
        self._conn.close()


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    store = SessionStore()

    # 1. Create a session
    sid = store.create_session()
    print(f"Created session: {sid}")

    # 2. Add two messages
    store.add_message(sid, "user", "What is retrieval-augmented generation?")
    store.add_message(sid, "assistant", "RAG combines a retrieval step with LLM generation.")

    # 3. Add one turn
    turn_id = store.add_turn(
        session_id=sid,
        query="What is retrieval-augmented generation?",
        search_queries=["RAG definition", "retrieval augmented generation explained"],
        urls_opened=["https://example.com/rag", "https://arxiv.org/abs/2005.11401"],
        snippets_selected=[1, 2],
        final_answer="RAG combines a retrieval step with LLM generation. [1]",
        context_breakdown={"pages_selected": 2, "total_tokens": 120},
    )
    print(f"Turn id: {turn_id}")
    print("Turn detail:", store.get_turn_detail(turn_id))

    # 4. Retrieve and print history
    history = store.get_history(sid)
    print("\n--- Conversation History ---")
    for msg in history:
        print(f"[{msg['timestamp']}] {msg['role'].upper()}: {msg['content'][:80]}")

    # 5. List all sessions
    sessions = store.list_sessions()
    print(f"\n--- Sessions ({len(sessions)} total) ---")
    for s in sessions:
        print(s)

    # 6. Summarization flag
    print(f"\nNeeds summarization (max_turns=5): {store.needs_summarization(sid)}")

    store.close()
    print("\nStore OK")
