import sqlite3
from datetime import datetime
from typing import List, Dict

DB_PATH = "chatbot.db"


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                ts         TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id)")
        conn.commit()


_init()


def save_message(session_id: str, role: str, content: str):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().isoformat()),
        )
        conn.commit()


def get_history(session_id: str) -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def get_all_sessions() -> List[str]:
    """Return all session IDs that have at least one message, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM chat_messages ORDER BY id DESC"
        ).fetchall()
    return [r["session_id"] for r in rows]
