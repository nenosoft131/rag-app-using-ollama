import sqlite3
from threading import Lock
from datetime import datetime
from collections import deque
from typing import List, Dict, Any

DB_PATH = "chatbot.db"


class AppStats:
    def __init__(self):
        self._lock = Lock()
        self._init_db()
        self._load_counters()
        self.recent_logs: deque = deque(maxlen=100)

    # ── DB helpers ─────────────────────────────────────────────────────────────

    def _conn(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stats_counters (
                    key TEXT PRIMARY KEY, value INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recent_queries (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        TEXT NOT NULL,
                    query     TEXT,
                    route     TEXT,
                    model     TEXT,
                    retries   INTEGER DEFAULT 0,
                    latency_ms INTEGER DEFAULT 0,
                    success   INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY, value TEXT
                )
            """)
            for key in ("total_queries", "retrieval_route", "general_route",
                        "grader_retries", "hallucination_retries", "errors", "docs_uploaded"):
                conn.execute(
                    "INSERT OR IGNORE INTO stats_counters (key, value) VALUES (?, 0)", (key,)
                )
            conn.execute(
                "INSERT OR IGNORE INTO app_meta (key, value) VALUES ('started_at', ?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
            )
            conn.commit()

    def _load_counters(self):
        with self._conn() as conn:
            for row in conn.execute("SELECT key, value FROM stats_counters"):
                setattr(self, row["key"], row["value"])
        for field in ("total_queries", "retrieval_route", "general_route",
                      "grader_retries", "hallucination_retries", "errors", "docs_uploaded"):
            if not hasattr(self, field):
                setattr(self, field, 0)

    # ── Write operations ───────────────────────────────────────────────────────

    def inc(self, field: str, amount: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field, 0) + amount)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO stats_counters (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = value + ?",
                (field, amount, amount),
            )
            conn.commit()

    def add_log(self, line: str):
        with self._lock:
            self.recent_logs.append({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "msg": line,
            })

    def add_query(self, query: str, route: str, model: str,
                  retries: int, latency_ms: int, success: bool):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO recent_queries (ts, query, route, model, retries, latency_ms, success) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), query[:200], route, model,
                 retries, latency_ms, 1 if success else 0),
            )
            conn.execute(
                "DELETE FROM recent_queries WHERE id NOT IN "
                "(SELECT id FROM recent_queries ORDER BY id DESC LIMIT 200)"
            )
            conn.commit()

    # ── Read operations ────────────────────────────────────────────────────────

    def get_recent_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, query, route, model, retries, latency_ms, success "
                "FROM recent_queries ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_started_at(self) -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM app_meta WHERE key = 'started_at'"
            ).fetchone()
        return row["value"] if row else "unknown"

    def snapshot(self) -> dict:
        with self._lock:
            total = max(self.total_queries, 1)
            data = {
                "total_queries":         self.total_queries,
                "retrieval_route":       self.retrieval_route,
                "general_route":         self.general_route,
                "grader_retries":        self.grader_retries,
                "hallucination_retries": self.hallucination_retries,
                "errors":                self.errors,
                "docs_uploaded":         self.docs_uploaded,
                "retrieval_pct":         round(self.retrieval_route / total * 100, 1),
                "general_pct":           round(self.general_route   / total * 100, 1),
                "grader_retry_pct":      round(self.grader_retries  / total * 100, 1),
                "halluc_retry_pct":      round(self.hallucination_retries / total * 100, 1),
                "success_rate":          round((self.total_queries - self.errors) / total * 100, 1),
                "recent_logs":           list(self.recent_logs),
            }

        data["recent_queries"] = self.get_recent_queries(50)
        data["started_at"] = self.get_started_at()

        queries = data["recent_queries"]
        data["avg_latency_s"] = (
            round(sum(q["latency_ms"] for q in queries) / len(queries) / 1000, 1)
            if queries else 0
        )
        return data


# Singleton shared across the app
stats = AppStats()
