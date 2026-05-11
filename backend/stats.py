from threading import Lock
from datetime import datetime
from collections import deque


class AppStats:
    def __init__(self):
        self._lock = Lock()
        self.total_queries = 0
        self.retrieval_route = 0
        self.general_route = 0
        self.grader_retries = 0
        self.hallucination_retries = 0
        self.errors = 0
        self.docs_uploaded = 0
        self.recent_logs: deque = deque(maxlen=50)  # last 50 log lines

    def inc(self, field: str, amount: int = 1):
        with self._lock:
            setattr(self, field, getattr(self, field) + amount)

    def add_log(self, line: str):
        with self._lock:
            self.recent_logs.append({
                "ts": datetime.now().strftime("%H:%M:%S"),
                "msg": line,
            })

    def snapshot(self) -> dict:
        with self._lock:
            total = self.total_queries or 1  # avoid div-by-zero
            return {
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
                "recent_logs":           list(self.recent_logs),
            }


# Singleton shared across the app
stats = AppStats()
