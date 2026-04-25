"""
server/session_manager.py — Thread-safe UUID-based session store.

Each /reset creates a new isolated DataCentricEnvironment instance
identified by a UUID session_id. Multiple concurrent clients can run
episodes without state corruption.

Sessions expire after SESSION_TTL_SECONDS (default: 30 min).
Old sessions are cleaned up on each new /reset call.
"""
import threading
import time
import uuid
from typing import Optional
from server.config import cfg
from server.logger import get_logger, log_event

logger = get_logger("session_manager")


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._total_sessions = 0
        self._total_resets = 0

    def create_session(self, env_instance) -> str:
        """Register a new environment instance, return its session_id."""
        self._cleanup_expired()

        with self._lock:
            if len(self._sessions) >= cfg.MAX_CONCURRENT_SESSIONS:
                # Evict the oldest session
                oldest_id = min(self._sessions, key=lambda k: self._sessions[k]["created_at"])
                del self._sessions[oldest_id]
                log_event(logger, "session_evicted", session_id=oldest_id, reason="max_sessions_reached")

            session_id = uuid.uuid4().hex
            self._sessions[session_id] = {
                "env": env_instance,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "step_count": 0,
            }
            self._total_sessions += 1
            self._total_resets += 1

        log_event(logger, "session_created", session_id=session_id,
                  total_active=len(self._sessions))
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return session dict or None if not found / expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if time.time() - session["created_at"] > cfg.SESSION_TTL_SECONDS:
                del self._sessions[session_id]
                log_event(logger, "session_expired", session_id=session_id)
                return None
            session["last_accessed"] = time.time()
            return session

    def get_env(self, session_id: str):
        """Return the environment for a session_id, or None."""
        session = self.get_session(session_id)
        return session["env"] if session else None

    def increment_steps(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["step_count"] += 1

    def delete_session(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def _cleanup_expired(self):
        """Remove sessions older than TTL."""
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if now - s["created_at"] > cfg.SESSION_TTL_SECONDS
            ]
            for sid in expired:
                del self._sessions[sid]
                log_event(logger, "session_cleaned_up", session_id=sid)

    def metrics(self) -> dict:
        with self._lock:
            active = len(self._sessions)
            step_counts = [s["step_count"] for s in self._sessions.values()]
        return {
            "active_sessions": active,
            "total_sessions_created": self._total_sessions,
            "total_resets": self._total_resets,
            "avg_steps_active": round(sum(step_counts) / max(len(step_counts), 1), 2),
        }


# Global singleton — imported by main.py
session_manager = SessionManager()
