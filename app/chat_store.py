"""会话历史存储：SQLite（Python 标准库，零依赖）。

表结构刻意保持与 MySQL 兼容，方便你后面换成 MySQL 时说一句"换过存储层"。
连接统一走 app/db.py：那里开了 WAL，读写不互相阻塞 ——
改造后 app.db 里同时有用量、缓存、文档台账三张表，并发写比原来多得多。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .db import connect

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatStore:
    def __init__(self, db_path: Path) -> None:
        self.path = db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return connect(self.path)

    def add(self, session_id: str, role: str, content: str, citations: list | None = None) -> None:
        payload = json.dumps(citations, ensure_ascii=False) if citations else None
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, citations, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, payload, _now()),
            )

    def history(self, session_id: str, max_turns: int = 5) -> list[dict]:
        """取最近 N 轮对话（一轮 = 一问一答），喂给模型做多轮记忆。"""
        limit = max(1, max_turns) * 2
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def messages(self, session_id: str, limit: int = 200) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self, session_id: str) -> int:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            return cursor.rowcount
