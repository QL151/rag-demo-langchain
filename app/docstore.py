"""文档台账：记录每份文档的指纹、片段数、所属部门，用来支撑增量入库和权限过滤。

解决的具体问题：
以前"重新上传同一份文档"= 把整份文档重新向量化一遍。
一份 500 段的制度手册改一个错别字，就要重新调用 500 次 embedding 接口 ——
既慢又费钱，而且用户完全感知不到为什么"就改了一个字要等 3 分钟"。

有了台账 + 片段级指纹以后：
    文档指纹没变 → 整篇跳过（0 次 embedding 调用）；
    文档指纹变了 → 只对"文字变了的片段"重新向量化，其余复用索引里的旧向量。

`embedding_signature` 也记在这里：换了向量模型就必须全量重建，
否则拿新模型的向量去和旧模型的向量算相似度，结果是错的但不会报错。
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .db import session

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    source               TEXT PRIMARY KEY,
    doc_hash             TEXT NOT NULL,
    embedding_signature  TEXT NOT NULL DEFAULT '',
    chunk_count          INTEGER NOT NULL DEFAULT 0,
    chars                INTEGER NOT NULL DEFAULT 0,
    dept                 TEXT NOT NULL DEFAULT '',
    doc_type             TEXT NOT NULL DEFAULT '',
    updated_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_dept ON documents (dept);
"""


def text_hash(text: str) -> str:
    """片段/文档指纹。用 sha256 前 16 位就够：几万段里的碰撞概率可以忽略。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


class DocumentStore:
    def __init__(self, settings: Settings) -> None:
        self.path = Path(settings.db_path)
        with session(self.path) as conn:
            conn.executescript(_SCHEMA)

    def get(self, source: str) -> dict | None:
        with session(self.path) as conn:
            row = conn.execute("SELECT * FROM documents WHERE source = ?", (source,)).fetchone()
        return dict(row) if row else None

    def upsert(
        self,
        *,
        source: str,
        doc_hash: str,
        embedding_signature: str,
        chunk_count: int,
        chars: int,
        dept: str = "",
        doc_type: str = "",
    ) -> None:
        with session(self.path) as conn:
            conn.execute(
                "INSERT INTO documents (source, doc_hash, embedding_signature, chunk_count,"
                " chars, dept, doc_type, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(source) DO UPDATE SET doc_hash = excluded.doc_hash,"
                " embedding_signature = excluded.embedding_signature,"
                " chunk_count = excluded.chunk_count, chars = excluded.chars,"
                " dept = excluded.dept, doc_type = excluded.doc_type,"
                " updated_at = excluded.updated_at",
                (
                    source, doc_hash, embedding_signature, int(chunk_count), int(chars),
                    dept, doc_type, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )

    def delete(self, source: str) -> None:
        with session(self.path) as conn:
            conn.execute("DELETE FROM documents WHERE source = ?", (source,))

    def clear(self) -> int:
        with session(self.path) as conn:
            cursor = conn.execute("DELETE FROM documents")
            return cursor.rowcount

    def all(self) -> list[dict]:
        with session(self.path) as conn:
            rows = conn.execute(
                "SELECT source, doc_hash, embedding_signature, chunk_count, chars, dept, doc_type, updated_at"
                " FROM documents ORDER BY source"
            ).fetchall()
        return [dict(row) for row in rows]

    def depts(self) -> list[str]:
        with session(self.path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT dept FROM documents WHERE dept <> '' ORDER BY dept"
            ).fetchall()
        return [str(row["dept"]) for row in rows]
