"""答案缓存：完全相同的问题直接命中，措辞不同但意思一样的问题走语义命中。

企业知识库里 30%~50% 的提问是重复的（"专业版多少钱 / 专业版的价格 / 专业版怎么收费"），
每次都跑一遍「向量化 + 检索 + 精排 + 大模型」既慢又费钱。这一层就是省这两样的。

三级命中策略（从便宜到贵）：
1. **精确命中**：问题归一化后完全相同 → 一次 SQL 查询，1ms；
2. **语义命中**：问题向量和缓存里的相似度 ≥ 阈值（默认 0.95）→ 一次线性扫描；
3. 都没命中 → 走完整链路，并把结果写回缓存。

两个必须做对的细节：
- **知识库一变，缓存必须整体失效。** 否则用户上传了新文档，提问还是拿到旧答案，
  这是最典型的"看起来很聪明的 bug"。这里用持久化的 `cache_version` 计数器实现：
  任何入库/删除/清空都 bump 一次，重启进程也不会漂。
- **embedding 指纹要进缓存键。** 换了向量模型以后，旧缓存的语义相似度不再可信。
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Settings
from .db import session
from .embedding_local import cosine

logger = logging.getLogger(__name__)

_PUNCT = re.compile(r"[\s，。！？、；：\u201c\u201d\u2018\u2019（）()\[\]{}<>《》~!?,.;:\-_/\\|]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answer_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    question     TEXT NOT NULL,
    question_key TEXT NOT NULL,
    cache_key    TEXT NOT NULL,          -- filters + embedding 指纹
    answer       TEXT NOT NULL,
    citations    TEXT,
    embedding    TEXT,
    hits         INTEGER NOT NULL DEFAULT 0,
    answer_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cache_lookup ON answer_cache (question_key, cache_key, created_at);
CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def normalize_question(question: str) -> str:
    """归一化：去掉空白与标点、统一大小写。让"专业版多少钱？"和"专业版多少钱"算同一个问题。"""
    return _PUNCT.sub("", (question or "").strip().lower())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AnswerCache:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.path = Path(settings.db_path)
        self.hits_exact = 0
        self.hits_semantic = 0
        self.misses = 0
        self.skipped = 0        # 因为知识库版本变化而作废的读取次数
        with session(self.path) as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------ 知识库版本
    def version(self) -> int:
        with session(self.path) as conn:
            row = conn.execute("SELECT value FROM app_meta WHERE key = 'cache_version'").fetchone()
        return int(row["value"]) if row else 0

    def bump_version(self) -> int:
        """知识库发生写入/删除时调用，让全部缓存立刻失效。"""
        with session(self.path) as conn:
            conn.execute(
                "INSERT INTO app_meta (key, value) VALUES ('cache_version', '1')"
                " ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
            )
            row = conn.execute("SELECT value FROM app_meta WHERE key = 'cache_version'").fetchone()
        new_value = int(row["value"]) if row else 0
        logger.info("知识库版本 +1 → %d，答案缓存整体失效", new_value)
        return new_value

    # ------------------------------------------------------------ 缓存键
    def _cache_key(self, filters: dict | None, embedding_signature: str) -> str:
        """缓存键 = 知识库版本 + embedding 指纹 + 过滤条件。

        为什么知识库版本必须在键里：用户上传了新文档，旧答案就不再正确。
        用"版本号进键"而不是"收到写入通知再去删缓存"，好处是不会漏、也不需要额外机制 ——
        没被删掉的旧行会在 TTL 和容量上限里自然淘汰。
        """
        parts = [f"v{self.version()}", embedding_signature]
        if filters:
            parts += [f"{k}={v}" for k, v in sorted(filters.items())]
        return "|".join(parts)

    # ------------------------------------------------------------ 读
    def get(
        self,
        question: str,
        embedding: list[float] | None = None,
        filters: dict | None = None,
        embedding_signature: str = "",
    ) -> dict | None:
        if not self.s.cache_enabled:
            return None
        cache_key = self._cache_key(filters, embedding_signature)
        cutoff = (_now() - timedelta(seconds=self.s.cache_ttl_seconds)).isoformat(timespec="seconds")
        question_key = normalize_question(question)

        with session(self.path) as conn:
            row = conn.execute(
                "SELECT id, question, answer, citations, created_at FROM answer_cache"
                " WHERE question_key = ? AND cache_key = ? AND created_at >= ?"
                " ORDER BY id DESC LIMIT 1",
                (question_key, cache_key, cutoff),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE answer_cache SET hits = hits + 1 WHERE id = ?", (row["id"],)
                )
                self.hits_exact += 1
                return {
                    "answer": row["answer"],
                    "citations": json.loads(row["citations"] or "[]"),
                    "match": "exact",
                    "similarity": 1.0,
                    "cached_question": row["question"],
                }

            if embedding and self.s.cache_semantic_threshold > 0:
                best = self._semantic_lookup(conn, cache_key, cutoff, embedding)
                if best is not None:
                    self.hits_semantic += 1
                    return best

        self.misses += 1
        return None

    def _semantic_lookup(self, conn, cache_key: str, cutoff: str, embedding: list[float]) -> dict | None:
        rows = conn.execute(
            "SELECT id, question, answer, citations, embedding FROM answer_cache"
            " WHERE cache_key = ? AND created_at >= ? AND embedding IS NOT NULL"
            " ORDER BY id DESC LIMIT ?",
            (cache_key, cutoff, max(1, self.s.cache_max_scan)),
        ).fetchall()

        best_score = 0.0
        best_row = None
        for row in rows:
            try:
                vector = json.loads(row["embedding"])
            except (TypeError, ValueError):
                continue
            if not vector:
                continue
            score = cosine(embedding, vector)
            if score > best_score:
                best_score, best_row = score, row

        if best_row is not None and best_score >= self.s.cache_semantic_threshold:
            conn.execute("UPDATE answer_cache SET hits = hits + 1 WHERE id = ?", (best_row["id"],))
            return {
                "answer": best_row["answer"],
                "citations": json.loads(best_row["citations"] or "[]"),
                "match": "semantic",
                "similarity": round(best_score, 4),
                "cached_question": best_row["question"],
            }
        return None

    # ------------------------------------------------------------ 写
    def put(
        self,
        *,
        question: str,
        answer: str,
        citations: list[dict] | None = None,
        embedding: list[float] | None = None,
        filters: dict | None = None,
        embedding_signature: str = "",
    ) -> None:
        if not self.s.cache_enabled or not answer.strip():
            return
        try:
            with session(self.path) as conn:
                conn.execute(
                    "INSERT INTO answer_cache (created_at, question, question_key, cache_key,"
                    " answer, citations, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        _now().isoformat(timespec="seconds"),
                        question,
                        normalize_question(question),
                        self._cache_key(filters, embedding_signature),
                        answer,
                        json.dumps(citations or [], ensure_ascii=False),
                        json.dumps(embedding) if embedding else None,
                    ),
                )
                # 只保留最近的 N 条，避免缓存表无限膨胀
                conn.execute(
                    "DELETE FROM answer_cache WHERE id NOT IN"
                    " (SELECT id FROM answer_cache ORDER BY id DESC LIMIT 5000)"
                )
        except Exception as exc:  # noqa: BLE001 —— 缓存写失败不能影响问答
            logger.warning("写缓存失败（不影响问答）：%s", exc)

    def prune(self) -> int:
        """清掉过期条目（可以挂在定时任务上）。"""
        cutoff = (_now() - timedelta(seconds=self.s.cache_ttl_seconds)).isoformat(timespec="seconds")
        with session(self.path) as conn:
            cursor = conn.execute("DELETE FROM answer_cache WHERE created_at < ?", (cutoff,))
            return cursor.rowcount

    def clear(self) -> int:
        with session(self.path) as conn:
            cursor = conn.execute("DELETE FROM answer_cache")
            return cursor.rowcount

    # ------------------------------------------------------------ 统计
    def stats(self) -> dict:
        with session(self.path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS total_hits FROM answer_cache"
            ).fetchone()
        lookups = self.hits_exact + self.hits_semantic + self.misses
        return {
            "enabled": self.s.cache_enabled,
            "entries": int(row["entries"] or 0),
            "total_hits": int(row["total_hits"] or 0),
            "process_lookups": lookups,
            "process_hits_exact": self.hits_exact,
            "process_hits_semantic": self.hits_semantic,
            "process_misses": self.misses,
            "hit_rate": round((self.hits_exact + self.hits_semantic) / lookups, 3) if lookups else 0.0,
            "version": self.version(),
            "ttl_seconds": self.s.cache_ttl_seconds,
            "semantic_threshold": self.s.cache_semantic_threshold,
        }
