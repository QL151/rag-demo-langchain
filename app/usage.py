"""用量与成本计量：把每一次模型调用都记下来。

为什么这个模块值得单独存在 —— 它是简历上所有成本数字的来源：

    "单次问答成本从 0.06 元降到 0.012 元" 这句话，
    如果没有这张表，就只能是拍脑袋；有了它，就是 `SELECT sum(cost)` 跑出来的。

设计取舍：
- 落 SQLite 而不是日志文件：要按天/按会话聚合，SQL 一句就够，日志还得自己写解析器；
- 记 `estimated` 标记：服务商没返回 usage 时用字符数估算，
  宁可标注"这是估算值"，也不要把它当成精确账单；
- 记 `first_token_ms`（首字延迟）而不只是总耗时：流式场景下用户体感取决于首字，
  总耗时 8 秒但首字 0.6 秒的体验，远好于总耗时 5 秒但首字 4 秒。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import Settings
from .db import session

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    kind              TEXT NOT NULL,          -- answer | rewrite | rerank | embed
    model             TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost              REAL NOT NULL DEFAULT 0,
    estimated         INTEGER NOT NULL DEFAULT 0,
    latency_ms        REAL NOT NULL DEFAULT 0,
    first_token_ms    REAL,
    cache_hit         INTEGER NOT NULL DEFAULT 0,
    session_id        TEXT,
    trace_id          TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_events (created_at);
CREATE INDEX IF NOT EXISTS idx_usage_kind ON usage_events (kind, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class UsageStore:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.path = Path(settings.db_path)
        with session(self.path) as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ 写入
    def record(
        self,
        *,
        kind: str,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        estimated: bool = False,
        latency_ms: float = 0.0,
        first_token_ms: float | None = None,
        cache_hit: bool = False,
        session_id: str = "",
        trace_id: str = "",
    ) -> float:
        """记一条用量，返回本次估算费用（元）。

        命中答案缓存时强制记 0 费用：那次请求根本没调用模型，
        哪怕调用方误传了 token 数也不该计入成本 —— 这个不变量比"信任入参"重要。
        """
        if cache_hit:
            prompt_tokens = completion_tokens = 0
        cost = (
            prompt_tokens / 1000.0 * self.s.price_prompt_per_1k
            + completion_tokens / 1000.0 * self.s.price_completion_per_1k
        )
        try:
            with session(self.path) as conn:
                conn.execute(
                    "INSERT INTO usage_events (created_at, kind, model, prompt_tokens,"
                    " completion_tokens, cost, estimated, latency_ms, first_token_ms,"
                    " cache_hit, session_id, trace_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _now(), kind, model, int(prompt_tokens), int(completion_tokens),
                        round(cost, 8), 1 if estimated else 0, round(float(latency_ms), 2),
                        None if first_token_ms is None else round(float(first_token_ms), 2),
                        1 if cache_hit else 0, session_id, trace_id,
                    ),
                )
        except Exception as exc:  # noqa: BLE001 —— 计量失败绝不能影响正常问答
            logger.warning("写入用量记录失败（不影响问答）：%s", exc)
        return cost

    # ------------------------------------------------------------------ 查询
    def summary(self, days: int = 7) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(days=max(1, days))).isoformat(timespec="seconds")
        with session(self.path) as conn:
            overall = conn.execute(
                "SELECT COUNT(*) AS calls,"
                " COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,"
                " COALESCE(SUM(completion_tokens), 0) AS completion_tokens,"
                " COALESCE(SUM(cost), 0) AS cost,"
                " COALESCE(AVG(latency_ms), 0) AS avg_latency,"
                " COALESCE(SUM(estimated), 0) AS estimated_calls"
                " FROM usage_events WHERE created_at >= ?",
                (since,),
            ).fetchone()
            by_kind = conn.execute(
                "SELECT kind, COUNT(*) AS calls, COALESCE(SUM(cost), 0) AS cost,"
                " COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens"
                " FROM usage_events WHERE created_at >= ? GROUP BY kind ORDER BY cost DESC",
                (since,),
            ).fetchall()
            daily = conn.execute(
                "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS calls,"
                " COALESCE(SUM(cost), 0) AS cost"
                " FROM usage_events WHERE created_at >= ? GROUP BY day ORDER BY day",
                (since,),
            ).fetchall()
            latency = self._latency_percentiles(conn, since)

        calls = int(overall["calls"] or 0)
        tokens = int(overall["prompt_tokens"] or 0) + int(overall["completion_tokens"] or 0)
        cost = float(overall["cost"] or 0)
        return {
            "window_days": days,
            "calls": calls,
            "prompt_tokens": int(overall["prompt_tokens"] or 0),
            "completion_tokens": int(overall["completion_tokens"] or 0),
            "tokens": tokens,
            "cost_yuan": round(cost, 6),
            "avg_cost_per_call_yuan": round(cost / calls, 6) if calls else 0.0,
            "avg_latency_ms": round(float(overall["avg_latency"] or 0), 1),
            "p95_latency_ms": latency.get("p95_latency_ms", 0.0),
            "p95_first_token_ms": latency.get("p95_first_token_ms", 0.0),
            "estimated_calls": int(overall["estimated_calls"] or 0),
            "estimated_ratio": round(int(overall["estimated_calls"] or 0) / calls, 3) if calls else 0.0,
            "by_kind": [dict(row) for row in by_kind],
            "daily": [dict(row) for row in daily],
        }

    def _latency_percentiles(self, conn, since: str) -> dict:
        """P95 用"取第 95% 位"的朴素做法 —— 单机几万行数据，不值得上近似算法。"""
        rows = conn.execute(
            "SELECT latency_ms, first_token_ms FROM usage_events"
            " WHERE created_at >= ? AND kind = 'answer' ORDER BY latency_ms",
            (since,),
        ).fetchall()
        if not rows:
            return {}
        latencies = [float(r["latency_ms"] or 0) for r in rows]
        firsts = sorted(float(r["first_token_ms"]) for r in rows if r["first_token_ms"] is not None)
        index = min(len(latencies) - 1, int(len(latencies) * 0.95))
        result = {"p95_latency_ms": round(latencies[index], 1)}
        if firsts:
            result["p95_first_token_ms"] = round(firsts[min(len(firsts) - 1, int(len(firsts) * 0.95))], 1)
        return result

    def recent(self, limit: int = 50) -> list[dict]:
        with session(self.path) as conn:
            rows = conn.execute(
                "SELECT created_at, kind, model, prompt_tokens, completion_tokens, cost,"
                " estimated, latency_ms, first_token_ms, cache_hit, session_id, trace_id"
                " FROM usage_events ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in rows]
