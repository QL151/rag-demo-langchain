"""请求级 Trace：一次提问的每个阶段耗时多少、召回了几条、分数多少、为什么拒答。

为什么不用 Langfuse 之类的现成方案：
它们确实更好，但需要额外部署一个服务 + 一个 Postgres/ClickHouse。
小厂第一阶段往往是"先能定位问题"，一个 JSONL 文件 + 一个查询接口就够了，
而且 JSONL 可以直接 `jq` / `grep` 捞，不依赖任何东西。
（真需要看板时，把 `_write()` 换成 Langfuse 的 SDK 调用即可，上层不用改。）

它解决的具体问题：用户说"它答错了"，你以前只能翻日志猜；
现在拿到 trace_id 就能看到：改写后的问句是什么、召回了哪些片段、各打多少分、
是命中了缓存还是走了大模型、card 卡在哪一段。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import Settings

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Trace:
    id: str
    session_id: str
    question: str
    started_at: float = field(default_factory=time.perf_counter)
    stages: list[dict] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    error: str = ""

    @contextmanager
    def stage(self, name: str, **extra) -> Iterator[dict]:
        """with trace.stage("retrieve", top_k=4): ... —— 自动记耗时。"""
        started = time.perf_counter()
        payload: dict = dict(extra)
        try:
            yield payload
        finally:
            self.stages.append({
                "stage": name,
                "ms": round((time.perf_counter() - started) * 1000, 2),
                **payload,
            })

    def mark(self, name: str, ms: float, **extra) -> None:
        self.stages.append({"stage": name, "ms": round(ms, 2), **extra})

    def set(self, **kwargs) -> None:
        self.data.update(kwargs)

    @property
    def total_ms(self) -> float:
        return round((time.perf_counter() - self.started_at) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.id,
            "session_id": self.session_id,
            "question": self.question,
            "created_at": _now().isoformat(timespec="seconds"),
            "total_ms": self.total_ms,
            "stages": self.stages,
            "error": self.error,
            **self.data,
        }


class Tracer:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.dir: Path = Path(settings.trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._recent: deque[dict] = deque(maxlen=max(10, settings.trace_memory_size))
        self.written = 0

    def start(self, question: str, session_id: str = "") -> Trace:
        trace_id = uuid.uuid4().hex[:12]
        return Trace(id=trace_id, session_id=session_id, question=question)

    def finish(self, trace: Trace) -> dict:
        payload = trace.to_dict()
        self._recent.append(payload)
        if self.s.trace_enabled:
            self._write(payload)
        return payload

    def _write(self, payload: dict) -> None:
        """按天分文件：data/traces/2026-02-14.jsonl —— 好删、好捞、好归档。"""
        day = payload["created_at"][:10]
        path = self.dir / f"{day}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.written += 1
        except Exception as exc:  # noqa: BLE001 —— trace 写失败不能影响问答
            logger.warning("写 trace 失败（不影响问答）：%s", exc)

    def get(self, trace_id: str) -> dict | None:
        for payload in reversed(self._recent):
            if payload["trace_id"] == trace_id:
                return payload
        # 内存里没有（进程重启过）→ 回文件里捞
        for path in sorted(self.dir.glob("*.jsonl"), reverse=True)[:3]:
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if trace_id in line:
                        return json.loads(line)
            except Exception:  # noqa: BLE001
                continue
        return None

    def recent(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(limit, self._recent.maxlen or 20))
        return list(reversed(self._recent))[:limit]

    def bad_cases(self, limit: int = 20) -> list[dict]:
        """筛出"值得复盘"的请求：拒答、出错、或者走了完整链路但没命中缓存。"""
        flagged = [
            payload for payload in reversed(self._recent)
            if payload.get("refused") or payload.get("error")
        ]
        return flagged[:limit]

    def stats(self) -> dict:
        return {
            "enabled": self.s.trace_enabled,
            "dir": str(self.dir),
            "in_memory": len(self._recent),
            "written_this_process": self.written,
        }
