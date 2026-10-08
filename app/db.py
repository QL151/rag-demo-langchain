"""SQLite 连接的小工具：统一超时、WAL 模式和 Row 工厂。

为什么单独抽出来：改造后 app.db 里多了三张表（用量、答案缓存、文档台账），
如果每个模块各写一遍 `sqlite3.connect(...)`，参数迟早会不一致 ——
比如漏掉 WAL，FastAPI 的线程池并发写就很容易 `database is locked`。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# WAL：读写不互相阻塞（默认的 delete 模式下，写会锁住整个库）
_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=5000",
)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        try:
            conn.execute(pragma)
        except sqlite3.Error:  # pragma: no cover —— 个别文件系统不支持 WAL
            pass
    return conn


@contextmanager
def session(path: Path) -> Iterator[sqlite3.Connection]:
    """一次连接 + 自动提交/回滚。"""
    conn = connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()
