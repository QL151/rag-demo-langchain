"""pytest 共享夹具。

两条原则：
1. **测试绝不联网。** 所有用例都用本地哈希向量化 + 关闭精排/改写，
   这样 `pytest` 跑几秒就完，也不会花掉你的 API 额度。
2. **用真实代码路径，只换掉外部依赖。** 夹具是把真实的 Settings 用
   `dataclasses.replace` 改成"内存向量库 + 临时目录"，而不是 mock 掉被测逻辑 ——
   否则测的是 mock，不是你的代码。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from app.cache import AnswerCache
from app.chat_store import ChatStore
from app.config import settings as base_settings
from app.docstore import DocumentStore
from app.llm import LLMClient
from app.rag import RagService
from app.trace import Tracer
from app.usage import UsageStore
from app.vectorstore import MemoryStore


@pytest.fixture()
def make_settings(tmp_path: Path):
    def _make(**overrides):
        defaults = dict(
            vector_backend="memory",
            memory_index_file=tmp_path / "index.json",
            chroma_dir=tmp_path / "chroma",
            upload_dir=tmp_path / "uploads",
            db_path=tmp_path / "app.db",
            trace_dir=tmp_path / "traces",
            embedding_mode="local",     # 本地哈希：不联网
            embedding_api_key="",
            llm_api_key="",             # 没有 Key → 只检索不生成
            rerank_enabled=False,       # 不联网
            rewrite_enabled=False,
            cache_enabled=True,
            min_vector_score=0.25,
            min_rerank_score=0.30,
            top_k=4,
            retrieve_k=20,
        )
        defaults.update(overrides)
        return dataclasses.replace(base_settings, **defaults)

    return _make


@pytest.fixture()
def rag(make_settings):
    """一套完整的 RagService（内存向量库 + 本地向量化 + 临时 SQLite）。"""
    return build_rag(make_settings())


@pytest.fixture()
def rag_strict(make_settings):
    """把拒答阈值提到 0.95 的版本。

    为什么要单独一个夹具：本地哈希向量化的相似度分布是"压缩"的，
    不相关的文本也能有 0.1~0.3 的余弦值（哈希桶碰撞），
    按真实 embedding 校准的 0.25 阈值在本地模式下拦不住任何东西。
    所以测"阈值拒答"这条逻辑时，用真实 embedding 的分数分布来构造场景。
    """
    return build_rag(make_settings(min_vector_score=0.95))


@pytest.fixture()
def rag_rewrite(make_settings):
    """打开多轮查询改写的版本（改写需要真的调用一次模型，所以默认关掉）。"""
    return build_rag(make_settings(rewrite_enabled=True))


def build_rag(cfg) -> RagService:
    cfg.ensure_dirs()
    llm = LLMClient(cfg)
    return RagService(
        cfg,
        llm,
        MemoryStore(cfg.memory_index_file),
        ChatStore(cfg.db_path),
        usage=UsageStore(cfg),
        cache=AnswerCache(cfg),
        tracer=Tracer(cfg),
        documents=DocumentStore(cfg),
    )


DOC_MANUAL = """# 产品手册

## 版本与价格

体验版每月可以提问 500 次，专业版 199 元每月。

## 错误码

遇到 40412 表示配额已用尽，请升级套餐或等待下月重置。

## 退款政策

付费版本支持 14 天内无理由退款。
"""

DOC_HR = """# 员工手册

## 考勤

实习生每周需要到岗 4 天，其余时间可以远程。

## 报销

差旅报销需要提供发票原件，30 天内提交。
"""


@pytest.fixture()
def seeded(rag):
    """入库两份不同部门的文档，返回 RagService。"""
    rag.ingest_text("产品手册.md", DOC_MANUAL, dept="product", doc_type="manual")
    rag.ingest_text("员工手册.md", DOC_HR, dept="hr", doc_type="policy")
    return rag


@pytest.fixture()
def seeded_strict(rag_strict):
    rag_strict.ingest_text("产品手册.md", DOC_MANUAL, dept="product", doc_type="manual")
    rag_strict.ingest_text("员工手册.md", DOC_HR, dept="hr", doc_type="policy")
    return rag_strict
