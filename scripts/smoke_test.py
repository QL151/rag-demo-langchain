"""一键自测（不需要浏览器、不需要启动服务）：

    python scripts/smoke_test.py

它会：入库示例文档（走增量入库）→ 打印混合检索命中的片段（含各路分数）→ 直接调用 RAG 流程提问。
没有配置 LLM_API_KEY 也能跑（只返回检索片段），配置了就是完整回答。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chat_store import ChatStore  # noqa: E402
from app.config import BASE_DIR, settings  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.rag import RagService  # noqa: E402
from app.vectorstore import build_store  # noqa: E402

QUESTIONS = [
    "小峰智能体有哪些功能？",
    "专业版多少钱？",
    "40412 是什么错误",
    "实习生每周需要到岗几天？",
]


def main() -> int:
    settings.ensure_dirs()
    llm = LLMClient(settings)
    store = build_store(settings)
    rag = RagService(settings, llm, store, ChatStore(settings.db_path))

    sample = BASE_DIR / "data" / "sample" / "产品手册示例.md"
    if not sample.exists():
        print(f"[×] 找不到示例文档：{sample}")
        return 1

    print("=" * 76)
    print("配置：向量库 =", store.backend, "| 向量化 =", llm.describe()["embedding_model"])
    print("     生成模型 =", settings.llm_model, "| LLM 就绪 =", llm.llm_ready)
    print(
        "     检索：混合 =", settings.hybrid_enabled,
        "| 精排 =", settings.rerank_ready, f"({settings.rerank_model})",
        "| 改写 =", settings.rewrite_enabled,
        "| 缓存 =", settings.cache_enabled,
    )
    for warning in settings.warnings():
        print("     [!]", warning)
    print("=" * 76)

    result = rag.ingest_file(sample.name, sample.read_bytes(), dept="product")
    state = "整篇跳过（内容未变化）" if result["skipped"] else "已入库"
    print(
        f"[√] {state}：{result['source']}，{result['chunks']} 段，{result['chars']} 字"
        f"（复用旧向量 {result['reused_chunks']} 段 / 新向量化 {result['embedded_chunks']} 段）\n"
    )

    for question in QUESTIONS:
        retrieval = rag.search(question)
        print("-" * 76)
        print("问题：", question)
        if retrieval.rewritten:
            print("改写后检索问句：", retrieval.query)
        if retrieval.note:
            print("检索说明：", retrieval.note)
        print(f"召回 {retrieval.recalled} 条 → 过阈值后 {len(retrieval.candidates)} 条"
              + ("（判定为知识库中没有依据 → 直接拒答，不调用大模型）" if retrieval.refused else ""))
        for i, candidate in enumerate(rag._select(retrieval), 1):
            snippet = " ".join(candidate.text.split())[:56]
            channels = "+".join(candidate.matched_by) or "-"
            rerank = "-" if candidate.rerank_score is None else f"{candidate.rerank_score:.3f}"
            print(
                f"  [{i}] 最终 {candidate.final_score:<7} 向量 {candidate.vector_score:<7} "
                f"精排 {rerank:<7} BM25 {candidate.bm25_score:<6.2f} [{channels}] "
                f"{candidate.source} 第{candidate.chunk_index + 1}段 | {snippet}…"
            )

        answer, citations, meta = rag.answer("smoke-test", question)
        print(f"\n回答（引用 {len(citations)} 条｜缓存 {meta['cache']}｜耗时 {meta['latency_ms']}ms"
              f"｜trace {meta['trace_id']}）：")
        print("  " + answer.replace("\n", "\n  "))
        print()

    print("=" * 76)
    print("[√] 自测通过。接下来：uvicorn app.main:app --port 8000 然后打开 http://127.0.0.1:8000")
    print("     想看用量与缓存统计：GET /api/stats；想看某次请求的链路：GET /api/traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
