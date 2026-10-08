"""POST 探测：验证 embedding 与 rerank（精排）配置能不能用。

用法（项目根目录下）：

    python scripts/check_embedding.py

它会真的调用接口（只花几分钱甚至免费额度），然后告诉你：
  1. 当前配置对不对（Key 只显示长度，**不会泄露内容**）
  2. 向量维度是多少、语义能力是否正常
  3. 精排接口能不能调通（调不通会说明原因，并提示会自动降级）
  4. 维度和你现有的索引是否兼容 —— 不兼容就必须重建索引

为什么需要它：批量导入一篇文档要发几十上百次请求，
先花十几秒验证配置，比跑到一半失败再排查省事得多。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.embedding_local import cosine  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.retrieval import Candidate, LLMReranker, Reranker  # noqa: E402
from app.vectorstore import MemoryStore  # noqa: E402


def existing_index_dim() -> int | None:
    """读一下内置索引里一条向量的维度（用来判断要不要重建）。"""
    path = settings.memory_index_file
    if not path.exists():
        return None
    try:
        return MemoryStore(path).dimension()
    except Exception:  # noqa: BLE001
        return None


def check_embedding(llm: LLMClient) -> int:
    key = llm.embedding_key()
    print("-" * 68)
    print("【1】向量化（embedding）")
    print(f"  EMBEDDING_MODE      = {settings.embedding_mode}")
    print(f"  EMBEDDING_MODEL     = {settings.embedding_model}")
    print(f"  EMBEDDING_BASE_URL  = {settings.embedding_base_url}")
    print(f"  EMBEDDING_BATCH_SIZE= {settings.embedding_batch_size}")
    print(f"  EMBEDDING_API_KEY   = {'已填写（长度 ' + str(len(settings.embedding_api_key)) + '）' if settings.embedding_api_key else '（空）'}")
    print(f"  实际使用的 Key 长度  = {len(key)}（内容已隐藏）")
    print(f"  现有索引维度        = {existing_index_dim() or '（还没有索引）'}")

    if settings.embedding_mode != "api":
        print("[!] 当前 EMBEDDING_MODE 不是 api，程序用的是本地哈希算法（效果明显弱于真实模型）。")
        return 1
    if not key:
        print("[×] 没有可用的 Key。")
        if settings.embedding_base_url != settings.llm_base_url:
            print("    ⚠ embedding 和对话**不是同一家服务商**（base_url 不同），")
            print("      不能复用 LLM_API_KEY —— 必须单独填 EMBEDDING_API_KEY。")
        return 1

    samples = ["专业版多少钱", "专业版的价格是多少", "今天天气不错"]
    print("  正在调用接口（3 条文本）...")
    try:
        vectors = llm.embed(samples)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[×] 调用失败：{type(exc).__name__}: {exc}")
        print("  常见原因：401 Key 不对；404 模型名写错；403 没勾选该模型；网络不通")
        return 1

    dims = {len(vector) for vector in vectors}
    print(f"  [√] 调用成功！向量维度 = {sorted(dims)}")
    similar = cosine(vectors[0], vectors[1])
    different = cosine(vectors[0], vectors[2])
    print(f"      相近句 = {similar:.4f}｜无关句 = {different:.4f}", end="  ")
    print("[√] 语义能力正常" if similar > different else "[!] 分数异常，检查模型名和 base_url")

    old_dim = existing_index_dim()
    new_dim = sorted(dims)[0]
    if old_dim is None:
        print("  [下一步] 可以导入文档了：python scripts/bulk_ingest.py \"你的文档目录\"")
    elif old_dim == new_dim:
        print(f"  [提示] 现有索引维度也是 {old_dim}，但**算法可能不同**，换过模型仍建议重建索引")
    else:
        print(f"  [!] 维度不一致：现有索引 {old_dim} 维，新模型 {new_dim} 维 —— 必须重建索引")
    return 0


def check_rerank() -> int:
    print("-" * 68)
    print("【2】精排（rerank）")
    print(f"  RERANK_ENABLED  = {settings.rerank_enabled}")
    print(f"  RERANK_PROVIDER = {settings.rerank_provider}")
    print(f"  RERANK_MODEL    = {settings.rerank_model}")
    print(f"  RERANK_BASE_URL = {settings.rerank_base_url}")
    print(f"  RERANK_API_KEY  = {'已配置（长度 ' + str(len(settings.rerank_api_key)) + '）' if settings.rerank_api_key else '（空）'}")
    print(f"  listwise 降级    = {settings.rerank_llm_ready}（依赖 LLM_API_KEY）")
    if not settings.rerank_enabled:
        print("  [--] 已关闭，检索会用「向量+BM25」融合结果（可用且有说明，不是静默降级）")
        return 0

    candidates = [
        Candidate(id="1", text="专业版 199 元每月，支持 14 天无理由退款。", metadata={}),
        Candidate(id="2", text="实习生每周需要到岗 4 天，其余时间可以远程。", metadata={}),
        Candidate(id="3", text="差旅报销需要提供发票原件，30 天内提交。", metadata={}),
    ]

    if settings.rerank_ready:
        reranker = Reranker(settings)
        print("  正在调用 cross-encoder 精排接口（3 条候选）...")
        if reranker.rerank("专业版多少钱", candidates, top_n=3):
            ranked = sorted(candidates, key=lambda item: item.rerank_score or 0, reverse=True)
            for item in ranked:
                print(f"      {item.rerank_score:.4f}  {item.text[:32]}…")
            print("  [√] cross-encoder 工作正常" if ranked[0].id == "1" else "  [!] 排序不符合预期")
            return 0 if ranked[0].id == "1" else 1
        print(f"  [×] cross-encoder 调用失败：{reranker.last_error}")
        print("      403 AccessDenied 通常表示**这个 Key 没有开通该模型**，")
        print("      去服务商控制台给 Key 加上 rerank 权限，或改用 listwise（RERANK_PROVIDER=llm）。")
    else:
        print("  [--] 没有可用的 cross-encoder Key，跳过接口调用")

    if settings.rerank_llm_ready:
        llm = LLMClient(settings)
        print("  正在测试 listwise 精排（用对话模型给候选排序）...")
        outcome = LLMReranker(settings, llm).rerank("专业版多少钱", candidates, top_n=3)
        if outcome.ok:
            ranked = sorted(candidates, key=lambda item: item.rerank_rank or 999)
            for item in ranked:
                print(f"      #{item.rerank_rank}  {item.text[:32]}…")
            print("  [√] listwise 精排工作正常（注意：它给的是顺序，不是分数）")
            return 0
        print(f"  [×] listwise 精排失败：{outcome.reason}")
        return 1

    print("  [!] 两种精排都用不了 —— 检索仍可用，只是少了最准的一层")
    return 1


def main() -> int:
    llm = LLMClient(settings)
    print("=" * 68)
    print("embedding / rerank 配置自检")
    print("=" * 68)
    embedding_ok = check_embedding(llm)
    print()
    rerank_ok = check_rerank()
    print("=" * 68)
    print(f"结论：向量化 {'OK' if embedding_ok == 0 else '异常'}；精排 {'OK' if rerank_ok == 0 else '不可用/异常'}")
    for warning in settings.warnings():
        print("  [!]", warning)
    return 0 if embedding_ok == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
