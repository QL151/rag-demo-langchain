"""检索策略评测：在真实语料上跑消融实验，输出可以贴进简历的对比表。

用法（项目根目录）：

    # 默认：按 Markdown 标题切分，跑「纯向量 → +BM25/RRF → +精排」三步消融
    python scripts/evaluate_retrieval.py "D:\\你的\\文档目录"

    # 便宜试跑
    python scripts/evaluate_retrieval.py "D:\\你的\\文档目录" --limit 8 --sample 20

    # 顺便把 4 种切分策略也比一遍（embedding 调用量 ×4，慢且费钱）
    python scripts/evaluate_retrieval.py "D:\\你的\\文档目录" --grid

它做什么：
    1. 读目录下所有 md / txt，按指定策略切分，**只向量化一次**；
    2. 用同一批向量分别跑三种检索策略，避免"重复向量化"把实验做贵；
    3. 自动生成两组问题：
       · 文档级：用文档标题提问（容易，区分度低）
       · 章节级：用 `##` 小节标题提问（难，区分度高），并剔除"背景/总结"这类通用标题；
    4. 打印对比表，并把报告存到 data/_eval/report-*.md（可复现、可归档）。

为什么必须做消融实验：
    "我做了混合检索" 是一句话；"混合检索让章节级 Top-1 从 61% 涨到 78%" 是一个结论。
    面试官信的只有后者，而且会接着问"你怎么测的" —— 这个脚本就是答案。
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import split_by_heading, split_text  # noqa: E402
from app.config import BASE_DIR, settings as base_settings  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402
from app.vectorstore import MemoryStore  # noqa: E402

SUFFIXES = {".md", ".markdown", ".txt"}
MIN_DOC_CHARS = 200
RANDOM_SEED = 42

# 通用小节标题：几乎每篇文档都有，用它提问无法判断检索对不对
GENERIC_TITLE = re.compile(
    r"^(背景|前言|简介|概述|导读|总结|小结|结语|展望|写在最后|参考文献|参考资料|"
    r"关于我们|团队介绍|作者|附录|目录)|"
    r"(总结|展望|背景|简介|概述|结语|前言)$"
)

STRATEGIES = [
    ("纯向量", dict(hybrid_enabled=False, rerank_enabled=False)),
    ("向量+BM25（RRF 融合）", dict(hybrid_enabled=True, rerank_enabled=False)),
    ("+ 精排（rerank）", dict(hybrid_enabled=True, rerank_enabled=True)),
]

CHUNKERS = {
    "chunk300": ("按句子切 CHUNK_SIZE=300", lambda t: split_text(t, 300, 60)),
    "chunk500": ("按句子切 CHUNK_SIZE=500（默认）", lambda t: split_text(t, 500, 80)),
    "chunk800": ("按句子切 CHUNK_SIZE=800", lambda t: split_text(t, 800, 100)),
    "heading": ("按 Markdown 标题切（超长再按句子）", lambda t: split_by_heading(t, 500, 80)),
}


def norm(text: str) -> str:
    """归一化：去掉 # 和标点空白，用于判断"标题是否出现在片段里"。"""
    return re.sub(r"[\s#、，。：:()（）\-—·|]", "", text)


def doc_title(text: str, filename: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line:
            break
    return Path(filename).stem


def is_generic(title: str) -> bool:
    stripped = re.sub(r"^[\d一二三四五六七八九十]+[、.．]\s*", "", title).strip()
    return bool(GENERIC_TITLE.search(stripped)) or len(stripped) <= 4


def option(name: str, default: int) -> int:
    if name in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(name) + 1])
        except (IndexError, ValueError):
            pass
    return default


def load_docs(target: Path, limit: int) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        body = path.read_text(encoding="utf-8", errors="ignore")
        if len(body) >= MIN_DOC_CHARS:
            docs.append((str(path.relative_to(target)).replace("\\", "/"), body))
    return docs[:limit] if limit else docs


def build_questions(docs: list[tuple[str, str]], sample_size: int):
    doc_queries = [(doc_title(body, name), name) for name, body in docs]
    sections: list[tuple[str, str]] = []
    for name, body in docs:
        for match in re.finditer(r"(?m)^##\s+(.*)$", body):
            title = match.group(1).strip()
            if 4 <= len(title) <= 30:
                sections.append((title, name))
    random.seed(RANDOM_SEED)
    sampled = random.sample(sections, min(sample_size, len(sections))) if sections else []
    specific = [(title, source) for title, source in sampled if not is_generic(title)]
    return doc_queries, sampled, specific


def evaluate(
    strategy_settings: dict,
    store: MemoryStore,
    questions: dict,
    embeddings: list,
    index: dict,
    top_k: int,
) -> dict:
    """用给定策略跑一遍全部问题，返回命中统计。

    embeddings 由调用方提前批量算好并按 id 索引 —— 三种策略复用同一批向量，
    这是把"四次向量化"压成"一次"的关键，也是这个脚本能便宜跑完的原因。
    """
    doc_questions = questions["doc"]
    sampled = questions["sampled"]
    specific = questions["specific"]
    cfg = dataclasses.replace(
        base_settings,
        vector_backend="memory",
        memory_index_file=store.path,
        embedding_mode="local",
        embedding_api_key="",
        # 注意：这里**不能**清空 llm_api_key —— listwise 精排要用对话模型排序。
        # 检索本身不会调用 chat，所以留着 Key 是安全的。
        min_vector_score=0.0,      # 评测只关心排序，不要让阈值把候选过滤掉
        min_rerank_score=0.0,
        top_k=top_k,
        **strategy_settings,
    )
    retriever = HybridRetriever(cfg, store, LLMClient(cfg))

    doc1 = doc4 = 0
    per_question: list[dict] = []
    rerank_failures = 0

    def hit_flags(question: str, key: str) -> tuple[list[str], bool, bool]:
        result = retriever.search(question, top_k, embedding=embeddings[index[question]])
        hits = result.top(top_k)  # Hit@K只能检查前K个候选，不能统计整个召回池。
        sources = [hit.source for hit in hits]
        if strategy_settings.get("rerank_enabled") and not result.rerank_used:
            nonlocal rerank_failures
            rerank_failures += 1
        flags = [key in norm(hit.text) for hit in hits]
        return sources, (bool(flags) and flags[0]), any(flags)

    for question, expected in doc_questions:
        sources, _, _ = hit_flags(question, "")
        doc1 += bool(sources) and sources[0] == expected
        doc4 += expected in sources

    sec1 = sec4 = spec1 = spec4 = 0
    for question, _ in sampled:
        key = norm(question)[:8]
        sources, first, any_hit = hit_flags(question, key)
        sec1 += first
        sec4 += any_hit
        per_question.append({"question": question, "top1": first, "top4": any_hit, "sources": sources[:top_k]})
    for question, _ in specific:
        key = norm(question)[:8]
        _, first, any_hit = hit_flags(question, key)
        spec1 += first
        spec4 += any_hit

    return {
        "doc1": doc1, "doc4": doc4,
        "sec1": sec1, "sec4": sec4, "spec1": spec1, "spec4": spec4,
        "rerank_failures": rerank_failures,
        "per_question": per_question,
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0]) if args else (BASE_DIR / "data" / "corpus")
    if not target.exists():
        print(f"[×] 目录不存在：{target}")
        print('    用法：python scripts/evaluate_retrieval.py "D:\\你的\\文档目录"')
        return 1

    limit = option("--limit", 0)
    sample_size = option("--sample", 120)
    top_k = option("--top-k", 4)
    grid = "--grid" in sys.argv

    docs = load_docs(target, limit)
    if not docs:
        print(f"[×] {target} 里没有可用的文档（需要 {MIN_DOC_CHARS} 字以上的 md/txt）")
        return 1

    total_chars = sum(len(body) for _, body in docs)
    doc_questions, sampled, specific = build_questions(docs, sample_size)
    questions = {"doc": doc_questions, "sampled": sampled, "specific": specific}
    print(f"语料：{target}")
    print(f"      {len(docs)} 篇，{total_chars} 字")
    print(f"问题：文档级 {len(doc_questions)}｜章节级 {len(sampled)}（剔通用后 {len(specific)}）")
    print(f"向量化：{base_settings.embedding_mode} / {base_settings.embedding_model}")
    print(f"检索策略：{' → '.join(name for name, _ in STRATEGIES)}")
    print(f"Top-K = {top_k}；切分配置 = {'4 种全跑' if grid else '仅 heading'}\n")

    settings = base_settings
    settings.ensure_dirs()
    llm = LLMClient(settings)
    if not llm.embedding_key() and settings.embedding_mode == "api":
        print("[×] EMBEDDING_MODE=api 但没有可用的 Key，先跑 python scripts/check_embedding.py")
        return 1

    work_dir = BASE_DIR / "data" / "_eval"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    configs = list(CHUNKERS.items()) if grid else [("heading", CHUNKERS["heading"])]
    all_questions = [question for question, _ in doc_questions] + [question for question, _ in sampled]
    if not all_questions:
        print("[×] 没能生成任何评测问题")
        return 1

    # 所有问题只向量化一次，三种策略共用
    print(f"正在向量化 {len(all_questions)} 个评测问题（一次，三种策略复用）...")
    started = time.time()
    question_vectors = llm.embed(all_questions)
    print(f"  完成，用时 {time.time() - started:.0f}s\n")
    index = {question: i for i, question in enumerate(all_questions)}

    report_lines: list[str] = [
        f"# 检索策略对比报告",
        "",
        f"- 语料：`{target}`（{len(docs)} 篇 / {total_chars} 字）",
        f"- 向量模型：`{llm.describe()['embedding_model']}`",
        f"- 问题数：文档级 {len(doc_questions)}｜章节级 {len(sampled)}（剔通用后 {len(specific)}）",
        f"- Top-K：{top_k}",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 切分 | 检索策略 | 片段数 | 文档级 Top-1 | 章节级 Top-1（剔通用） | 章节级 Top-4（剔通用） |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for label, (display, chunker) in configs:
        started = time.time()
        index_file = work_dir / f"{label}.json"
        store = MemoryStore(index_file)
        ids, texts, metadatas = [], [], []
        for name, body in docs:
            for position, chunk in enumerate(chunker(body)):
                ids.append(f"{name}::{position}")
                texts.append(chunk)
                metadatas.append({"source": name, "chunk_index": position, "chars": len(chunk)})
        print(f"[{display}] {len(ids)} 段，正在向量化...")
        store.add(ids, texts, llm.embed(texts), metadatas, persist=False)
        store.flush()
        print(f"  入库完成，用时 {time.time() - started:.0f}s")

        for name, strategy in STRATEGIES:
            started = time.time()
            metrics = evaluate(strategy, store, questions, question_vectors, index, top_k)
            spec1_rate = metrics["spec1"] / max(1, len(specific))
            spec4_rate = metrics["spec4"] / max(1, len(specific))
            print(
                f"  · {name:<22} 章节级 Top-1 {spec1_rate:.0%}｜Top-4 {spec4_rate:.0%}"
                f"｜用时 {time.time() - started:.0f}s"
            )
            if metrics["rerank_failures"]:
                print(f"    [!] 有 {metrics['rerank_failures']} 个问题没走成精排（服务不可用），结果已降级")
            report_lines.append(
                f"| {display} | {name} | {len(ids)} | {metrics['doc1'] / max(1, len(doc_questions)):.0%} |"
                f" **{spec1_rate:.0%}** | {spec4_rate:.0%} |"
            )

            report_path = work_dir / f"questions-{label}-{len(report_lines)}.json"
            report_path.write_text(json.dumps(metrics["per_question"], ensure_ascii=False, indent=2), "utf-8")
        print()

    report_lines += [
        "",
        "## 怎么读这张表",
        "",
        "- **章节级 Top-1（剔通用）才是真实能力**：随机抽 `##` 小节标题当问题，",
        "  看第一名是否命中了对应小节；'背景/总结'这类每篇都有的标题已剔除，因为它们没有区分度。",
        "- 文档级太容易（命中同一篇文章就算对），只作参考。",
        "- 逐题明细在同目录的 `questions-*.json` 里，答错的题可以直接拉出来复盘。",
    ]
    report = "\n".join(report_lines)
    report_file = work_dir / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    report_file.write_text(report, "utf-8")

    print("=" * 100)
    print(report)
    print("=" * 100)
    print(f"报告已保存：{report_file}")
    print("（data/_eval 已加入 .gitignore；要长期归档请把 report-*.md 复制到别处）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
