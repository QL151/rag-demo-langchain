"""在真实语料上评测检索命中率（全自动、可复现）。

> ⚠️ v0.2 起推荐用 `scripts/evaluate_retrieval.py`（功能是它的超集：
> 除了 4 种切分策略的对比，还能做「纯向量 → +BM25/RRF → +精排」的消融实验，
> 并且三种策略共用同一批向量，不会把 embedding 调用量翻倍）。
> 本脚本保留下来，是因为它讲清了"怎么设计一个不会自欺欺人的评测集"——
> 通用标题剔除、两级指标、问题采样，这些思路在新脚本里是一样的。
> 想跑 4 种切分的对比：`python scripts/evaluate_retrieval.py "目录" --grid`

用法（在项目根目录执行）：

    # 完整跑一遍（74 篇 64 万字的语料大约 5 分钟）
    python scripts/evaluate_corpus.py "D:\\workbuddywork\\2026-09-16-14-36-17\\rag-lab\\data\\docs_ai"

    # 先小规模试跑，确认脚本能跑通（快）
    python scripts/evaluate_corpus.py "D:\\path\\to\\docs" --limit 8 --sample 20

参数：
    --limit N    最多用前 N 篇文档（默认全部）
    --sample N   章节级评测抽多少个标题（默认 120）
    --top-k N    检索取前 k 条（默认 4）

它做什么：
    1. 读取目录下所有 md / txt
    2. 用 4 种切分配置分别建索引（索引文件写在 data/_eval/ 下，**不会动你的正式知识库**）
    3. 自动生成两组问题：
       · 文档级：用文档标题提问 → 看能否命中"正确的那篇文章"（较容易，区分度低）
       · 章节级：用 `##` 小节标题提问 → 看命中片段里是否包含该小节（较难，区分度高）
       并把"背景 / 总结 / 展望"这类**每篇都有的通用标题**单独剔除统计
    4. 打印对比表

为什么要剔除"通用标题"？
    如果拿"一、背景"去检索，命中的是哪篇文章的"背景"都不算错 ——
    这种问题**没有区分度**，会把真实成绩拉低十几个百分点。这是评测集设计的常见坑。

怎么用这个结果：
    · 章节级 Top-1（剔除通用标题后）才是"能不能定位到对的那一段"的真实指标；
    · 换切分策略 / 换 embedding 模型后重跑，看数字有没有提升 —— 这就是简历上的量化成果。
"""

from __future__ import annotations

import dataclasses
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import split_text  # noqa: E402
from app.config import BASE_DIR, settings  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.vectorstore import MemoryStore  # noqa: E402

try:  # 做了「实验 9」之后，这里会自动用你自己写进 chunking.py 的版本
    from app.chunking import split_by_heading  # type: ignore[attr-defined]  # noqa: E402
except ImportError:  # 还没做实验 9，就用脚本里这份等价实现

    def split_by_heading(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
        """优先按 Markdown 标题切分；某一段还是太长，再对它按句子切。"""
        parts = re.split(r"(?m)^(?=#{1,6}\s)", text)
        chunks: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            chunks.extend([part] if len(part) <= chunk_size else split_text(part, chunk_size, overlap))
        return chunks


SUFFIXES = {".md", ".markdown", ".txt"}
MIN_DOC_CHARS = 200
RANDOM_SEED = 42

# 通用小节标题：几乎每篇文档都有，用它提问无法判断检索对不对
GENERIC_TITLE = re.compile(
    r"^(背景|前言|简介|概述|导读|总结|小结|结语|展望|写在最后|参考文献|参考资料|"
    r"关于我们|团队介绍|作者|附录|目录)|"
    r"(总结|展望|背景|简介|概述|结语|前言)$"
)


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


def build_store_for(label: str, docs: list[tuple[str, str]], chunker, llm: LLMClient, tmp_dir: Path) -> tuple[MemoryStore, int]:
    """按给定切分方式把语料灌进一个独立的临时索引。"""
    index_file = tmp_dir / f"{label}.json"
    store = MemoryStore(index_file)
    ids, texts, metadatas = [], [], []
    for name, body in docs:
        for i, chunk in enumerate(chunker(body)):
            ids.append(f"{name}::{i}")
            texts.append(chunk)
            metadatas.append({"source": name, "chunk_index": i, "chars": len(chunk)})
    store.add(ids, texts, llm.embed(texts), metadatas)
    return store, len(ids)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    def opt(name: str, default: int) -> int:
        if name in sys.argv:
            try:
                return int(sys.argv[sys.argv.index(name) + 1])
            except (IndexError, ValueError):
                pass
        return default

    target = Path(args[0]) if args else (BASE_DIR / "data" / "corpus")
    if not target.exists():
        print(f"[×] 目录不存在：{target}")
        print('    用法：python scripts/evaluate_corpus.py "D:\\你的\\文档目录"')
        return 1

    limit = opt("--limit", 0)
    sample_size = opt("--sample", 120)
    top_k = opt("--top-k", 4)

    # --only chunk500,heading  → 只跑指定的切分配置（省钱省时间）
    only: set[str] | None = None
    if "--only" in sys.argv:
        try:
            only = {name.strip() for name in sys.argv[sys.argv.index("--only") + 1].split(",")}
        except IndexError:
            only = None

    files = [p for p in sorted(target.rglob("*")) if p.is_file() and p.suffix.lower() in SUFFIXES]
    docs: list[tuple[str, str]] = []
    for path in files:
        body = path.read_text(encoding="utf-8", errors="ignore")
        if len(body) >= MIN_DOC_CHARS:
            docs.append((str(path.relative_to(target)).replace("\\", "/"), body))
    if limit:
        docs = docs[:limit]
    if not docs:
        print(f"[×] {target} 里没有可用的文档（需要 {MIN_DOC_CHARS} 字以上的 md/txt）")
        return 1

    total_chars = sum(len(b) for _, b in docs)
    print(f"语料：{target}")
    print(f"      {len(docs)} 篇，{total_chars} 字")
    print(f"      向量化方式：{settings.embedding_mode}；向量库：MemoryStore（临时索引，不影响正式知识库）\n")

    # ---- 生成问题 ----
    doc_queries = [(doc_title(body, name), name) for name, body in docs]
    sections: list[tuple[str, str]] = []
    for name, body in docs:
        for m in re.finditer(r"(?m)^##\s+(.*)$", body):
            title = m.group(1).strip()
            if 4 <= len(title) <= 30:
                sections.append((title, name))
    random.seed(RANDOM_SEED)
    sampled = random.sample(sections, min(sample_size, len(sections))) if sections else []
    specific = [(t, s) for t, s in sampled if not is_generic(t)]
    print(f"文档级问题：{len(doc_queries)} 个（用文档标题提问）")
    print(f"章节级问题：{len(sampled)} 个（从 {len(sections)} 个 ## 标题中随机抽），其中非通用标题 {len(specific)} 个\n")

    configs = [
        ("chunk300", "按句子切 CHUNK_SIZE=300", lambda t: split_text(t, 300, 60)),
        ("chunk500", "按句子切 CHUNK_SIZE=500（默认）", lambda t: split_text(t, 500, 80)),
        ("chunk800", "按句子切 CHUNK_SIZE=800", lambda t: split_text(t, 800, 100)),
        ("heading", "按 Markdown 标题切（超长再按句子）", lambda t: split_by_heading(t, 500, 80)),
    ]
    if only:
        configs = [c for c in configs if c[0] in only]
        if not configs:
            print(f"[×] --only 指定的配置不存在，可选：chunk300, chunk500, chunk800, heading")
            return 1
        print(f"（--only：只评测 {', '.join(c[0] for c in configs)}）\n")

    tmp_dir = BASE_DIR / "data" / "_eval"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    settings.ensure_dirs()
    llm = LLMClient(settings)
    rows: list[str] = []
    nd, ns, nn = len(doc_queries), len(sampled), len(specific)

    try:
        for label, display, chunker in configs:
            started = time.time()
            store, chunks = build_store_for(label, docs, chunker, llm, tmp_dir)

            doc1 = doc4 = 0
            for question, expected in doc_queries:
                sources = [h.metadata.get("source") for h in store.query(llm.embed([question])[0], top_k)]
                doc1 += bool(sources) and sources[0] == expected
                doc4 += expected in sources

            def section_hit(question: str) -> tuple[bool, bool]:
                hits = store.query(llm.embed([question])[0], top_k)
                key = norm(question)[:8]
                flags = [key in norm(h.text) for h in hits]
                return (bool(flags) and flags[0]), any(flags)

            sec1 = sec4 = 0
            for question, _ in sampled:
                first, any_hit = section_hit(question)
                sec1 += first
                sec4 += any_hit

            spec1 = spec4 = 0
            for question, _ in specific:
                first, any_hit = section_hit(question)
                spec1 += first
                spec4 += any_hit

            rows.append(
                f"| {display} | {chunks} | {doc1 / nd:.0%} | {doc4 / nd:.0%} | "
                f"{sec1 / ns:.0%} | **{spec1 / nn:.0%}** | {spec4 / nn:.0%} |"
            )
            print(
                f"[完成] {display}：片段 {chunks}｜文档级 Top-1 {doc1 / nd:.0%}｜"
                f"章节级 Top-1(剔通用) {spec1 / nn:.0%}｜用时 {time.time() - started:.0f}s"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)   # 评测结束清掉临时索引

    print("\n" + "=" * 96)
    print("| 切分配置 | 总片段数 | 文档级 Top-1 | 文档级 Top-4 | 章节级 Top-1(全部) | 章节级 Top-1(剔通用) | 章节级 Top-4(剔通用) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(row)
    print("=" * 96)
    print(f"问题数：文档级 {nd}｜章节级 {ns}（其中非通用 {nn}）｜Top-K = {top_k}")
    print("读法：'章节级 Top-1(剔通用)' 才是真实能力；'文档级' 太容易，区分度低。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
