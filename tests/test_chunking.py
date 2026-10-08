"""切分策略的测试。

切分是 RAG 里最容易被忽略、又最影响召回的一环，所以这几条必须锁死：
1. 每个 chunk 不超过 chunk_size；
2. 相邻 chunk 之间有重叠（防止答案正好落在接缝处）；
3. Markdown 标题优先成段（标题是天然语义边界）；
4. 过短的碎片被丢掉（"## 一、"这种没有信息量的段落留着只会污染召回）。
"""

from __future__ import annotations

from app.chunking import split_by_heading, split_text

LONG = "。".join(f"这是第{i}句关于产品价格的说明，内容长度刻意写长一点" for i in range(30)) + "。"


def test_chunks_respect_max_size():
    chunks = split_text(LONG, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_adjacent_chunks_overlap():
    chunks = split_text(LONG, chunk_size=200, overlap=60)
    # 取上一块的末尾若干字，它应当出现在下一块的开头
    tail = chunks[0][-20:]
    assert tail in chunks[1]


def test_overlap_is_clamped_to_half_of_chunk_size():
    """overlap 大于 chunk_size/2 时必须被夹住，否则新块反而会超过 chunk_size。"""
    chunks = split_text(LONG, chunk_size=100, overlap=999)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_heading_split_keeps_sections_whole():
    text = "# 标题\n\n## 第一节\n\n短内容。\n\n## 第二节\n\n另一段短内容。\n"
    chunks = split_by_heading(text, chunk_size=500, overlap=80)
    joined = "\n".join(chunks)
    assert "## 第一节" in joined and "## 第二节" in joined
    assert len(chunks) == 3          # 标题 + 两个小节（标题单独一段）


def test_heading_split_falls_back_for_long_section():
    text = "## 一大段\n\n" + LONG
    chunks = split_by_heading(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_empty_and_whitespace_input():
    assert split_text("", 500, 80) == []
    assert split_text("   \n\n  ", 500, 80) == []
    assert split_by_heading("", 500, 80) == []


def test_tiny_fragments_are_dropped():
    chunks = split_text("短。\n\n" + LONG, chunk_size=200, overlap=40)
    assert all(len(chunk) >= 8 for chunk in chunks)
