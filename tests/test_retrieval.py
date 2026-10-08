"""检索层测试：混合召回、阈值拒答、部门隔离、精排降级。

这些用例对应的都是"出问题不报错、只是悄悄变差"的场景 ——
正是最需要测试守住的类型。
"""

from __future__ import annotations

from app.llm import LLMClient
from app.retrieval import Candidate, HybridRetriever, dedupe_by_source, rrf_fuse
from app.vectorstore import MemoryStore


def test_exact_value_query_hits_keyword_channel(seeded):
    """错误码这种精确值，纯向量很容易跑偏，必须有 BM25 兜住。"""
    result = seeded.search("40412 是什么错误")
    sources = [candidate.source for candidate in result.candidates]
    assert "产品手册.md" in sources
    top = result.candidates[0]
    assert "40412" in top.text
    assert "关键词" in top.matched_by


def test_semantic_query_still_works_without_keyword_overlap(seeded):
    """换一种说法（字面不重合）时靠向量路召回，这是混合检索的另一半。"""
    result = seeded.search("专业版的售价是多少钱")
    assert result.candidates, "向量路应当能召回价格那一段"
    assert any("199" in candidate.text for candidate in result.candidates)


def test_unrelated_question_is_refused_by_threshold(seeded_strict):
    """知识库里没有依据时必须拒答，而不是硬凑一段给大模型。

    用 rag_strict（阈值 0.95）而不是默认阈值：本地哈希向量化的相似度是压缩分布，
    按真实 embedding 校准的 0.25 在本地模式下拦不住东西 —— 那是向量模型的问题，
    不是阈值逻辑的问题，所以这里用真实模型的分数分布来测这条逻辑。
    """
    result = seeded_strict.search("请问怎么养一只柯基犬")
    assert result.refused is True
    assert result.candidates == []
    assert result.recalled > 0      # 确实召回了东西，是阈值把它们挡掉的


def test_common_single_character_match_does_not_bypass_the_threshold(seeded_strict):
    """单字"问"这种常见字命中，不该被当成"强关键词命中"而豁免阈值。"""
    result = seeded_strict.search("问")
    assert result.refused is True


def test_department_filter_isolates_documents(seeded):
    hr_only = seeded.search("实习生每周到岗几天", dept="hr")
    assert hr_only.candidates
    assert all(candidate.metadata.get("dept") == "hr" for candidate in hr_only.candidates)

    wrong_dept = seeded.search("实习生每周到岗几天", dept="product")
    assert all(candidate.source != "员工手册.md" for candidate in wrong_dept.candidates)


def test_doc_type_filter(seeded):
    result = seeded.search("退款政策", doc_type="policy")
    assert all(candidate.metadata.get("doc_type") == "policy" for candidate in result.candidates)


def test_rerank_unavailable_degrades_with_a_visible_note(seeded):
    """精排没配 Key 时必须能跑，而且要让调用方知道"这次没有精排"。"""
    result = seeded.search("专业版多少钱")
    assert result.rerank_used is False
    assert result.note  # 有说明，不是静默降级
    assert result.candidates


def test_timings_are_recorded(seeded):
    result = seeded.search("专业版多少钱")
    assert "bm25_ms" in result.timings
    assert "fuse_ms" in result.timings


def test_threshold_decision_is_purely_the_vector_score(make_settings):
    """不依赖任何 embedding 模型，直接构造正交向量来验证阈值逻辑本身。

    这条测试的价值：它把"阈值该不该拦"和"向量模型好不好"彻底解耦 ——
    换 embedding 模型后这条测试永远稳定，其他检索测试可能会飘。
    """
    cfg = make_settings(hybrid_enabled=False, min_vector_score=0.25)
    store = MemoryStore(cfg.memory_index_file)
    store.add(
        ["doc"],
        ["这是一段和问题毫不相干的文本"],
        [[1.0, 0.0, 0.0]],
        [{"source": "x.md", "chunk_index": 0, "chars": 20}],
    )
    retriever = HybridRetriever(cfg, store, LLMClient(cfg))

    orthogonal = retriever.search("任意问题", embedding=[0.0, 1.0, 0.0])
    assert orthogonal.refused is True

    aligned = retriever.search("任意问题", embedding=[1.0, 0.0, 0.0])
    assert aligned.refused is False
    assert aligned.candidates[0].text.startswith("这是一段")


def test_dedupe_limits_chunks_per_source():
    items = [
        Candidate(id="1", text="", metadata={"source": "a"}),
        Candidate(id="2", text="", metadata={"source": "a"}),
        Candidate(id="3", text="", metadata={"source": "a"}),
        Candidate(id="4", text="", metadata={"source": "b"}),
    ]
    kept = dedupe_by_source(items, limit_per_source=2)
    assert [item.source for item in kept] == ["a", "a", "b"]


def test_rrf_fuse_is_importable_from_retrieval():
    assert rrf_fuse([["a"], ["a"]])["a"] > rrf_fuse([["a"]])["a"]


def test_search_top_k_is_respected(seeded):
    result = seeded.search("产品价格和退款", top_k=1)
    assert len(seeded._select(result)) <= 1
