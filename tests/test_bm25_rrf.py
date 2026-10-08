"""BM25 与 RRF 的单元测试。

这两个是混合检索的地基，出问题会让"检索质量"整体下降但**不报错**，
所以必须有单元测试盯着它们，而不是靠肉眼看检索结果。
"""

from __future__ import annotations

from app.retrieval import BM25, rrf_fuse, tokenize


def test_tokenize_keeps_bigrams_for_chinese():
    tokens = tokenize("退款政策")
    assert "退" in tokens and "款" in tokens
    assert "退款" in tokens and "款政" in tokens and "政策" in tokens


def test_tokenize_lowercases_and_keeps_alphanumeric():
    tokens = tokenize("Model X200-PRO 40412")
    assert "model" in tokens and "x200" in tokens and "pro" in tokens and "40412" in tokens


def test_bm25_ranks_exact_code_first():
    docs = {
        "a": "遇到 40412 表示配额已用尽。",
        "b": "遇到 50001 表示服务内部错误。",
        "c": "这一段和错误码完全无关，讲的是公司的报销流程。",
    }
    ranked = BM25(docs).search("40412 是什么意思", top_k=3)
    assert ranked[0][0] == "a"
    assert ranked[0][1] > 0


def test_bm25_idf_is_never_negative():
    """idf 用了 log(1+...) 平滑，高频词不会得到负分（否则命中它反而扣分）。"""
    docs = {str(i): "价格 价格 价格" for i in range(5)}
    index = BM25(docs)
    assert all(value >= 0 for value in index.idf.values())


def test_bm25_returns_nothing_for_unrelated_query():
    docs = {"a": "退款政策说明"}
    assert BM25(docs).search("量子力学", top_k=3) == []


def test_rrf_rewards_agreement_between_channels():
    """同时被两路召回的结果，必须排在只被一路召回的前面 —— 这是 RRF 的全部意义。"""
    fused = rrf_fuse([["x", "y"], ["y", "z"]])
    assert fused["y"] > fused["x"]
    assert fused["y"] > fused["z"]


def test_rrf_handles_empty_and_single_channel():
    assert rrf_fuse([]) == {}
    assert rrf_fuse([[]]) == {}
    fused = rrf_fuse([["a", "b"]])
    assert fused["a"] > fused["b"]


def test_rrf_scores_are_rank_based_not_score_based():
    """RRF 只看名次：两路各自的第一名得分必须相同，与原始分数无关。"""
    fused = rrf_fuse([["a"], ["b"]])
    assert fused["a"] == fused["b"]
