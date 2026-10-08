"""精排测试：cross-encoder 的熔断、listwise 降级、以及"顺序不等于分数"这条纪律。

精排是最容易"看起来在工作、其实在帮倒忙"的一层，所以这几条必须锁死：
- 精排服务挂了不能把整个问答拖垮（要降级，而且要留下说明）；
- 连续失败要熔断，否则每次提问都白等一次超时/403；
- listwise 精排只给顺序不给分数，**不能**拿名次去冒充相关度分去过阈值。
"""

from __future__ import annotations

import dataclasses

import pytest

from app.llm import LLMClient
from app.retrieval import Candidate, LLMReranker, Reranker, RetrievalResult
from app.vectorstore import MemoryStore


class StubLLM:
    def __init__(self, reply: str, ready: bool = True):
        self.reply = reply
        self.llm_ready = ready
        self.signature = "local:hash-1024"
        self.calls = 0

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def chat(self, messages, stream=False, usage_sink=None):
        self.calls += 1
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 300, "completion_tokens": 8, "estimated": False})
        return self.reply


def candidates() -> list[Candidate]:
    return [
        Candidate(id="a", text="实习生每周需要到岗 4 天。", metadata={"source": "hr.md"}),
        Candidate(id="b", text="专业版 199 元每月。", metadata={"source": "p.md"}),
        Candidate(id="c", text="差旅报销需要发票原件。", metadata={"source": "hr.md"}),
    ]


# ------------------------------------------------------------------ 解析
@pytest.mark.parametrize(
    "reply,expected",
    [
        ("[2,1]", [2, 1]),
        ("```json\n[2, 1]\n```", [2, 1]),
        ("我认为相关的是：[2, 1]。", [2, 1]),
        ("[2, 2, 1]", [2, 1]),          # 去重
        ("[9, 0]", []),                 # 越界全部丢掉 → 等于"没有相关的"
        ("[]", []),                     # 模型明确说全部无关
        ("没有相关内容", None),           # 解析失败 ≠ 全部无关，这两者必须区分
        ("", None),
    ],
)
def test_listwise_parse(reply, expected):
    assert LLMReranker._parse(reply, 3) == expected


def test_listwise_empty_result_is_a_refusal_signal(make_settings):
    """模型说"都不相关"时必须当成拒答信号，不能当成解析失败。"""
    settings = make_settings(rerank_enabled=True, rerank_provider="llm", llm_api_key="fake")
    outcome = LLMReranker(settings, StubLLM("[]")).rerank("怎么养柯基", candidates(), 3)
    assert outcome.ok is True
    assert outcome.no_relevant is True
    assert outcome.mode == "listwise"


# ------------------------------------------------------------------ listwise 精排
def test_listwise_rerank_sets_rank_not_score(make_settings):
    settings = make_settings(rerank_enabled=True, rerank_provider="llm", llm_api_key="fake", rewrite_enabled=False)
    stub = StubLLM("[2,1,3]")
    options = candidates()
    outcome = LLMReranker(settings, stub).rerank("专业版多少钱", options, top_n=3)

    assert outcome.ok is True
    assert outcome.mode == "listwise"
    assert outcome.usage["prompt_tokens"] == 300
    assert [item.rerank_rank for item in options] == [2, 1, 3]
    # 关键纪律：listwise 不产生分数
    assert all(item.rerank_score is None for item in options)


def test_listwise_rerank_skips_when_llm_unavailable(make_settings):
    settings = make_settings(rerank_enabled=True, rerank_provider="llm", llm_api_key="fake")
    outcome = LLMReranker(settings, StubLLM("[1]", ready=False)).rerank("q", candidates(), 3)
    assert outcome.ok is False


def test_listwise_rerank_survives_model_exception(make_settings):
    class Broken(StubLLM):
        def chat(self, messages, stream=False, usage_sink=None):
            raise RuntimeError("模型服务挂了")

    settings = make_settings(rerank_enabled=True, rerank_provider="llm", llm_api_key="fake")
    outcome = LLMReranker(settings, Broken("[1]")).rerank("q", candidates(), 3)
    assert outcome.ok is False
    assert "挂了" in outcome.reason


# ------------------------------------------------------------------ 熔断
def test_cross_encoder_breaker_opens_after_repeated_failures(make_settings):
    settings = make_settings(
        rerank_enabled=True, rerank_provider="http", rerank_api_key="k",
        rerank_breaker_threshold=2, rerank_breaker_seconds=60,
    )
    reranker = Reranker(settings)
    reranker._post = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("403 Forbidden"))

    assert reranker.ready is True
    assert reranker.rerank("q", candidates(), 3) is False
    assert reranker.rerank("q", candidates(), 3) is False
    # 连续失败到阈值 → 熔断，ready 变 False，后续不会再白等
    assert reranker.breaker_open is True
    assert reranker.ready is False
    assert reranker.rerank("q", candidates(), 3) is False


def test_breaker_resets_after_success(make_settings):
    settings = make_settings(rerank_enabled=True, rerank_provider="http", rerank_api_key="k", rerank_breaker_threshold=3)
    reranker = Reranker(settings)
    reranker._post = lambda *args, **kwargs: {"results": [{"index": 1, "relevance_score": 0.9}]}
    options = candidates()
    assert reranker.rerank("q", options, 3) is True
    assert reranker.failures == 0
    assert options[1].rerank_score == 0.9
    assert options[0].rerank_score is None


def test_reranker_extracts_both_response_shapes():
    assert Reranker._extract_results({"output": {"results": [{"index": 0}]}}) == [{"index": 0}]
    assert Reranker._extract_results({"results": [{"index": 0}]}) == [{"index": 0}]
    assert Reranker._extract_results({"unexpected": 1}) == []


# ------------------------------------------------------------------ 和阈值的关系
def test_retriever_refuses_when_listwise_says_nothing_relevant(make_settings):
    """端到端：向量分过线，但 listwise 说"都不相关" → 仍然拒答。

    这条对应真实场景：1700 多段的语料里，随便什么问题都能找到向量分 0.35 的片段，
    固定阈值拦不住；让模型看一眼再决定，比调阈值可靠得多。
    """
    settings = make_settings(
        hybrid_enabled=False, rerank_enabled=True, rerank_provider="llm",
        rerank_llm_fallback=True, llm_api_key="fake", min_vector_score=0.1,
    )
    store = MemoryStore(settings.memory_index_file)
    store.add(
        ["a"],
        ["公司专业版 199 元每月，支持 14 天退款。"],
        [[1.0, 0.0]],
        [{"source": "p.md", "chunk_index": 0}],
    )
    from app.retrieval import HybridRetriever

    retriever = HybridRetriever(settings, store, StubLLM("[]"))
    result = retriever.search("怎么养柯基", embedding=[1.0, 0.0])

    assert result.rerank_mode == "listwise"
    assert result.refused is True
    assert result.candidates == []
    assert "listwise" in result.note


def test_listwise_mode_uses_vector_threshold_not_rank(make_settings):
    """listwise 精排给的是顺序，拒答阈值必须继续看向量分。

    注意这条测的是"模型认为有相关片段"的情形（返回了 [1,2]）：
    这时不能因为"排了名次"就放行，仍然要看向量分。
    """
    settings = make_settings(
        hybrid_enabled=False, rerank_enabled=True, rerank_provider="llm", rerank_llm_fallback=True,
        min_vector_score=0.5, min_rerank_score=0.3, llm_api_key="fake",
    )
    store = MemoryStore(settings.memory_index_file)
    store.add(
        ["a", "b"],
        ["完全不相关的内容", "另一个不相关的内容"],
        [[1.0, 0.0], [0.0, 1.0]],
        [{"source": "x.md", "chunk_index": 0}, {"source": "y.md", "chunk_index": 1}],
    )
    from app.retrieval import HybridRetriever

    retriever = HybridRetriever(settings, store, StubLLM("[1,2]"))
    result = retriever.search("任意问题", embedding=[0.0, 0.0])

    assert result.rerank_mode == "listwise"
    # 向量分全是 0（第三个维度的向量），低于 0.5 → 应当全部拒答，
    # 哪怕 listwise 把它们都排上了名次
    assert result.refused is True
    assert "listwise" in result.note
