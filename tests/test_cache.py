"""答案缓存的测试。

缓存的价值完全取决于"该命中的命中、该失效的失效"，所以两类都要测：
- 精确命中 / 语义命中（省钱的来源）；
- 知识库一改就整体失效（不失效的话，用户上传了新文档还拿到旧答案，
  这是最难排查的一类线上问题）。
"""

from __future__ import annotations

import dataclasses

from app.cache import AnswerCache, normalize_question


def test_normalize_question_ignores_punctuation_and_space():
    assert normalize_question("专业版多少钱？") == normalize_question("专业版多少钱")
    assert normalize_question(" 专业版 多少钱 ") == normalize_question("专业版多少钱")
    assert normalize_question("Pro 版") == normalize_question("pro版")


def test_exact_hit(make_settings):
    cache = AnswerCache(make_settings())
    cache.put(question="专业版多少钱？", answer="199 元/月", citations=[{"index": 1}])

    hit = cache.get("专业版多少钱")
    assert hit is not None
    assert hit["match"] == "exact"
    assert hit["answer"] == "199 元/月"
    assert hit["citations"] == [{"index": 1}]


def test_semantic_hit_with_similar_embedding(make_settings):
    cache = AnswerCache(make_settings(cache_semantic_threshold=0.9))
    cache.put(
        question="专业版的价格是多少",
        answer="199 元/月",
        embedding=[1.0, 0.0, 0.0],
    )

    hit = cache.get("专业版怎么收费", embedding=[0.99, 0.05, 0.0])
    assert hit is not None
    assert hit["match"] == "semantic"
    assert hit["similarity"] >= 0.9


def test_semantic_miss_when_embedding_is_far(make_settings):
    cache = AnswerCache(make_settings(cache_semantic_threshold=0.9))
    cache.put(question="专业版多少钱", answer="199 元/月", embedding=[1.0, 0.0, 0.0])
    assert cache.get("实习生几天到岗", embedding=[0.0, 1.0, 0.0]) is None


def test_version_bump_invalidates_everything(make_settings):
    """知识库更新后，旧答案必须立刻查不到。"""
    cache = AnswerCache(make_settings())
    cache.put(question="专业版多少钱", answer="199 元/月")
    assert cache.get("专业版多少钱") is not None

    cache.bump_version()
    assert cache.get("专业版多少钱") is None


def test_filters_are_part_of_the_key(make_settings):
    """同一个问题在不同部门下必须分开缓存，否则会串数据。"""
    cache = AnswerCache(make_settings())
    cache.put(question="报销标准", answer="销售 500/天", filters={"dept": "sales"})
    assert cache.get("报销标准", filters={"dept": "hr"}) is None
    assert cache.get("报销标准", filters={"dept": "sales"}) is not None


def test_embedding_signature_is_part_of_the_key(make_settings):
    """换了向量模型，旧缓存的语义相似度不再可信。"""
    cache = AnswerCache(make_settings())
    cache.put(question="专业版多少钱", answer="199 元/月", embedding_signature="api:model-a")
    assert cache.get("专业版多少钱", embedding_signature="api:model-b") is None


def test_expired_entries_are_not_returned(make_settings, tmp_path):
    settings = make_settings()
    cache = AnswerCache(settings)
    cache.put(question="专业版多少钱", answer="199 元/月")

    expired = AnswerCache(dataclasses.replace(settings, cache_ttl_seconds=-1))
    assert expired.get("专业版多少钱") is None


def test_disabled_cache_never_hits(make_settings):
    cache = AnswerCache(make_settings(cache_enabled=False))
    cache.put(question="专业版多少钱", answer="199 元/月")
    assert cache.get("专业版多少钱") is None


def test_stats_reports_hit_rate(make_settings):
    cache = AnswerCache(make_settings())
    cache.put(question="专业版多少钱", answer="199 元/月")
    cache.get("专业版多少钱")
    cache.get("完全没问过的问题")

    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["process_hits_exact"] == 1
    assert stats["process_misses"] == 1
    assert stats["hit_rate"] == 0.5
