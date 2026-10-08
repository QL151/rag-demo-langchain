"""用量与成本计量的测试。

这些数字会直接写进简历/汇报，所以计算逻辑必须锁死：
单次费用、按 kind 拆分、缓存命中不产生费用、P95 延迟口径。
"""

from __future__ import annotations

from app.usage import UsageStore


def test_cost_math(make_settings):
    settings = make_settings(price_prompt_per_1k=0.002, price_completion_per_1k=0.008)
    store = UsageStore(settings)
    cost = store.record(kind="answer", prompt_tokens=1000, completion_tokens=500)
    assert cost == 0.002 + 0.004

    summary = store.summary()
    assert summary["calls"] == 1
    assert summary["tokens"] == 1500
    assert summary["cost_yuan"] == round(0.006, 6)
    assert summary["avg_cost_per_call_yuan"] == round(0.006, 6)


def test_cache_hit_costs_nothing(make_settings):
    store = UsageStore(make_settings())
    store.record(kind="answer", prompt_tokens=800, completion_tokens=200, cache_hit=True)
    summary = store.summary()
    assert summary["cost_yuan"] == 0
    assert summary["calls"] == 1


def test_by_kind_breakdown(make_settings):
    store = UsageStore(make_settings())
    store.record(kind="rewrite", prompt_tokens=100, completion_tokens=20)
    store.record(kind="answer", prompt_tokens=1000, completion_tokens=300)
    store.record(kind="rerank", prompt_tokens=500)
    kinds = {row["kind"]: row for row in store.summary()["by_kind"]}
    assert set(kinds) == {"rewrite", "answer", "rerank"}
    assert kinds["answer"]["tokens"] == 1300


def test_p95_latency_is_reported(make_settings):
    store = UsageStore(make_settings())
    for index in range(1, 21):
        store.record(kind="answer", latency_ms=index * 10, first_token_ms=index)
    summary = store.summary()
    assert summary["p95_latency_ms"] >= 190
    assert summary["p95_first_token_ms"] >= 19


def test_estimated_calls_are_flagged(make_settings):
    """服务商没返回 usage 时要标出来，避免把估算值当成真实账单。"""
    store = UsageStore(make_settings())
    store.record(kind="answer", prompt_tokens=100, completion_tokens=50, estimated=True)
    store.record(kind="answer", prompt_tokens=100, completion_tokens=50, estimated=False)
    summary = store.summary()
    assert summary["estimated_calls"] == 1
    assert summary["estimated_ratio"] == 0.5


def test_recent_returns_latest_first(make_settings):
    store = UsageStore(make_settings())
    store.record(kind="answer", model="m1")
    store.record(kind="answer", model="m2")
    recent = store.recent(limit=5)
    assert [row["model"] for row in recent] == ["m2", "m1"]
