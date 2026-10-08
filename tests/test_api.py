"""HTTP 接口测试：路由、状态码、SSE 事件流、统计接口。

用 FastAPI 的 TestClient 打真实路由，但依赖全部换成"临时目录 + 内存向量库"，
所以整个测试文件不联网、不碰你的正式知识库。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.cache import AnswerCache
from app.chat_store import ChatStore
from app.docstore import DocumentStore
from app.llm import LLMClient
from app.rag import RagService
from app.trace import Tracer
from app.usage import UsageStore
from app.vectorstore import MemoryStore
from tests.conftest import DOC_HR, DOC_MANUAL


def build_client(cfg, monkeypatch) -> TestClient:
    """把 app.main 里的模块级单例换成"临时目录 + 内存向量库"的实例。

    为什么能这么替换：路由函数在被调用时才去查模块全局变量，
    所以只要在构造 TestClient 之前换掉，请求就会走测试用的依赖，全程不联网。
    """
    cfg.ensure_dirs()
    llm = LLMClient(cfg)
    store = MemoryStore(cfg.memory_index_file)
    chat = ChatStore(cfg.db_path)
    usage, cache, tracer, docs = UsageStore(cfg), AnswerCache(cfg), Tracer(cfg), DocumentStore(cfg)
    service = RagService(
        cfg, llm, store, chat, usage=usage, cache=cache, tracer=tracer, documents=docs
    )

    monkeypatch.setattr(main_module, "settings", cfg)
    for name, value in {
        "llm_client": llm, "vector_store": store, "chat_store": chat, "usage_store": usage,
        "answer_cache": cache, "tracer": tracer, "document_store": docs, "rag": service,
    }.items():
        monkeypatch.setattr(main_module, name, value)

    return TestClient(main_module.app)


@pytest.fixture()
def client(make_settings, monkeypatch):
    with build_client(make_settings(), monkeypatch) as test_client:
        yield test_client


@pytest.fixture()
def strict_client(make_settings, monkeypatch):
    """阈值 0.95：本地哈希向量的分数分布拦不住不相关内容，测阈值得换一种构造方式。"""
    with build_client(make_settings(min_vector_score=0.95), monkeypatch) as test_client:
        yield test_client


def upload(client, name: str, text: str, dept: str = "") -> dict:
    response = client.post(
        "/api/documents",
        files={"file": (name, text.encode("utf-8"), "text/markdown")},
        data={"dept": dept},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_health_reports_retrieval_configuration(client):
    payload = client.get("/api/health").json()
    assert payload["vector_backend"] == "memory"
    assert payload["retrieval"]["hybrid"] is True
    assert payload["retrieval"]["rerank_ready"] is False
    assert isinstance(payload["warnings"], list)


def test_upload_and_list_documents(client):
    result = upload(client, "产品手册.md", DOC_MANUAL, dept="product")
    assert result["chunks"] > 0
    assert result["embedded_chunks"] == result["chunks"]

    listed = client.get("/api/documents").json()
    assert [doc["source"] for doc in listed] == ["产品手册.md"]
    assert listed[0]["dept"] == "product"


def test_upload_is_idempotent(client):
    upload(client, "产品手册.md", DOC_MANUAL)
    again = upload(client, "产品手册.md", DOC_MANUAL)
    assert again["skipped"] is True
    assert again["embedded_chunks"] == 0


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/api/documents", files={"file": ("empty.md", b"", "text/markdown")}
    )
    assert response.status_code == 400


def test_search_mode_returns_citations_without_generating(client):
    upload(client, "产品手册.md", DOC_MANUAL)
    payload = client.post(
        "/api/chat", json={"question": "40412 是什么", "session_id": "t1", "mode": "search"}
    ).json()
    assert payload["citations"]
    assert payload["citations"][0]["matched_by"]
    assert "仅检索模式" in payload["answer"]


def test_chat_answer_and_cache_hit(client):
    upload(client, "产品手册.md", DOC_MANUAL)
    body = {"question": "专业版多少钱？", "session_id": "t2"}

    first = client.post("/api/chat", json=body).json()
    assert first["cache"] == "miss"
    assert first["citations"]
    assert first["trace_id"]

    second = client.post("/api/chat", json=body).json()
    assert second["cache"] == "exact"
    assert second["answer"] == first["answer"]


def test_chat_refuses_when_nothing_passes_the_threshold(strict_client):
    upload(strict_client, "产品手册.md", DOC_MANUAL)
    payload = strict_client.post(
        "/api/chat", json={"question": "请问怎么养一只柯基犬", "session_id": "t3"}
    ).json()
    assert payload["refused"] is True
    assert payload["citations"] == []


def test_department_filter_over_http(client):
    upload(client, "员工手册.md", DOC_HR, dept="hr")
    payload = client.post(
        "/api/chat",
        json={"question": "实习生每周到岗几天", "session_id": "t4", "mode": "search", "dept": "product"},
    ).json()
    assert payload["citations"] == []

    hr = client.post(
        "/api/chat",
        json={"question": "实习生每周到岗几天", "session_id": "t4", "mode": "search", "dept": "hr"},
    ).json()
    assert hr["citations"]


def test_stream_endpoint_emits_expected_events(client):
    upload(client, "产品手册.md", DOC_MANUAL)
    with client.stream(
        "POST", "/api/chat/stream", json={"question": "专业版多少钱？", "session_id": "t5"}
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    events = [
        json.loads(line[5:].strip())
        for line in body.splitlines()
        if line.startswith("data:")
    ]
    types = [event["type"] for event in events]
    assert types[0] == "meta"
    assert "citations" in types
    assert types[-1] == "done"
    assert events[-1]["answer"]


def test_stats_endpoint_aggregates_everything(client):
    upload(client, "产品手册.md", DOC_MANUAL)
    client.post("/api/chat", json={"question": "专业版多少钱？", "session_id": "t6"})

    payload = client.get("/api/stats").json()
    assert payload["knowledge"]["documents"] == 1
    assert payload["knowledge"]["chunks"] > 0
    assert payload["usage"]["calls"] >= 1
    assert payload["cache"]["entries"] >= 1
    assert payload["trace"]["in_memory"] >= 1


def test_trace_lookup_by_id(client):
    upload(client, "产品手册.md", DOC_MANUAL)
    chat = client.post("/api/chat", json={"question": "专业版多少钱？", "session_id": "t7"}).json()
    trace = client.get(f"/api/traces/{chat['trace_id']}").json()
    assert trace["question"] == "专业版多少钱？"
    assert any(stage["stage"] == "retrieve" for stage in trace["stages"])
    assert trace["kept"] >= 1

    assert client.get("/api/traces/does-not-exist").status_code == 404


def test_clear_cache_and_documents(client):
    upload(client, "产品手册.md", DOC_MANUAL)
    assert client.delete("/api/cache").status_code == 200
    assert client.delete("/api/documents/产品手册.md").json()["deleted_chunks"] > 0
    assert client.get("/api/documents").json() == []
