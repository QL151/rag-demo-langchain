"""RagService 的端到端测试：增量入库、查询改写、拒答、缓存、Trace。

全部离线运行（本地哈希向量化 + 不联网），几秒跑完。
"""

from __future__ import annotations

from tests.conftest import DOC_MANUAL


class StubLLM:
    """把"生成"换掉，其余（embed / signature）继续用真实实现。

    为什么不 mock 整个 LLMClient：我们要测的是编排逻辑，
    向量化和指纹逻辑必须是真的，否则测不出增量入库这类行为。

    replies 支持按调用顺序返回不同内容 —— 查询改写和最终生成是两次独立调用，
    用同一句话会让断言变得自我实现、测不出东西。
    """

    def __init__(self, real, reply: str | None = None, replies: list[str] | None = None):
        self._real = real
        self.replies = list(replies) if replies else [reply or "这是一个测试回答。"]
        self.llm_ready = True
        self.signature = real.signature
        self.calls: list[list[dict]] = []

    def embed(self, texts):
        return self._real.embed(texts)

    def _next(self) -> str:
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    def chat(self, messages, stream=False, usage_sink=None):
        self.calls.append(list(messages))
        if usage_sink is not None:
            usage_sink.update({"prompt_tokens": 120, "completion_tokens": 30, "estimated": False})
        reply = self._next()
        if stream:
            return iter([reply])
        return reply


# ------------------------------------------------------------------ 增量入库
def test_first_ingest_embeds_everything(rag):
    result = rag.ingest_text("手册.md", DOC_MANUAL, dept="product")
    assert result["skipped"] is False
    assert result["embedded_chunks"] == result["chunks"] > 0
    assert result["reused_chunks"] == 0


def test_unchanged_document_is_skipped(rag):
    first = rag.ingest_text("手册.md", DOC_MANUAL)
    second = rag.ingest_text("手册.md", DOC_MANUAL)
    assert second["skipped"] is True
    assert second["embedded_chunks"] == 0
    assert second["chunks"] == first["chunks"]


def test_only_changed_chunks_are_re_embedded(rag):
    """改一个错别字不该让整份文档重新向量化 —— 这是增量入库的核心价值。"""
    first = rag.ingest_text("手册.md", DOC_MANUAL)
    assert first["chunks"] >= 3

    edited = DOC_MANUAL.replace("199 元每月", "299 元每月")
    second = rag.ingest_text("手册.md", edited)
    assert second["skipped"] is False
    assert second["embedded_chunks"] == 1
    assert second["reused_chunks"] == first["chunks"] - 1


def test_force_rebuild_embeds_everything(rag):
    rag.ingest_text("手册.md", DOC_MANUAL)
    forced = rag.ingest_text("手册.md", DOC_MANUAL, force=True)
    assert forced["skipped"] is False
    assert forced["embedded_chunks"] == forced["chunks"]


def test_ingest_bumps_cache_version(rag):
    """入库必须让答案缓存失效，否则用户上传了新文档还拿到旧答案。"""
    before = rag.cache.version()
    rag.ingest_text("手册.md", DOC_MANUAL)
    assert rag.cache.version() == before + 1


def test_delete_document_removes_ledger_and_bumps_version(rag):
    rag.ingest_text("手册.md", DOC_MANUAL)
    version = rag.cache.version()
    deleted = rag.delete_document("手册.md")
    assert deleted > 0
    assert rag.store.count() == 0
    assert rag.documents.get("手册.md") is None
    assert rag.cache.version() == version + 1


# ------------------------------------------------------------------ 查询改写
def test_rewrite_uses_history_to_resolve_reference(rag_rewrite):
    rag_rewrite.llm = StubLLM(rag_rewrite.llm, "学生版多少钱？")
    history = [
        {"role": "user", "content": "专业版多少钱？"},
        {"role": "assistant", "content": "专业版 199 元/月。"},
    ]
    rewritten, changed = rag_rewrite.rewrite_query(history, "那学生呢？")
    assert changed is True
    assert rewritten == "学生版多少钱？"


def test_rewrite_skipped_on_first_turn(rag_rewrite):
    rag_rewrite.llm = StubLLM(rag_rewrite.llm, "不该被调用")
    rewritten, changed = rag_rewrite.rewrite_query([], "专业版多少钱？")
    assert changed is False
    assert rewritten == "专业版多少钱？"


def test_rewrite_failure_falls_back_to_original_question(rag_rewrite):
    """改写失败绝不能阻塞主链路。"""

    class BrokenLLM(StubLLM):
        def chat(self, messages, stream=False, usage_sink=None):
            raise RuntimeError("改写服务挂了")

    rag_rewrite.llm = BrokenLLM(rag_rewrite.llm)
    history = [{"role": "user", "content": "专业版多少钱？"}]
    rewritten, changed = rag_rewrite.rewrite_query(history, "那学生呢？")
    assert changed is False
    assert rewritten == "那学生呢？"


def test_rewrite_result_is_used_for_retrieval_only(rag_rewrite):
    """改写后的问句只用于检索，生成侧拿到的仍然是原问题 + 历史。

    这条很容易做错：如果生成侧也换成改写后的问句，回答就会丢掉用户的原始措辞，
    而且多轮历史会被重复表述一遍。
    """
    stub = StubLLM(rag_rewrite.llm, replies=["学生版多少钱？", "学生版 99 元/月。[1]"])
    rag_rewrite.llm = stub
    rag_rewrite.ingest_text("产品手册.md", DOC_MANUAL, dept="product")
    rag_rewrite.chat.add("rw", "user", "专业版多少钱？")
    rag_rewrite.chat.add("rw", "assistant", "199 元/月。")

    events = list(rag_rewrite.stream_answer("rw", "那它多少钱？"))
    meta = events[0]
    assert meta["rewritten_query"] == "学生版多少钱？"

    generation_messages = stub.calls[-1]
    assert generation_messages[-1]["role"] == "user"
    assert "那它多少钱？" in generation_messages[-1]["content"]


# ------------------------------------------------------------------ 问答
def test_answer_without_llm_key_returns_retrieval_only(seeded):
    text, citations, meta = seeded.answer("s1", "专业版多少钱？")
    assert citations
    assert "未配置 LLM_API_KEY" in text
    assert meta["refused"] is False


def test_answer_refuses_when_knowledge_base_has_no_answer(seeded_strict):
    """阈值拒答：所有候选都不够相关时，直接说"资料里没有"，不调用大模型。"""
    text, citations, meta = seeded_strict.answer("s2", "请问怎么养一只柯基犬")
    assert meta["refused"] is True
    assert citations == []
    assert "资料中没有提到" in text


def test_second_identical_question_hits_cache(seeded):
    first_text, _, first_meta = seeded.answer("s3", "专业版多少钱？")
    second_text, second_citations, second_meta = seeded.answer("s3", "专业版多少钱？")
    assert first_meta["cache"] == "miss"
    assert second_meta["cache"] == "exact"
    assert second_text == first_text
    assert second_citations


def test_answer_records_usage_and_trace(seeded):
    seeded.llm = StubLLM(seeded.llm, "专业版 199 元/月。[1]")
    _, _, meta = seeded.answer("s4", "专业版多少钱？")
    assert meta["usage"]["prompt_tokens"] == 120
    assert meta["usage"]["cost_yuan"] > 0
    assert meta["trace_id"]

    trace = seeded.tracer.get(meta["trace_id"])
    assert trace is not None
    assert trace["question"] == "专业版多少钱？"
    assert any(stage["stage"] == "retrieve" for stage in trace["stages"])


def test_department_isolation_end_to_end(seeded):
    text, citations, _ = seeded.answer("s5", "实习生每周到岗几天", dept="product")
    assert all(citation.dept != "hr" for citation in citations)


def test_stream_answer_emits_meta_citations_delta_done(seeded):
    seeded.llm = StubLLM(seeded.llm, "专业版 199 元/月。[1]")
    events = list(seeded.stream_answer("s6", "专业版多少钱？"))
    types = [event["type"] for event in events]
    assert types[0] == "meta"
    assert "citations" in types
    assert "delta" in types
    assert types[-1] == "done"
    assert events[-1]["answer"] == "专业版 199 元/月。[1]"
    assert events[-1]["usage"]["prompt_tokens"] == 120


def test_stream_answer_refusal_does_not_call_the_model(seeded_strict):
    stub = StubLLM(seeded_strict.llm)
    seeded_strict.llm = stub
    events = list(seeded_strict.stream_answer("s7", "请问怎么养一只柯基犬"))
    assert events[-1]["refused"] is True
    assert stub.calls == []          # 拒答时一次模型都没调用 → 这次提问成本为 0
    assert events[0]["cache"] == "miss"
