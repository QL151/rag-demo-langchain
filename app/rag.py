"""RAG 主流程：入库（增量）与问答（改写 → 混合检索 → 精排 → 阈值 → 生成 → 缓存 → 计量）。

整个文件就是面试时你要能白板讲出来的东西，所以刻意写得很直白。

一次提问的完整链路（改造后 9 步）：
    1. 取多轮历史（**必须在检索之前取**，因为改写要用它）
    2. 把问题向量化 —— 这一次向量同时喂给"答案缓存"和"向量检索"，不重复调用
    3. 查答案缓存：精确命中 → 直接返回；语义命中（相似度 ≥ 阈值）→ 直接返回
    4. 未命中 → 查询改写（"那学生呢？" → "学生版多少钱？"）
    5. 混合检索：BM25 + 向量双路召回 → RRF 融合 →（可选）rerank 精排
    6. 阈值拒答：所有候选都不够相关 → 直接说"资料里没有"，**不调用大模型**（省钱 + 不编）
    7. 拼 Prompt（带编号的【参考资料】+ 最近 N 轮历史）→ 生成
    8. 结果写回缓存、记 token/成本、写 trace
    9. 返回回答 + 引用（带向量分、精排分、命中通道）

每一步都对应一个能讲清的取舍，注释里都写明了。
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Iterator

from .cache import AnswerCache
from .chat_store import ChatStore
from .chunking import split_by_heading
from .config import Settings
from .docstore import DocumentStore, text_hash
from .llm import EmbeddingUnavailable, LLMClient, estimate_tokens
from .loaders import read_text
from .retrieval import Candidate, HybridRetriever, RetrievalResult, dedupe_by_source
from .schemas import Citation
from .trace import Trace, Tracer
from .usage import UsageStore
from .vectorstore import BaseVectorStore, Hit

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 160

SYSTEM_PROMPT = """你是一个严谨的企业知识库问答助手。

必须遵守：
1. 只依据【参考资料】回答；资料里没有的信息，直接回答"资料中没有提到"，不要编造。
2. 引用资料时在句末标注编号，例如 [1][2]；不要编造不存在的编号。
3. 用中文回答；能分点时用短句分点，不要长篇大论。
4. 问题与资料无关时，礼貌说明并引导用户换个问法。
5. 不要复述【参考资料】的原文段落，要给出结论。"""

REWRITE_PROMPT = """你是检索查询改写器。根据对话历史，把用户的最新问题改写成一个不依赖上下文、可以独立检索的问句。

规则：
- 补全指代：把"那学生呢？""这个多少钱？"里的省略信息从历史里补回来；
- 只输出改写后的问句本身，不要解释、不要加引号、不要分点；
- 如果最新问题本身已经完整，原样输出。"""

NO_LLM_NOTICE = "（未配置 LLM_API_KEY，以下仅返回检索到的原文片段；配置 Key 后即可生成自然语言回答）"

REFUSAL_TEMPLATE = (
    "资料中没有提到相关内容。\n\n"
    "可以这样做：① 换一种说法再问；② 确认相关文档已经上传"
    "（当前知识库 {documents} 篇 / {chunks} 段）；③ 如果确实应该有，"
    "说明它可能没被正确解析或切分。"
)


class RagService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        store: BaseVectorStore,
        chat: ChatStore,
        usage: UsageStore | None = None,
        cache: AnswerCache | None = None,
        tracer: Tracer | None = None,
        documents: DocumentStore | None = None,
    ) -> None:
        self.s = settings
        self.llm = llm
        self.store = store
        self.chat = chat
        self.usage = usage or UsageStore(settings)
        self.cache = cache or AnswerCache(settings)
        self.tracer = tracer or Tracer(settings)
        self.documents = documents or DocumentStore(settings)
        self.retriever = HybridRetriever(settings, store, llm)
        from .workflow import build_workflow
        self.workflow = build_workflow(self)
        self.stream_workflow = build_workflow(self,streaming=True)

    # ------------------------------------------------------------ 入库（增量）
    def ingest_text(
        self,
        source: str,
        text: str,
        *,
        dept: str = "",
        doc_type: str = "",
        force: bool = False,
        persist: bool = True,
    ) -> dict:
        """增量入库：文档没变整篇跳过；变了只重新向量化"文字变了"的片段。

        `persist=False` 用于批量导入：只标脏不落盘，最后统一 flush 一次。
        内置向量库每次 add 都要把整个索引重写一遍，逐篇落盘就是 O(n²) 的磁盘写入。
        """
        dept = dept or self.s.default_dept
        normalized = text or ""
        fingerprint = text_hash(normalized)
        ledger = self.documents.get(source)

        # ---- 第一级：整篇没变，直接跳过 ----
        if (
            not force
            and ledger
            and ledger["doc_hash"] == fingerprint
            and ledger["embedding_signature"] == self.llm.signature
            and self.store.count() > 0
        ):
            logger.info("文档未变化，跳过入库：%s", source)
            return {
                "source": source, "chunks": ledger["chunk_count"], "chars": ledger["chars"],
                "dept": dept, "skipped": True, "reused_chunks": ledger["chunk_count"],
                "embedded_chunks": 0,
            }

        chunks = split_by_heading(normalized, self.s.chunk_size, self.s.chunk_overlap)
        if not chunks:
            return {
                "source": source, "chunks": 0, "chars": 0, "dept": dept,
                "skipped": False, "reused_chunks": 0, "embedded_chunks": 0,
            }

        # ---- 第二级：片段级复用 ----
        # force=True 表示"要全量重建"（典型场景：换了 embedding 模型），
        # 这时必须连片段级复用一起关掉，否则复用的还是旧模型的向量，问题原样存在。
        reusable: dict[str, list[float]] = {}
        if not force and ledger and ledger["embedding_signature"] == self.llm.signature:
            reusable = self.store.embeddings_by_source(source)

        chunk_hashes = [text_hash(chunk) for chunk in chunks]
        pending: list[int] = []
        vectors: list[list[float] | None] = []
        for index, chunk_hash in enumerate(chunk_hashes):
            vector = reusable.get(chunk_hash)
            if vector is None:
                pending.append(index)
                vectors.append(None)
            else:
                vectors.append(vector)

        if pending:
            started = time.perf_counter()
            fresh = self.llm.embed([chunks[i] for i in pending])
            for offset, index in enumerate(pending):
                vectors[index] = fresh[offset]
            self.usage.record(
                kind="embed",
                model=self.llm.signature,
                prompt_tokens=sum(estimate_tokens(chunks[i]) for i in pending),
                estimated=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        ids = [f"{source}::{i}::{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": source,
                "chunk_index": i,
                "chars": len(chunk),
                "text_hash": chunk_hashes[i],
                "dept": dept,
                "doc_type": doc_type,
                "embed_sig": self.llm.signature,
            }
            for i, chunk in enumerate(chunks)
        ]

        self.store.delete_source(source)   # 同名文件重复上传 → 覆盖而不是堆叠
        self.store.add(
            ids, chunks, [v or [] for v in vectors], metadatas, persist=persist
        )  # type: ignore[arg-type]
        self.documents.upsert(
            source=source,
            doc_hash=fingerprint,
            embedding_signature=self.llm.signature,
            chunk_count=len(chunks),
            chars=sum(len(c) for c in chunks),
            dept=dept,
            doc_type=doc_type,
        )
        self.cache.bump_version()          # 知识库变了，缓存必须整体失效
        self.retriever.invalidate()

        result = {
            "source": source, "chunks": len(chunks), "chars": sum(len(c) for c in chunks),
            "dept": dept, "skipped": False, "reused_chunks": len(chunks) - len(pending),
            "embedded_chunks": len(pending),
        }
        logger.info(
            "入库完成：%s，%d 段（复用 %d / 新向量化 %d）",
            source, len(chunks), result["reused_chunks"], result["embedded_chunks"],
        )
        return result

    def ingest_file(
        self, filename: str, raw: bytes, *, dept: str = "", doc_type: str = "", persist: bool = True
    ) -> dict:
        self.s.ensure_dirs()
        safe_name = Path(filename).name or "unnamed.txt"
        target = self.s.upload_dir / safe_name
        target.write_bytes(raw)

        text = read_text(target)
        if not text.strip():
            raise ValueError("文件内容为空，或这是扫描版 PDF（需要 OCR 才能提取文字）")
        return self.ingest_text(safe_name, text, dept=dept, doc_type=doc_type, persist=persist)

    def delete_document(self, source: str) -> int:
        deleted = self.store.delete_source(source)
        self.documents.delete(source)
        self.cache.bump_version()
        self.retriever.invalidate()
        return deleted

    def clear_documents(self) -> int:
        before = self.store.count()
        self.store.clear()
        self.documents.clear()
        self.cache.bump_version()
        self.retriever.invalidate()
        return before

    # ------------------------------------------------------------ 检索
    def _filters(self, dept: str | None = None, doc_type: str | None = None) -> dict | None:
        filters: dict = {}
        if dept:
            filters["dept"] = dept
        if doc_type:
            filters["doc_type"] = doc_type
        return filters or None

    def search(
        self,
        question: str,
        top_k: int | None = None,
        *,
        dept: str | None = None,
        doc_type: str | None = None,
        query: str | None = None,
        embedding: list[float] | None = None,
    ) -> RetrievalResult:
        from .framework import HybridSearchRetriever
        # 每次调用创建独立retriever，result不会被另一个请求覆盖。
        adapter=HybridSearchRetriever(backend=self.retriever, options=dict(
            top_k=top_k,filters=self._filters(dept,doc_type),query=query,embedding=embedding))
        adapter.invoke(question)  # 真正经过LangChain的Retriever执行入口。
        return adapter.result

    def retrieve(self, question: str, top_k: int | None = None, dept: str | None = None) -> list[Hit]:
        """兼容旧调用：只想要最终片段列表时用它。"""
        result = self.search(question, top_k, dept=dept)
        return [candidate.to_hit() for candidate in self._select(result)]

    def search_only(
        self, question: str, top_k: int | None = None, dept: str | None = None, doc_type: str | None = None
    ) -> tuple[list[Citation], RetrievalResult]:
        """只检索、不调用大模型。用来调试检索质量，也省 Token。"""
        result = self.search(question, top_k, dept=dept, doc_type=doc_type)
        return self._citations(self._select(result)), result

    # ------------------------------------------------------------ 组装
    @staticmethod
    def _snippet(text: str) -> str:
        flat = " ".join((text or "").split())
        return flat[:SNIPPET_CHARS] + ("…" if len(flat) > SNIPPET_CHARS else "")

    def _select(self, result: RetrievalResult) -> list[Candidate]:
        """先按来源去重，再截断到 top_k。

        顺序很重要：如果先截断再去重，Top-4 里 4 段同属一篇文档时会被砍到只剩 1 段，
        白白浪费了引用位。先在更大的候选池里去重，取出来的 4 条覆盖面更广。
        """
        return dedupe_by_source(result.candidates, limit_per_source=2)[: result.top_k]

    def _citations(self, candidates: list[Candidate]) -> list[Citation]:
        return [
            Citation(
                index=index + 1,
                source=candidate.source,
                chunk_index=candidate.chunk_index,
                score=candidate.final_score,
                vector_score=round(candidate.vector_score, 4),
                rerank_score=None if candidate.rerank_score is None else round(candidate.rerank_score, 4),
                rerank_rank=candidate.rerank_rank or None,
                matched_by=candidate.matched_by,
                dept=str(candidate.metadata.get("dept", "")),
                snippet=self._snippet(candidate.text),
            )
            for index, candidate in enumerate(candidates)
        ]

    def _context(self, candidates: list[Candidate]) -> str:
        """把命中的片段拼成【参考资料】，并限制总长度以控制 Token 成本。"""
        blocks: list[str] = []
        used = 0
        for index, candidate in enumerate(candidates, 1):
            block = (
                f"[{index}] （来源：{candidate.source} 第 {candidate.chunk_index + 1} 段）\n"
                f"{candidate.text}"
            )
            if blocks and used + len(block) > self.s.max_context_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def _messages(self, history: list[dict], question: str, candidates: list[Candidate]) -> list[dict]:
        context = self._context(candidates) or "（无可用资料）"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)  # 多轮记忆
        messages.append(
            {"role": "user", "content": f"【参考资料】\n{context}\n\n【问题】\n{question}"}
        )
        return messages

    @staticmethod
    def _retrieval_only_answer(citations: list[Citation]) -> str:
        if not citations:
            return "知识库里还没有内容，请先上传文档再提问。"
        lines = [NO_LLM_NOTICE, ""]
        lines += [f"[{c.index}] （{c.source}）{c.snippet}" for c in citations]
        return "\n".join(lines)

    def _refusal(self) -> str:
        return REFUSAL_TEMPLATE.format(
            documents=len(self.store.sources()), chunks=self.store.count()
        )

    # ------------------------------------------------------------ 多轮查询改写
    def rewrite_query(self, history: list[dict], question: str, sink: dict | None = None) -> tuple[str, bool]:
        """把"那学生呢？"这类指代问题改写成可独立检索的问句。

        为什么必须做：不改写的话，检索用的就是"那学生呢？"这 5 个字，
        向量检索完全召不回"学生版 99 元/月"。这是多轮 RAG 最经典的 bug。
        改写失败/超时都直接返回原问题 —— 增强功能绝不能阻塞主链路。
        """
        if not self.s.rewrite_enabled or not history or not self.llm.llm_ready:
            return question, False
        started = time.perf_counter()
        try:
            messages = [
                {"role": "system", "content": REWRITE_PROMPT},
                *history[-4:],
                {"role": "user", "content": question},
            ]
            rewritten = (self.llm.chat(messages, usage_sink=sink) or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询改写失败，改用原问题检索：%s", exc)
            return question, False
        rewritten = rewritten.splitlines()[0].strip().strip('"“”') if rewritten else ""
        if not rewritten or len(rewritten) > 200 or rewritten == question:
            return question, False
        if sink is not None:
            sink.setdefault("latency_ms", (time.perf_counter() - started) * 1000)
        logger.info("查询改写：%s → %s", question, rewritten)
        return rewritten, True

    # ------------------------------------------------------------ 缓存
    def _cache_lookup(
        self, question: str, embedding: list[float], filters: dict | None, use_cache: bool
    ) -> dict | None:
        if not use_cache:
            return None
        return self.cache.get(
            question,
            embedding=embedding,
            filters=filters,
            embedding_signature=self.llm.signature,
        )

    # ------------------------------------------------------------ 问答（一次性）
    def answer(self, session_id, question, top_k=None, *, dept=None, doc_type=None, use_cache=True):
        """执行编译后的LangGraph；缓存/拒答分支由条件边选择。"""
        context=dict(session_id=session_id,question=question,top_k=top_k,dept=dept,doc_type=doc_type,use_cache=use_cache)
        result=self.workflow.invoke({'ctx':context})
        return result['ctx']['output']

    def stream_answer(self, session_id, question, top_k=None, *, dept=None, doc_type=None, use_cache=True):
        """图中节点实时写custom事件；外部SSE接口无需改字段。"""
        trace=self.tracer.start(question,session_id)
        context=dict(session_id=session_id,question=question,top_k=top_k,dept=dept,doc_type=doc_type,
                     use_cache=use_cache,trace=trace,filters=self._filters(dept,doc_type),
                     started=time.perf_counter(),first_token_ms=None)
        try:
            yield from self.stream_workflow.stream({'ctx':context},stream_mode='custom')
        except Exception as exc:
            trace.error=f'{type(exc).__name__}: {exc}'
            self.tracer.finish(trace)
            logger.exception('流式生成失败')
            yield {'type':'error','message':str(exc),'trace_id':trace.id}


    # ------------------------------------------------------------ 辅助
    def _record_rerank(self, trace: Trace, session_id: str) -> None:
        """把精排这次调用的 token 也计入成本。

        精排是最容易被忽略的成本项：一次提问多一次模型调用，
        候选 20 条、每条 500 字，输入就是 1 万字符量级 —— 不计量的话，
        "成本降了"的结论会凭空多出一块看不见的支出。
        """
        outcome = self.retriever.last_rerank
        if not outcome.ok:
            return
        usage = outcome.usage or {}
        tokens = int(usage.get("total_tokens", 0) or 0)
        model = self.s.rerank_model
        if outcome.provider == "llm":
            model = self.s.llm_model
            tokens = tokens or (int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0)))
        if not tokens:
            return
        self.usage.record(
            kind="rerank",
            model=f"{model}({outcome.mode})",
            prompt_tokens=int(usage.get("prompt_tokens", tokens) or tokens),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            estimated=bool(usage.get("estimated", outcome.provider == "http")),
            session_id=session_id,
            trace_id=trace.id,
        )

    def _meta(
        self,
        trace: Trace,
        cache: str,
        rewritten: str | None,
        rerank_used: bool,
        refused: bool,
        citations: int,
        started: float,
        sink: dict | None = None,
        cost: float = 0.0,
        first_token_ms: float | None = None,
    ) -> dict:
        sink = sink or {}
        return {
            "trace_id": trace.id,
            "cache": cache,
            "rewritten_query": rewritten,
            "rerank_used": rerank_used,
            "rerank_mode": trace.data.get("rerank_mode", ""),
            "refused": refused,
            "citation_count": citations,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "first_token_ms": None if first_token_ms is None else round(first_token_ms, 1),
            "usage": {
                "prompt_tokens": sink.get("prompt_tokens", 0),
                "completion_tokens": sink.get("completion_tokens", 0),
                "estimated": bool(sink.get("estimated", False)),
                "cost_yuan": round(cost, 6),
            },
        }


def _top_score(result: RetrievalResult) -> float:
    return round(result.candidates[0].final_score, 4) if result.candidates else 0.0
