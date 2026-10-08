"""FastAPI 入口：上传 / 检索问答 / 会话历史 / 统计 / Trace。

启动：uvicorn app.main:app --reload --port 8000
文档：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .cache import AnswerCache
from .chat_store import ChatStore
from .config import BASE_DIR, settings
from .docstore import DocumentStore
from .llm import EmbeddingUnavailable, LLMClient, LLMNotConfigured
from .loaders import UnsupportedFile
from .rag import RagService
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    HealthResponse,
    Message,
    StatsResponse,
    UploadResponse,
)
from .trace import Tracer
from .usage import UsageStore
from .vectorstore import build_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag")

MAX_UPLOAD_MB = 20

# ---------------------------------------------------------------- 组装依赖
settings.ensure_dirs()
llm_client = LLMClient(settings)
vector_store = build_store(settings)
chat_store = ChatStore(settings.db_path)
usage_store = UsageStore(settings)
answer_cache = AnswerCache(settings)
tracer = Tracer(settings)
document_store = DocumentStore(settings)
rag = RagService(
    settings, llm_client, vector_store, chat_store,
    usage=usage_store, cache=answer_cache, tracer=tracer, documents=document_store,
)

app = FastAPI(
    title="企业知识库智能问答系统",
    version="0.2.0",
    description=(
        "混合检索（BM25 + 向量 + RRF + 精排）→ 阈值拒答 → 带引用生成 → 答案缓存 → 成本计量。"
    ),
)
# 跨域：开发时用 *，上线把 CORS_ORIGINS 收窄成具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

api = APIRouter(prefix="/api")


def require_auth(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """可选的接口鉴权：配了 API_KEY 就校验，没配就放行（本地开发零门槛）。

    为什么不做成强制：练手/内网演示时强制鉴权只会让人卡在 401 上；
    但把开关留着，就说明你知道这东西上线必须有。
    """
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Key")


# ------------------------------------------------------------------- 页面
@app.get("/", include_in_schema=False)
def index():
    page = STATIC_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return {"message": "RAG demo API is running", "docs": "/docs"}


# ------------------------------------------------------------------- 状态
@api.get("/health", response_model=HealthResponse, summary="健康检查 / 当前配置")
def health() -> HealthResponse:
    # 索引是用哪个向量模型建的？和当前配置不一致就说明检索相似度不可比，
    # 这种"不报错但结果错"的状态必须显式暴露出来。
    stale = [
        doc["source"] for doc in document_store.all()
        if doc.get("embedding_signature") and doc["embedding_signature"] != llm_client.signature
    ]
    warnings = settings.warnings()
    if stale:
        warnings.append(
            f"{len(stale)} 份文档是用旧向量模型建的索引（例如 {stale[0]}），"
            "相似度不可比，建议重新上传这些文档"
        )
    tracked = {doc["source"] for doc in document_store.all()}
    untracked = [item["source"] for item in vector_store.sources() if item["source"] not in tracked]
    if untracked:
        warnings.append(
            f"{len(untracked)} 份文档没有台账记录（早期版本入库的数据，例如 {untracked[0]}）："
            "增量入库和部门过滤对它们不生效，重新上传一次即可补齐"
        )
    return HealthResponse(
        vector_backend=vector_store.backend,
        documents=len(vector_store.sources()),
        chunks=vector_store.count(),
        retrieval={
            "hybrid": settings.hybrid_enabled,
            "retrieve_k": settings.retrieve_k,
            "rrf_k": settings.rrf_k,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_ready": settings.rerank_ready,
            "rerank_llm_ready": settings.rerank_llm_ready,
            "rerank_provider": settings.rerank_provider,
            "rerank_model": settings.rerank_model,
            "rewrite_enabled": settings.rewrite_enabled,
            "min_vector_score": settings.min_vector_score,
            "min_rerank_score": settings.min_rerank_score,
            "cache_enabled": settings.cache_enabled,
            "auth_enabled": bool(settings.api_key),
        },
        warnings=warnings,
        **llm_client.describe(),
    )


@api.get("/stats", response_model=StatsResponse, summary="用量 / 成本 / 缓存 / 知识库统计")
def stats(days: int = Query(default=7, ge=1, le=90)) -> StatsResponse:
    depts = {doc["dept"] for doc in document_store.all() if doc.get("dept")}
    return StatsResponse(
        usage=usage_store.summary(days=days),
        cache=answer_cache.stats(),
        trace=tracer.stats(),
        knowledge={
            "documents": len(vector_store.sources()),
            "chunks": vector_store.count(),
            "backend": vector_store.backend,
            "depts": sorted(depts),
            "dept_count": len(depts),
        },
    )


# ------------------------------------------------------------------- 文档
@api.post(
    "/documents",
    response_model=UploadResponse,
    summary="上传文档并入库（增量：内容没变整篇跳过）",
    dependencies=[Depends(require_auth)],
)
async def upload_document(
    file: UploadFile = File(...),
    dept: str = Form(default="", description="所属部门，用于检索时的数据隔离"),
    doc_type: str = Form(default="", description="文档类型，如 policy / manual"),
) -> UploadResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="文件为空")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_UPLOAD_MB}MB 限制")

    try:
        result = rag.ingest_file(file.filename or "unnamed.txt", raw, dept=dept, doc_type=doc_type)
    except UnsupportedFile as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmbeddingUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("入库失败")
        raise HTTPException(status_code=500, detail=f"入库失败：{exc}") from exc

    if result["chunks"] == 0:
        raise HTTPException(status_code=400, detail="切分后没有有效内容")
    return UploadResponse(**result)


@api.get("/documents", response_model=list[DocumentInfo], summary="列出已入库文档")
def list_documents() -> list[DocumentInfo]:
    ledger = {doc["source"]: doc for doc in document_store.all()}
    items = []
    for item in vector_store.sources():
        meta = ledger.get(item["source"], {})
        items.append(DocumentInfo(
            source=item["source"],
            chunks=item["chunks"],
            chars=item["chars"],
            dept=item.get("dept") or meta.get("dept", ""),
            doc_type=meta.get("doc_type", ""),
            updated_at=meta.get("updated_at", ""),
        ))
    return items


@api.delete("/documents/{source}", summary="删除某个文档及其向量", dependencies=[Depends(require_auth)])
def delete_document(source: str) -> dict:
    return {"source": source, "deleted_chunks": rag.delete_document(source)}


@api.delete("/documents", summary="清空整个知识库", dependencies=[Depends(require_auth)])
def clear_documents() -> dict:
    return {"deleted_chunks": rag.clear_documents()}


# ------------------------------------------------------------------- 问答
@api.post(
    "/chat",
    response_model=ChatResponse,
    summary="提问（一次性返回）",
    dependencies=[Depends(require_auth)],
)
def chat(request: ChatRequest) -> ChatResponse:
    if request.mode == "search":
        citations, result = rag.search_only(
            request.question, request.top_k, request.dept, request.doc_type
        )
        return ChatResponse(
            session_id=request.session_id,
            answer=(
                f"（仅检索模式：命中 {len(citations)} 条，未调用大模型）"
                + (f"｜{result.note}" if result.note else "")
                + (f"｜改写后：{result.query}" if result.rewritten else "")
            ),
            citations=citations,
            refused=result.refused,
            rewritten_query=result.query if result.rewritten else None,
            rerank_used=result.rerank_used,
            rerank_mode=result.rerank_mode,
        )
    try:
        answer, citations, meta = rag.answer(
            request.session_id, request.question, request.top_k,
            dept=request.dept, doc_type=request.doc_type, use_cache=request.use_cache,
        )
    except LLMNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("生成失败")
        raise HTTPException(status_code=502, detail=f"生成失败：{exc}") from exc
    return ChatResponse(
        session_id=request.session_id, answer=answer, citations=citations,
        **_pick(meta),
    )


def _pick(meta: dict) -> dict:
    """从 rag 返回的 meta 里挑出响应模型认识的字段（meta 还带 first_token_ms / citation_count 等）。"""
    keep = {"trace_id", "cache", "rewritten_query", "rerank_used", "rerank_mode",
            "refused", "latency_ms", "usage"}
    return {key: value for key, value in meta.items() if key in keep}


@api.post("/chat/stream", summary="提问（SSE 流式返回）", dependencies=[Depends(require_auth)])
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def event_source():
        try:
            for event in rag.stream_answer(
                request.session_id, request.question, request.top_k,
                dept=request.dept, doc_type=request.doc_type, use_cache=request.use_cache,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式生成失败")
            payload = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------- 会话
@api.get("/sessions/{session_id}/messages", response_model=list[Message], summary="查看会话历史")
def session_messages(session_id: str) -> list[Message]:
    return [Message(**row) for row in chat_store.messages(session_id)]


@api.delete("/sessions/{session_id}", summary="清空会话历史")
def clear_session(session_id: str) -> dict:
    return {"session_id": session_id, "deleted": chat_store.clear(session_id)}


# ------------------------------------------------------------------- 观测
@api.get("/traces", summary="最近的请求链路（倒序）")
def recent_traces(limit: int = Query(default=20, ge=1, le=200)) -> list[dict]:
    return tracer.recent(limit)


@api.get("/traces/bad-cases", summary="需要复盘的请求：拒答 / 报错")
def bad_cases(limit: int = Query(default=20, ge=1, le=200)) -> list[dict]:
    return tracer.bad_cases(limit)


@api.get("/traces/{trace_id}", summary="按 trace_id 查一次请求的完整链路")
def get_trace(trace_id: str) -> dict:
    payload = tracer.get(trace_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="没有这条 trace（可能已被清理，只保留最近若干条）")
    return payload


@api.delete("/cache", summary="清空答案缓存", dependencies=[Depends(require_auth)])
def clear_cache() -> dict:
    return {"deleted": answer_cache.clear()}


app.include_router(api)
