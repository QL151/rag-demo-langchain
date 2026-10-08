"""接口的请求 / 响应模型（Pydantic 会自动生成 /docs 里的在线文档）。

改造后 Citation 里多了三个字段，都是"可解释性"：
- `vector_score`：向量路的余弦相似度（原始分数，不参与最终排序时也保留，方便对比）
- `rerank_score`：精排分数（没走精排时为 null）
- `matched_by`：命中通道（["向量"] / ["关键词"] / ["向量","关键词"]）

有了这三个字段，用户问"为什么给我这个答案"时，你能直接指给前端看，
而不是只能说"模型觉得相关"。—— 这是 RAG 系统能不能上生产的关键之一。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    index: int = Field(description="引用编号，对应回答里的 [1][2]")
    source: str = Field(description="来源文件名")
    chunk_index: int = Field(description="在原文中的片段序号，从 0 开始")
    score: float = Field(description="最终排序分（有精排用精排分，否则用向量相似度）")
    vector_score: float = Field(default=0.0, description="向量路余弦相似度")
    rerank_score: Optional[float] = Field(default=None, description="精排相关性分，未启用或 listwise 精排时为 null")
    rerank_rank: Optional[int] = Field(default=None, description="listwise 精排给出的名次（cross-encoder 时为 null）")
    matched_by: List[str] = Field(default_factory=list, description="命中通道：向量 / 关键词")
    dept: str = Field(default="", description="文档所属部门（数据隔离用）")
    snippet: str = Field(description="命中的原文片段（截断）")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str = Field(default="default", max_length=64, description="会话 ID，用于多轮记忆")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="覆盖默认检索条数")
    dept: Optional[str] = Field(default=None, max_length=64, description="只看某个部门的文档（数据隔离）")
    doc_type: Optional[str] = Field(default=None, max_length=32, description="只看某类文档，如 policy / manual")
    mode: str = Field(default="answer", pattern="^(answer|search)$", description="answer=生成回答；search=只检索")
    use_cache: bool = Field(default=True, description="是否允许命中答案缓存")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Citation] = []
    refused: bool = Field(default=False, description="true = 知识库中没有依据，系统拒答（未调用大模型）")
    cache: str = Field(default="miss", description="exact / semantic / miss")
    rewritten_query: Optional[str] = Field(default=None, description="多轮改写后实际用于检索的问句")
    rerank_used: bool = Field(default=False, description="本次是否走了精排")
    rerank_mode: str = Field(default="", description="cross-encoder / listwise / 空")
    trace_id: str = Field(default="", description="用它可以查这次请求的完整链路")
    latency_ms: float = Field(default=0.0, description="服务端总耗时")
    usage: Dict[str, float] = Field(default_factory=dict, description="本次 token 与费用")


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    chars: int = 0
    dept: str = ""
    doc_type: str = ""
    updated_at: str = ""


class UploadResponse(BaseModel):
    source: str
    chunks: int
    chars: int
    dept: str = ""
    skipped: bool = Field(default=False, description="true = 内容指纹没变，整篇跳过（增量入库）")
    reused_chunks: int = Field(default=0, description="复用了旧向量的片段数，未重复调用 embedding")
    embedded_chunks: int = Field(default=0, description="本次真正重新向量化的片段数")


class Message(BaseModel):
    role: str
    content: str
    created_at: str


class HealthResponse(BaseModel):
    status: str = "ok"
    vector_backend: str
    documents: int
    chunks: int
    llm_ready: bool
    llm_model: str
    embedding_mode: str
    embedding_model: str
    embedding_signature: str = ""
    retrieval: Dict[str, object] = Field(default_factory=dict, description="当前检索配置（混合/精排/阈值）")
    warnings: List[str] = Field(default_factory=list, description="启动自检发现的问题")


class StatsResponse(BaseModel):
    usage: Dict[str, object] = Field(default_factory=dict)
    cache: Dict[str, object] = Field(default_factory=dict)
    trace: Dict[str, object] = Field(default_factory=dict)
    knowledge: Dict[str, object] = Field(default_factory=dict)
