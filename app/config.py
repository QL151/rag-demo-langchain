"""集中管理配置：所有可变项都来自环境变量 / .env 文件，代码里不写死。

改造后新增的三组配置，都是"企业知识库"区别于"练手 Demo"的地方：

1. 检索（hybrid / rrf / rerank / 阈值）—— 决定答得准不准；
2. 成本与缓存（单价、缓存 TTL、语义缓存阈值）—— 决定跑得起跑不起；
3. 治理（部门过滤、接口鉴权、CORS、trace）—— 决定敢不敢给客户用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

try:  # 没装 python-dotenv 也能跑
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False

BASE_DIR = Path(__file__).resolve().parent.parent


def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    value = Path(_str(name, default))
    return value if value.is_absolute() else (BASE_DIR / value)


def _list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _str(name)
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _same_host(a: str, b: str) -> bool:
    """两个接口地址是不是同一家服务商（同一域名）。

    只有同源时才允许复用 Key —— 拿 DeepSeek 的 Key 去调百炼的 rerank，
    结果必然是 401，而且报错信息很难看出原因。
    """
    host_a, host_b = urlparse(a).hostname, urlparse(b).hostname
    return bool(host_a) and host_a == host_b


DEFAULT_RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


@dataclass(frozen=True)
class Settings:
    # --- LLM：负责"生成回答" ---
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_timeout: float

    # --- Embedding：负责"向量化"，local 模式无需 Key ---
    embedding_mode: str  # local | api
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_batch_size: int
    local_embedding_dim: int

    # --- 存储 ---
    vector_backend: str  # chroma | memory
    chroma_dir: Path
    memory_index_file: Path
    upload_dir: Path
    db_path: Path
    trace_dir: Path

    # --- RAG 参数 ---
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chars: int
    max_history_turns: int

    # --- 检索：混合 + 融合 + 精排 + 拒答（新增）---
    retrieve_k: int = 20              # 每一路召回的候选数（融合前的池子）
    hybrid_enabled: bool = True       # BM25 + 向量双路召回
    rrf_k: int = 60                   # RRF 平滑常数，论文经验值
    rerank_enabled: bool = True       # 精排总开关
    rerank_provider: str = "auto"     # auto | http | llm | off
    rerank_base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    rerank_api_key: str = ""
    rerank_model: str = "gte-rerank-v2"
    rerank_timeout: float = 20.0
    rerank_candidates: int = 20       # 送去精排的候选数
    rerank_llm_fallback: bool = True  # cross-encoder 不可用时，用对话模型做 listwise 精排
    rerank_llm_max_docs: int = 12     # listwise 精排最多塞几个候选（控 token）
    rerank_breaker_threshold: int = 3 # 连续失败几次就熔断（不再每次提问都白等一次 403）
    rerank_breaker_seconds: int = 300 # 熔断持续多久，之后放一次请求试探
    min_vector_score: float = 0.40    # 没有精排时用它拒答；**必须用 calibrate_threshold.py 标定**
    min_rerank_score: float = 0.30    # 有 cross-encoder 精排时以它为准拒答
    min_bm25_idf_ratio: float = 0.6   # 关键词"强命中"的相对门槛（见 retrieval.py 的说明）
    rewrite_enabled: bool = True      # 多轮指代 → 独立问句

    # --- 成本 / 缓存（新增）---
    # 单价请按你实际用的服务商改：默认取 DeepSeek 的常见价（输入 2 元/百万、输出 8 元/百万）
    price_prompt_per_1k: float = 0.002     # 元 / 千 token
    price_completion_per_1k: float = 0.008
    cache_enabled: bool = True
    cache_ttl_seconds: int = 24 * 3600
    cache_semantic_threshold: float = 0.95
    cache_max_scan: int = 500

    # --- 治理（新增）---
    api_key: str = ""                      # 空 = 不鉴权（本地开发）
    cors_origins: tuple[str, ...] = ("*",)
    trace_enabled: bool = True
    trace_memory_size: int = 200
    default_dept: str = "public"

    @property
    def llm_ready(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def rerank_ready(self) -> bool:
        """cross-encoder 精排是否可用：开关打开 + 用 http/auto + 有 Key。"""
        return (
            self.rerank_enabled
            and self.rerank_provider in {"auto", "http"}
            and bool(self.rerank_api_key)
        )

    @property
    def rerank_llm_ready(self) -> bool:
        """listwise 精排（用对话模型排序）是否可用：它只依赖 LLM Key。"""
        return (
            self.rerank_enabled
            and self.rerank_provider in {"auto", "llm"}
            and self.rerank_llm_fallback
            and bool(self.llm_api_key)
        )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BASE_DIR / ".env")
        rerank_base = _str("RERANK_BASE_URL", DEFAULT_RERANK_URL)
        rerank_key = _str("RERANK_API_KEY")
        if not rerank_key and _same_host(rerank_base, _str(
            "EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )):
            rerank_key = _str("EMBEDDING_API_KEY")
        return cls(
            llm_api_key=_str("LLM_API_KEY"),
            llm_base_url=_str("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_model=_str("LLM_MODEL", "deepseek-chat"),
            llm_temperature=_float("LLM_TEMPERATURE", 0.3),
            llm_timeout=_float("LLM_TIMEOUT", 60.0),
            embedding_mode=_str("EMBEDDING_MODE", "local").lower(),
            embedding_api_key=_str("EMBEDDING_API_KEY"),
            embedding_base_url=_str(
                "EMBEDDING_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            embedding_model=_str("EMBEDDING_MODEL", "text-embedding-v3"),
            embedding_batch_size=_int("EMBEDDING_BATCH_SIZE", 10),
            local_embedding_dim=_int("LOCAL_EMBEDDING_DIM", 1024),
            vector_backend=_str("VECTOR_BACKEND", "chroma").lower(),
            chroma_dir=_path("CHROMA_DIR", "data/chroma"),
            memory_index_file=_path("MEMORY_INDEX_FILE", "data/memory_index.json"),
            upload_dir=_path("UPLOAD_DIR", "data/uploads"),
            db_path=_path("DB_PATH", "data/app.db"),
            trace_dir=_path("TRACE_DIR", "data/traces"),
            chunk_size=_int("CHUNK_SIZE", 500),
            chunk_overlap=_int("CHUNK_OVERLAP", 80),
            top_k=_int("TOP_K", 4),
            max_context_chars=_int("MAX_CONTEXT_CHARS", 6000),
            max_history_turns=_int("MAX_HISTORY_TURNS", 5),
            retrieve_k=_int("RETRIEVE_K", 20),
            hybrid_enabled=_bool("HYBRID_ENABLED", True),
            rrf_k=_int("RRF_K", 60),
            rerank_enabled=_bool("RERANK_ENABLED", True),
            rerank_provider=_str("RERANK_PROVIDER", "auto").lower(),
            rerank_base_url=rerank_base,
            rerank_api_key=rerank_key,
            rerank_model=_str("RERANK_MODEL", "gte-rerank-v2"),
            rerank_timeout=_float("RERANK_TIMEOUT", 20.0),
            rerank_candidates=_int("RERANK_CANDIDATES", 20),
            rerank_llm_fallback=_bool("RERANK_LLM_FALLBACK", True),
            rerank_llm_max_docs=_int("RERANK_LLM_MAX_DOCS", 12),
            rerank_breaker_threshold=_int("RERANK_BREAKER_THRESHOLD", 3),
            rerank_breaker_seconds=_int("RERANK_BREAKER_SECONDS", 300),
            min_vector_score=_float("MIN_VECTOR_SCORE", 0.40),
            min_rerank_score=_float("MIN_RERANK_SCORE", 0.30),
            min_bm25_idf_ratio=_float("MIN_BM25_IDF_RATIO", 0.6),
            rewrite_enabled=_bool("REWRITE_ENABLED", True),
            price_prompt_per_1k=_float("PRICE_PROMPT_PER_1K", 0.002),
            price_completion_per_1k=_float("PRICE_COMPLETION_PER_1K", 0.008),
            cache_enabled=_bool("CACHE_ENABLED", True),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 24 * 3600),
            cache_semantic_threshold=_float("CACHE_SEMANTIC_THRESHOLD", 0.95),
            cache_max_scan=_int("CACHE_MAX_SCAN", 500),
            api_key=_str("API_KEY"),
            cors_origins=_list("CORS_ORIGINS", ("*",)),
            trace_enabled=_bool("TRACE_ENABLED", True),
            trace_memory_size=_int("TRACE_MEMORY_SIZE", 200),
            default_dept=_str("DEFAULT_DEPT", "public"),
        )

    def ensure_dirs(self) -> None:
        for path in (
            self.chroma_dir,
            self.upload_dir,
            self.db_path.parent,
            self.memory_index_file.parent,
            self.trace_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def warnings(self) -> list[str]:
        """启动时自检，把"看起来配了其实调不通"的组合提前说清楚。

        为什么需要它：以前缺少 Key 是静默降级，生产上会把配置错误藏到用户提问那一刻。
        """
        problems: list[str] = []
        if self.embedding_mode == "api" and not self.embedding_api_key:
            problems.append(
                "EMBEDDING_MODE=api 但 EMBEDDING_API_KEY 为空 —— 将回退本地哈希向量化（效果明显变差）"
            )
        if self.embedding_mode == "api" and self.embedding_base_url == self.llm_base_url \
                and not self.embedding_api_key:
            problems.append("embedding 与对话同源时可直接复用 LLM_API_KEY，当前两边都没配")
        if self.rerank_enabled and not self.rerank_ready and not self.rerank_llm_ready:
            problems.append(
                "RERANK_ENABLED=true 但两种精排都用不了（cross-encoder 没 Key、"
                "listwise 也没 LLM Key）—— 将只用「向量+BM25」融合结果"
            )
        if not self.llm_api_key:
            problems.append("LLM_API_KEY 为空 —— 只能检索、不能生成回答（本地演示没问题）")
        if "*" in self.cors_origins and self.api_key:
            problems.append('CORS_ORIGINS=* 且配置了 API_KEY：生产环境应把来源收窄到具体域名')
        if self.min_vector_score <= 0 and self.min_rerank_score <= 0:
            problems.append("拒答阈值全为 0 —— 知识库里没有答案时模型会硬编，建议保留默认值")
        if self.embedding_mode != "api" and not self.rerank_ready:
            problems.append(
                "本地哈希向量 + 无精排：相似度分布和真实 embedding 完全不同，"
                "MIN_VECTOR_SCORE 的默认值是按真实模型校准的，本地模式需要自己重新标定"
            )
        return problems

    def public_info(self) -> dict:
        """给前端/健康检查用的安全信息（不含任何 Key）。"""
        return {
            "llm_ready": self.llm_ready,
            "llm_model": self.llm_model,
            "embedding_mode": self.embedding_mode,
            "vector_backend": self.vector_backend,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "retrieve_k": self.retrieve_k,
            "hybrid_enabled": self.hybrid_enabled,
            "rerank_enabled": self.rerank_enabled,
            "rerank_ready": self.rerank_ready,
            "rerank_llm_ready": self.rerank_llm_ready,
            "rerank_provider": self.rerank_provider,
            "rerank_model": self.rerank_model,
            "rewrite_enabled": self.rewrite_enabled,
            "cache_enabled": self.cache_enabled,
            "min_vector_score": self.min_vector_score,
            "min_rerank_score": self.min_rerank_score,
            "auth_enabled": bool(self.api_key),
        }


settings = Settings.from_env()
