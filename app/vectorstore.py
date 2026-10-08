"""向量库：优先 Chroma（真实向量数据库、可持久化），装不上时用纯 Python 实现兜底。

为什么准备两套？
- Chroma 是简历上的关键词，也是真实项目该用的东西；
- 但它在部分 Python 版本上装不上，学生本地环境很容易卡住。
  所以：导入失败就自动降级到 MemoryStore，一行配置都不用改，流程照样跑通。

改造后这一层多做了四件事（都是为了撑住"混合检索 + 增量入库 + 部门隔离"）：
1. `Hit` 带上 id —— RRF 融合要靠 id 对齐两路召回的结果；
2. `query(where=...)` —— 支持按部门/类型过滤，企业知识库的刚需；
3. `all_items()/get_item()/embeddings_by_source()` —— 给 BM25 建索引和增量入库复用向量用；
4. `revision` 版本号 —— 任何写操作都 +1，上层据此判断要不要重建 BM25 索引，
   不用再搞一套"文档变了要通知谁"的机制。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .config import Settings
from .embedding_local import cosine

try:  # numpy 是可选的：装了就用矩阵化检索，没装也能跑
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    """一条检索结果。"""

    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    id: str = ""

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def chunk_index(self) -> int:
        try:
            return int(self.metadata.get("chunk_index", -1))
        except (TypeError, ValueError):
            return -1


class DimensionMismatch(ValueError):
    """向量维度和索引里的不一致（换过 embedding 模型）。"""


def _dimension_error(actual: int, expected: int) -> DimensionMismatch:
    return DimensionMismatch(
        f"向量维度不一致：索引里是 {expected} 维，这次算出来是 {actual} 维。"
        "原因：你切换过 EMBEDDING_MODE / embedding 模型，但旧向量还在库里。"
        "解决：删掉 data/chroma 目录（或 data/memory_index.json）后重新导入文档。"
    )


def _friendly_error(exc: Exception) -> Exception:
    message = str(exc)
    if "dimension" in message.lower():
        return DimensionMismatch(
            "向量维度不一致：你切换过 EMBEDDING_MODE / embedding 模型，但旧向量还在库里。"
            "解决：删除 data/chroma 目录（或 data/memory_index.json）后重新上传文档。"
            f"原始错误：{message}"
        )
    return exc if isinstance(exc, ValueError) else ValueError(message)


def _clean_filters(where: dict | None) -> dict | None:
    """过滤条件里的空值直接丢掉；全空就返回 None（Chroma 不接受空 dict）。"""
    if not where:
        return None
    cleaned = {k: v for k, v in where.items() if v not in (None, "", [])}
    return cleaned or None


class BaseVectorStore:
    backend = "base"

    def __init__(self) -> None:
        self._revision = 0

    @property
    def revision(self) -> int:
        """写操作版本号：上层用它判断缓存/BM25 索引要不要重建。"""
        return self._revision

    def _touch(self) -> None:
        self._revision += 1

    # ---- 子类必须实现 ----
    def add(self, ids: Sequence[str], texts: Sequence[str], embeddings: Sequence[Sequence[float]],
            metadatas: Sequence[dict], persist: bool = True) -> None:
        raise NotImplementedError

    def query(self, embedding: Sequence[float], top_k: int, where: dict | None = None) -> list[Hit]:
        raise NotImplementedError

    def all_items(self) -> list[dict]:
        """全部片段（id/text/metadata，不含向量）—— BM25 建索引用它。"""
        raise NotImplementedError

    def get_item(self, item_id: str) -> dict | None:
        raise NotImplementedError

    def embeddings_by_source(self, source: str) -> dict[str, list[float]]:
        """某个文档现有片段的「文本指纹 → 向量」—— 增量入库复用未变片段的向量。"""
        raise NotImplementedError

    def sources(self) -> list[dict]:
        raise NotImplementedError

    def delete_source(self, source: str) -> int:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    # ---- 可选 ----
    def flush(self) -> None:
        """把缓存的写入落盘（默认无操作：Chroma 是即时写）。"""

    def dimension(self) -> int | None:
        return None


class ChromaStore(BaseVectorStore):
    backend = "chroma"

    def __init__(self, directory: Path) -> None:
        super().__init__()
        import chromadb  # 惰性导入：没装 chromadb 也能 import 本模块

        directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(directory))
        from langchain_chroma import Chroma
        self.langchain=Chroma(client=self.client,collection_name='knowledge',embedding_function=None,
                              collection_metadata={'hnsw:space':'cosine'})
        self.collection=self.langchain._collection  # 增量指纹/维度维护继续使用底层数据读取。

    def add(self, ids, texts, embeddings, metadatas, persist: bool = True) -> None:
        try:
            # upsert：id 相同就覆盖，避免重复上传报错
            from langchain_chroma import Chroma
            from .framework import ProvidedBatchEmbeddings
            # 原增量入库已经算好/复用了向量；传入预计算适配器，避免add_texts再次付费算向量。
            writer=Chroma(client=self.client,collection_name='knowledge',
                embedding_function=ProvidedBatchEmbeddings(list(texts),[list(e) for e in embeddings]))
            writer.add_texts(texts=list(texts),ids=list(ids),metadatas=[dict(m) for m in metadatas])
        except Exception as exc:
            raise _friendly_error(exc) from exc
        self._touch()

    def query(self,embedding,top_k,where=None)->list[Hit]:
        if self.count()==0:return []
        try:
            pairs=self.langchain.similarity_search_by_vector_with_relevance_scores(
                list(embedding),k=top_k,filter=_clean_filters(where))
        except Exception as exc:
            raise _friendly_error(exc) from exc
        # LangChain Chroma此方法返回距离；继续使用原版1-distance的分数口径。
        return [Hit(text=doc.page_content,metadata=dict(doc.metadata),
                    score=round(1.0-float(distance),4),id=str(doc.id or '')) for doc,distance in pairs]

    def all_items(self) -> list[dict]:
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            {"id": str(item_id), "text": text or "", "metadata": dict(meta or {})}
            for item_id, text, meta in zip(ids, documents, metadatas)
        ]

    def get_item(self, item_id: str) -> dict | None:
        result = self.collection.get(ids=[item_id], include=["documents", "metadatas"])
        ids = result.get("ids") or []
        if not ids:
            return None
        return {
            "id": str(ids[0]),
            "text": (result.get("documents") or [""])[0] or "",
            "metadata": dict((result.get("metadatas") or [{}])[0] or {}),
        }

    def embeddings_by_source(self, source: str) -> dict[str, list[float]]:
        result = self.collection.get(
            where={"source": source}, include=["embeddings", "metadatas"]
        )
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        if embeddings is None:
            return {}
        mapping: dict[str, list[float]] = {}
        for meta, vector in zip(metadatas, embeddings):
            fingerprint = (meta or {}).get("text_hash")
            if fingerprint:
                mapping[str(fingerprint)] = list(vector)
        return mapping

    def sources(self) -> list[dict]:
        result = self.collection.get(include=["metadatas"])
        return _aggregate(result.get("metadatas") or [])

    def delete_source(self, source: str) -> int:
        result = self.collection.get(where={"source": source}, include=[])
        ids = result.get("ids") or []
        if ids:
            self.collection.delete(ids=ids)
            self._touch()
        return len(ids)

    def count(self) -> int:
        return int(self.collection.count())

    def clear(self) -> None:
        self.client.delete_collection("knowledge")
        from langchain_chroma import Chroma
        self.langchain=Chroma(client=self.client,collection_name='knowledge',embedding_function=None,
                              collection_metadata={'hnsw:space':'cosine'})
        self.collection=self.langchain._collection
        self._touch()

    def dimension(self) -> int | None:
        """从库里取一条向量看维度（用于启动自检）。"""
        result = self.collection.get(limit=1, include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])


class MemoryStore(BaseVectorStore):
    """纯 Python 实现：JSON 持久化 + 余弦检索。数据量几千条以内完全够用。

    两个工程细节：
    - 装了 numpy 就走矩阵化检索（1726 段从 ~0.2s 降到毫秒级），没装则逐条算；
    - `persist=False` 时只标脏不落盘，批量导入末尾调一次 flush()，
      把"每加一批就重写整个索引文件"从 O(n²) 降到 O(n)。
    """

    backend = "memory"

    def __init__(self, index_file: Path) -> None:
        super().__init__()
        self.path = index_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._items: dict[str, dict] = {}
        self._dirty = False
        self._matrix = None          # numpy 缓存
        self._matrix_revision = -1
        self._dim: int | None = None
        self._load()

    # ------------------------------------------------------------ 持久化
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._items = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("索引文件损坏，已忽略：%s", self.path)
            self._items = {}
        for item in self._items.values():
            vector = item.get("embedding") or []
            if vector:
                self._dim = len(vector)
                break

    def _save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            self._flush_locked()

    def _flush_locked(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._items, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)  # 先写临时文件再替换，避免写一半崩掉
        self._dirty = False

    def flush(self) -> None:
        self._save()

    # ------------------------------------------------------------ 写入
    def add(self, ids, texts, embeddings, metadatas, persist: bool = True) -> None:
        with self._lock:
            for item_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
                vector = list(embedding)
                if vector:
                    if self._dim is not None and len(vector) != self._dim:
                        raise _dimension_error(len(vector), self._dim)
                    self._dim = len(vector)
                self._items[str(item_id)] = {
                    "text": text,
                    "embedding": vector,
                    "metadata": dict(metadata),
                }
            self._dirty = True
            self._touch()
            if persist:
                self._flush_locked()

    def delete_source(self, source: str) -> int:
        with self._lock:
            victims = [k for k, v in self._items.items() if v["metadata"].get("source") == source]
            for key in victims:
                self._items.pop(key, None)
            if victims:
                self._dirty = True
                self._touch()
                self._flush_locked()
        return len(victims)

    def clear(self) -> None:
        with self._lock:
            self._items = {}
            self._dim = None
            self._dirty = True
            self._touch()
            self._flush_locked()

    # ------------------------------------------------------------ 读取
    def _matrix_cache(self):
        """把全部向量拼成一个归一化矩阵并缓存，写操作后自动失效。"""
        if _np is None:
            return None
        if self._matrix is not None and self._matrix_revision == self._revision:
            return self._matrix
        ids = list(self._items)
        if not ids:
            return None
        matrix = _np.array([self._items[i]["embedding"] for i in ids], dtype=_np.float32)
        norms = _np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = (ids, matrix / norms)
        self._matrix_revision = self._revision
        return self._matrix

    def query(self, embedding, top_k, where=None) -> list[Hit]:
        with self._lock:
            items = list(self._items.items())
        where = _clean_filters(where)
        if where:
            items = [
                (k, v) for k, v in items
                if all(str(v["metadata"].get(key, "")) == str(value) for key, value in where.items())
            ]
        if not items:
            return []

        query_vector = list(embedding)
        cached = self._matrix_cache()
        if cached is not None and not where:
            ids, matrix = cached
            if matrix.shape[1] != len(query_vector):
                raise _dimension_error(len(query_vector), int(matrix.shape[1]))
            query_array = _np.array(query_vector, dtype=_np.float32)
            norm = float(_np.linalg.norm(query_array)) or 1.0
            scores = matrix @ (query_array / norm)
            k = min(top_k, len(ids))
            # argpartition 只做部分排序，比全量 sort 快一个量级
            top = _np.argpartition(-scores, k - 1)[:k]
            top = top[_np.argsort(-scores[top])]
            return [
                Hit(
                    text=self._items[ids[i]]["text"],
                    metadata=dict(self._items[ids[i]]["metadata"]),
                    score=round(float(scores[i]), 4),
                    id=ids[i],
                )
                for i in top
            ]

        hits: list[Hit] = []
        if self._dim is not None and query_vector and len(query_vector) != self._dim:
            raise _dimension_error(len(query_vector), self._dim)
        for item_id, item in items:
            vector = item["embedding"]
            if self._dim is not None and vector and len(vector) != self._dim:
                raise _dimension_error(len(vector), self._dim)
            hits.append(Hit(
                text=item["text"],
                metadata=dict(item["metadata"]),
                score=round(cosine(query_vector, vector), 4),
                id=item_id,
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def all_items(self) -> list[dict]:
        with self._lock:
            return [
                {"id": item_id, "text": item["text"], "metadata": dict(item["metadata"])}
                for item_id, item in self._items.items()
            ]

    def get_item(self, item_id: str) -> dict | None:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return None
            return {"id": item_id, "text": item["text"], "metadata": dict(item["metadata"])}

    def embeddings_by_source(self, source: str) -> dict[str, list[float]]:
        with self._lock:
            return {
                str(item["metadata"]["text_hash"]): list(item["embedding"])
                for item in self._items.values()
                if item["metadata"].get("source") == source and item["metadata"].get("text_hash")
            }

    def sources(self) -> list[dict]:
        with self._lock:
            metadatas = [item["metadata"] for item in self._items.values()]
        return _aggregate(metadatas)

    def count(self) -> int:
        return len(self._items)

    def dimension(self) -> int | None:
        return self._dim

    # 兼容旧脚本：旧的 check_embedding.py 直接读 JSON 文件
    def peek_dimension(self) -> int | None:
        return self.dimension()


def _aggregate(metadatas: Sequence[dict]) -> list[dict]:
    summary: dict[str, dict] = {}
    for meta in metadatas:
        source = str((meta or {}).get("source", "unknown"))
        item = summary.setdefault(source, {"source": source, "chunks": 0, "chars": 0, "dept": ""})
        item["chunks"] += 1
        try:
            item["chars"] += int((meta or {}).get("chars", 0))
        except (TypeError, ValueError):
            pass
        if not item["dept"]:
            item["dept"] = str((meta or {}).get("dept", ""))
    return sorted(summary.values(), key=lambda x: x["source"])


def build_store(settings: Settings) -> BaseVectorStore:
    if settings.vector_backend == "memory":
        logger.info("向量库：MemoryStore（纯 Python，%s）", settings.memory_index_file)
        return MemoryStore(settings.memory_index_file)
    try:
        store = ChromaStore(settings.chroma_dir)
        logger.info("向量库：Chroma（%s）", settings.chroma_dir)
        return store
    except Exception as exc:
        logger.warning("Chroma 初始化失败（%s），自动回退 MemoryStore。", exc)
        logger.warning("如果 pip install chromadb 失败，这属于正常现象，功能不受影响。")
        return MemoryStore(settings.memory_index_file)
