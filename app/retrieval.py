"""混合检索：向量召回 + LangChain BM25Retriever → RRF → 可选精排 → 阈值判断。

BM25按词项统计匹配，向量检索按embedding相似度匹配，两者可以互补；
没有一种检索器保证所有精确词或语义问题都必定命中。
通用余弦相似度范围为[-1,1]，BM25没有统一固定分数上限。
RRF按名次融合，避免直接混用原始分数；经过校准的分数也可以加权融合。
BM25不是神经双塔。cross-encoder联合比较问题与文档；本项目也支持LLM列表排序。
精排失败可沿用召回结果，embedding失败则明确报错，避免混用向量空间。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi

from .config import Settings
from .embedding_local import cosine
from .vectorstore import BaseVectorStore, Hit

logger = logging.getLogger(__name__)

# 中文按单字 + 相邻 bigram；英文/数字按整词。
# 用 bigram 是为了缓解"单字歧义"（"银" 和 "行" 分开看毫无意义，"银行"才有）。
_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")

# 送去精排的单条文本上限：rerank 模型都有 token 上限，
# 超长文档会被截断甚至报错，先自己截一刀更可控。
RERANK_DOC_CHARS = 3000


def tokenize(text: str) -> list[str]:
    """极简分词：英文数字按词，中文按字 + 相邻 bigram。零依赖，几千段毫秒级。"""
    text = (text or "").lower()
    raw = _TOKEN.findall(text)
    han = [t for t in raw if "\u4e00" <= t <= "\u9fff"]
    words = [t for t in raw if not ("\u4e00" <= t <= "\u9fff")]
    return words + han + ["".join(pair) for pair in zip(han, han[1:])]


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion：把多路排名的名次融成一个分数。

    只吃名次、不吃原始分数 —— 这就是它能跨量纲融合的原因。
    k 越大，靠前的名次之间差距越小（k=60 时第 1 名和第 2 名只差 0.0002），
    作用是削弱"某一路的第一名必须排第一"的强假设。
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class SmoothBM25(BM25Okapi):
    """保留原非负IDF规则；词频饱和与长度归一化交给rank_bm25。"""
    def _calc_idf(self,nd):
        self.idf={word:math.log(1+(self.corpus_size-count+0.5)/(count+0.5)) for word,count in nd.items()}
        self.average_idf=sum(self.idf.values())/len(self.idf) if self.idf else 0.0

class BM25:
    """兼容原类接口，内部真正调用LangChain BM25Retriever和rank_bm25。"""
    k1=1.5
    b=0.75
    def __init__(self,documents:dict[str,str]):
        self.ids=list(documents)
        self.n=len(self.ids)
        self.docs=[Document(page_content=documents[key],metadata={'chunk_id':key},id=key) for key in self.ids]
        corpus=[tokenize(d.page_content) or [f'__empty_{i}__'] for i,d in enumerate(self.docs)]
        self.engine=SmoothBM25(corpus,k1=self.k1,b=self.b) if corpus else None
        self.idf=self.engine.idf if self.engine else {}
        self.avgdl=self.engine.avgdl if self.engine else 1.0
        self.adapter=BM25Retriever(vectorizer=self.engine,docs=self.docs,preprocess_func=tokenize,k=max(1,self.n)) if self.docs else None
    def search(self,query,top_k):
        if self.adapter is None:return []
        scores=dict(zip(self.ids,self.engine.get_scores(tokenize(query))))
        # invoke输出Document；不把零匹配片段当成有效关键词命中。
        docs=self.adapter.invoke(query)
        selected={d.id for d in docs}
        # 稳定同分排序保持原插入顺序；前K裁剪在这里明确执行。
        ranked=[(key,float(scores[key])) for key in self.ids if key in selected and scores[key]>0]
        ranked.sort(key=lambda x:x[1],reverse=True)
        return ranked[:top_k]


@dataclass
class Candidate:
    """一个候选片段，带齐"每一路给我打了多少分"，方便调试和前端展示。"""

    id: str
    text: str
    metadata: dict
    vector_score: float = 0.0
    vector_rank: int = 0          # 0 = 没被向量路召回
    bm25_score: float = 0.0
    bm25_rank: int = 0            # 0 = 没被关键词路召回
    rrf_score: float = 0.0
    rerank_score: float | None = None
    rerank_rank: int = 0          # listwise 精排给出的名次（0 = 没进精排名单）

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def chunk_index(self) -> int:
        try:
            return int(self.metadata.get("chunk_index", -1))
        except (TypeError, ValueError):
            return -1

    @property
    def matched_by(self) -> list[str]:
        """这个片段是被哪几路召回的 —— 前端会显示成 [向量][关键词]。"""
        channels = []
        if self.vector_rank:
            channels.append("向量")
        if self.bm25_rank:
            channels.append("关键词")
        return channels

    @property
    def final_score(self) -> float:
        """对外展示与排序用的分数：有精排就用精排，否则退回向量相似度。"""
        if self.rerank_score is not None:
            return round(self.rerank_score, 4)
        return round(self.vector_score, 4)

    def to_hit(self) -> Hit:
        return Hit(text=self.text, metadata=self.metadata, score=self.final_score, id=self.id)


@dataclass
class RetrievalResult:
    question: str
    query: str                    # 实际用于检索的问句（可能是改写后的）
    rewritten: bool
    candidates: list[Candidate]   # 已按最终分数排好序、已过阈值（**未截断**，截断交给去重后做）
    top_k: int
    recalled: int                 # 融合后的候选总数（过阈值前）
    rerank_used: bool
    rerank_mode: str              # "" | cross-encoder | listwise
    refused: bool                 # 阈值判定：知识库里没有依据
    timings: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def top(self, n: int | None = None) -> list[Candidate]:
        return self.candidates[: (n or self.top_k)]


class Reranker:
    """精排客户端：走 HTTP 直调，不引额外 SDK（企业内网部署友好）。

    兼容两种常见的返回结构：
      · 阿里云百炼：{"output": {"results": [{"index": 0, "relevance_score": 0.9}]}}
      · 硅基流动等：{"results": [{"index": 0, "relevance_score": 0.9}]}

    内置一个**熔断器**：连续失败 N 次后停止尝试 M 秒。
    为什么必须有：Key 没开通权限时每次都返回 403，
    如果每次提问都先白等一次失败的 HTTP 往返，用户会以为"这系统就是慢"。
    熔断后直接走 listwise 降级，几百毫秒就省下来了。
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.last_error: str = ""
        self.last_usage: dict = {}
        self.failures: int = 0
        self.opened_until: float = 0.0

    @property
    def breaker_open(self) -> bool:
        return time.time() < self.opened_until

    @property
    def ready(self) -> bool:
        return self.s.rerank_ready and not self.breaker_open

    def _record_failure(self, message: str) -> None:
        self.last_error = message
        self.failures += 1
        if self.failures >= max(1, self.s.rerank_breaker_threshold):
            self.opened_until = time.time() + max(1, self.s.rerank_breaker_seconds)
            logger.warning(
                "精排连续失败 %d 次，熔断 %d 秒（期间直接用 listwise 降级）：%s",
                self.failures, self.s.rerank_breaker_seconds, message,
            )

    def rerank(self, question: str, candidates: list[Candidate], top_n: int) -> bool:
        """就地写入 rerank_score。返回是否成功（失败不抛异常，由调用方降级）。"""
        if not self.ready or not candidates:
            return False
        documents = [(c.text or " ")[:RERANK_DOC_CHARS] for c in candidates]
        started = time.perf_counter()
        try:
            payload = self._post(question, documents, top_n)
        except Exception as exc:  # noqa: BLE001 —— 精排是可选增强，绝不能让它拖垮主链路
            self._record_failure(f"{type(exc).__name__}: {exc}")
            logger.warning("精排调用失败，已降级为「向量+BM25」融合结果：%s", self.last_error)
            return False

        results = self._extract_results(payload)
        if not results:
            self._record_failure(f"返回结构无法解析：{json.dumps(payload, ensure_ascii=False)[:200]}")
            logger.warning("精排返回异常：%s", self.last_error)
            return False

        for item in results:
            try:
                index = int(item["index"])
                score = float(item.get("relevance_score", item.get("score", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(candidates):
                candidates[index].rerank_score = score
        self.last_usage = payload.get("usage") or {}
        self.failures = 0
        logger.info("精排完成：%d 条候选，用时 %.0fms", len(results), (time.perf_counter() - started) * 1000)
        return True

    def _post(self, question: str, documents: list[str], top_n: int) -> dict:
        body = {
            "model": self.s.rerank_model,
            "input": {"query": question, "documents": documents},
            "parameters": {"return_documents": False, "top_n": top_n},
        }
        headers = {
            "Authorization": f"Bearer {self.s.rerank_api_key}",
            "Content-Type": "application/json",
        }
        try:
            import httpx
        except ImportError:  # pragma: no cover —— 没装 httpx 时用标准库兜底
            return self._post_urllib(self.s.rerank_base_url, body, headers)
        with httpx.Client(timeout=self.s.rerank_timeout) as client:
            resp = client.post(self.s.rerank_base_url, headers=headers, json=body)
            if resp.status_code >= 400:
                # 把服务端返回的错误体带上：403 到底是"Key 不对"还是"没开通这个模型"，
                # 全在 body 的 code 字段里（比如 Model.AccessDenied / InvalidApiKey）。
                # 不带 body 的报错会让人在"是不是 Key 填错了"上浪费半小时。
                raise RuntimeError(
                    f"HTTP {resp.status_code} {resp.reason_phrase}：{resp.text[:300]}"
                )
            return resp.json()

    @staticmethod
    def _post_urllib(url: str, body: dict, headers: dict) -> dict:  # pragma: no cover
        """标准库兜底：企业内网经常装不上 httpx，这时候也得能跑。"""
        import urllib.request

        request = urllib.request.Request(
            url=url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _extract_results(payload: dict) -> list[dict]:
        if not isinstance(payload, dict):
            return []
        for path in (("output", "results"), ("results",)):
            node = payload
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, list):
                return node
        return []


LLM_RERANK_PROMPT = """你是检索结果排序器。请判断候选片段与问题的相关程度。

【问题】{question}

【候选片段】
{documents}

输出要求：
- 输出一个 JSON 数组，按相关度从高到低排列**确实与问题相关**的片段编号；
- 与问题无关的片段不要出现在数组里；如果全部无关，输出 []；
- 只输出数组本身，不要解释、不要加 markdown 代码块。例如：[3,1,7]"""

_JSON_ARRAY = re.compile(r"\[[\s\d,]*\]")


@dataclass
class RerankOutcome:
    ok: bool
    mode: str = ""            # cross-encoder | listwise
    provider: str = ""        # http | llm
    reason: str = ""
    usage: dict = field(default_factory=dict)
    no_relevant: bool = False # listwise 明确回答"没有一个候选相关" —— 这是最强的拒答信号


class LLMReranker:
    """用对话模型做 listwise 精排（cross-encoder 服务不可用时的降级方案）。

    为什么需要它：`gte-rerank` 这类接口要么没开通权限、要么公司内网调不出去，
    而"跨编码器排序"这件事，本质上对话模型也能做 —— 把候选编号给它，
    让它输出一个按相关度排序的数组即可。

    和 cross-encoder 的关键区别：**它给的是顺序，不是分数**。
    所以拒答阈值仍然用向量分（见 `_apply_threshold`），
    不能拿"第 1 名"去冒充"相关度 0.9" —— 那是自欺欺人。

    代价：一次提问多一次模型调用，候选越多越贵，所以只用前 N 条（默认 12）。
    """

    def __init__(self, settings: Settings, llm) -> None:
        self.s = settings
        self.llm = llm

    @property
    def ready(self) -> bool:
        return self.s.rerank_llm_ready and bool(getattr(self.llm, "llm_ready", False))

    def rerank(self, question: str, candidates: list[Candidate], top_n: int) -> RerankOutcome:
        if not self.ready or not candidates:
            return RerankOutcome(False, reason="listwise 精排不可用")
        pool = candidates[: max(1, self.s.rerank_llm_max_docs)]
        documents = "\n".join(
            f"{index}. {(candidate.text or '')[:300]}" for index, candidate in enumerate(pool, 1)
        )
        messages = [
            {"role": "system", "content": "你只输出 JSON 数组，不输出任何其他内容。"},
            {
                "role": "user",
                "content": LLM_RERANK_PROMPT.format(question=question, documents=documents),
            },
        ]
        sink: dict = {}
        try:
            reply = self.llm.chat(messages, usage_sink=sink) or ""
        except Exception as exc:  # noqa: BLE001 —— 精排是增强，不能拖垮主链路
            return RerankOutcome(False, reason=f"{type(exc).__name__}: {exc}")

        order = self._parse(reply, len(pool))
        if order is None:
            return RerankOutcome(False, reason=f"返回内容无法解析为数组：{reply[:120]}", usage=sink)
        if not order:
            # 模型明确回答"没有一个候选相关"。这是一个非常有价值的信号：
            # 向量相似度在真实语料上区分度有限（上千个片段里，随便什么问题都能找到 0.3+ 的），
            # 而"让模型看一眼再决定要不要"比任何一个固定阈值都可靠。
            return RerankOutcome(
                True, mode="listwise", provider="llm", usage=sink, no_relevant=True
            )
        for rank, index in enumerate(order, 1):
            pool[index - 1].rerank_rank = rank
        return RerankOutcome(True, mode="listwise", provider="llm", usage=sink)

    @staticmethod
    def _parse(reply: str, pool_size: int) -> list[int] | None:
        """返回 None = 解析失败（不能当成"没有相关内容"），返回 [] = 模型说全部无关。"""
        match = _JSON_ARRAY.search(reply or "")
        if not match:
            return None
        try:
            raw = json.loads(match.group(0))
        except ValueError:
            return None
        seen: set[int] = set()
        order: list[int] = []
        for item in raw:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= pool_size and index not in seen:
                seen.add(index)
                order.append(index)
        return order


class HybridRetriever:
    """把「向量路 + 关键词路 → RRF → 精排 → 阈值」串起来的门面类。

    BM25 索引是在内存里现算的，所以依赖 store.revision：
    只要向量库被写过（上传/删除/清空），revision 加一，下次提问自动重建索引。
    这样就不用为了"文档改了要刷新 BM25"再加一套通知机制。
    """

    def __init__(self, settings: Settings, store: BaseVectorStore, llm) -> None:
        self.s = settings
        self.store = store
        self.llm = llm
        self.reranker = Reranker(settings)
        self.llm_reranker = LLMReranker(settings, llm)
        self._bm25: BM25 | None = None
        self._bm25_revision: int = -1
        self.last_timing: dict[str, float] = {}
        self.last_rerank = RerankOutcome(False, reason="未执行")

    # ---------------------------------------------------------------- BM25 索引
    def _bm25_index(self) -> BM25:
        revision = getattr(self.store, "revision", 0)
        if self._bm25 is None or revision != self._bm25_revision:
            started = time.perf_counter()
            documents = {item["id"]: item["text"] for item in self.store.all_items()}
            self._bm25 = BM25(documents)
            self._bm25_revision = revision
            logger.info(
                "重建 BM25 索引：%d 段，用时 %.0fms", len(documents), (time.perf_counter() - started) * 1000
            )
        return self._bm25

    def invalidate(self) -> None:
        """显式失效（测试与手动刷新用）。"""
        self._bm25 = None
        self._bm25_revision = -1

    # ---------------------------------------------------------------- 检索主流程
    def search(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
        query: str | None = None,
        embedding: Sequence[float] | None = None,
    ) -> RetrievalResult:
        """question 用于生成侧，query 用于检索侧（查询改写后两者不同）。

        `embedding` 可由调用方传入以复用刚算好的问题向量（缓存查一次、检索用同一份），
        传了就不重复调用 embedding 接口 —— 一次提问省一次接口往返。
        """
        top_k = top_k or self.s.top_k
        search_query = (query or question).strip() or question
        timings: dict[str, float] = {}
        note = ""

        # ---- 向量路 ----
        started = time.perf_counter()
        vector = list(embedding) if embedding is not None else self.llm.embed([search_query])[0]
        hits = self.store.query(vector, self.s.retrieve_k, where=filters)
        timings["embed_ms"] = 0.0 if embedding is not None else (time.perf_counter() - started) * 1000

        candidates: dict[str, Candidate] = {}
        vector_ranking: list[str] = []
        for rank, hit in enumerate(hits, 1):
            key = hit.id or f"{hit.source}::{hit.chunk_index}"
            candidates[key] = Candidate(
                id=key,
                text=hit.text,
                metadata=dict(hit.metadata),
                vector_score=float(hit.score),
                vector_rank=rank,
            )
            vector_ranking.append(key)

        # ---- 关键词路 ----
        bm25_ranking: list[str] = []
        if self.s.hybrid_enabled:
            started = time.perf_counter()
            ranked = self._bm25_index().search(search_query, self.s.retrieve_k)
            for rank, (doc_id, score) in enumerate(ranked, 1):
                item = candidates.get(doc_id)
                if item is None:
                    record = self.store.get_item(doc_id)
                    if record is None:      # 索引和数据不同步（理论上不会发生）
                        continue
                    if not _match_filters(record["metadata"], filters):
                        continue
                    item = candidates[doc_id] = Candidate(
                        id=doc_id, text=record["text"], metadata=dict(record["metadata"])
                    )
                item.bm25_score = float(score)
                item.bm25_rank = rank
                bm25_ranking.append(doc_id)
            timings["bm25_ms"] = (time.perf_counter() - started) * 1000

        # ---- RRF 融合 ----
        started = time.perf_counter()
        fused = rrf_fuse([vector_ranking, bm25_ranking], self.s.rrf_k) if bm25_ranking else {
            doc_id: 1.0 / (self.s.rrf_k + rank) for rank, doc_id in enumerate(vector_ranking, 1)
        }
        for doc_id, score in fused.items():
            if doc_id in candidates:
                candidates[doc_id].rrf_score = score
        ordered = sorted(candidates.values(), key=lambda c: c.rrf_score, reverse=True)
        timings["fuse_ms"] = (time.perf_counter() - started) * 1000

        # ---- 精排：优先 cross-encoder，不可用时降级为 listwise（用对话模型排序）----
        rerank_used = False
        rerank_mode = ""
        channels = "向量+BM25" if bm25_ranking else "纯向量"
        outcome = RerankOutcome(False, reason="精排已关闭")
        if self.s.rerank_enabled and ordered:
            started = time.perf_counter()
            pool = ordered[: self.s.rerank_candidates]
            if self.reranker.ready:
                ok = self.reranker.rerank(search_query, pool, top_k)
                if ok:
                    outcome = RerankOutcome(True, mode="cross-encoder", provider="http",
                                            usage=getattr(self.reranker, "last_usage", {}) or {})
                else:
                    outcome = RerankOutcome(False, reason=self.reranker.last_error or "cross-encoder 调用失败")
            if not outcome.ok and self.llm_reranker.ready:
                fallback = self.llm_reranker.rerank(search_query, pool, top_k)
                if fallback.ok:
                    outcome = fallback
                else:
                    outcome.reason = f"{outcome.reason}；listwise 也不可用：{fallback.reason}"
            timings["rerank_ms"] = (time.perf_counter() - started) * 1000
            rerank_used = outcome.ok
            rerank_mode = outcome.mode
            if not rerank_used:
                note = f"精排不可用（{outcome.reason}），已降级为「{channels}」融合结果"
            elif rerank_mode == "listwise":
                note = f"cross-encoder 不可用，已用对话模型做 listwise 精排（宁可用顺序，也不静默跳过）"
        elif not self.s.rerank_enabled:
            note = f"精排已关闭（RERANK_ENABLED=false），当前为「{channels}」融合结果"
        else:
            note = f"没有可精排的候选，当前为「{channels}」融合结果"
        self.last_rerank = outcome

        if rerank_mode == "cross-encoder":
            ordered = sorted(
                [c for c in ordered if c.rerank_score is not None],
                key=lambda c: c.rerank_score or 0.0,
                reverse=True,
            )
        elif rerank_mode == "listwise":
            # 只有被模型点名的候选有名次，其余排在后面（保持 RRF 的相对顺序）
            ordered = sorted(
                ordered,
                key=lambda c: (c.rerank_rank if c.rerank_rank else 10**6),
            )

        # ---- 阈值拒答 ----
        if outcome.ok and outcome.no_relevant:
            # 最强的拒答信号：listwise 精排看完全部候选，明确回答"都不相关"。
            # 它比任何固定阈值都可靠 —— 因为阈值是"分数分布"上的一个点，
            # 而这是在"读懂问题"之后做的判断。
            kept: list[Candidate] = []
            note = "listwise 精排判定：所有候选都与问题无关 → 拒答（未调用大模型生成）"
        else:
            kept = self._apply_threshold(
                ordered, rerank_mode, self._strong_terms(search_query, bool(bm25_ranking))
            )

        return RetrievalResult(
            question=question,
            query=search_query,
            rewritten=search_query != question,
            candidates=kept,
            top_k=top_k,
            recalled=len(ordered),
            rerank_used=rerank_used,
            rerank_mode=rerank_mode,
            refused=not kept,
            timings=timings,
            note=note,
        )

    def _strong_terms(self, query: str, has_keyword_channel: bool) -> set[str]:
        """找出这次查询里"够格豁免阈值"的关键词。

        规则一：**长度至少 2**。中文单字歧义太大 ——
        查"请问怎么养一只柯基犬"时，单字"问"会命中"提问"，
        这种命中没有任何信息量，不该被当成强证据。
        规则二：**idf 要达到本次查询里最大 idf 的 60%**（相对门槛，不是绝对值）。
        为什么用相对值：绝对 idf 会随语料规模漂移（同一个词在 4 段语料里 idf=1.2，
        在 20 万段语料里 idf=8），写死一个绝对门槛换个数据集就失效。
        """
        if not has_keyword_channel:
            return set()
        index = self._bm25_index()
        scored = [(term, index.idf.get(term, 0.0)) for term in tokenize(query)]
        best = max((idf for _, idf in scored), default=0.0)
        if best <= 0:
            return set()
        floor = best * self.s.min_bm25_idf_ratio
        return {term for term, idf in scored if len(term) >= 2 and idf >= floor}

    def _apply_threshold(
        self,
        ordered: list[Candidate],
        rerank_mode: str,
        strong_terms: set[str],
    ) -> list[Candidate]:
        """把"分数太低"的候选挡在大模型之前 —— 这是防幻觉最可靠的一层。

        - cross-encoder 精排：以它的相关性分为准（最可信）；
        - listwise 精排：**它只给顺序不给分数**，所以阈值仍按向量分/强关键词判定，
          绝不能拿"排第 1 名"去冒充"相关度 0.95"；
        - 都没精排：向量相似度过线，**或者**是 BM25 前 2 名且命中了强关键词。

        为什么要给 BM25 开后门：错误码 / 型号这类查询的向量分数天然很低，
        但它们恰恰是最该答对的，丢掉它们等于把混合检索白做了。
        """
        strong_keyword: set[str] = set()
        if strong_terms:
            for candidate in ordered:
                if candidate.bm25_rank in (1, 2) and any(
                    term in candidate.text.lower() for term in strong_terms
                ):
                    strong_keyword.add(candidate.id)

        kept: list[Candidate] = []
        for candidate in ordered:
            if rerank_mode == "cross-encoder":
                if (candidate.rerank_score or 0.0) >= self.s.min_rerank_score:
                    kept.append(candidate)
            elif candidate.vector_score >= self.s.min_vector_score or candidate.id in strong_keyword:
                kept.append(candidate)
        return kept


def _match_filters(metadata: dict, filters: dict | None) -> bool:
    """内存侧过滤（BM25 候选走这里；向量库侧由 where 参数各自实现）。"""
    if not filters:
        return True
    return all(str(metadata.get(key, "")) == str(value) for key, value in filters.items())


def dedupe_by_source(candidates: Iterable[Candidate], limit_per_source: int = 2) -> list[Candidate]:
    """同一个文档最多留 N 段。

    不这样做的话，Top-4 可能全是同一篇文档的相邻片段 —— 资料看着有 4 条，
    实际只覆盖了一个知识点，反而是噪声。
    """
    seen: Counter = Counter()
    kept: list[Candidate] = []
    for candidate in candidates:
        if seen[candidate.source] >= limit_per_source:
            continue
        seen[candidate.source] += 1
        kept.append(candidate)
    return kept


def best_vector_score(candidates: Sequence[Candidate]) -> float:
    return max((c.vector_score for c in candidates), default=0.0)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """给语义缓存复用（转成 list 后走同一套实现，避免两份逻辑不一致）。"""
    return cosine(list(a), list(b))
