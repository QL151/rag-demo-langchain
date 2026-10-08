"""LangChain业务适配器：接入已有服务，保留分数、重排与阈值的业务约定。"""
from typing import Any
from pydantic import PrivateAttr
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from .embedding_local import embed_local

class LocalHashEmbeddings(Embeddings):
    """保留原离线哈希，但实现LangChain标准Embeddings接口。"""
    def __init__(self,dim=1024):
        self.dim=dim
    def embed_documents(self,texts):
        return [embed_local(text,self.dim) for text in texts]
    def embed_query(self,text):
        return self.embed_documents([text])[0]

class HybridSearchRetriever(BaseRetriever):
    """LangChain检索器：执行原双路融合和重排，输出标准Document并保留完整诊断。"""
    backend: Any
    options: dict
    _result: Any = PrivateAttr(default=None)

    @property
    def result(self):
        return self._result

    def _get_relevant_documents(self,query:str,*,run_manager:CallbackManagerForRetrieverRun):
        self._result=self.backend.search(query,**self.options)
        return [Document(page_content=c.text,metadata={**c.metadata,
                    'candidate_id':c.id,'vector_score':c.vector_score,'bm25_score':c.bm25_score,
                    'rrf_score':c.rrf_score,'rerank_score':c.rerank_score},id=c.id)
                for c in self._result.candidates]

class ProvidedBatchEmbeddings(Embeddings):
    """让LangChain写库复用原项目已算好的批次向量，不重算未修改片段。"""
    def __init__(self,texts,vectors):
        self.texts,self.vectors=texts,vectors
    def embed_documents(self,texts):
        if list(texts)!=self.texts or len(texts)!=len(self.vectors):
            raise ValueError('预计算批次与LangChain写入文本不一致')
        return self.vectors
    def embed_query(self,text):
        raise ValueError('预计算批次只用于写入；查询应使用实际模型向量')
