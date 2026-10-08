"""LangChain模型封装，保留原项目embed/chat/usage_sink接口与本地演示模式。

真实模型用ChatOpenAI、OpenAIEmbeddings；提示词用ChatPromptTemplate；
普通生成走RunnableSequence.invoke，流式生成走RunnableSequence.stream。
"""
from __future__ import annotations
import logging
import os
from typing import Sequence, Iterator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import convert_to_messages
from .config import Settings
from .framework import LocalHashEmbeddings
logger=logging.getLogger(__name__)

class LLMNotConfigured(Exception):
    """没有可用聊天Key。"""
class EmbeddingUnavailable(Exception):
    """真实向量接口失败；不换成本地向量污染索引。"""

def estimate_tokens(text:str)->int:
    text=text or ''
    if not text:return 0
    han=sum(1 for ch in text if '\u4e00'<=ch<='\u9fff')
    other=len(text)-han
    return han+max(1,other//4) if other else han

class LLMClient:
    def __init__(self,settings:Settings):
        self.s=settings
        self._chains={}
        self._embeddings={}
        self.local_embeddings=LocalHashEmbeddings(settings.local_embedding_dim)

    @property
    def llm_ready(self):
        return self.s.llm_ready

    def _chain(self,include_usage=True):
        if not self.s.llm_api_key:
            raise LLMNotConfigured('未配置LLM_API_KEY')
        key=(self.s.llm_api_key,self.s.llm_base_url,self.s.llm_model,include_usage)
        if key not in self._chains:
            model=ChatOpenAI(model=self.s.llm_model,api_key=self.s.llm_api_key,
                base_url=self.s.llm_base_url,temperature=self.s.llm_temperature,
                timeout=self.s.llm_timeout,max_retries=2,stream_usage=include_usage,use_responses_api=False)
            # 原_messages返回dict消息；先转换为LangChain消息，再填入占位模板。
            prompt=ChatPromptTemplate.from_messages([MessagesPlaceholder('messages')])
            self._chains[key]=prompt | model
        return self._chains[key]

    @staticmethod
    def _content(response):
        if isinstance(response.content,str):return response.content
        return ''.join(item.get('text','') for item in response.content if isinstance(item,dict))

    def chat(self,messages:Sequence[dict],stream=False,usage_sink=None):
        prompt_text='\n'.join(str(m.get('content','')) for m in messages)
        inputs={'messages':convert_to_messages(list(messages))}
        if stream:
            return self._stream(inputs,prompt_text,usage_sink)
        reply=self._chain().invoke(inputs)
        text=self._content(reply)
        if usage_sink is not None:
            usage_sink.update(self._usage_of(getattr(reply,'usage_metadata',None),prompt_text,text))
        return text

    def _stream(self,inputs,prompt_text,usage_sink)->Iterator[str]:
        parts=[];usage=None
        include=os.getenv('STREAM_USAGE','true').lower() not in {'false','0','no'}
        for attempt in range(2):
            try:
                for chunk in self._chain(include_usage=include).stream(inputs):
                    if getattr(chunk,'usage_metadata',None):usage=chunk.usage_metadata
                    text=self._content(chunk)
                    if text:
                        parts.append(text)
                        yield text
                break
            except Exception as exc:
                # 仅在服务拒绝stream_options且尚未发出文字时取消usage再试。
                message=str(exc).lower()
                if not parts and include and attempt==0 and any(w in message for w in ['stream_options','include_usage']):
                    logger.warning('服务不支持流式usage，关闭该参数重试')
                    include=False
                else:raise
        if usage_sink is not None:
            usage_sink.update(self._usage_of(usage,prompt_text,''.join(parts)))

    @staticmethod
    def _usage_of(usage,prompt_text,completion_text):
        usage=usage or {}
        prompt=int(usage.get('input_tokens',usage.get('prompt_tokens',0)) or 0)
        completion=int(usage.get('output_tokens',usage.get('completion_tokens',0)) or 0)
        if prompt or completion:
            return {'prompt_tokens':prompt,'completion_tokens':completion,'estimated':False}
        return {'prompt_tokens':estimate_tokens(prompt_text),'completion_tokens':estimate_tokens(completion_text),'estimated':True}

    def embedding_key(self):
        if self.s.embedding_api_key:return self.s.embedding_api_key
        from urllib.parse import urlparse
        if urlparse(self.s.embedding_base_url).hostname==urlparse(self.s.llm_base_url).hostname:
            return self.s.llm_api_key
        return ''

    @property
    def signature(self):
        # 修正原版缺Key实际用哈希却标成api的身份问题。
        if self.s.embedding_mode=='api' and self.embedding_key():
            return f'api:{self.s.embedding_model}'
        return f'local:hash-{self.s.local_embedding_dim}d'

    def embed(self,texts:Sequence[str])->list[list[float]]:
        if not texts:return []
        key=self.embedding_key()
        if self.s.embedding_mode=='api' and key:
            return self._embed_api(list(texts),key)
        if self.s.embedding_mode=='api':logger.warning('没有可用embedding Key，使用原本的本地演示模式')
        return self.local_embeddings.embed_documents(list(texts))

    def _embed_api(self,texts,api_key):
        key=(api_key,self.s.embedding_base_url,self.s.embedding_model)
        if key not in self._embeddings:
            self._embeddings[key]=OpenAIEmbeddings(model=self.s.embedding_model,api_key=api_key,
                base_url=self.s.embedding_base_url,check_embedding_ctx_length=False,
                model_kwargs={'encoding_format':'float'},chunk_size=max(1,self.s.embedding_batch_size),
                request_timeout=self.s.llm_timeout,max_retries=2)
        try:
            # 保留原批次的单段2000字符上限与空白兜底。
            return self._embeddings[key].embed_documents([t[:2000] or ' ' for t in texts])
        except Exception as exc:
            raise EmbeddingUnavailable(f'LangChain向量化失败：{type(exc).__name__}: {exc}\n模型={self.s.embedding_model}；地址={self.s.embedding_base_url}') from exc

    def describe(self):
        actual_api=self.s.embedding_mode=='api' and bool(self.embedding_key())
        return {'llm_ready':self.llm_ready,'llm_model':self.s.llm_model,
            'embedding_mode':self.s.embedding_mode,
            'embedding_model':self.s.embedding_model if actual_api else f'local-hash-{self.s.local_embedding_dim}d',
            'embedding_signature':self.signature,'model_framework':'langchain'}
