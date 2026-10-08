"""新增框架集成回归：验证真正经过LangChain与LangGraph，不只检查导入。"""
import json
import pytest
from dataclasses import replace
from langgraph.graph.state import CompiledStateGraph
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence
from app.framework import HybridSearchRetriever, LocalHashEmbeddings
from app.vectorstore import ChromaStore
from app.llm import LLMClient


def test_original_service_uses_compiled_graphs(rag):
    assert isinstance(rag.workflow,CompiledStateGraph)
    assert isinstance(rag.stream_workflow,CompiledStateGraph)
    assert {'prepare','cached','rewrite','retrieve','select','refuse','generate'} <= set(rag.workflow.nodes)

def test_generate_refuse_and_cache_paths_are_recorded(seeded,seeded_strict):
    seeded.answer('g','专业版价格是多少？',use_cache=False)
    traces=seeded.tracer.recent(1)
    trace=traces[0]
    # 各Trace持久化格式直接从具体对象读取，不借助测试猜测。
    assert 'langgraph' in json.dumps(trace,ensure_ascii=False)
    assert 'generate' in json.dumps(trace,ensure_ascii=False)
    seeded.answer('g2','专业版价格是多少？')
    assert 'cached' in json.dumps(seeded.tracer.recent(1),ensure_ascii=False)
    _,_,meta=seeded_strict.answer('refuse','请问怎么养一只柯基犬',use_cache=False)
    assert meta['refused'] and 'refuse' in json.dumps(seeded_strict.tracer.recent(1),ensure_ascii=False)

def test_retriever_returns_langchain_documents(seeded):
    adapter=HybridSearchRetriever(backend=seeded.retriever,options={'top_k':4})
    assert isinstance(adapter,BaseRetriever)
    docs=adapter.invoke('40412')
    assert docs and all(isinstance(d,Document) for d in docs)
    assert adapter.result.query=='40412'
    assert 'bm25_score' in docs[0].metadata

def test_local_embedding_interface_preserves_old_values():
    from app.embedding_local import embed_local
    embed=LocalHashEmbeddings(1024)
    assert embed.embed_query('退款政策')==embed_local('退款政策',1024)

def test_chroma_langchain_write_query_reuse_and_clear(tmp_path,make_settings):
    from tests.conftest import build_rag, DOC_MANUAL
    from langchain_chroma import Chroma
    service=build_rag(make_settings(vector_backend='chroma',chroma_dir=tmp_path/'chroma-real'))
    # fixture工厂固定MemoryStore，这里显式替换真实Chroma并重建检索器。
    from app.retrieval import HybridRetriever
    service.store=ChromaStore(tmp_path/'chroma-real')
    service.retriever=HybridRetriever(service.s,service.store,service.llm)
    assert isinstance(service.store.langchain,Chroma)
    a=service.ingest_text('手册.md',DOC_MANUAL)
    b=service.ingest_text('手册.md',DOC_MANUAL.replace('199 元每月','299 元每月'))
    assert b['embedded_chunks']==1 and b['reused_chunks']==a['chunks']-1
    assert service.retrieve('专业版价格')
    service.clear_documents()
    assert service.store.count()==0
    service.ingest_text('新文档.md','设备错误码40412表示配额已用尽。')
    assert service.retrieve('40412')

def test_stream_graph_emits_original_event_protocol(seeded):
    packets=list(seeded.stream_answer('stream','40412',use_cache=False))
    assert [p['type'] for p in packets][0:2]==['meta','citations']
    assert packets[-1]['type']=='done'
    assert ''.join(p['text'] for p in packets if p['type']=='delta')==packets[-1]['answer']

def test_signature_matches_missing_key_fallback(make_settings):
    client=LLMClient(make_settings(embedding_mode='api',embedding_api_key='',llm_api_key=''))
    assert client.signature.startswith('local:')

def test_real_langchain_models_and_prompt_with_mock_http(make_settings,monkeypatch):
    import httpx2 as httpx
    import app.llm as module
    from langchain_openai import ChatOpenAI,OpenAIEmbeddings
    seen=[]
    def handler(request):
        payload=json.loads(request.content);seen.append(payload)
        if request.url.path.endswith('/embeddings'):
            texts=payload['input'];assert all(isinstance(t,str) for t in texts)
            vectors=LocalHashEmbeddings().embed_documents(texts)
            return httpx.Response(200,json={'object':'list','model':'test','data':[{'index':i,'object':'embedding','embedding':v} for i,v in enumerate(vectors)]})
        if not payload.get('stream'):
            return httpx.Response(200,json={'id':'test','object':'chat.completion','created':1,'model':'test','choices':[{'index':0,'message':{'role':'assistant','content':'专业版199元[1]。'},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})
        chunks=[{'id':'test','object':'chat.completion.chunk','created':1,'model':'test','choices':[{'index':0,'delta':{'content':s},'finish_reason':None}]} for s in ['专业版','199元[1]。']]
        chunks.append({'id':'test','object':'chat.completion.chunk','created':1,'model':'test','choices':[],'usage':{'prompt_tokens':12,'completion_tokens':6,'total_tokens':18}})
        wire=''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n'
        return httpx.Response(200,text=wire,headers={'Content-Type':'text/event-stream'})
    transport=httpx.MockTransport(handler);sync=httpx.Client(transport=transport);async_client=httpx.AsyncClient(transport=transport)
    monkeypatch.setattr(module,'ChatOpenAI',lambda **kw:ChatOpenAI(**kw,http_client=sync,http_async_client=async_client))
    monkeypatch.setattr(module,'OpenAIEmbeddings',lambda **kw:OpenAIEmbeddings(**kw,http_client=sync,http_async_client=async_client))
    cfg=make_settings(embedding_mode='api',embedding_api_key='fake-test-key',llm_api_key='fake-test-key',
                      embedding_base_url='https://mock.test/v1',llm_base_url='https://mock.test/v1')
    client=LLMClient(cfg)
    assert len(client.embed(['专业版价格199元'])[0])==1024
    messages=[{'role':'system','content':'只依据资料回答'},{'role':'user','content':'专业版价格？资料：199元'}]
    sink={};assert client.chat(messages,usage_sink=sink)=='专业版199元[1]。'
    assert isinstance(client._chain(),RunnableSequence) and sink['prompt_tokens']==10
    sink={};assert list(client.chat(messages,stream=True,usage_sink=sink))==['专业版','199元[1]。']
    assert sink['completion_tokens']==6
    sync.close()
    import asyncio
    asyncio.run(async_client.aclose())


def test_document_listing_contains_embedding_signature(seeded):
    docs=seeded.documents.all()
    assert docs and all(d['embedding_signature']==seeded.llm.signature for d in docs)

def test_hit_at_k_does_not_count_fifth_candidate(monkeypatch,tmp_path):
    import scripts.evaluate_retrieval as module
    from types import SimpleNamespace
    from app.vectorstore import MemoryStore
    hits=[SimpleNamespace(source=str(i),text='其他资料') for i in range(4)]
    hits.append(SimpleNamespace(source='目标.md',text='目标问题'))
    result=SimpleNamespace(rerank_used=False,top=lambda n:hits[:n])
    monkeypatch.setattr(module,'HybridRetriever',lambda *args:SimpleNamespace(search=lambda *a,**kw:result))
    report=module.evaluate({},MemoryStore(tmp_path/'index.json'),
      {'doc':[('目标问题','目标.md')],'sampled':[('目标问题','目标.md')],'specific':[('目标问题','目标.md')]},
      [[1.0]],{'目标问题':0},4)
    assert report['doc4']==0 and report['sec4']==0 and report['spec4']==0
