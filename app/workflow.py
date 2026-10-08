"""原RAG流程的LangGraph版：原步骤拆成节点，图负责缓存和拒答的条件分支。

ctx存原方法的局部状态，避免迁移时改变外部API与计量字段。未配置checkpointer；
SQLite仍保存会话，不把包含Trace对象的临时状态当作可恢复运行检查点。
"""
from typing import TypedDict, Any
import time
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from .schemas import Citation
from .llm import EmbeddingUnavailable
logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    ctx: dict[str, Any]

def emit(packet):
    get_stream_writer()(packet)
STATE_FIELDS = ['c', 'cached', 'changed', 'citations', 'cost', 'delta', 'dept', 'doc_type', 'done', 'filters', 'final', 'first_token_ms', 'graph_nodes', 'history', 'item', 'latency_ms', 'meta', 'output', 'parts', 'payload', 'question', 'question_vector', 'result', 'rewrite_sink', 'rewritten', 'selected', 'session_id', 'sink', 'stage', 'started', 'started_generate', 'text', 'top_k', 'trace', 'use_cache']

def build_workflow(service, streaming=False):
    from .rag import _top_score
    self = service

    def a_prepare(state):
        """读取历史、记录用户问题、计算问题向量并查询答案缓存，输出缓存判定与本轮状态。"""
        ctx = dict(state['ctx'])
        dept = ctx.get('dept')
        doc_type = ctx.get('doc_type')
        filters = ctx.get('filters')
        history = ctx.get('history')
        question = ctx.get('question')
        question_vector = ctx.get('question_vector')
        session_id = ctx.get('session_id')
        trace = ctx.get('trace')
        use_cache = ctx.get('use_cache')
        graph_nodes = ctx.get('graph_nodes', []) + ['prepare']
        trace = self.tracer.start(question, session_id)
        filters = self._filters(dept, doc_type)
        started = time.perf_counter()
        history = self.chat.history(session_id, self.s.max_history_turns)
        self.chat.add(session_id, 'user', question)
        trace.set(history_turns=len(history), dept=dept or '', doc_type=doc_type or '')
        question_vector: list[float] | None = None
        try:
            with trace.stage('embed'):
                question_vector = self.llm.embed([question])[0]
        except EmbeddingUnavailable as exc:
            trace.error = str(exc)
            trace.set(workflow='langgraph', graph_nodes=graph_nodes)
            self.tracer.finish(trace)
            raise
        cached = self._cache_lookup(question, question_vector, filters, use_cache)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def a_cached(state):
        """复用缓存回答，写入会话与费用记录，输出答案、引用和缓存命中元信息。"""
        ctx = dict(state['ctx'])
        c = ctx.get('c')
        cached = ctx.get('cached')
        citations = ctx.get('citations')
        item = ctx.get('item')
        meta = ctx.get('meta')
        session_id = ctx.get('session_id')
        started = ctx.get('started')
        text = ctx.get('text')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['cached']
        citations = [Citation(**item) for item in cached['citations']]
        text = cached['answer']
        self.chat.add(session_id, 'assistant', text, [c.model_dump() for c in citations])
        self.usage.record(kind='answer', model=self.s.llm_model, cache_hit=True, session_id=session_id, trace_id=trace.id, latency_ms=(time.perf_counter() - started) * 1000)
        trace.set(cache=cached['match'], cache_similarity=cached['similarity'], refused=False, citation_count=len(citations), reused_question=cached['cached_question'])
        meta = self._meta(trace, cached['match'], None, False, False, len(citations), started)
        trace.set(workflow='langgraph', graph_nodes=graph_nodes)
        self.tracer.finish(trace)
        output = (text, citations, meta)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def a_rewrite(state):
        """结合历史补全问句，记录改写的状态与计量；首轮或失败时保留原问题。"""
        ctx = dict(state['ctx'])
        changed = ctx.get('changed')
        history = ctx.get('history')
        question = ctx.get('question')
        rewrite_sink = ctx.get('rewrite_sink')
        session_id = ctx.get('session_id')
        stage = ctx.get('stage')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['rewrite']
        rewrite_sink: dict = {}
        with trace.stage('rewrite') as stage:
            rewritten, changed = self.rewrite_query(history, question, rewrite_sink)
            stage['changed'] = changed
        if rewrite_sink:
            self.usage.record(kind='rewrite', model=self.s.llm_model, prompt_tokens=rewrite_sink.get('prompt_tokens', 0), completion_tokens=rewrite_sink.get('completion_tokens', 0), estimated=bool(rewrite_sink.get('estimated', False)), latency_ms=rewrite_sink.get('latency_ms', 0.0), session_id=session_id, trace_id=trace.id)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def a_retrieve(state):
        """调用LangChain检索适配器，执行原混合召回、RRF、精排和阈值规则，输出检索结果。"""
        ctx = dict(state['ctx'])
        changed = ctx.get('changed')
        dept = ctx.get('dept')
        doc_type = ctx.get('doc_type')
        question = ctx.get('question')
        question_vector = ctx.get('question_vector')
        result = ctx.get('result')
        rewritten = ctx.get('rewritten')
        session_id = ctx.get('session_id')
        stage = ctx.get('stage')
        top_k = ctx.get('top_k')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['retrieve']
        with trace.stage('retrieve') as stage:
            result = self.search(question, top_k, dept=dept, doc_type=doc_type, query=rewritten if changed else None, embedding=question_vector if not changed else None)
            stage.update({'recalled': result.recalled, 'kept': len(result.candidates), 'rerank_used': result.rerank_used})
        self._record_rerank(trace, session_id)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def a_select(state):
        """限制同来源片段并截取Top-K，产生最终引用，供条件边判断是否有足够资料。"""
        ctx = dict(state['ctx'])
        result = ctx.get('result')
        selected = ctx.get('selected')
        graph_nodes = ctx.get('graph_nodes', []) + ['select']
        selected = self._select(result)
        citations = self._citations(selected)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def a_refuse(state):
        """资料不足时生成拒答、保存历史和trace；不会进入正常回答生成节点。"""
        ctx = dict(state['ctx'])
        changed = ctx.get('changed')
        meta = ctx.get('meta')
        result = ctx.get('result')
        rewritten = ctx.get('rewritten')
        session_id = ctx.get('session_id')
        started = ctx.get('started')
        text = ctx.get('text')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['refuse']
        text = self._refusal()
        self.chat.add(session_id, 'assistant', text, [])
        trace.set(cache='miss', refused=True, refused_by='threshold', rewritten_query=rewritten if changed else None, rerank_used=result.rerank_used, rerank_mode=result.rerank_mode, recalled=result.recalled, kept=len(result.candidates), top_vector_score=_top_score(result), note=result.note)
        meta = self._meta(trace, 'miss', rewritten if changed else None, result.rerank_used, True, 0, started)
        trace.set(workflow='langgraph', graph_nodes=graph_nodes)
        self.tracer.finish(trace)
        output = (text, [], meta)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def a_generate(state):
        """将问题、历史和检索原文组成提示词，调用LangChain聊天链，保存答案、引用和计量。"""
        ctx = dict(state['ctx'])
        c = ctx.get('c')
        changed = ctx.get('changed')
        citations = ctx.get('citations')
        cost = ctx.get('cost')
        filters = ctx.get('filters')
        history = ctx.get('history')
        latency_ms = ctx.get('latency_ms')
        meta = ctx.get('meta')
        question = ctx.get('question')
        question_vector = ctx.get('question_vector')
        result = ctx.get('result')
        rewritten = ctx.get('rewritten')
        selected = ctx.get('selected')
        session_id = ctx.get('session_id')
        sink = ctx.get('sink')
        stage = ctx.get('stage')
        started = ctx.get('started')
        text = ctx.get('text')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['generate']
        sink: dict = {}
        with trace.stage('generate') as stage:
            if not self.llm.llm_ready:
                text = self._retrieval_only_answer(citations)
            else:
                text = self.llm.chat(self._messages(history, question, selected), usage_sink=sink)
            stage['chars'] = len(text)
        latency_ms = (time.perf_counter() - started) * 1000
        cost = self.usage.record(kind='answer', model=self.s.llm_model, prompt_tokens=sink.get('prompt_tokens', 0), completion_tokens=sink.get('completion_tokens', 0), estimated=bool(sink.get('estimated', False)), latency_ms=latency_ms, session_id=session_id, trace_id=trace.id)
        self.chat.add(session_id, 'assistant', text, [c.model_dump() for c in citations])
        self.cache.put(question=question, answer=text, citations=[c.model_dump() for c in citations], embedding=question_vector, filters=filters, embedding_signature=self.llm.signature)
        trace.set(cache='miss', refused=False, rewritten_query=rewritten if changed else None, rerank_used=result.rerank_used, rerank_mode=result.rerank_mode, citation_count=len(citations), recalled=result.recalled, kept=len(result.candidates), prompt_tokens=sink.get('prompt_tokens', 0), completion_tokens=sink.get('completion_tokens', 0), cost_yuan=round(cost, 6), note=result.note, top_vector_score=_top_score(result))
        meta = self._meta(trace, 'miss', rewritten if changed else None, result.rerank_used, False, len(citations), started, sink, cost)
        trace.set(workflow='langgraph', graph_nodes=graph_nodes)
        self.tracer.finish(trace)
        output = (text, citations, meta)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def s_prepare(state):
        """读取历史、记录用户问题、计算问题向量并查询答案缓存，输出缓存判定与本轮状态。流式版通过custom writer发送原协议事件。"""
        ctx = dict(state['ctx'])
        dept = ctx.get('dept')
        doc_type = ctx.get('doc_type')
        filters = ctx.get('filters')
        history = ctx.get('history')
        question = ctx.get('question')
        question_vector = ctx.get('question_vector')
        session_id = ctx.get('session_id')
        trace = ctx.get('trace')
        use_cache = ctx.get('use_cache')
        graph_nodes = ctx.get('graph_nodes', []) + ['prepare']
        history = self.chat.history(session_id, self.s.max_history_turns)
        self.chat.add(session_id, 'user', question)
        trace.set(history_turns=len(history), dept=dept or '', doc_type=doc_type or '')
        with trace.stage('embed'):
            question_vector = self.llm.embed([question])[0]
        cached = self._cache_lookup(question, question_vector, filters, use_cache)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def s_cached(state):
        """复用缓存回答，写入会话与费用记录，输出答案、引用和缓存命中元信息。流式版通过custom writer发送原协议事件。"""
        ctx = dict(state['ctx'])
        c = ctx.get('c')
        cached = ctx.get('cached')
        citations = ctx.get('citations')
        item = ctx.get('item')
        meta = ctx.get('meta')
        session_id = ctx.get('session_id')
        started = ctx.get('started')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['cached']
        citations = [Citation(**item) for item in cached['citations']]
        self.usage.record(kind='answer', model=self.s.llm_model, cache_hit=True, session_id=session_id, trace_id=trace.id, latency_ms=(time.perf_counter() - started) * 1000)
        trace.set(cache=cached['match'], cache_similarity=cached['similarity'], citation_count=len(citations), reused_question=cached['cached_question'])
        meta = self._meta(trace, cached['match'], None, False, False, len(citations), started)
        emit({'type': 'meta', **meta})
        emit({'type': 'citations', 'citations': [c.model_dump() for c in citations]})
        emit({'type': 'delta', 'text': cached['answer']})
        self.chat.add(session_id, 'assistant', cached['answer'], [c.model_dump() for c in citations])
        trace.set(workflow='langgraph', graph_nodes=graph_nodes)
        self.tracer.finish(trace)
        emit({'type': 'done', 'answer': cached['answer'], **meta})
        pass
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def s_rewrite(state):
        """结合历史补全问句，记录改写的状态与计量；首轮或失败时保留原问题。流式版通过custom writer发送原协议事件。"""
        ctx = dict(state['ctx'])
        changed = ctx.get('changed')
        history = ctx.get('history')
        question = ctx.get('question')
        rewrite_sink = ctx.get('rewrite_sink')
        session_id = ctx.get('session_id')
        stage = ctx.get('stage')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['rewrite']
        rewrite_sink: dict = {}
        with trace.stage('rewrite') as stage:
            rewritten, changed = self.rewrite_query(history, question, rewrite_sink)
            stage['changed'] = changed
        if rewrite_sink:
            self.usage.record(kind='rewrite', model=self.s.llm_model, prompt_tokens=rewrite_sink.get('prompt_tokens', 0), completion_tokens=rewrite_sink.get('completion_tokens', 0), estimated=bool(rewrite_sink.get('estimated', False)), latency_ms=rewrite_sink.get('latency_ms', 0.0), session_id=session_id, trace_id=trace.id)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def s_retrieve(state):
        """调用LangChain检索适配器，执行原混合召回、RRF、精排和阈值规则，输出检索结果。流式版通过custom writer发送原协议事件。"""
        ctx = dict(state['ctx'])
        changed = ctx.get('changed')
        dept = ctx.get('dept')
        doc_type = ctx.get('doc_type')
        question = ctx.get('question')
        question_vector = ctx.get('question_vector')
        result = ctx.get('result')
        rewritten = ctx.get('rewritten')
        session_id = ctx.get('session_id')
        stage = ctx.get('stage')
        top_k = ctx.get('top_k')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['retrieve']
        with trace.stage('retrieve') as stage:
            result = self.search(question, top_k, dept=dept, doc_type=doc_type, query=rewritten if changed else None, embedding=question_vector if not changed else None)
            stage.update({'recalled': result.recalled, 'kept': len(result.candidates), 'rerank_used': result.rerank_used})
        self._record_rerank(trace, session_id)
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def s_select(state):
        """限制同来源片段并截取Top-K，产生最终引用，供条件边判断是否有足够资料。流式版通过custom writer发送原协议事件。"""
        ctx = dict(state['ctx'])
        c = ctx.get('c')
        citations = ctx.get('citations')
        result = ctx.get('result')
        selected = ctx.get('selected')
        graph_nodes = ctx.get('graph_nodes', []) + ['select']
        selected = self._select(result)
        citations = self._citations(selected)
        payload = [c.model_dump() for c in citations]
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def s_refuse(state):
        """资料不足时生成拒答、保存历史和trace；不会进入正常回答生成节点。流式版通过custom writer发送原协议事件。"""
        ctx = dict(state['ctx'])
        changed = ctx.get('changed')
        meta = ctx.get('meta')
        result = ctx.get('result')
        rewritten = ctx.get('rewritten')
        session_id = ctx.get('session_id')
        started = ctx.get('started')
        text = ctx.get('text')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['refuse']
        text = self._refusal()
        trace.set(cache='miss', refused=True, refused_by='threshold', rewritten_query=rewritten if changed else None, rerank_used=result.rerank_used, rerank_mode=result.rerank_mode, recalled=result.recalled, kept=len(result.candidates), note=result.note, top_vector_score=_top_score(result))
        meta = self._meta(trace, 'miss', rewritten if changed else None, result.rerank_used, True, 0, started)
        emit({'type': 'meta', **meta})
        emit({'type': 'citations', 'citations': []})
        emit({'type': 'delta', 'text': text})
        self.chat.add(session_id, 'assistant', text, [])
        trace.set(workflow='langgraph', graph_nodes=graph_nodes)
        self.tracer.finish(trace)
        emit({'type': 'done', 'answer': text, **meta})
        pass
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}

    def s_generate(state):
        """将问题、历史和检索原文组成提示词，调用LangChain聊天链，保存答案、引用和计量。流式版通过custom writer发送原协议事件。"""
        ctx = dict(state['ctx'])
        changed = ctx.get('changed')
        citations = ctx.get('citations')
        cost = ctx.get('cost')
        delta = ctx.get('delta')
        done = ctx.get('done')
        filters = ctx.get('filters')
        final = ctx.get('final')
        first_token_ms = ctx.get('first_token_ms')
        history = ctx.get('history')
        latency_ms = ctx.get('latency_ms')
        meta = ctx.get('meta')
        parts = ctx.get('parts')
        payload = ctx.get('payload')
        question = ctx.get('question')
        question_vector = ctx.get('question_vector')
        result = ctx.get('result')
        rewritten = ctx.get('rewritten')
        selected = ctx.get('selected')
        session_id = ctx.get('session_id')
        sink = ctx.get('sink')
        started = ctx.get('started')
        started_generate = ctx.get('started_generate')
        text = ctx.get('text')
        trace = ctx.get('trace')
        graph_nodes = ctx.get('graph_nodes', []) + ['generate']
        meta = self._meta(trace, 'miss', rewritten if changed else None, result.rerank_used, False, len(citations), started)
        emit({'type': 'meta', **meta})
        emit({'type': 'citations', 'citations': payload})
        sink: dict = {}
        parts: list[str] = []
        if not self.llm.llm_ready:
            text = self._retrieval_only_answer(citations)
            parts.append(text)
            emit({'type': 'delta', 'text': text})
        else:
            started_generate = time.perf_counter()
            with trace.stage('generate'):
                for delta in self.llm.chat(self._messages(history, question, selected), stream=True, usage_sink=sink):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    parts.append(delta)
                    emit({'type': 'delta', 'text': delta})
            trace.mark('generate_total', (time.perf_counter() - started_generate) * 1000)
        final = ''.join(parts)
        latency_ms = (time.perf_counter() - started) * 1000
        cost = self.usage.record(kind='answer', model=self.s.llm_model, prompt_tokens=sink.get('prompt_tokens', 0), completion_tokens=sink.get('completion_tokens', 0), estimated=bool(sink.get('estimated', False)), latency_ms=latency_ms, first_token_ms=first_token_ms, session_id=session_id, trace_id=trace.id)
        self.chat.add(session_id, 'assistant', final, payload)
        self.cache.put(question=question, answer=final, citations=payload, embedding=question_vector, filters=filters, embedding_signature=self.llm.signature)
        trace.set(cache='miss', refused=False, citation_count=len(citations), recalled=result.recalled, kept=len(result.candidates), rerank_used=result.rerank_used, rerank_mode=result.rerank_mode, prompt_tokens=sink.get('prompt_tokens', 0), completion_tokens=sink.get('completion_tokens', 0), first_token_ms=first_token_ms, cost_yuan=round(cost, 6), note=result.note, top_vector_score=_top_score(result))
        done = self._meta(trace, 'miss', rewritten if changed else None, result.rerank_used, False, len(citations), started, sink, cost, first_token_ms)
        trace.set(workflow='langgraph', graph_nodes=graph_nodes)
        self.tracer.finish(trace)
        emit({'type': 'done', 'answer': final, **done})
        values = locals()
        ctx.update({key: values[key] for key in STATE_FIELDS if key in values})
        return {'ctx': ctx}
    nodes = ['prepare', 'cached', 'rewrite', 'retrieve', 'select', 'refuse', 'generate']
    graph = StateGraph(WorkflowState)
    functions = [s_prepare, s_cached, s_rewrite, s_retrieve, s_select, s_refuse, s_generate] if streaming else [a_prepare, a_cached, a_rewrite, a_retrieve, a_select, a_refuse, a_generate]
    for name, function in zip(nodes, functions):
        graph.add_node(name, function)
    graph.add_edge(START, 'prepare')
    graph.add_conditional_edges('prepare', lambda s: 'cached' if s['ctx'].get('cached') else 'rewrite')
    graph.add_edge('cached', END)
    graph.add_edge('rewrite', 'retrieve')
    graph.add_edge('retrieve', 'select')
    graph.add_conditional_edges('select', lambda s: 'refuse' if s['ctx']['result'].refused or not s['ctx']['selected'] else 'generate')
    graph.add_edge('refuse', END)
    graph.add_edge('generate', END)
    return graph.compile()
