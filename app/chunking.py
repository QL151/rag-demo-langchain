"""标题边界保留原规则，超长小节改用LangChain递归切分器。"""
from __future__ import annotations
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
MIN_CHUNK_CHARS=8
_WHITESPACE=re.compile(r'[ \t\u3000]+')

def split_text(text:str,chunk_size:int=500,overlap:int=80)->list[str]:
    text=_WHITESPACE.sub(' ',(text or '').replace('\r\n','\n').replace('\r','\n')).strip()
    if not text:return []
    size=max(50,chunk_size)
    overlap=max(0,min(overlap,size//2))
    splitter=RecursiveCharacterTextSplitter(chunk_size=size,chunk_overlap=overlap,
        separators=['\n\n','\n','。','！','？','!','?',' ',''],keep_separator='end')
    # 真正调用LangChain的Document切分入口，而非仅导入类名。
    chunks=splitter.split_documents([Document(page_content=text)])
    return [d.page_content for d in chunks if len(d.page_content)>=MIN_CHUNK_CHARS]

def split_by_heading(text:str,chunk_size:int=500,overlap:int=80)->list[str]:
    # 原项目测试要求标题独立小节也保留，包括不足8字符的短标题。
    parts=re.split(r'(?m)^(?=#{1,6}\s)',text)
    chunks=[]
    for part in parts:
        part=part.strip()
        if not part:continue
        if len(part)<=chunk_size:chunks.append(part)
        else:chunks.extend(split_text(part,chunk_size,overlap))
    return chunks
