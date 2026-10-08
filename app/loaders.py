"""文件读取：txt / md / csv / json 直接读，pdf 用 pypdf。"""

from __future__ import annotations

from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".log", ".py", ".html", ".yml", ".yaml"}


class UnsupportedFile(Exception):
    """文件类型不支持或读取失败。"""


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    # 其余类型都按文本尝试；真正的二进制会被 _looks_binary 拦下来
    return _read_plain(path)


def _read_plain(path: Path) -> str:
    data = path.read_bytes()
    if _looks_binary(data):
        raise UnsupportedFile(
            f"《{path.name}》看起来不是文本文件，无法解析。"
            f"支持的类型：{', '.join(sorted(TEXT_SUFFIXES))} 以及文字版 PDF。"
        )
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            from langchain_community.document_loaders import TextLoader
            documents=TextLoader(str(path),encoding=encoding).load()
            return documents[0].page_content if documents else ''
        except RuntimeError as exc:
            if isinstance(exc.__cause__,UnicodeDecodeError):
                continue
            raise UnsupportedFile(f'LangChain文本读取失败：{path.name}') from exc
    return data.decode("utf-8", errors="ignore")


def _looks_binary(data: bytes) -> bool:
    """粗判二进制：出现 NUL 字节，或控制字符占比过高（图片/压缩包/可执行文件）。"""
    if not data:
        return False
    sample = data[:4096]
    if b"\x00" in sample:
        return True
    control = sum(1 for byte in sample if byte < 9 or 13 < byte < 32)
    return control / len(sample) > 0.10


def _read_pdf(path: Path) -> str:
    from langchain_community.document_loaders import PyPDFLoader
    documents=PyPDFLoader(str(path),mode='page').load()
    # 保留原接口的纯文本返回类型，页间仍用两个换行连接。
    return '\n\n'.join(d.page_content.strip() for d in documents if d.page_content.strip())
