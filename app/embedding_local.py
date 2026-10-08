"""无 API Key 也能跑通的本地向量化：哈希词袋 + 中文 bigram。

用途：让你在没配任何 Key 的情况下就能把「上传 → 检索 → 引用」整条链路跑起来，
      方便先看到效果、先录演示视频。

注意：这只是**召回可用**的轻量实现，语义泛化能力明显弱于真实 embedding 模型
      （比如问"多少钱"不一定能命中"价格 199 元/月"）。
      正式使用请把 EMBEDDING_MODE 设为 api，并在 README 里写清你对比过两者 —— 这本身是加分项。
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

_EN_TOKEN = re.compile(r"[a-z0-9_]+")
_CN_CHAR = re.compile(r"[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    text = (text or "").lower()
    han = _CN_CHAR.findall(text)
    tokens = _EN_TOKEN.findall(text) + han
    tokens += ["".join(pair) for pair in zip(han, han[1:])]  # 中文 bigram，缓解单字歧义
    return tokens


def embed_local(text: str, dim: int = 1024) -> list[float]:
    """返回 L2 归一化后的稀疏哈希向量（用列表表示，便于 JSON 持久化）。"""
    vector = [0.0] * dim
    tokens = _tokens(text)
    if not tokens:
        return vector
    for token, count in Counter(tokens).items():
        # 用 crc32 而不是内置 hash()：hash() 每个进程的盐不同，持久化后会失效
        index = zlib.crc32(token.encode("utf-8")) % dim
        vector[index] += 1.0 + math.log(count)  # 次线性 TF，抑制高频词
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
