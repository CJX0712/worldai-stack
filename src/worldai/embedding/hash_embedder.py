"""哈希嵌入器（零依赖默认实现）。

采用 **有符号哈希嵌入（signed hash embedding）** 技巧（HashingVectorizer 同族）：
- 固定维度向量，文本 token 经 BLAKE2b 散列映射到某一维并带正负号累加；
- 中文走字符 bigram、拉丁走整词，二者共享同一散列空间；
- 输出 L2 归一化，使余弦相似度反映词法重叠程度。

这是**真实可用的嵌入**，而非占位 stub；默认链路下检索即依赖它。维度固定、
确定性、无需模型权重，保证 ``git clone`` 后在裸 CPython 上即可复现。

作者：晨星
"""

from __future__ import annotations

import hashlib
import math

from worldai.util.tokens import tokenize

_DEFAULT_DIM = 256


class HashEmbedder:
    """有符号哈希嵌入器。"""

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for tok in tokenize(text):
            idx, sign = self._signed_hash(tok)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    def _signed_hash(self, token: str) -> tuple[int, int]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % self._dim
        sign = 1 if digest[4] % 2 == 0 else -1
        return idx, sign
