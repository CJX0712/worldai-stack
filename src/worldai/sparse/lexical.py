"""词法稀疏检索器（零依赖默认实现）。

采用 **BM25 公式 + Robertson 非否定 IDF**：

    idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))   # 恒 >= 0

动机（实测坑）：``rank_bm25`` 的 IDF 在词恰好出现于半数文档时等于 ``ln(1)=0``，
会令小语料检索得分全 0、排序反转；Robertson 形式数学上保证非负，且对长尾词更稳健。
本实现即为默认链路的稀疏召回，与稠密检索做 RRF 互补融合。

作者：晨星
"""

from __future__ import annotations

import math

from worldai.models import Chunk, RetrievalHit
from worldai.util.tokens import term_frequencies


class LexicalRetriever:
    """BM25 词法检索器（零依赖）。"""

    def __init__(self, k: float = 1.2, b: float = 0.75) -> None:
        self.k = k
        self.b = b
        self._chunks: list[Chunk] = []
        self._tf: list[dict[str, int]] = []
        self._dl: list[int] = []
        self._df: dict[str, int] = {}
        self._n = 0
        self._avgdl = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tf = []
        self._dl = []
        self._df = {}
        for c in self._chunks:
            tf = term_frequencies(c.text)
            self._tf.append(tf)
            self._dl.append(sum(tf.values()))
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        self._n = len(self._chunks)
        self._avgdl = (sum(self._dl) / self._n) if self._n else 0.0

    def _idf(self, term: str) -> float:
        n_t = self._df.get(term, 0)
        if n_t == 0:
            return 0.0
        return math.log(1.0 + (self._n - n_t + 0.5) / (n_t + 0.5))

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        qtf = term_frequencies(query)
        if not qtf or self._n == 0:
            return []
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            dl = self._dl[i]
            norm = 1.0 - self.b + self.b * (dl / self._avgdl) if self._avgdl > 0 else 1.0
            score = 0.0
            for t, qc in qtf.items():
                if t in tf:
                    idf = self._idf(t)
                    score += idf * (tf[t] * (self.k + 1)) / (tf[t] + self.k * norm)
            scored.append((score, i))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, i in scored[:k]:
            if score > 0.0:
                hits.append(RetrievalHit(chunk=self._chunks[i], score=score))
        return hits
