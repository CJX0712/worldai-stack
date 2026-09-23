"""Rank-BM25 稀疏检索器（生产实现）。

- 复用 ``rank-bm25``（Okapi BM25 经典实现），适合中大语料；
- 与默认的词法检索器（Robertson 非否定 IDF）互补：生产环境语料充足时，
  rank-bm25 统计更标准；默认实现则保证小语料与离线场景的鲁棒性；
- 构造惰性，索引在 ``index`` 时构建。

作者：晨星
"""

from __future__ import annotations

from worldai.errors import ProviderError
from worldai.models import Chunk, RetrievalHit
from worldai.util.tokens import tokenize


class BM25Retriever:
    """Rank-BM25 检索器（生产实现）。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def index(self, chunks: list[Chunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ProviderError("BM25Retriever 需要 rank-bm25：pip install rank-bm25") from exc
        self._chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[RetrievalHit] = []
        for i in order[:k]:
            if scores[i] > 0.0:
                hits.append(RetrievalHit(chunk=self._chunks[i], score=float(scores[i])))
        return hits
