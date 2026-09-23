"""交叉编码器重排器（生产实现，FastEmbed TextMatching）。

- 复用 ``fastembed`` 的 ``TextMatching``（cross-encoder 重排，ONNX）；
- 相比默认的词法重排，交叉编码器对「查询-文档」做联合编码，精度更高，
  尤其擅长区分语义相近的候选；
- 模型惰性加载，构造不下载。

作者：晨星
"""

from __future__ import annotations

from worldai.errors import ProviderError
from worldai.models import RetrievalHit


class CrossEncoderReranker:
    """交叉编码器重排器（生产实现）。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", cache_dir: str | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            try:
                from fastembed import TextMatching
            except ImportError as exc:  # pragma: no cover
                raise ProviderError(
                    "CrossEncoderReranker 需要 fastembed：pip install fastembed"
                ) from exc
            self._model = TextMatching(model_name=self.model_name, cache_dir=self.cache_dir)

    def rerank(self, query: str, hits: list[RetrievalHit], k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        self._ensure()
        pairs = [(query, h.chunk.text) for h in hits]
        raw = self._model.predict(pairs)  # type: ignore[union-attr]
        scores = [float(x) for x in raw]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievalHit] = []
        for i in order[:k]:
            out.append(RetrievalHit(chunk=hits[i].chunk, score=scores[i]))
        return out
