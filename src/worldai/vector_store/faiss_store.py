"""FAISS 向量索引（生产实现，faiss-cpu）。

- 复用 Meta 出品的 ``faiss``，精确（IndexFlat）近邻，CPU 即可高速检索；
- 模型/索引**惰性加载**，构造不触网；
- FAISS 不支持增量删除，``drop`` 通过「过滤 + 重建索引」实现，保证重复摄入
  不产生孤儿分块。

作者：晨星
"""

from __future__ import annotations

from worldai.errors import ProviderError
from worldai.models import Chunk, RetrievalHit


class FaissVectorStore:
    """FAISS 精确向量索引（生产实现）。"""

    def __init__(self, metric: str = "l2", dim: int | None = None) -> None:
        if metric not in ("l2", "ip"):
            raise ValueError("metric 仅支持 l2 / ip")
        self.metric = metric
        self._dim = dim
        self._faiss = None
        self._index = None
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._payloads: list[Chunk] = []

    def _ensure(self, dim: int) -> None:
        if self._index is None:
            try:
                import faiss
            except ImportError as exc:  # pragma: no cover
                raise ProviderError("FaissVectorStore 需要 faiss-cpu：pip install faiss-cpu") from exc
            self._faiss = faiss
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim) if self.metric == "ip" else faiss.IndexFlatL2(dim)

    def add(self, ids: list[str], vectors: list[list[float]], payloads: list[Chunk]) -> None:
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids / vectors / payloads 长度必须一致")
        dim = len(vectors[0])
        self._ensure(dim)
        import numpy as np

        arr = np.array(vectors, dtype="float32")
        if self.metric != "ip":
            # L2 索引要求向量已归一化以保证「距离小 = 相似度高」语义一致；
            # 这里对输入做行归一化（幂等，对 IP 索引同样无害）
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        self._index.add(arr)  # type: ignore[union-attr]
        self._ids.extend(ids)
        self._vectors.extend(vectors)
        self._payloads.extend(payloads)

    def search(self, vector: list[float], k: int) -> list[RetrievalHit]:
        if self._index is None or self._index.ntotal == 0:  # type: ignore[union-attr]
            return []
        k = min(k, self._index.ntotal)  # type: ignore[union-attr]
        import numpy as np

        q = np.array([vector], dtype="float32")
        if self.metric != "ip":
            n = float(np.linalg.norm(q))
            if n > 0:
                q = q / n
        scores, idx = self._index.search(q, k)  # type: ignore[union-attr]
        hits: list[RetrievalHit] = []
        for s, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            # L2 距离越小越相似 -> 取负；IP 越大越相似
            score = -float(s) if self.metric != "ip" else float(s)
            hits.append(RetrievalHit(chunk=self._payloads[int(i)], score=score))
        return hits

    def drop(self, doc_id: str) -> None:
        keep = [
            (i, v, p)
            for i, v, p in zip(self._ids, self._vectors, self._payloads)
            if p.doc_id != doc_id
        ]
        self._ids = [x[0] for x in keep]
        self._vectors = [x[1] for x in keep]
        self._payloads = [x[2] for x in keep]
        if self._index is not None and self._ids:
            import numpy as np

            dim = len(self._vectors[0])
            self._ensure(dim)
            arr = np.array(self._vectors, dtype="float32")
            if self.metric != "ip":
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms
            self._index.reset()
            self._index.add(arr)
        elif self._index is not None:
            self._index.reset()

    def count(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0
