"""内存向量索引（零依赖默认实现）。

- 精确余弦相似度（非近似），保证默认链路检索结果确定可复现；
- 以 Chunk 为负载（payload），检索命中直接返回带分块的 :class:`RetrievalHit`；
- ``drop`` 按 doc_id 删除，避免重复摄入产生孤儿分块。

作者：晨星
"""

from __future__ import annotations

from worldai.models import Chunk, RetrievalHit
from worldai.util.mathx import cosine


class MemoryVectorStore:
    """进程内精确余弦向量库。"""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: list[list[float]] = []
        self._payloads: list[Chunk] = []

    def add(self, ids: list[str], vectors: list[list[float]], payloads: list[Chunk]) -> None:
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError("ids / vectors / payloads 长度必须一致")
        self._ids.extend(ids)
        self._vectors.extend(vectors)
        self._payloads.extend(payloads)

    def search(self, vector: list[float], k: int) -> list[RetrievalHit]:
        if k <= 0:
            return []
        scored: list[tuple[float, int]] = []
        for i, v in enumerate(self._vectors):
            scored.append((cosine(vector, v), i))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        hits: list[RetrievalHit] = []
        for score, i in scored[:k]:
            hits.append(RetrievalHit(chunk=self._payloads[i], score=score))
        return hits

    def drop(self, doc_id: str) -> None:
        keep_ids, keep_vecs, keep_pl = [], [], []
        for cid, cv, cp in zip(self._ids, self._vectors, self._payloads):
            if cp.doc_id != doc_id:
                keep_ids.append(cid)
                keep_vecs.append(cv)
                keep_pl.append(cp)
        self._ids, self._vectors, self._payloads = keep_ids, keep_vecs, keep_pl

    def count(self) -> int:
        return len(self._ids)
