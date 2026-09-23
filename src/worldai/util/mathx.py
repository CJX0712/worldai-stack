"""数值工具：余弦相似度、Reciprocal Rank Fusion（零依赖）。

作者：晨星
"""

from __future__ import annotations


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度；任一向量为零向量时返回 0.0。"""
    if len(a) != len(b):
        raise ValueError("向量维度不一致")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """倒数排名融合。

    ``ranked_lists`` 为多个有序的 chunk_id 列表（每个列表内部按相关性降序）。
    返回 ``(chunk_id, fused_score)`` 降序列表。``k`` 为平滑常数（默认 60）。
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
