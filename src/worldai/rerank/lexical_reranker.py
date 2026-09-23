"""词法重排器（零依赖默认实现）。

对初检索返回的候选集，以**候选集自身为语料**重新计算 BM25 词法得分并精排。
相比初检的全局索引，重排聚焦少量候选、对查询词权重更敏感，可提升首位命中率。

作者：晨星
"""

from __future__ import annotations

import math

from worldai.models import RetrievalHit
from worldai.util.tokens import term_frequencies


class LexicalReranker:
    """BM25 词法重排器（零依赖）。"""

    def __init__(self, k: float = 1.2, b: float = 0.75) -> None:
        self.k = k
        self.b = b

    def rerank(self, query: str, hits: list[RetrievalHit], k: int) -> list[RetrievalHit]:
        if not hits:
            return []
        cands = [h.chunk for h in hits]
        tf_list: list[dict[str, int]] = []
        dl_list: list[int] = []
        df: dict[str, int] = {}
        for c in cands:
            tf = term_frequencies(c.text)
            tf_list.append(tf)
            dl_list.append(sum(tf.values()))
            for t in tf:
                df[t] = df.get(t, 0) + 1
        n = len(cands)
        avgdl = (sum(dl_list) / n) if n else 0.0
        qtf = term_frequencies(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(tf_list):
            dl = dl_list[i]
            norm = 1.0 - self.b + self.b * (dl / avgdl) if avgdl > 0 else 1.0
            score = 0.0
            for t, _ in qtf.items():
                n_t = df.get(t, 0)
                if n_t > 0 and t in tf:
                    idf = math.log(1.0 + (n - n_t + 0.5) / (n_t + 0.5))
                    score += idf * (tf[t] * (self.k + 1)) / (tf[t] + self.k * norm)
            scored.append((score, i))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        out: list[RetrievalHit] = []
        for score, i in scored[:k]:
            if score > 0.0:
                out.append(RetrievalHit(chunk=cands[i], score=score))
        return out
