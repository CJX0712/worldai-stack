"""检索与接地评估度量（零依赖）。

头条指标（来自实测教训）：
- ``doc_hit_rate`` / ``doc_mrr``：文档级召回与排序，避免被「整篇文档多分块、块级标注缺失」
  低估真实表现；
- ``block_recall@k``：块级召回，仅作诊断项；
- ``groundedness``：答案与证据的词法接地比例（先剥离引用标记，避免把出处当断言）；
- ``deterministic_view``：剔除延迟等墙钟字段，供确定性断言。

作者：晨星
"""

from __future__ import annotations

import re

from worldai.models import RetrievalHit
from worldai.util.tokens import tokenize


class EvalQuery:
    """单条评估查询及其已知相关文档。"""

    def __init__(self, question: str, relevant_doc_ids: list[str]) -> None:
        self.question = question
        self.relevant_doc_ids = relevant_doc_ids


def _doc_hit(hits: list[RetrievalHit], relevant: list[str]) -> bool:
    return any(h.chunk.doc_id in relevant for h in hits)


def _doc_mrr(hits: list[RetrievalHit], relevant: list[str]) -> float:
    for rank, h in enumerate(hits, 1):
        if h.chunk.doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _block_recall(hits: list[RetrievalHit], relevant: list[str]) -> float:
    relevant_chunks = sum(1 for h in hits if h.chunk.doc_id in relevant)
    if not relevant:
        return 0.0
    return relevant_chunks / len(relevant)


def strip_citations(text: str) -> str:
    """剥离引用/出处标记（``[id]`` 与 ``(来源/依据/source: ...)``）。

    引用是元数据不是断言，接地判定前必须移除，否则会被判为无支撑。
    """
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\((?:来源|依据|source)[:：][^)]*\)", "", text, flags=re.IGNORECASE)
    return text


def groundedness_score(answer: str, evidence_texts: list[str]) -> float:
    """答案词法接地比例：答案 token 落在证据集合中的占比。"""
    ans_tokens = tokenize(strip_citations(answer))
    if not ans_tokens:
        return 0.0
    ev_tokens = set()
    for ev in evidence_texts:
        ev_tokens.update(tokenize(ev))
    if not ev_tokens:
        return 0.0
    supported = sum(1 for t in ans_tokens if t in ev_tokens)
    return supported / len(ans_tokens)


class RetrievalEvaluator:
    """对给定检索函数跑检索评估。"""

    def __init__(self, retrieve_fn) -> None:
        self.retrieve_fn = retrieve_fn

    def evaluate(self, queries: list[EvalQuery], k: int = 6) -> dict:
        per_query: list[dict] = []
        hit_sum = 0.0
        mrr_sum = 0.0
        recall_sum = 0.0
        for q in queries:
            hits = self.retrieve_fn(q.question, k)
            dh = _doc_hit(hits, q.relevant_doc_ids)
            dm = _doc_mrr(hits, q.relevant_doc_ids)
            br = _block_recall(hits, q.relevant_doc_ids)
            hit_sum += float(dh)
            mrr_sum += dm
            recall_sum += br
            per_query.append(
                {
                    "question": q.question,
                    "doc_hit": dh,
                    "doc_mrr": round(dm, 4),
                    "block_recall": round(br, 4),
                }
            )
        n = max(len(queries), 1)
        return {
            "doc_hit_rate": round(hit_sum / n, 4),
            "doc_mrr": round(mrr_sum / n, 4),
            "block_recall": round(recall_sum / n, 4),
            "num_queries": len(queries),
            "per_query": per_query,
        }


def deterministic_view(obj):
    """递归剔除含延迟/时间字段的键，供确定性断言（同一对象不同运行应相等）。"""
    if isinstance(obj, dict):
        return {
            k: deterministic_view(v)
            for k, v in obj.items()
            if not re.search(r"(latency|time|duration|elapsed|cost)", k, re.IGNORECASE)
        }
    if isinstance(obj, list):
        return [deterministic_view(x) for x in obj]
    return obj
