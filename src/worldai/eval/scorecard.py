"""评分卡：内置评估数据集 + 门限判定 + JSON 报告。

- 评估**始终在全新默认管道**上跑，保证确定性，不复用运行期单例；
- 头条指标 ``doc_hit_rate`` / ``doc_mrr`` 贴着实测基线设门限，让真回归必挂、数值抖动不挂；
- 评估适配器显式捕获本次检索结果，不读任何 ``last_*`` 缓存字段。

作者：晨星
"""

from __future__ import annotations

from worldai.config import Settings, build_pipeline
from worldai.eval.metrics import EvalQuery, RetrievalEvaluator, groundedness_score

# 内置评估语料（>=3 篇，避免小语料稀疏检索退化为全 0；段落长度刻意非均匀）
_EVAL_DOCS = [
    (
        "qc",
        "量子计算",
        "# 量子计算\n量子计算是一种基于量子力学原理的计算范式。\n\n"
        "量子计算利用量子叠加态与量子纠缠实现并行运算，能够在特定问题上获得指数级加速。\n\n"
        + "段落补充%d：量子比特是量子计算机的基本信息单元，可同时处于多种状态的叠加。" % 0
        + " 量子算法如 Shor 算法可用于大整数分解，Grover 算法可加速无序搜索。\n",
    ),
    (
        "cm",
        "古典音乐",
        "# 古典音乐\n古典音乐强调和声与旋律的平衡，重视结构与织体。\n\n"
        "交响乐团由弦乐、木管、铜管与打击乐四个声部组成，指挥负责统一诠释。\n\n"
        + "段落补充%d：巴洛克时期的复调音乐以巴赫的赋格为代表，古典时期则转向主调清晰的结构。" % 0
        + " 浪漫主义扩展了和声与表情范围。\n",
    ),
    (
        "cc",
        "气候变暖",
        "# 气候变暖\n气候变暖指全球平均气温的长期上升趋势。\n\n"
        "化石燃料燃烧释放的二氧化碳等温室气体被认为是气候变暖的主要原因。\n\n"
        + "段落补充%d：冰川消融与海平面上升是气候变暖的直接后果，影响沿海城市安全。" % 0
        + " 减少碳排放需要能源结构转型与国际合作。\n",
    ),
]

_EVAL_QUERIES = [
    EvalQuery("量子计算利用什么实现并行运算？", ["qc"]),
    EvalQuery("古典音乐强调什么的平衡？", ["cm"]),
    EvalQuery("气候变暖的主要原因是什么？", ["cc"]),
]

_THRESHOLDS = {
    "doc_hit_rate": 0.95,
    "doc_mrr": 0.50,
    "groundedness": 0.50,
}


def _build_eval_pipeline():
    # 全新默认管道，保证评估确定性、不复用单例
    pipeline = build_pipeline(Settings())
    for doc_id, title, text in _EVAL_DOCS:
        pipeline.ingest(doc_id, title, "inline", text=text)
    return pipeline


def run(k: int = 6) -> dict:
    pipeline = _build_eval_pipeline()
    evaluator = RetrievalEvaluator(retrieve_fn=lambda q, kk=k: pipeline.retrieve(q, kk))
    metrics = evaluator.evaluate(_EVAL_QUERIES, k=k)

    # 接地度量（取首个查询的答案与证据）
    sample = pipeline.query(_EVAL_QUERIES[0].question, k)
    evidence_texts = [h.chunk.text for h in sample.hits]
    groundedness = groundedness_score(sample.answer, evidence_texts)

    # 确定性：同管道再评估一次，结果应一致
    metrics_again = evaluator.evaluate(_EVAL_QUERIES, k=k)
    deterministic = metrics == metrics_again

    passed = (
        metrics["doc_hit_rate"] >= _THRESHOLDS["doc_hit_rate"]
        and metrics["doc_mrr"] >= _THRESHOLDS["doc_mrr"]
        and groundedness >= _THRESHOLDS["groundedness"]
        and deterministic
    )

    return {
        "metrics": metrics,
        "groundedness": round(groundedness, 4),
        "deterministic": deterministic,
        "thresholds": _THRESHOLDS,
        "passed": passed,
    }
