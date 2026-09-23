"""WorldAI 核心数据模型。

全部使用标准库 ``dataclasses``，不依赖任何第三方包，确保默认链路在裸 CPython
上即可运行（零依赖可复现）。API 层另有 pydantic schema 用于 REST 契约。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """文档分块。

    ``position`` 为该块在原文档中的字符起始位置，用于标题继承与顺序还原。
    """

    chunk_id: str
    doc_id: str
    text: str
    heading: str = ""
    position: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """一篇被摄入的文档。"""

    doc_id: str
    title: str
    text: str
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class Query:
    """一次检索/问答请求。"""

    text: str
    top_k: int = 6


@dataclass
class RetrievalHit:
    """一条检索命中结果。"""

    chunk: Chunk
    score: float


@dataclass
class AgentStep:
    """ReAct 循环中的单步：思考 -> 动作 -> 观察。"""

    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    """智能体最终输出。"""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    used_tools: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    """摄入结果摘要。"""

    doc_id: str
    title: str
    num_chunks: int


@dataclass
class QueryResult:
    """RAG 问答结果。"""

    answer: str
    hits: list[RetrievalHit] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
