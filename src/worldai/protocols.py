"""WorldAI 模块接口契约（Protocol）。

设计原则（单一职责 + 依赖反转）：
- 每个功能模块只声明一个 Protocol，定义其对外能力；
- 模块之间只依赖 Protocol，不依赖具体实现；
- 具体实现在组合根（``config.build_stack``）处按配置注入；
- 零依赖默认实现与生产实现（Ollama/FAISS/FastEmbed ...）可互相替换，
  且每个模块都能用 fake/默认实现独立单测。

所有 Protocol 标注 ``@runtime_checkable``，便于运行时做接口符合性校验。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from worldai.models import (
    AgentResult,
    Chunk,
    Document,
    Query,
    QueryResult,
    RetrievalHit,
)


@runtime_checkable
class Extractor(Protocol):
    """文本抽取器：把外部文件/字符串变成纯文本。"""

    def extract(self, source: str) -> str:
        """返回抽取出的纯文本。``source`` 可为文件路径或原始文本。"""
        ...


@runtime_checkable
class Chunker(Protocol):
    """语义分块器：把一篇文档切成带标题继承的块。"""

    def chunk(self, text: str, title: str, doc_id: str) -> list[Chunk]:
        """返回分块列表（顺序即文档顺序）。"""
        ...


@runtime_checkable
class Embedder(Protocol):
    """文本嵌入器：把文本映射为定长向量。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入，返回与输入等长的向量列表。"""
        ...

    def dim(self) -> int:
        """返回向量维度。"""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """稠密向量索引：负责精确的近邻召回。"""

    def add(self, ids: list[str], vectors: list[list[float]], payloads: list[Chunk]) -> None:
        ...

    def search(self, vector: list[float], k: int) -> list[RetrievalHit]:
        ...

    def drop(self, doc_id: str) -> None:
        ...

    def count(self) -> int:
        ...


@runtime_checkable
class SparseRetriever(Protocol):
    """稀疏检索器：基于词法/统计的召回（与稠密互补）。"""

    def index(self, chunks: list[Chunk]) -> None:
        ...

    def search(self, query: str, k: int) -> list[RetrievalHit]:
        ...


@runtime_checkable
class Reranker(Protocol):
    """重排器：对初检索结果精排。"""

    def rerank(self, query: str, hits: list[RetrievalHit], k: int) -> list[RetrievalHit]:
        ...


@runtime_checkable
class LLM(Protocol):
    """大语言模型：完成文本生成。"""

    def complete(self, prompt: str) -> str:
        """给定 prompt，返回补全文本。"""
        ...


@runtime_checkable
class Repository(Protocol):
    """文档仓储：持久化文档与分块。"""

    def put(self, doc: Document) -> None:
        ...

    def get(self, doc_id: str) -> Document | None:
        ...

    def all(self) -> list[Document]:
        ...

    def delete(self, doc_id: str) -> None:
        ...


@runtime_checkable
class Tool(Protocol):
    """智能体可调用的工具。"""

    name: str
    description: str

    def run(self, action_input: str) -> str:
        """执行工具，返回观察文本。"""
        ...


@runtime_checkable
class Agent(Protocol):
    """智能体：基于任务驱动多步推理与工具调用。"""

    def run(self, task: str) -> AgentResult:
        ...
