"""WorldAI —— 端到端可运行的 RAG 知识库 + ReAct 工具智能体系统。

模块按单一职责 + Protocol 注入划分，默认零依赖真实实现保证离线可验证，
生产实现（FastEmbed / FAISS / Rank-BM25 / llama.cpp / Ollama）经环境变量切换。

作者：晨星
"""

__version__ = "1.0.0"
__author__ = "晨星"

from worldai.config import Settings, build_pipeline, build_stack
from worldai.models import (
    AgentResult,
    Chunk,
    Document,
    IngestResult,
    Query,
    QueryResult,
    RetrievalHit,
)

__all__ = [
    "Settings",
    "build_pipeline",
    "build_stack",
    "Chunk",
    "Document",
    "Query",
    "RetrievalHit",
    "AgentResult",
    "IngestResult",
    "QueryResult",
]
