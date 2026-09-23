"""REST 接口数据契约（pydantic）。

仅用于 API 边界；核心数据模型（见 ``worldai.models``）保持零依赖，二者职责分离。

作者：晨星
"""

from __future__ import annotations

from pydantic import BaseModel


class IngestRequest(BaseModel):
    doc_id: str
    title: str
    source: str = ""
    text: str | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class AgentRequest(BaseModel):
    task: str
    top_k: int | None = None


class HealthModuleInfo(BaseModel):
    embedder: str
    vector_store: str
    sparse: str
    reranker: str | None
    llm: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    modules: HealthModuleInfo


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    num_chunks: int


class HitItem(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    hits: list[HitItem]


class StepItem(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class AgentResponse(BaseModel):
    answer: str
    used_tools: list[str]
    retrieved_chunk_ids: list[str]
    steps: list[StepItem]
