"""FastAPI 应用工厂（服务层入口）。

- :func:`create_app` 在**内部完成路由注册**并返回可直接挂载/启动的 app（避免工厂函数
  返回空壳 app 导致全部 404 的坑）；
- 组合根（``build_stack``）在启动时构建一次，挂载于 ``app.state``，路由直接消费；
- 默认配置即零依赖真实实现，无需模型/密钥即可启动与问答。

作者：晨星
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from worldai.api.schemas import (
    AgentRequest,
    AgentResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    StepItem,
)
from worldai.config import Settings, build_stack


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    stack = build_stack(settings)

    app = FastAPI(title="WorldAI", version="1.0.0")
    app.state.stack = stack
    app.state.settings = settings

    @app.get("/health", response_model=HealthResponse, summary="健康检查")
    def health():  # noqa: ANN202
        s = stack
        return HealthResponse(
            modules={
                "embedder": type(s["embedder"]).__name__,
                "vector_store": type(s["vector_store"]).__name__,
                "sparse": type(s["sparse"]).__name__,
                "reranker": type(s["reranker"]).__name__ if s["reranker"] else None,
                "llm": type(s["llm"]).__name__,
            }
        )

    @app.post("/ingest", response_model=IngestResponse, summary="摄入文档")
    def ingest(req: IngestRequest):  # noqa: ANN201
        try:
            res = stack["pipeline"].ingest(req.doc_id, req.title, req.source, text=req.text)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc))
        return IngestResponse(doc_id=res.doc_id, title=res.title, num_chunks=res.num_chunks)

    @app.post("/query", response_model=QueryResponse, summary="RAG 直接问答")
    def query(req: QueryRequest):  # noqa: ANN201
        result = stack["pipeline"].query(req.question, req.top_k)
        return QueryResponse(
            answer=result.answer,
            hits=[
                {"doc_id": h.chunk.doc_id, "chunk_id": h.chunk.chunk_id, "score": round(h.score, 4), "text": h.chunk.text[:200]}
                for h in result.hits
            ],
        )

    @app.post("/agent", response_model=AgentResponse, summary="ReAct 智能体问答")
    def agent(req: AgentRequest):  # noqa: ANN201
        result = stack["agent"].run(req.task)
        return AgentResponse(
            answer=result.answer,
            used_tools=result.used_tools,
            retrieved_chunk_ids=result.retrieved_chunk_ids,
            steps=[StepItem(thought=s.thought, action=s.action, action_input=s.action_input, observation=s.observation) for s in result.steps],
        )

    return app
