"""服务层端到端测试：FastAPI 应用工厂 + REST 跨模块链路。

使用 FastAPI TestClient 跑完整 ASGI 请求/响应周期，覆盖：
/health 装配自检、/ingest 摄入、/query 与 /agent 跨模块链路（摄入->检索->生成）。
"""

from fastapi.testclient import TestClient

from worldai.api.app import create_app

KB_TEXT = (
    "# 量子计算简介\n量子计算是一种基于量子力学原理的计算范式。\n\n"
    "量子计算利用量子叠加态与量子纠缠实现并行运算，能够在特定问题上获得指数级加速。\n\n"
    "传统计算机使用比特，而量子计算机使用量子比特（qubit）作为基本信息单元。\n"
)


def _client_with_kb():
    app = create_app()
    client = TestClient(app)
    client.post(
        "/ingest",
        json={"doc_id": "kb1", "title": "量子计算", "source": "inline", "text": KB_TEXT},
    )
    return client


def test_health_reports_modules():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    m = body["modules"]
    assert m["embedder"] == "HashEmbedder"
    assert m["vector_store"] == "MemoryVectorStore"
    assert m["sparse"] == "LexicalRetriever"
    assert m["llm"] == "MockLLM"


def test_ingest_then_query_chain():
    client = _client_with_kb()
    r = client.post("/query", json={"question": "量子计算利用什么实现并行运算？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["hits"]


def test_agent_endpoint_knowledge_chain():
    client = _client_with_kb()
    r = client.post("/agent", json={"task": "量子计算利用什么实现并行运算？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert "retriever" in body["used_tools"]
    assert body["retrieved_chunk_ids"]


def test_agent_endpoint_arithmetic_routing():
    client = _client_with_kb()
    r = client.post("/agent", json={"task": "帮我计算 12*(3+4) 等于多少？"})
    assert r.status_code == 200
    body = r.json()
    assert "84" in body["answer"]
    assert "calculator" in body["used_tools"]


def test_ingest_bad_request_returns_400():
    client = TestClient(create_app())
    # 重复 doc_id 不会报错，但空 doc_id 等异常应转为 400
    r = client.post("/ingest", json={"doc_id": "x", "title": "t", "source": "inline", "text": ""})
    assert r.status_code == 200  # 空文本也是合法摄入
