# WorldAI 接口规范（SPEC）

> 作者：**晨星**

---

## 一、REST API

基地址：`http://<host>:<port>`，由 `scripts/serve.py` 或 `uvicorn worldai.api.app:create_app --factory` 启动。

### 1. `GET /health`
健康检查 + 模块装配自检。

响应（200）：
```json
{
  "status": "ok",
  "version": "1.0.0",
  "modules": {
    "embedder": "HashEmbedder",
    "vector_store": "MemoryVectorStore",
    "sparse": "LexicalRetriever",
    "reranker": "LexicalReranker",
    "llm": "MockLLM"
  }
}
```

### 2. `POST /ingest`
摄入文档。

请求体：
```json
{ "doc_id": "kb1", "title": "量子计算", "source": "inline", "text": "..." }
```
- `source` 为 `"inline"` 时取 `text`；为文件路径时由 `TextExtractor` 抽取（PDF/TXT/MD）。

响应（200）：
```json
{ "doc_id": "kb1", "title": "量子计算", "num_chunks": 3 }
```

### 3. `POST /query`
RAG 直接问答（不使用 ReAct 智能体）。

请求体：`{ "question": "...", "top_k": 6 }`
响应（200）：
```json
{ "answer": "...", "hits": [ { "doc_id": "kb1", "chunk_id": "kb1::c0", "score": 0.83, "text": "..." } ] }
```

### 4. `POST /agent`
ReAct 工具智能体问答（支持检索 / 算术 / 日期前置路由）。

请求体：`{ "task": "...", "top_k": 6 }`
响应（200）：
```json
{
  "answer": "...",
  "used_tools": ["retriever"],
  "retrieved_chunk_ids": ["kb1::c0"],
  "steps": [ { "thought": "...", "action": "retriever", "action_input": "...", "observation": "..." } ]
}
```

> 完整 OpenAPI 文档由 `create_app().openapi()` 生成，见 `docs/openapi.yaml`。

---

## 二、配置契约（`worldai.config.Settings`）

所有字段均有合理默认，全部经 `WORLDAI_*` 环境变量驱动（详见 `README.md` 第四节）。

| 字段 | 默认 | 说明 |
|---|---|---|
| `embedder` | `hash` | 嵌入实现选择 |
| `vector_store` | `memory` | 稠密向量库选择 |
| `sparse` | `lexical` | 稀疏检索选择 |
| `reranker` | `lexical` | 重排器选择（`none` 可关闭） |
| `llm` | `mock` | 大模型选择 |
| `top_k` / `rerank_k` | `6` / `4` | 检索 / 重排返回数 |
| `chunk_size` / `chunk_overlap` | `480` / `80` | 分块窗口 / 重叠 |
| `max_agent_steps` | `6` | ReAct 最大步数 |

---

## 三、模块接口（Protocol，摘要）

- `Embedder.embed(texts) -> list[list[float]]`；`Embedder.dim() -> int`
- `VectorStore.add(ids, vectors, payloads)`；`search(vector, k)`；`drop(doc_id)`；`count()`
- `SparseRetriever.index(chunks)`；`search(query, k)`
- `Reranker.rerank(query, hits, k)`
- `LLM.complete(prompt) -> str`
- `Repository.put/get/all/delete`
- `Extractor.extract(source)`；`Chunker.chunk(text, title, doc_id)`
- `Tool.run(action_input) -> str`；`Agent.run(task) -> AgentResult`

每个 Protocol 均以 `@runtime_checkable` 声明，组合根据此注入实现，各模块可独立单测。

---

作者：晨星
