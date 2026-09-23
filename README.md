# WorldAI —— 端到端可运行的 RAG 知识库 + ReAct 工具智能体系统

<p align="center">
  <a href="https://github.com/CJX0712/worldai-stack-4d9b99c2/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/worldai-stack-4d9b99c2/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/worldai-stack-4d9b99c2/releases"><img src="https://img.shields.io/github/v/release/CJX0712/worldai-stack-4d9b99c2?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/worldai-stack-4d9b99c2/blob/master/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/worldai-stack-4d9b99c2" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

> 复用业界领先开源成果、按单一职责 + Protocol 注入划分的模块化 AI 系统。
> 默认零依赖真实实现，**无需 GPU / 模型 / API Key / 网络**即可一键复现并全绿验证。
> 生产实现（FastEmbed / FAISS / Rank-BM25 / llama.cpp / Ollama）经环境变量一键切换。
>
> 作者：**晨星**

---

## 一、系统定位

WorldAI 是一套**完整可运行**的 AI 系统开发环境，覆盖从文档摄入、向量/稀疏检索、重排、大语言模型推理，到 ReAct 工具智能体与 REST 服务网关的端到端链路。设计目标：

- **复用而非从零自研**：嵌入用 FastEmbed（ONNX）、向量用 FAISS、稀疏用 Rank-BM25、本地推理用 llama.cpp、服务用 FastAPI——均为开源翘楚。
- **单一职责 + 依赖反转**：每个模块只声明一个 Protocol 接口，运行时按配置注入实现；模块间只依赖接口，不依赖具体实现。
- **可独立验证、可协同成链**：每个模块都可用零依赖实现（哈希嵌入 / 内存余弦 / IDF 词法 / 确定性 Mock LLM）独立单测，又能组合成完整链路。
- **稳定可复现**：锁版依赖 + `verify.py` 八阶段一键验证，在干净环境中可完全复现。

---

## 二、模块架构（六层 + 契约层）

| 层 | 模块 | 职责 | 默认实现（零依赖） | 生产实现 |
|---|---|---|---|---|
| 0 契约 | `config` / `protocols` / `models` / `errors` | 配置、接口契约、数据模型、异常 | — | — |
| 1 摄入 | `ingestion` | PDF/TXT/MD 抽取 + 语义分块（标题继承） | 纯 Python 抽取 / 语义分块 | pypdf 抽取 |
| 2 表征 | `embedding` / `vector_store` / `sparse` / `rerank` | 嵌入、稠密检索、稀疏检索、重排 | 哈希嵌入 / 内存余弦 / IDF 词法 / IDF 余弦 | FastEmbed / FAISS / BM25 / 交叉编码器 |
| 3 生成 | `llm` | 大模型推理 | 确定性 Mock LLM（真 ReAct） | Ollama / llama.cpp |
| 4 编排 | `agent` / `pipeline` | ReAct 循环 + 工具路由、RAG 链路 | 确定性算术/日期前置路由 | — |
| 5 服务 | `api` / `repository` | REST 网关、文档仓储 | 内存仓储 + FastAPI 工厂 | Postgres（同接口） |
| 6 验证 | `eval` / `scripts` | 检索/接地度量、8 阶段 verify | — | — |

数据流（默认链路）：

```
文档 → [摄入] 抽取+分块+标题继承
     → [嵌入] 哈希向量 → [稠密] 内存余弦
     → [稀疏] IDF 词法 BM25  ──┐
     → [融合] RRF 倒数排名融合 ├→ [重排] IDF 余弦 ─→ 证据
     → [生成] Mock LLM 抽取式回答   （或 ReAct 智能体多步调用 retriever/calculator/date）
     → [服务] REST /health /ingest /query /agent
```

---

## 三、快速开始

### 3.1 安装依赖

```bash
# 方式 A：顶层钉版（仅运行时）
pip install -r requirements.txt

# 方式 B：完全复现（与开发/验证环境一致，含全部传递依赖锁版）
pip install -r requirements.lock.txt

# 方式 C：开发 / 验证（含 pytest，运行 verify.py 与单测所需）
pip install -r requirements-dev.txt
```

> 默认链路（零依赖）只需要标准库；`requirements*.txt` 中的第三方包仅在生产 provider 切换、运行 REST 服务或运行 `verify.py` 单测时需要。
> 运行任何脚本前请先激活含上述依赖的 Python 环境（虚拟环境），否则 `verify.py` 的 pytest 阶段将明确 FAIL（不再假绿）。

### 3.2 一键验证（核心交付物）

```bash
# 需先安装开发依赖（含 pytest）：pip install -r requirements-dev.txt
python scripts/verify.py
```

八阶段：P0 字符门禁 → 逐模块 import → pytest 单测 + API 端到端 → 评分卡门限 → 运行时不变量 → 确定性 → 复现清单。全部通过即 `最终=PASS`，并输出 `scripts/verify_report.json`。

### 3.3 运行 REST 服务

```bash
# 默认零依赖配置
python scripts/serve.py --port 8000

# 或借助 uvicorn 工厂模式
uvicorn worldai.api.app:create_app --factory --port 8000
```

然后：

```bash
curl -X POST http://localhost:8000/ingest -H 'Content-Type: application/json' \
  -d '{"doc_id":"kb1","title":"量子计算","source":"inline","text":"量子计算利用量子叠加态实现并行运算。"}'

curl -X POST http://localhost:8000/agent -H 'Content-Type: application/json' \
  -d '{"task":"量子计算利用什么实现并行运算？"}'

curl -X POST http://localhost:8000/agent -H 'Content-Type: application/json' \
  -d '{"task":"帮我计算 12*(3+4) 等于多少？"}'
```

---

## 四、切换生产 Provider（环境变量）

WorldAI 的所有能力均可经 `WORLDAI_*` 环境变量切换到业界领先开源实现，无需改代码：

| 变量 | 可选值 | 默认 | 说明 |
|---|---|---|---|
| `WORLDAI_EMBEDDER` | `hash` / `fastembed` | `hash` | 哈希嵌入 / FastEmbed（BGE 双语） |
| `WORLDAI_VECTOR_STORE` | `memory` / `faiss` | `memory` | 内存余弦 / FAISS 精确检索 |
| `WORLDAI_SPARSE` | `lexical` / `bm25` | `lexical` | IDF 词法 / Rank-BM25 |
| `WORLDAI_RERANKER` | `lexical` / `cross_encoder` / `none` | `lexical` | IDF 余弦 / 交叉编码器 |
| `WORLDAI_LLM` | `mock` / `ollama` / `llamacpp` | `mock` | Mock / Ollama / 本地 GGUF |
| `WORLDAI_FASTEMBED_MODEL` | 模型名 | `BAAI/bge-small-zh-v1.5` | FastEmbed 模型 |
| `WORLDAI_OLLAMA_URL` / `WORLDAI_OLLAMA_MODEL` | URL / 模型 | `:11434` / `qwen2.5:0.5b` | Ollama 接入 |
| `WORLDAI_LLAMACPP_MODEL_PATH` | 本地 .gguf 路径 | 空 | llama.cpp 本地推理 |

示例（启用全部生产实现，需已装对应模型/服务）：

```bash
export WORLDAI_EMBEDDER=fastembed
export WORLDAI_VECTOR_STORE=faiss
export WORLDAI_SPARSE=bm25
export WORLDAI_RERANKER=cross_encoder
export WORLDAI_LLM=ollama
python scripts/serve.py
```

---

## 五、测试

```bash
# 需先安装开发依赖：pip install -r requirements-dev.txt
PYTHONPATH=src python -m pytest tests/ -q
```

覆盖：摄入（抽取+分块+标题继承）、表征（哈希嵌入/内存向量/词法检索/词法重排）、生产 Provider 装配与离线检索（FAISS/BM25）、LLM（Mock 三模式）、编排（ReAct 收敛/算术与日期路由/RAG 接地）、服务层（FastAPI 端到端跨模块链路）。

---

## 六、交付物清单

- `src/worldai/`：完整可运行源代码（按层 + Protocol 注入划分）
- `requirements.txt` / `requirements.lock.txt`：钉版 + 锁版依赖，干净环境一键复现
- `README.md` / `ARCHITECTURE.md` / `docs/SPEC.md` / `docs/openapi.yaml` / `docs/decisions/`：架构 / 部署 / 使用文档
- `scripts/verify.py`：八阶段一键验证链
- `scripts/serve.py`：REST 服务启动入口
- `tools/scan_emoji.py`：P0 源码字符门禁
- `LICENSE`：MIT（作者 晨星）

---

## 七、设计原则（来自多轮实测）

1. **默认实现要是真的实现，不能是 stub**：每个外部端口都配零依赖真实实现（抽取式回答 / 哈希嵌入 / 精确余弦 / IDF 词法 / 确定性计算器），默认配置即被持续验证的配置。
2. **确定性前置路由**：算术表达式与日期类问题在调用 LLM 前由规则路由到计算器/日期工具，禁止小模型重算。
3. **重复摄入先 drop**：向量库按 doc_id 删除旧分块，稀疏索引基于全局分块全量重建，杜绝孤儿分块被检索。
4. **Prompt 分隔符全局唯一且不在指令中字面出现**：避免与内容块撞名导致抽取错位（已踩坑）。
5. **评估始终在全新管道上跑**：不读任何 `last_*` 缓存，保证确定性与无偏。

详见 `ARCHITECTURE.md` 的「失败模式表」。

---

作者：晨星
