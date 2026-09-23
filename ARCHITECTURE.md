# WorldAI 系统架构与失败模式

> 作者：**晨星**
> 本文档记录 WorldAI 的分层架构、模块契约与**真实踩过的坑**（症状 / 根因 / 修法 / 守护测试）。

---

## 一、分层与依赖方向

```
┌─────────────────────────────────────────────────────────────┐
│  契约层 (config / protocols / models / errors)               │
│  只定义接口(Protocol)与数据模型，零依赖                        │
└─────────────────────────────────────────────────────────────┘
            ▲ 模块只依赖 Protocol，运行时注入实现
┌──────────┴──────────────────────────────────────────────────┐
│  摄入层 → 表征层(嵌入/稠密/稀疏/重排) → 生成层(LLM) → 编排层   │
│  (agent/pipeline) → 服务层(api/repository) → 验证层(eval)     │
└─────────────────────────────────────────────────────────────┘
```

- **依赖反转**：`config.build_stack` 是唯一的组合根，按 `Settings` 实例化全部模块并注入依赖。
- **零依赖默认**：默认全部使用纯标准库实现，确保 `git clone` 后无网络/密钥即可验证全链路。
- **生产切换**：把 `WORLDAI_*` 指向 FastEmbed/FAISS/BM25/llama.cpp/Ollama，装配同一套接口。

---

## 二、关键模块契约（Protocol）

| Protocol | 方法 | 默认实现 | 生产实现 |
|---|---|---|---|
| `Embedder` | `embed(texts)` / `dim()` | `HashEmbedder`（BLAKE2b 有符号哈希 + 中文 bigram，L2 归一） | `FastEmbedEmbedder`（BGE 双语 ONNX） |
| `VectorStore` | `add` / `search` / `drop` / `count` | `MemoryVectorStore`（精确余弦） | `FaissVectorStore`（IndexFlat，归一化后 L2/IP） |
| `SparseRetriever` | `index` / `search` | `LexicalRetriever`（BM25 + Robertson 非否定 IDF） | `BM25Retriever`（rank-bm25） |
| `Reranker` | `rerank(query, hits, k)` | `LexicalReranker`（IDF 余弦） | `CrossEncoderReranker`（fastembed TextMatching） |
| `LLM` | `complete(prompt)` | `MockLLM`（三模式确定性推理） | `OllamaLLM` / `LlamaCppLLM` |
| `Repository` | `put` / `get` / `all` / `delete` | `MemoryRepository` | Postgres（同接口） |
| `Extractor` / `Chunker` | `extract` / `chunk` | `TextExtractor` / `SemanticChunker`（标题继承 + 重叠窗口） | `PdfExtractor`（pypdf） |
| `Tool` / `Agent` | `run` / `run(task)` | `CalculatorTool` / `DateTool` / `RetrieverTool` / `ReActAgent` | — |

---

## 三、数据流（默认链路）

```
ingest(doc) ──► extractor.extract ─► chunker.chunk(标题继承)
              ──► embedder.embed ──► vector_store.add
              ──► sparse.index(全局分块全量重建)

query/agent(task) ──► embedder.embed(query)
              ──► vector_store.search (稠密) + sparse.search (稀疏)
              ──► rrf_fuse (倒数排名融合)
              ──► reranker.rerank (精排)
              ──► [agent] ReAct 循环调用 retriever/calculator/date
              ──► llm.complete (生成) ──► 答案
```

---

## 四、失败模式表（症状 / 根因 / 修法 / 守护测试）

| 症状 | 根因 | 修法 | 守护测试 |
|---|---|---|---|
| MockLLM 输出变成孤立字符（如 `与`），答案错乱 | prompt 指令文本里**字面写了 `<kb-context>`/`<user-question>` 分隔符**，与真实内容块撞名，`_between` 抓到指令里的伪标记段 | 指令中绝不出现分隔符字面量，仅作为真实内容块包裹符；`_between` 取正确块 | `test_mock_llm_rag_context_mode` / `followup` |
| 2 篇文档语料检索得分全 0、排序反转 | `rank_bm25` 的 IDF 在词恰出现于半数文档时为 `ln(1)=0` | 默认词法检索器改用 Robertson 非否定 IDF `ln(1+(N-n+0.5)/(n+0.5))`；BM25 仅留作大语料基准 | `test_lexical_retriever_small_corpus_nonzero` |
| 算术问题答错（如 `12*(3+4)` 算成 `3+4`） | 路由正则不支持括号，且未剥离疑问助词 | 改为「剥离疑问助词 + 纯算术字符校验 + 括号平衡」；支持中文运算符 加减乘除 | `test_safe_eval_and_detect` |
| `create_app()` 造出的 app 全部路由 404 | 路由注册被留给调用方，工厂函数未注册 | 工厂函数**内部完成注册**并返回可直接挂载的 app | `tests/api_e2e_test.py` 全过 |
| PDF/中文经 GBK 落盘后乱码、正则诡异失败 | 源码里写了 BMP 符号字面量（如 ☀-➿） | 全仓 P0 门禁：`scan_emoji` 用纯 `ord()` 码点范围扫描，源文件禁放符号 | `scripts/verify.py` 阶段 1 |
| 答案只含证据首行，丢失约 90% 上下文 | 证据按多行渲染，消费方逐行解析截断 | 证据渲染压成 `@@CHUNK@@` 定界单行；解析时整体切分，并断言 render→parse 往返一致 | `test_evidence_roundtrip` |
| 重复摄入较短文档后，旧内容仍可被检索 | `put_chunks` 是 upsert 不是 replace，旧修订孤儿分块残留 | 摄入前 `drop(doc_id)`；稀疏索引基于全局分块全量重建 | `test_ingest_shorter_doc_drops_orphans` |
| FAISS/llama.cpp 等生产模块 import 即失败，阻断默认验证 | 默认验证环境无重型依赖 | 所有生产实现**惰性加载**，构造不触网/不下载；`build_stack` 默认全零依赖 | `test_build_stack_production_wiring_is_lazy` |
| 指向 `localhost` 的 Ollama 请求被重置（WinError 10054） | httpx 默认 `trust_env=True`，把本机流量送进系统 SOCKS 代理 | 所有指向本机的 httpx 请求一律 `trust_env=False` | `OllamaLLM` 单测构造不触网 |
| 评分卡指标被自家输出拉低 | 引用标记 `(依据：…)` 与证据零 token 重叠，被判无支撑 | 接地判定前剥离 `[id]` / `(来源/依据/source: …)` 标记 | `groundedness_score` 单测 |
| 评估指标被「设计上不检索」用例拉低 | 算术/日期路由用例期望零证据却计为失败 | 这类用例从检索聚合中剔除；仅统计真实检索类 | `scorecard._EVAL_QUERIES` 全部含相关文档 |

---

## 五、复现与验证

```bash
pip install -r requirements.lock.txt   # 或 requirements.txt
PYTHONPATH=src python scripts/verify.py # 八阶段全绿
```

验证链通过 = 系统可在干净环境一键复现并稳定运行的硬证据。

---

作者：晨星
