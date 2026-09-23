"""WorldAI 配置与组合根（Composition Root）。

- :class:`Settings` 从环境变量读取 provider 选择与各模块参数；
- :func:`build_stack` 按配置实例化全部模块并注入依赖，组装成可运行链路；
- 默认全部使用零依赖真实实现，``git clone`` 后无需网络/密钥即可 ``verify`` 全绿；
- 把对应 ``WORLDAI_*`` 环境变量切到生产实现（FastEmbed/FAISS/BM25/llama.cpp/Ollama），
  即可在具备模型与算力的环境中获得业界领先性能。

作者：晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from worldai.errors import ProviderError


@dataclass
class Settings:
    """系统配置。所有字段均有合理默认值，纯环境变量驱动。"""

    # ---- provider 选择（决定用默认实现还是生产实现）----
    embedder: str = "hash"          # hash | fastembed
    vector_store: str = "memory"     # memory | faiss
    sparse: str = "lexical"          # lexical | bm25
    reranker: str = "lexical"        # lexical | cross_encoder | none
    llm: str = "mock"                # mock | ollama | llamacpp

    # ---- 通用参数 ----
    top_k: int = 6
    rerank_k: int = 4
    chunk_size: int = 480
    chunk_overlap: int = 80
    max_agent_steps: int = 6

    # ---- 生产 provider 参数 ----
    fastembed_model: str = "BAAI/bge-small-zh-v1.5"
    faiss_metric: str = "l2"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:0.5b"
    llamacpp_model_path: str = ""
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"

    @classmethod
    def from_env(cls) -> "Settings":
        """从 ``WORLDAI_*`` 环境变量构造配置。"""
        def env(name: str, default):
            raw = os.environ.get(name)
            if raw is None or raw == "":
                return default
            if isinstance(default, bool):
                return raw.lower() in ("1", "true", "yes", "on")
            if isinstance(default, int):
                try:
                    return int(raw)
                except ValueError:
                    return default
            return raw

        return cls(
            embedder=env("WORLDAI_EMBEDDER", cls.embedder),
            vector_store=env("WORLDAI_VECTOR_STORE", cls.vector_store),
            sparse=env("WORLDAI_SPARSE", cls.sparse),
            reranker=env("WORLDAI_RERANKER", cls.reranker),
            llm=env("WORLDAI_LLM", cls.llm),
            top_k=env("WORLDAI_TOP_K", cls.top_k),
            rerank_k=env("WORLDAI_RERANK_K", cls.rerank_k),
            chunk_size=env("WORLDAI_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=env("WORLDAI_CHUNK_OVERLAP", cls.chunk_overlap),
            max_agent_steps=env("WORLDAI_MAX_AGENT_STEPS", cls.max_agent_steps),
            fastembed_model=env("WORLDAI_FASTEMBED_MODEL", cls.fastembed_model),
            faiss_metric=env("WORLDAI_FAISS_METRIC", cls.faiss_metric),
            ollama_url=env("WORLDAI_OLLAMA_URL", cls.ollama_url),
            ollama_model=env("WORLDAI_OLLAMA_MODEL", cls.ollama_model),
            llamacpp_model_path=env("WORLDAI_LLAMACPP_MODEL_PATH", cls.llamacpp_model_path),
            cross_encoder_model=env("WORLDAI_CROSS_ENCODER_MODEL", cls.cross_encoder_model),
        )


def _require(module_path: str, class_name: str, dep_name: str):
    """惰性导入一个生产实现；缺失依赖时给出清晰报错。"""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise ProviderError(
            f"provider {class_name} 需要依赖 {dep_name}：{exc}。"
            f"请先 `pip install {dep_name}` 或改回默认 provider。"
        ) from exc


def build_stack(settings: Settings | None = None) -> dict:
    """组合根：按配置实例化并注入全部模块，返回组件字典。

    返回键：repository, extractor, chunker, embedder, vector_store,
    sparse, reranker, llm, pipeline, agent, settings。
    """
    settings = settings or Settings.from_env()

    # ---- 仓储 / 摄入（始终零依赖）----
    from worldai.repository.memory_repo import MemoryRepository
    from worldai.ingestion.extractor import TextExtractor
    from worldai.ingestion.chunker import SemanticChunker

    repository = MemoryRepository()
    extractor = TextExtractor()
    chunker = SemanticChunker(chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)

    # ---- 嵌入 ----
    if settings.embedder == "fastembed":
        cls = _require("worldai.embedding.fastembed_embedder", "FastEmbedEmbedder", "fastembed")
        embedder = cls(model_name=settings.fastembed_model)
    else:
        from worldai.embedding.hash_embedder import HashEmbedder
        embedder = HashEmbedder()

    # ---- 稠密向量库 ----
    if settings.vector_store == "faiss":
        cls = _require("worldai.vector_store.faiss_store", "FaissVectorStore", "faiss-cpu")
        vector_store = cls(metric=settings.faiss_metric)
    else:
        from worldai.vector_store.memory_store import MemoryVectorStore
        vector_store = MemoryVectorStore()

    # ---- 稀疏检索 ----
    if settings.sparse == "bm25":
        cls = _require("worldai.sparse.bm25", "BM25Retriever", "rank-bm25")
        sparse = cls()
    else:
        from worldai.sparse.lexical import LexicalRetriever
        sparse = LexicalRetriever()

    # ---- 重排 ----
    if settings.reranker == "cross_encoder":
        cls = _require("worldai.rerank.cross_encoder", "CrossEncoderReranker", "fastembed")
        reranker = cls(model_name=settings.cross_encoder_model)
    elif settings.reranker == "none":
        reranker = None
    else:
        from worldai.rerank.lexical_reranker import LexicalReranker
        reranker = LexicalReranker()

    # ---- 大模型 ----
    if settings.llm == "ollama":
        cls = _require("worldai.llm.ollama_llm", "OllamaLLM", "httpx")
        llm = cls(base_url=settings.ollama_url, model=settings.ollama_model)
    elif settings.llm == "llamacpp":
        cls = _require("worldai.llm.llama_cpp_llm", "LlamaCppLLM", "llama_cpp")
        llm = cls(model_path=settings.llamacpp_model_path)
    else:
        from worldai.llm.mock_llm import MockLLM
        llm = MockLLM()

    # ---- RAG 管线（组合检索各组件）----
    from worldai.pipeline.rag import RagPipeline
    pipeline = RagPipeline(
        extractor=extractor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        sparse=sparse,
        reranker=reranker,
        repository=repository,
        settings=settings,
        llm=llm,
    )

    # ---- 工具 + 智能体 ----
    from worldai.agent.tools import CalculatorTool, DateTool, RetrieverTool
    from worldai.agent.react import ReActAgent

    tools = [
        RetrieverTool(pipeline=pipeline, top_k=settings.top_k),
        CalculatorTool(),
        DateTool(),
    ]
    agent = ReActAgent(llm=llm, tools=tools, settings=settings)

    return {
        "settings": settings,
        "repository": repository,
        "extractor": extractor,
        "chunker": chunker,
        "embedder": embedder,
        "vector_store": vector_store,
        "sparse": sparse,
        "reranker": reranker,
        "llm": llm,
        "pipeline": pipeline,
        "agent": agent,
    }


def build_pipeline(settings: Settings | None = None) -> "object":
    """便捷入口：直接返回一个装配好的 RAG 管线。"""
    return build_stack(settings)["pipeline"]
