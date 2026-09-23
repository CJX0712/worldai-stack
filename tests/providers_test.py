"""生产 Provider 测试：构造装配（惰性、零网络）+ FAISS/BM25 真实离线检索。"""

import numpy as np

from worldai.config import Settings, build_stack
from worldai.embedding.fastembed_embedder import FastEmbedEmbedder
from worldai.llm.llama_cpp_llm import LlamaCppLLM
from worldai.llm.ollama_llm import OllamaLLM
from worldai.models import Chunk
from worldai.protocols import Embedder, LLM, Reranker, SparseRetriever, VectorStore
from worldai.rerank.cross_encoder import CrossEncoderReranker
from worldai.sparse.bm25 import BM25Retriever
from worldai.vector_store.faiss_store import FaissVectorStore


def test_build_stack_production_wiring_is_lazy():
    # 全部切到生产 provider，验证装配不触发模型下载/网络
    s = Settings(
        embedder="fastembed",
        vector_store="faiss",
        sparse="bm25",
        reranker="cross_encoder",
        llm="ollama",
    )
    stack = build_stack(s)
    assert isinstance(stack["embedder"], Embedder)
    assert isinstance(stack["vector_store"], VectorStore)
    assert isinstance(stack["sparse"], SparseRetriever)
    assert isinstance(stack["reranker"], Reranker)
    assert isinstance(stack["llm"], LLM)
    assert stack["pipeline"] is not None
    assert stack["agent"] is not None


def test_fastembed_construct_lazy():
    emb = FastEmbedEmbedder(model_name="BAAI/bge-small-zh-v1.5")
    assert hasattr(emb, "embed") and hasattr(emb, "dim")


def test_cross_encoder_construct_lazy():
    rer = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")
    assert hasattr(rer, "rerank")


def test_ollama_construct_no_network():
    llm = OllamaLLM(base_url="http://localhost:11434", model="qwen2.5:0.5b")
    assert hasattr(llm, "complete")


def test_llamacpp_construct_no_load():
    llm = LlamaCppLLM(model_path="/tmp/absent.gguf")
    assert hasattr(llm, "complete")


def test_faiss_store_real_search_and_drop():
    rng = np.random.default_rng(42)
    vecs = [list(rng.standard_normal(16)) for _ in range(3)]
    store = FaissVectorStore(metric="l2")
    chunks = [Chunk(chunk_id=f"d{i}::c0", doc_id=f"d{i}", text=str(i)) for i in range(3)]
    store.add([c.chunk_id for c in chunks], vecs, chunks)
    assert store.count() == 3
    hits = store.search(vecs[0], k=1)
    assert hits[0].chunk.doc_id == "d0"
    store.drop("d0")
    assert store.count() == 2
    remaining = {h.chunk.doc_id for h in store.search(vecs[1], k=5)}
    assert "d0" not in remaining


def test_bm25_real_search():
    # rank-bm25 的 IDF 在「词恰出现于半数文档」时为 ln(1)=0；
    # 故生产 BM25 语料须 >=3 篇且目标词不在半数文档（小语料请用默认词法检索器）。
    chunks = [
        Chunk(chunk_id="d0::c0", doc_id="d0", text="量子计算利用叠加态实现并行运算"),
        Chunk(chunk_id="d1::c0", doc_id="d1", text="古典音乐强调和声与旋律的平衡"),
        Chunk(chunk_id="d2::c0", doc_id="d2", text="篮球运动强调团队配合与投篮命中率"),
    ]
    retr = BM25Retriever()
    retr.index(chunks)
    hits = retr.search("量子计算 叠加态", k=3)
    assert hits and hits[0].chunk.doc_id == "d0"
    assert all(h.score > 0.0 for h in hits)
