"""表征层（零依赖默认实现）测试：哈希嵌入 / 内存向量 / 词法检索 / 词法重排。"""

import math

from worldai.embedding.hash_embedder import HashEmbedder
from worldai.models import Chunk, RetrievalHit
from worldai.rerank.lexical_reranker import LexicalReranker
from worldai.sparse.lexical import LexicalRetriever
from worldai.util.mathx import cosine
from worldai.vector_store.memory_store import MemoryVectorStore


def test_hash_embedder_dim_and_unit_norm():
    emb = HashEmbedder(dim=128)
    assert emb.dim() == 128
    v = emb.embed(["机器学习是人工智能的一个分支"])[0]
    assert len(v) == 128
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-9


def test_hash_embedder_deterministic():
    emb = HashEmbedder(dim=128)
    a = emb.embed(["相同文本"])[0]
    b = emb.embed(["相同文本"])[0]
    assert a == b


def test_hash_embedder_similarity_rank():
    emb = HashEmbedder(dim=256)
    texts = [
        "机器学习模型训练方法",
        "咖啡的冲泡技巧",
        "深度学习神经网络结构",
    ]
    vecs = emb.embed(texts)
    q = emb.embed(["如何训练机器学习模型"])[0]
    sims = [cosine(q, v) for v in vecs]
    assert sims[0] == max(sims)  # 与首条文本最相似


def test_memory_store_search_returns_nearest():
    emb = HashEmbedder(dim=256)
    texts = ["机器学习模型训练方法", "咖啡的冲泡技巧", "深度学习神经网络"]
    vecs = emb.embed(texts)
    store = MemoryVectorStore()
    chunks = [Chunk(chunk_id=f"d{i}::c0", doc_id=f"d{i}", text=texts[i]) for i in range(3)]
    store.add([c.chunk_id for c in chunks], vecs, chunks)
    assert store.count() == 3
    q = emb.embed(["如何训练机器学习模型"])[0]
    hits = store.search(q, k=1)
    assert hits[0].chunk.doc_id == "d0"


def test_memory_store_drop_removes_orphans():
    emb = HashEmbedder(dim=64)
    store = MemoryVectorStore()
    chunks = [
        Chunk(chunk_id="da::c0", doc_id="da", text="a"),
        Chunk(chunk_id="db::c0", doc_id="db", text="b"),
    ]
    vecs = emb.embed(["a", "b"])
    store.add(["da::c0", "db::c0"], vecs, chunks)
    store.drop("da")
    assert store.count() == 1
    # 删除后再检索不应命中被删文档
    hits = store.search(emb.embed(["a"])[0], k=5)
    assert all(h.chunk.doc_id == "db" for h in hits)


def test_lexical_retriever_small_corpus_nonzero():
    # 2 篇文档小语料：目标词只出现在其中一篇
    chunks = [
        Chunk(chunk_id="d0::c0", doc_id="d0", text="量子计算利用叠加态进行并行运算"),
        Chunk(chunk_id="d1::c0", doc_id="d1", text="古典音乐强调和声与旋律的平衡"),
    ]
    retr = LexicalRetriever()
    retr.index(chunks)
    hits = retr.search("量子计算 叠加态", k=2)
    assert len(hits) >= 1
    assert hits[0].chunk.doc_id == "d0"
    assert hits[0].score > 0.0


def test_lexical_retriever_absent_term_no_hit():
    chunks = [Chunk(chunk_id="d0::c0", doc_id="d0", text="苹果是一种水果")]
    retr = LexicalRetriever()
    retr.index(chunks)
    assert retr.search("火箭 发动机", k=3) == []


def test_lexical_reranker_orders_by_match():
    chunks = [
        Chunk(chunk_id="d0::c0", doc_id="d0", text="神经网络的反向传播算法用于梯度下降"),
        Chunk(chunk_id="d1::c0", doc_id="d1", text="天气晴朗适合户外运动"),
    ]
    hits = [
        RetrievalHit(chunk=chunks[1], score=0.1),
        RetrievalHit(chunk=chunks[0], score=0.05),
    ]
    reranked = LexicalReranker().rerank("反向传播 梯度下降 神经网络", hits, k=2)
    assert reranked[0].chunk.doc_id == "d0"
