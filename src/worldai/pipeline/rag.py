"""RAG 管线（组合检索链路，零依赖默认实现）。

链路：摄入(抽取+分块) -> 嵌入 -> [稠密检索 + 稀疏检索] -> RRF 融合 -> 重排 -> 生成。

设计要点（来自实测坑）：
- **重复摄入先 drop**：向量库按 doc_id 删除旧分块，稀疏索引基于全局分块全量重建，
  避免 upsert 产生的孤儿分块被再次检索到；
- **稠密/稀疏互补融合**用 Reciprocal Rank Fusion（倒数排名融合），不依赖分数量纲对齐；
- **检索结果与生成解耦**：``retrieve`` 只做检索，``query`` 负责生成，便于独立评估。

作者：晨星
"""

from __future__ import annotations

from worldai.llm.prompt import build_rag_prompt
from worldai.models import Chunk, Document, IngestResult, QueryResult, RetrievalHit
from worldai.util.evidence import render_evidence
from worldai.util.mathx import rrf_fuse


class RagPipeline:
    """端到端 RAG 管线。"""

    def __init__(self, extractor, chunker, embedder, vector_store, sparse, reranker, repository, settings, llm=None) -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse = sparse
        self.reranker = reranker
        self.repository = repository
        self.settings = settings
        self.llm = llm
        self._all_chunks: list[Chunk] = []

    def ingest(self, doc_id: str, title: str, source: str, text: str | None = None) -> IngestResult:
        if text is None:
            text = self.extractor.extract(source)
        chunks = self.chunker.chunk(text, title, doc_id)
        doc = Document(doc_id=doc_id, title=title, text=text, source=source, chunks=chunks)
        self.repository.put(doc)

        # 先清除该文档的旧分块，避免孤儿分块
        self._all_chunks = [c for c in self._all_chunks if c.doc_id != doc_id]
        self._all_chunks.extend(chunks)
        self.vector_store.drop(doc_id)

        if chunks:
            vectors = self.embedder.embed([c.text for c in chunks])
            self.vector_store.add([c.chunk_id for c in chunks], vectors, chunks)

        # 稀疏索引基于全局分块全量重建（Lexical/BM25 均按整库建索引）
        self.sparse.index(self._all_chunks)
        return IngestResult(doc_id=doc_id, title=title, num_chunks=len(chunks))

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        top_k = top_k or self.settings.top_k
        if self.vector_store.count() == 0 or not self._all_chunks:
            return []
        qvec = self.embedder.embed([query])[0]
        dense = self.vector_store.search(qvec, top_k)
        sparse_hits = self.sparse.search(query, top_k)

        dense_ids = [h.chunk.chunk_id for h in dense]
        sparse_ids = [h.chunk.chunk_id for h in sparse_hits]
        fused = rrf_fuse([dense_ids, sparse_ids], k=60)

        by_id = {c.chunk_id: c for c in self._all_chunks}
        fused_hits = [RetrievalHit(chunk=by_id[cid], score=score) for cid, score in fused if cid in by_id]
        if self.reranker is not None:
            fused_hits = self.reranker.rerank(query, fused_hits, top_k)
        return fused_hits[:top_k]

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        hits = self.retrieve(question, top_k)
        evidence = render_evidence(hits)
        prompt = build_rag_prompt(question, evidence)
        answer = self.llm.complete(prompt)
        return QueryResult(answer=answer, hits=hits)
