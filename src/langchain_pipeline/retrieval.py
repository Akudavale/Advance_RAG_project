"""
LangChain-oriented retrieval engine (vector / BM25 / hybrid).
"""

import uuid
from typing import Any, Dict, List, Optional, Sequence

from src.langchain_pipeline.storage import SearchResult


class VectorRetriever:
    """Dense retriever adapter over storage engine."""

    def __init__(self, storage):
        self.storage = storage

    def retrieve(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        return self.storage.search_vector(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )


class SparseRetriever:
    """Sparse retriever adapter over storage engine."""

    def __init__(self, storage):
        self.storage = storage

    def retrieve(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self.storage.search_bm25(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )


class HybridRetriever:
    """Hybrid retriever using weighted reciprocal rank fusion (RRF)."""

    def __init__(self):
        self.rrf_k = 60.0

    def fuse(
        self,
        vector_results: Sequence[SearchResult],
        sparse_results: Sequence[Dict[str, Any]],
        top_k: int,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        alpha = max(0.0, min(1.0, float(alpha)))
        w_dense = alpha
        w_sparse = 1.0 - alpha

        merged: Dict[str, Dict[str, Any]] = {}

        for rank, result in enumerate(vector_results, start=1):
            doc_id = getattr(result, "document_id", "") or _make_fallback_doc_id(
                getattr(result, "content", ""),
                getattr(result, "metadata", {}) or {},
            )
            score = w_dense * (1.0 / (self.rrf_k + rank))
            entry = merged.setdefault(
                doc_id,
                {
                    "document_id": doc_id,
                    "content": getattr(result, "content", ""),
                    "metadata": getattr(result, "metadata", {}) or {},
                    "score": 0.0,
                },
            )
            entry["score"] += score

        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result.get("document_id", "") or _make_fallback_doc_id(
                result.get("content", ""),
                result.get("metadata", {}) or {},
            )
            score = w_sparse * (1.0 / (self.rrf_k + rank))
            entry = merged.setdefault(
                doc_id,
                {
                    "document_id": doc_id,
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}) or {},
                    "score": 0.0,
                },
            )
            entry["score"] += score

        ranked = list(merged.values())
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]


class LangChainRetrievalEngine:
    """
    Central retrieval engine used by orchestrator.

    Modes:
    - vector
    - bm25
    - hybrid
    """

    def __init__(self, storage, logger=None):
        self.storage = storage
        self.logger = logger

        self.vector_retriever = VectorRetriever(storage)
        self.sparse_retriever = SparseRetriever(storage)
        self.hybrid_retriever = HybridRetriever()

        self._bm25_bootstrapped = False

    def retrieve(
        self,
        query: str,
        top_k: int,
        retrieval_mode: str = "hybrid",
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        hybrid_alpha: float = 0.5,
        filter_filenames: Optional[List[str]] = None,
        use_reranking: bool = True,
        reranker: Optional[Any] = None,
        rerank_query: Optional[str] = None,
        rerank_top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        mode = (retrieval_mode or "hybrid").strip().lower()
        if mode not in {"vector", "bm25", "hybrid"}:
            mode = "hybrid"

        dense_k = dense_top_k or top_k
        sparse_k = sparse_top_k or top_k

        if filter_filenames is not None and len(filter_filenames) == 0:
            return []

        self._bootstrap_bm25_if_needed()

        filter_metadata = None
        if filter_filenames and len(filter_filenames) == 1:
            filter_metadata = {"filename": filter_filenames[0]}

        vector_results: List[SearchResult] = []
        sparse_results: List[Dict[str, Any]] = []

        if mode in {"vector", "hybrid"}:
            vector_results = self.vector_retriever.retrieve(
                query=query,
                top_k=max(top_k, dense_k),
                filter_metadata=filter_metadata,
            )

        if mode in {"bm25", "hybrid"}:
            sparse_results = self.sparse_retriever.retrieve(
                query=query,
                top_k=max(top_k, sparse_k),
                filter_metadata=filter_metadata,
            )

        if filter_filenames and len(filter_filenames) > 1:
            allowed = set(filter_filenames)
            vector_results = [
                r
                for r in vector_results
                if getattr(r, "metadata", {}).get("filename") in allowed
            ]
            sparse_results = [
                r
                for r in sparse_results
                if (r.get("metadata", {}) or {}).get("filename") in allowed
            ]

        final_top_k = int(rerank_top_k) if rerank_top_k else top_k
        final_top_k = max(1, final_top_k)

        candidates: List[SearchResult]
        if mode == "vector":
            candidates = vector_results[:top_k]
        elif mode == "bm25":
            candidates = _dict_results_to_search_results(sparse_results[:top_k])
        else:
            fused_dicts = self.hybrid_retriever.fuse(
                vector_results=vector_results,
                sparse_results=sparse_results,
                top_k=top_k,
                alpha=hybrid_alpha,
            )
            candidates = _dict_results_to_search_results(fused_dicts)

        if use_reranking and reranker is not None and len(candidates) > 1:
            try:
                ranked = reranker.rerank(
                    query=rerank_query or query,
                    results=candidates,
                    top_k=final_top_k,
                )
                return ranked[:final_top_k]
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Retrieval-stage reranking failed: {e}")
                return candidates[:final_top_k]

        return candidates[:final_top_k]

    def _bootstrap_bm25_if_needed(self):
        if self._bm25_bootstrapped:
            return
        self._bm25_bootstrapped = True

        if self.storage.bm25_size() > 0:
            return

        try:
            docs = self.storage.get_all_documents()
            if docs:
                added = self.storage.add_bm25_documents(docs)
                if self.logger:
                    self.logger.info(f"Bootstrapped BM25 with {added} chunks from vector store")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"BM25 bootstrap skipped: {e}")


def _dict_results_to_search_results(results: List[Dict[str, Any]]) -> List[SearchResult]:
    converted: List[SearchResult] = []
    for r in results:
        converted.append(
            SearchResult(
                content=r.get("content", ""),
                metadata=r.get("metadata", {}) or {},
                score=float(r.get("score", 0.0)),
                document_id=r.get("document_id", ""),
            )
        )
    return converted


def _make_fallback_doc_id(content: str, metadata: Dict[str, Any]) -> str:
    raw = (
        content
        + str(metadata.get("filename", ""))
        + str(metadata.get("page_number", ""))
        + str(metadata.get("chunk_id", ""))
    )
    return uuid.uuid5(uuid.NAMESPACE_DNS, raw).hex
