"""
LangChain-native storage layer for vector + sparse retrieval.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_core.embeddings import Embeddings


@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    document_id: str = ""


class _ProjectEmbedderAdapter(Embeddings):
    """Adapter to reuse the project's embedder inside LangChain vector store."""

    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embedder.embed(text)


class LangChainStorageEngine:
    """
    Unified storage engine:
    - Vector storage via LangChain Chroma integration
    - Sparse storage via persistent BM25 corpus
    """

    def __init__(self, config=None, embedder=None, logger=None):
        from config.config import Config

        self.config = config or Config()
        self.embedder = embedder
        self.logger = logger

        self.persist_directory = Path(getattr(self.config, "CHROMA_PERSIST_DIR", "./chroma_db"))
        self.collection_name = getattr(self.config, "CHROMA_COLLECTION", "rag_documents")

        self._registry_path = self.persist_directory / "document_registry_langchain.json"
        self._bm25_path = self.persist_directory / "bm25_langchain.json"

        self._indexed_documents: Set[str] = set()
        self._bm25_docs: List[Dict[str, Any]] = []
        self._bm25_doc_ids: Set[str] = set()
        self._bm25 = None

        self._vector_store = None
        self._embeddings = None

        self._load_registry()
        self._load_bm25()
        self._rebuild_bm25()
        self._bootstrap_registry_from_existing_vectors()

    @property
    def embeddings(self):
        if self._embeddings is None:
            if self.embedder is None:
                from src.embedding.embedder import Embedder

                self.embedder = Embedder(self.config)
            self._embeddings = _ProjectEmbedderAdapter(self.embedder)
        return self._embeddings

    @property
    def vector_store(self):
        if self._vector_store is None:
            try:
                from langchain_chroma import Chroma
            except Exception:
                from langchain_community.vectorstores import Chroma

            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
            )
        return self._vector_store

    def _bootstrap_registry_from_existing_vectors(self):
        """
        Rebuild filename registry from existing persisted vectors when registry is empty.
        """
        if self._indexed_documents:
            return

        try:
            collection = self.vector_store._collection  # noqa: SLF001
            rows = collection.get(include=["metadatas"])
            metas = rows.get("metadatas", []) if rows else []
            filenames = set()
            for metadata in metas:
                md = metadata or {}
                fn = md.get("filename", "")
                if fn:
                    filenames.add(fn)
            if filenames:
                for fn in filenames:
                    self._indexed_documents.add(self._doc_fingerprint(fn))
                self._save_registry()
        except Exception:
            pass

    def _doc_fingerprint(self, filename: str) -> str:
        return hashlib.md5(filename.encode("utf-8")).hexdigest()

    def _chunk_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        raw = (
            content
            + str(metadata.get("filename", ""))
            + str(metadata.get("page_number", ""))
            + str(metadata.get("chunk_id", ""))
        )
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _load_registry(self):
        try:
            if self._registry_path.exists():
                data = json.loads(self._registry_path.read_text(encoding="utf-8"))
                self._indexed_documents = set(data.get("document_ids", []))
        except Exception:
            self._indexed_documents = set()

    def _save_registry(self):
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "document_ids": list(self._indexed_documents),
                "updated_at": datetime.now().isoformat(),
                "count": len(self._indexed_documents),
            }
            self._registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_bm25(self):
        try:
            if not self._bm25_path.exists():
                return
            data = json.loads(self._bm25_path.read_text(encoding="utf-8"))
            self._bm25_docs = data.get("documents", [])
            self._bm25_doc_ids = {d.get("document_id", "") for d in self._bm25_docs if d.get("document_id")}
        except Exception:
            self._bm25_docs = []
            self._bm25_doc_ids = set()

    def _save_bm25(self):
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            payload = {"documents": self._bm25_docs}
            self._bm25_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _tokenize(self, text: str) -> List[str]:
        import re

        return re.findall(r"\b\w+\b", (text or "").lower())

    def _rebuild_bm25(self):
        if not self._bm25_docs:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi

            corpus = [self._tokenize(d.get("content", "")) for d in self._bm25_docs]
            self._bm25 = BM25Okapi(corpus)
        except Exception:
            self._bm25 = None

    def is_document_indexed(self, filename: str) -> bool:
        return self._doc_fingerprint(filename) in self._indexed_documents

    def add_documents(self, documents: List[Dict[str, Any]], skip_duplicates: bool = True) -> Dict[str, Any]:
        if not documents:
            return {"status": "success", "added": 0, "skipped": 0, "message": "No documents to add"}

        incoming = documents
        if skip_duplicates:
            filenames = {
                (d.get("metadata", {}) or {}).get("filename", "")
                for d in incoming
                if (d.get("metadata", {}) or {}).get("filename", "")
            }
            already = [fn for fn in filenames if self.is_document_indexed(fn)]
            if already:
                incoming = [
                    d
                    for d in incoming
                    if (d.get("metadata", {}) or {}).get("filename", "") not in set(already)
                ]
                if not incoming:
                    return {
                        "status": "success",
                        "added": 0,
                        "skipped": len(already),
                        "message": "All documents already indexed",
                        "skipped_files": already,
                    }

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        added_bm25 = 0
        for doc in incoming:
            content = (doc.get("content", "") or "").strip()
            metadata = (doc.get("metadata", {}) or {}).copy()
            if not content:
                continue

            doc_id = self._chunk_doc_id(content, metadata)
            texts.append(content)
            metadatas.append(metadata)
            ids.append(doc_id)

            if doc_id not in self._bm25_doc_ids:
                self._bm25_docs.append(
                    {"document_id": doc_id, "content": content, "metadata": metadata}
                )
                self._bm25_doc_ids.add(doc_id)
                added_bm25 += 1

        if ids:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        if added_bm25:
            self._rebuild_bm25()
            self._save_bm25()

        new_filenames = {
            m.get("filename", "")
            for m in metadatas
            if m.get("filename", "")
        }
        for fn in new_filenames:
            self._indexed_documents.add(self._doc_fingerprint(fn))
        self._save_registry()

        return {
            "status": "success",
            "added": len(ids),
            "skipped": 0,
            "message": f"Added {len(ids)} documents",
            "new_files": list(new_filenames),
        }

    def add_bm25_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add documents only to BM25 store (used for bootstrap)."""
        added = 0
        for doc in documents:
            content = (doc.get("content", "") or "").strip()
            metadata = (doc.get("metadata", {}) or {}).copy()
            if not content:
                continue
            doc_id = doc.get("document_id") or self._chunk_doc_id(content, metadata)
            if doc_id in self._bm25_doc_ids:
                continue
            self._bm25_docs.append(
                {"document_id": doc_id, "content": content, "metadata": metadata}
            )
            self._bm25_doc_ids.add(doc_id)
            added += 1
        if added:
            self._rebuild_bm25()
            self._save_bm25()
        return added

    def search_vector(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not query or not query.strip():
            return []
        try:
            pairs = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter_metadata,
            )
            results: List[SearchResult] = []
            for doc, distance in pairs:
                doc_id = self._chunk_doc_id(doc.page_content or "", doc.metadata or {})
                score = 1.0 / (1.0 + float(distance))
                results.append(
                    SearchResult(
                        content=doc.page_content or "",
                        metadata=doc.metadata or {},
                        score=score,
                        document_id=doc_id,
                    )
                )
            return results
        except Exception:
            return []

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip() or not self._bm25:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        raw_scores = self._bm25.get_scores(tokens)
        if len(raw_scores) == 0:
            return []

        min_s = float(min(raw_scores))
        max_s = float(max(raw_scores))
        denom = max(max_s - min_s, 1e-9)

        ranked = []
        for idx, score in enumerate(raw_scores):
            doc = self._bm25_docs[idx]
            metadata = doc.get("metadata", {}) or {}
            if filter_metadata:
                mismatch = any(metadata.get(k) != v for k, v in filter_metadata.items())
                if mismatch:
                    continue
            ranked.append(
                {
                    "content": doc.get("content", ""),
                    "metadata": metadata,
                    "score": (float(score) - min_s) / denom,
                    "document_id": doc.get("document_id", ""),
                }
            )
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def delete_document(self, filename: str) -> Dict[str, Any]:
        try:
            # Vector delete
            self.vector_store.delete(where={"filename": filename})

            # BM25 delete
            before = len(self._bm25_docs)
            self._bm25_docs = [
                d for d in self._bm25_docs if (d.get("metadata", {}) or {}).get("filename") != filename
            ]
            if len(self._bm25_docs) != before:
                self._bm25_doc_ids = {
                    d.get("document_id", "") for d in self._bm25_docs if d.get("document_id")
                }
                self._rebuild_bm25()
                self._save_bm25()

            self._indexed_documents.discard(self._doc_fingerprint(filename))
            self._save_registry()
            return {"status": "success", "message": f"Deleted document {filename}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def clear(self) -> Dict[str, Any]:
        try:
            # Clear vector collection
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(self.persist_directory))
                client.delete_collection(self.collection_name)
            except Exception:
                pass

            self._vector_store = None
            self._indexed_documents = set()
            self._save_registry()

            self._bm25_docs = []
            self._bm25_doc_ids = set()
            self._bm25 = None
            self._save_bm25()
            return {"status": "success", "message": "Storage cleared"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_all_documents(self) -> List[Dict[str, Any]]:
        try:
            collection = self.vector_store._collection  # noqa: SLF001
            rows = collection.get(include=["documents", "metadatas"])
            ids = rows.get("ids", []) if rows else []
            docs = rows.get("documents", []) if rows else []
            metas = rows.get("metadatas", []) if rows else []
            out = []
            for doc_id, content, metadata in zip(ids, docs, metas):
                out.append(
                    {
                        "content": content or "",
                        "metadata": metadata or {},
                        "document_id": doc_id or "",
                    }
                )
            return out
        except Exception:
            return []

    def bm25_size(self) -> int:
        return len(self._bm25_docs)

    def get_stats(self) -> Dict[str, Any]:
        count = 0
        try:
            count = self.vector_store._collection.count()  # noqa: SLF001
        except Exception:
            count = 0
        return {
            "indexed_documents": len(self._indexed_documents),
            "persist_directory": str(self.persist_directory),
            "collection_name": self.collection_name,
            "total_vectors": count,
        }
