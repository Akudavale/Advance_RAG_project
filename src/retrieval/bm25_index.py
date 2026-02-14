"""
src/retrieval/bm25_index.py
---------------------------
Lightweight persistent BM25 index for keyword-style retrieval.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BM25Index:
    """Persistent BM25 index over chunked document texts."""

    def __init__(self, config=None):
        from config.config import Config

        self.config = config or Config()
        self.persist_directory = Path(getattr(self.config, "CHROMA_PERSIST_DIR", "./chroma_db"))
        self.index_path = self.persist_directory / "bm25_index.json"

        self._documents: List[Dict[str, Any]] = []
        self._doc_ids: set = set()
        self._tokenized_corpus: List[List[str]] = []
        self._bm25 = None

        self._load()
        self._rebuild_index()

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def _make_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        hash_input = content
        hash_input += str(metadata.get("filename", ""))
        hash_input += str(metadata.get("page_number", ""))
        hash_input += str(metadata.get("chunk_id", ""))
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    def _rebuild_index(self):
        if not self._documents:
            self._bm25 = None
            self._tokenized_corpus = []
            return

        try:
            from rank_bm25 import BM25Okapi

            self._tokenized_corpus = [self._tokenize(d.get("content", "")) for d in self._documents]
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except Exception as e:
            logger.warning(f"Failed to rebuild BM25 index: {e}")
            self._bm25 = None

    def _save(self):
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            data = {"documents": self._documents}
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save BM25 index: {e}")

    def _load(self):
        try:
            if not self.index_path.exists():
                return
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._documents = data.get("documents", [])
            self._doc_ids = {d.get("document_id", "") for d in self._documents if d.get("document_id")}
            logger.info(f"Loaded BM25 index with {len(self._documents)} chunks")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")
            self._documents = []
            self._doc_ids = set()

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Add chunk documents to BM25 index."""
        added = 0
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {}) or {}
            if not content.strip():
                continue
            doc_id = self._make_doc_id(content, metadata)
            if doc_id in self._doc_ids:
                continue

            self._documents.append(
                {
                    "document_id": doc_id,
                    "content": content,
                    "metadata": metadata,
                }
            )
            self._doc_ids.add(doc_id)
            added += 1

        if added:
            self._rebuild_index()
            self._save()
        return added

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search BM25 index and return ranked chunks."""
        if not query or not query.strip() or not self._bm25:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        if len(raw_scores) == 0:
            return []

        # min-max normalize to [0,1] for easier fusion with vector scores
        min_s = float(min(raw_scores))
        max_s = float(max(raw_scores))
        denom = max(max_s - min_s, 1e-9)

        scored: List[Dict[str, Any]] = []
        for idx, s in enumerate(raw_scores):
            doc = self._documents[idx]
            metadata = doc.get("metadata", {}) or {}

            if filter_metadata:
                mismatch = False
                for k, v in filter_metadata.items():
                    if metadata.get(k) != v:
                        mismatch = True
                        break
                if mismatch:
                    continue

            score = (float(s) - min_s) / denom
            scored.append(
                {
                    "content": doc.get("content", ""),
                    "metadata": metadata,
                    "score": score,
                    "document_id": doc.get("document_id", ""),
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete_document(self, filename: str) -> int:
        """Delete all chunks of one source filename."""
        before = len(self._documents)
        self._documents = [
            d for d in self._documents if (d.get("metadata", {}) or {}).get("filename") != filename
        ]
        removed = before - len(self._documents)
        if removed:
            self._doc_ids = {d.get("document_id", "") for d in self._documents if d.get("document_id")}
            self._rebuild_index()
            self._save()
        return removed

    def clear(self):
        self._documents = []
        self._doc_ids = set()
        self._tokenized_corpus = []
        self._bm25 = None
        self._save()

    def size(self) -> int:
        return len(self._documents)
