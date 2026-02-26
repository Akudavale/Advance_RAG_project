"""
LangChain-based document ingestion and chunking for PDF sources.
"""

import re
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChainIndexingEngine:
    """Build indexable chunk dictionaries from PDFs using LangChain loaders/splitters."""

    def __init__(self, config=None):
        from config.config import Config

        self.config = config or Config()
        self.chunk_size = int(getattr(self.config, "CHUNK_SIZE", 1000))
        self.chunk_overlap = int(getattr(self.config, "CHUNK_OVERLAP", 200))

        self.table_extraction_enabled = bool(
            getattr(self.config, "PDF_TABLE_EXTRACTION_ENABLED", True)
        )
        self.table_min_cells = max(1, int(getattr(self.config, "PDF_TABLE_MIN_CELLS", 6)))
        self.table_max_rows = max(1, int(getattr(self.config, "PDF_TABLE_MAX_ROWS", 60)))

    def load_pdf_documents(self, file_path: str) -> List[Document]:
        """Load and split PDF into LangChain Documents, including optional table docs."""
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        page_docs = loader.load()
        filename = file_path.split("\\")[-1].split("/")[-1]

        # Normalize base metadata.
        normalized_pages: List[Document] = []
        for d in page_docs:
            metadata = dict(d.metadata or {})
            metadata["filename"] = filename
            metadata["source"] = filename
            # Normalize page numbering to 1-based.
            if "page" in metadata and "page_number" not in metadata:
                try:
                    metadata["page_number"] = int(metadata["page"]) + 1
                except Exception:
                    metadata["page_number"] = metadata["page"]
            metadata.setdefault("modality", "text")
            normalized_pages.append(Document(page_content=d.page_content or "", metadata=metadata))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        text_chunks = splitter.split_documents(normalized_pages)

        table_docs: List[Document] = []
        if self.table_extraction_enabled:
            table_docs = self._extract_table_documents(file_path=file_path, filename=filename)

        return text_chunks + table_docs

    def to_chunk_dicts(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Convert LangChain Documents to project chunk dictionaries.
        """
        out: List[Dict[str, Any]] = []
        for idx, doc in enumerate(documents):
            content = (doc.page_content or "").strip()
            if not content:
                continue

            metadata = dict(doc.metadata or {})
            metadata.setdefault("chunk_id", idx)
            metadata.setdefault("filename", metadata.get("source", ""))
            metadata.setdefault("source", metadata.get("filename", ""))
            metadata.setdefault("start_char", 0)
            metadata.setdefault("end_char", len(content))

            out.append({"content": content, "metadata": metadata})
        return out

    def _extract_table_documents(self, file_path: str, filename: str) -> List[Document]:
        """
        Extract tables + captions from PDF using pdfplumber and return as Documents.
        """
        try:
            import pdfplumber
        except Exception:
            return []

        docs: List[Document] = []
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                if not tables:
                    continue

                captions = self._extract_table_caption_candidates(page.extract_text() or "")
                for table_index, raw_rows in enumerate(tables, start=1):
                    rows = self._normalize_table_rows(raw_rows)
                    if not rows:
                        continue

                    cell_count = sum(len(r) for r in rows)
                    if cell_count < self.table_min_cells:
                        continue

                    caption = captions[table_index - 1] if table_index - 1 < len(captions) else (captions[-1] if captions else "")
                    body = self._format_table_rows(rows)
                    if not body:
                        continue

                    parts = []
                    if caption:
                        parts.append(f"Table caption: {caption}")
                    parts.append("Table content:")
                    parts.append(body)
                    content = "\n".join(parts).strip()

                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "filename": filename,
                                "source": filename,
                                "page_number": page_number,
                                "modality": "table",
                                "table_index": table_index,
                                "table_caption": caption,
                                "table_rows": len(rows),
                                "table_cols": max((len(r) for r in rows), default=0),
                            },
                        )
                    )
        return docs

    def _extract_table_caption_candidates(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.split("\n") if line and line.strip()]
        primary: List[str] = []
        secondary: List[str] = []
        for line in lines:
            if re.match(r"^\s*(table|tbl)\s*[\d\.\-:]", line, flags=re.IGNORECASE):
                primary.append(line)
            elif "table" in line.lower():
                secondary.append(line)
        return primary + secondary

    def _normalize_table_rows(self, raw_rows: List[List[Any]]) -> List[List[str]]:
        normalized: List[List[str]] = []
        for row in raw_rows or []:
            clean_row = []
            for cell in row or []:
                value = str(cell).strip() if cell is not None else ""
                value = re.sub(r"\s+", " ", value).strip()
                clean_row.append(value)
            if any(cell for cell in clean_row):
                normalized.append(clean_row)
            if len(normalized) >= self.table_max_rows:
                break
        return normalized

    def _format_table_rows(self, rows: List[List[str]]) -> str:
        if not rows:
            return ""
        width = max((len(r) for r in rows), default=0)
        if width <= 0:
            return ""
        rendered = []
        for row in rows:
            padded = row + ([""] * (width - len(row)))
            rendered.append(" | ".join(padded))
        return "\n".join(rendered)

