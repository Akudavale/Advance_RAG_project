"""  
src/data_processing/pdf_processor.py  
------------------------------------  
PDF document processing with chunking and metadata extraction.  
"""  
  
import logging  
import hashlib  
import json  
import os  
from pathlib import Path  
from typing import List, Dict, Any, Optional  
from dataclasses import dataclass, field  
from datetime import datetime  
  
logger = logging.getLogger(__name__)  
  
  
@dataclass  
class ProcessedPage:  
    """A processed PDF page."""  
    page_number: int  
    content: str  
    metadata: Dict[str, Any] = field(default_factory=dict)  
      
    def to_dict(self) -> Dict[str, Any]:  
        """Convert to JSON-serializable dictionary."""  
        return {  
            "page_number": self.page_number,  
            "content": self.content,  
            "metadata": self._serialize_metadata(self.metadata)  
        }  
      
    @staticmethod  
    def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:  
        """Ensure metadata is JSON-serializable."""  
        serialized = {}  
        for key, value in metadata.items():  
            try:  
                # Test if serializable  
                json.dumps(value)  
                serialized[key] = value  
            except (TypeError, ValueError):  
                # Convert non-serializable types to string  
                serialized[key] = str(value)  
        return serialized  
  
  
@dataclass  
class ProcessedChunk:  
    """A text chunk from a PDF."""  
    chunk_id: int  
    content: str  
    page_number: int  
    start_char: int  
    end_char: int  
    metadata: Dict[str, Any] = field(default_factory=dict)  
      
    def to_dict(self) -> Dict[str, Any]:  
        """Convert to dictionary format for vector store."""  
        return {  
            "content": self.content,  
            "metadata": {  
                **self.metadata,  
                "chunk_id": self.chunk_id,  
                "page_number": self.page_number,  
                "start_char": self.start_char,  
                "end_char": self.end_char  
            }  
        }  
  
  
class PDFProcessor:  
    """  
    Processes PDF documents into text chunks.  
      
    Features:  
    - Multiple PDF library support (pymupdf, pdfplumber, pypdf)  
    - Configurable chunking with overlap  
    - Metadata extraction  
    - Result caching (caches chunks, not pages)  
    - Sentence-boundary-aware chunking  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the PDF processor.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self.chunk_size = getattr(self.config, "CHUNK_SIZE", 1000)  
        self.chunk_overlap = getattr(self.config, "CHUNK_OVERLAP", 150)  
          
        # Validate chunk parameters  
        if self.chunk_overlap >= self.chunk_size:  
            logger.warning(  
                f"Chunk overlap ({self.chunk_overlap}) >= chunk size ({self.chunk_size}). "  
                f"Setting overlap to {self.chunk_size // 4}"  
            )  
            self.chunk_overlap = self.chunk_size // 4  
          
        # Caching  
        self.cache_enabled = getattr(self.config, "PDF_CACHE_ENABLED", True)  
        self.cache_dir = Path(getattr(self.config, "PDF_CACHE_DIR", "./.cache/pdf"))  
          
        if self.cache_enabled:  
            self.cache_dir.mkdir(parents=True, exist_ok=True)  
          
        # Detect available PDF library  
        self._pdf_library = self._detect_pdf_library()  
        logger.info(f"PDFProcessor initialized with {self._pdf_library}, "  
                   f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")  
      
    def _detect_pdf_library(self) -> str:  
        """Detect available PDF library."""  
        try:  
            import fitz  # pymupdf  
            return "pymupdf"  
        except ImportError:  
            pass  
          
        try:  
            import pdfplumber  
            return "pdfplumber"  
        except ImportError:  
            pass  
          
        try:  
            from pypdf import PdfReader  
            return "pypdf"  
        except ImportError:  
            pass  
          
        logger.warning("No PDF library found. Install pymupdf, pdfplumber, or pypdf.")  
        return "none"  
      
    def _get_file_hash(self, file_path: str) -> str:  
        """  
        Get hash of file for caching.  
          
        Includes file modification time to invalidate cache on changes.  
        """  
        hasher = hashlib.md5()  
          
        # Include file path and modification time  
        file_stat = os.stat(file_path)  
        hasher.update(file_path.encode())  
        hasher.update(str(file_stat.st_mtime).encode())  
        hasher.update(str(file_stat.st_size).encode())  
          
        # Include chunk parameters (cache invalidates if these change)  
        hasher.update(str(self.chunk_size).encode())  
        hasher.update(str(self.chunk_overlap).encode())  
          
        # Include first 8KB of file content  
        with open(file_path, "rb") as f:  
            hasher.update(f.read(8192))  
          
        return hasher.hexdigest()  
      
    def _get_cache_path(self, file_hash: str) -> Path:  
        """Get cache file path."""  
        return self.cache_dir / f"{file_hash}.json"  
      
    def _load_from_cache(self, file_path: str) -> Optional[List[ProcessedChunk]]:  
        """  
        Load processed chunks from cache.  
          
        Returns:  
            List of ProcessedChunk if cache hit, None otherwise  
        """  
        if not self.cache_enabled:  
            return None  
          
        try:  
            file_hash = self._get_file_hash(file_path)  
            cache_path = self._get_cache_path(file_hash)  
              
            if not cache_path.exists():  
                return None  
              
            with open(cache_path, "r", encoding="utf-8") as f:  
                data = json.load(f)  
              
            # Validate cache version  
            if data.get("version") != "1.0":  
                logger.debug("Cache version mismatch, ignoring cache")  
                return None  
              
            # Reconstruct chunks  
            chunks = [  
                ProcessedChunk(  
                    chunk_id=c["chunk_id"],  
                    content=c["content"],  
                    page_number=c["page_number"],  
                    start_char=c["start_char"],  
                    end_char=c["end_char"],  
                    metadata=c.get("metadata", {})  
                )  
                for c in data.get("chunks", [])  
            ]  
              
            logger.info(f"Loaded {len(chunks)} chunks from cache for {os.path.basename(file_path)}")  
            return chunks  
              
        except json.JSONDecodeError as e:  
            logger.warning(f"Cache JSON decode error: {e}")  
            return None  
        except Exception as e:  
            logger.debug(f"Cache load failed: {e}")  
            return None  
      
    def _save_to_cache(self, file_path: str, chunks: List[ProcessedChunk]):  
        """  
        Save processed chunks to cache.  
          
        Args:  
            file_path: Original PDF file path  
            chunks: Processed chunks to cache  
        """  
        if not self.cache_enabled:  
            return  
          
        try:  
            file_hash = self._get_file_hash(file_path)  
            cache_path = self._get_cache_path(file_hash)  
              
            # Build cache data with version for future compatibility  
            data = {  
                "version": "1.0",  
                "file_path": os.path.abspath(file_path),  
                "filename": os.path.basename(file_path),  
                "processed_at": datetime.now().isoformat(),  
                "chunk_size": self.chunk_size,  
                "chunk_overlap": self.chunk_overlap,  
                "chunk_count": len(chunks),  
                "chunks": [  
                    {  
                        "chunk_id": c.chunk_id,  
                        "content": c.content,  
                        "page_number": c.page_number,  
                        "start_char": c.start_char,  
                        "end_char": c.end_char,  
                        "metadata": c.metadata  
                    }  
                    for c in chunks  
                ]  
            }  
              
            with open(cache_path, "w", encoding="utf-8") as f:  
                json.dump(data, f, ensure_ascii=False, indent=2)  
              
            logger.debug(f"Saved {len(chunks)} chunks to cache")  
              
        except Exception as e:  
            logger.warning(f"Cache save failed: {e}")  
      
    def process(self, file_path: str) -> List[Dict[str, Any]]:  
        """  
        Process a PDF file into chunks.  
          
        Args:  
            file_path: Path to PDF file  
              
        Returns:  
            List of chunk dictionaries with 'content' and 'metadata' keys  
              
        Raises:  
            FileNotFoundError: If file doesn't exist  
            RuntimeError: If no PDF library is available  
            ValueError: If file is not a valid PDF  
        """  
        # Validate file exists  
        if not os.path.exists(file_path):  
            raise FileNotFoundError(f"PDF file not found: {file_path}")  
          
        # Validate file extension  
        if not file_path.lower().endswith('.pdf'):  
            raise ValueError(f"File is not a PDF: {file_path}")  
          
        # Validate PDF library  
        if self._pdf_library == "none":  
            raise RuntimeError(  
                "No PDF library available. Install one of: pymupdf, pdfplumber, pypdf"  
            )  
          
        # Check cache first  
        cached_chunks = self._load_from_cache(file_path)  
        if cached_chunks is not None:  
            return [c.to_dict() for c in cached_chunks]  
          
        # Extract pages from PDF  
        try:  
            pages = self._extract_pages(file_path)  
        except Exception as e:  
            logger.error(f"PDF extraction failed: {e}")  
            raise ValueError(f"Failed to extract text from PDF: {e}")  
          
        if not pages:  
            logger.warning(f"No content extracted from {file_path}")  
            return []  
          
        # Get filename for metadata  
        filename = os.path.basename(file_path)  
          
        # Chunk the content  
        chunks = self._chunk_pages(pages, filename)  
          
        if not chunks:  
            logger.warning(f"No chunks created from {file_path}")  
            return []  
          
        # Cache results  
        self._save_to_cache(file_path, chunks)  
          
        logger.info(f"Processed {file_path}: {len(pages)} pages -> {len(chunks)} chunks")  
          
        return [c.to_dict() for c in chunks]  
      
    def _extract_pages(self, file_path: str) -> List[ProcessedPage]:  
        """  
        Extract text from PDF pages.  
          
        Args:  
            file_path: Path to PDF file  
              
        Returns:  
            List of ProcessedPage objects  
        """  
        if self._pdf_library == "pymupdf":  
            return self._extract_pymupdf(file_path)  
        elif self._pdf_library == "pdfplumber":  
            return self._extract_pdfplumber(file_path)  
        elif self._pdf_library == "pypdf":  
            return self._extract_pypdf(file_path)  
        else:  
            raise RuntimeError("No PDF library available")  
      
    def _extract_pymupdf(self, file_path: str) -> List[ProcessedPage]:  
        """Extract using PyMuPDF (fitz)."""  
        import fitz  
          
        pages = []  
        doc = None  
          
        try:  
            doc = fitz.open(file_path)  
              
            for page_num in range(len(doc)):  
                page = doc[page_num]  
                text = page.get_text("text")  # Explicitly request text format  
                  
                # Clean up text  
                text = self._clean_text(text)  
                  
                if text:  
                    pages.append(ProcessedPage(  
                        page_number=page_num + 1,  
                        content=text,  
                        metadata={  
                            "width": float(page.rect.width),  
                            "height": float(page.rect.height),  
                            "rotation": page.rotation  
                        }  
                    ))  
                      
        finally:  
            if doc:  
                doc.close()  
          
        return pages  
      
    def _extract_pdfplumber(self, file_path: str) -> List[ProcessedPage]:  
        """Extract using pdfplumber."""  
        import pdfplumber  
          
        pages = []  
          
        with pdfplumber.open(file_path) as pdf:  
            for i, page in enumerate(pdf.pages):  
                text = page.extract_text() or ""  
                text = self._clean_text(text)  
                  
                if text:  
                    pages.append(ProcessedPage(  
                        page_number=i + 1,  
                        content=text,  
                        metadata={  
                            "width": float(page.width),  
                            "height": float(page.height)  
                        }  
                    ))  
          
        return pages  
      
    def _extract_pypdf(self, file_path: str) -> List[ProcessedPage]:  
        """Extract using pypdf."""  
        from pypdf import PdfReader  
          
        pages = []  
        reader = PdfReader(file_path)  
          
        for i, page in enumerate(reader.pages):  
            text = page.extract_text() or ""  
            text = self._clean_text(text)  
              
            if text:  
                pages.append(ProcessedPage(  
                    page_number=i + 1,  
                    content=text,  
                    metadata={}  
                ))  
          
        return pages  
      
    def _clean_text(self, text: str) -> str:  
        """  
        Clean extracted text.  
          
        Args:  
            text: Raw extracted text  
              
        Returns:  
            Cleaned text  
        """  
        if not text:  
            return ""  
          
        # Replace multiple whitespace with single space  
        import re  
        text = re.sub(r'[ \t]+', ' ', text)  
          
        # Replace multiple newlines with double newline  
        text = re.sub(r'\n{3,}', '\n\n', text)  
          
        # Remove leading/trailing whitespace from lines  
        lines = [line.strip() for line in text.split('\n')]  
        text = '\n'.join(lines)  
          
        # Remove leading/trailing whitespace  
        text = text.strip()  
          
        return text  
      
    def _chunk_pages(  
        self,  
        pages: List[ProcessedPage],  
        filename: str  
    ) -> List[ProcessedChunk]:  
        """  
        Chunk pages into smaller pieces.  
          
        Args:  
            pages: List of processed pages  
            filename: Source filename for metadata  
              
        Returns:  
            List of ProcessedChunk objects  
        """  
        chunks = []  
        chunk_id = 0  
          
        for page in pages:  
            page_chunks = self._chunk_text(  
                text=page.content,  
                page_number=page.page_number,  
                filename=filename,  
                start_chunk_id=chunk_id  
            )  
            chunks.extend(page_chunks)  
            chunk_id += len(page_chunks)  
          
        return chunks  
      
    def _chunk_text(  
        self,  
        text: str,  
        page_number: int,  
        filename: str,  
        start_chunk_id: int = 0  
    ) -> List[ProcessedChunk]:  
        """  
        Chunk text with overlap, respecting sentence boundaries.  
          
        Args:  
            text: Text to chunk  
            page_number: Source page number  
            filename: Source filename  
            start_chunk_id: Starting chunk ID  
              
        Returns:  
            List of ProcessedChunk objects  
        """  
        if not text or not text.strip():  
            return []  
          
        chunks = []  
        start = 0  
        chunk_id = start_chunk_id  
        text_length = len(text)  
          
        while start < text_length:  
            # Calculate end position  
            end = min(start + self.chunk_size, text_length)  
              
            # Try to break at sentence boundary if not at end of text  
            if end < text_length:  
                end = self._find_break_point(text, start, end)  
              
            # Extract chunk text  
            chunk_text = text[start:end].strip()  
              
            if chunk_text:  
                chunks.append(ProcessedChunk(  
                    chunk_id=chunk_id,  
                    content=chunk_text,  
                    page_number=page_number,  
                    start_char=start,  
                    end_char=end,  
                    metadata={  
                        "filename": filename,  
                        "source": filename  
                    }  
                ))  
                chunk_id += 1  
              
            # Move start position with overlap  
            if end >= text_length:  
                break  
              
            start = end - self.chunk_overlap  
              
            # Ensure we make progress  
            if start <= chunks[-1].start_char if chunks else 0:  
                start = end  
          
        return chunks  
      
    def _find_break_point(self, text: str, start: int, end: int) -> int:  
        """  
        Find a good break point near the end position.  
          
        Prefers breaking at:  
        1. Paragraph boundaries (double newline)  
        2. Sentence boundaries (. ! ?)  
        3. Clause boundaries (, ; :)  
        4. Word boundaries (space)  
          
        Args:  
            text: Full text  
            start: Start position  
            end: Proposed end position  
              
        Returns:  
            Adjusted end position  
        """  
        search_text = text[start:end]  
        min_chunk = self.chunk_size // 2  # Don't break before half the chunk size  
          
        # Look for paragraph break  
        para_break = search_text.rfind('\n\n')  
        if para_break > min_chunk:  
            return start + para_break + 2  
          
        # Look for sentence end  
        for sep in ['. ', '.\n', '? ', '?\n', '! ', '!\n']:  
            sent_break = search_text.rfind(sep)  
            if sent_break > min_chunk:  
                return start + sent_break + len(sep)  
          
        # Look for clause break  
        for sep in [', ', '; ', ': ', ',\n']:  
            clause_break = search_text.rfind(sep)  
            if clause_break > min_chunk:  
                return start + clause_break + len(sep)  
          
        # Look for word break  
        word_break = search_text.rfind(' ')  
        if word_break > min_chunk:  
            return start + word_break + 1  
          
        # No good break found, use original end  
        return end  
      
    def get_document_metadata(self, file_path: str) -> Dict[str, Any]:  
        """  
        Get PDF document metadata without full processing.  
          
        Args:  
            file_path: Path to PDF file  
              
        Returns:  
            Document metadata dictionary  
        """  
        if not os.path.exists(file_path):  
            return {"error": f"File not found: {file_path}"}  
          
        try:  
            if self._pdf_library == "pymupdf":  
                import fitz  
                doc = fitz.open(file_path)  
                metadata = dict(doc.metadata) if doc.metadata else {}  
                page_count = len(doc)  
                doc.close()  
                  
            elif self._pdf_library == "pdfplumber":  
                import pdfplumber  
                with pdfplumber.open(file_path) as pdf:  
                    metadata = dict(pdf.metadata) if pdf.metadata else {}  
                    page_count = len(pdf.pages)  
                      
            elif self._pdf_library == "pypdf":  
                from pypdf import PdfReader  
                reader = PdfReader(file_path)  
                metadata = dict(reader.metadata) if reader.metadata else {}  
                page_count = len(reader.pages)  
                  
            else:  
                return {"error": "No PDF library available"}  
              
            # Clean metadata (remove None values and convert to strings)  
            clean_metadata = {}  
            for key, value in metadata.items():  
                if value is not None:  
                    # Remove leading slash from pypdf keys  
                    clean_key = key.lstrip('/')  
                    clean_metadata[clean_key] = str(value)  
              
            return {  
                "filename": os.path.basename(file_path),  
                "file_size_bytes": os.path.getsize(file_path),  
                "page_count": page_count,  
                "metadata": clean_metadata,  
                "pdf_library": self._pdf_library  
            }  
              
        except Exception as e:  
            logger.error(f"Failed to get metadata: {e}")  
            return {"error": str(e)}  
      
    def clear_cache(self, file_path: Optional[str] = None) -> int:  
        """  
        Clear cached data.  
          
        Args:  
            file_path: If provided, clear only this file's cache.  
                      If None, clear all cache.  
                        
        Returns:  
            Number of cache files removed  
        """  
        if not self.cache_enabled or not self.cache_dir.exists():  
            return 0  
          
        removed = 0  
          
        if file_path:  
            # Clear specific file's cache  
            try:  
                file_hash = self._get_file_hash(file_path)  
                cache_path = self._get_cache_path(file_hash)  
                if cache_path.exists():  
                    cache_path.unlink()  
                    removed = 1  
            except Exception as e:  
                logger.warning(f"Failed to clear cache for {file_path}: {e}")  
        else:  
            # Clear all cache  
            for cache_file in self.cache_dir.glob("*.json"):  
                try:  
                    cache_file.unlink()  
                    removed += 1  
                except Exception as e:  
                    logger.warning(f"Failed to remove {cache_file}: {e}")  
          
        logger.info(f"Cleared {removed} cache file(s)")  
        return removed  