# """  
# src/retrieval/vector_store.py  
# -----------------------------  
# Vector store implementation with support for multiple backends.  
# Thread-safe with consistent document format.  
# """  
  
# import logging  
# import threading  
# from typing import List, Dict, Any, Optional, Tuple  
# from dataclasses import dataclass, field  
# from datetime import datetime  
# import hashlib  
  
# import numpy as np  
  
# logger = logging.getLogger(__name__)  
  
  
# @dataclass  
# class Document:  
#     """Standardized document format for the vector store."""  
#     content: str  
#     metadata: Dict[str, Any] = field(default_factory=dict)  
#     doc_id: Optional[str] = None  
#     embedding: Optional[np.ndarray] = None  
      
#     def __post_init__(self):  
#         if self.doc_id is None:  
#             # Generate deterministic ID from content  
#             self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:16]  
      
#     def to_dict(self) -> Dict[str, Any]:  
#         """Convert to dictionary format."""  
#         return {  
#             "content": self.content,  
#             "metadata": self.metadata,  
#             "doc_id": self.doc_id  
#         }  
  
  
# class VectorStore:  
#     """  
#     Thread-safe vector store with multiple backend support.  
      
#     Supported backends:  
#     - chroma: ChromaDB (default)  
#     - faiss: FAISS  
#     - memory: In-memory numpy-based store  
#     """  
      
#     def __init__(self, config=None):  
#         """  
#         Initialize the vector store.  
          
#         Args:  
#             config: Configuration object  
#         """  
#         from config.config import Config  
          
#         self.config = config or Config()  
#         self.backend = getattr(self.config, "VECTOR_DB", "chroma").lower()  
#         self.collection_name = getattr(self.config, "COLLECTION_NAME", "pdf_documents")  
#         self.dimension = getattr(self.config, "EMBEDDING_DIMENSION", 1024)  
          
#         # Thread safety  
#         self._lock = threading.RLock()  
          
#         # Storage  
#         self._documents: List[Document] = []  
#         self._embeddings: Optional[np.ndarray] = None  
#         self._id_to_index: Dict[str, int] = {}  
          
#         # Backend-specific client  
#         self._client = None  
#         self._collection = None  
          
#         self._initialize_backend()  
          
#         logger.info(f"VectorStore initialized with backend: {self.backend}")  
      
#     def _initialize_backend(self):  
#         """Initialize the storage backend."""  
#         if self.backend == "chroma":  
#             self._init_chroma()  
#         elif self.backend == "faiss":  
#             self._init_faiss()  
#         else:  
#             # Default to in-memory  
#             logger.info("Using in-memory vector store")  
      
#     def _init_chroma(self):  
#         """Initialize ChromaDB backend."""  
#         try:  
#             import chromadb  
#             from chromadb.config import Settings  
              
#             persist_dir = getattr(self.config, "CHROMA_PERSIST_DIR", "./chroma_db")  
              
#             self._client = chromadb.Client(Settings(  
#                 chroma_db_impl="duckdb+parquet",  
#                 persist_directory=persist_dir,  
#                 anonymized_telemetry=False  
#             ))  
              
#             self._collection = self._client.get_or_create_collection(  
#                 name=self.collection_name,  
#                 metadata={"hnsw:space": "cosine"}  
#             )  
              
#             logger.info(f"ChromaDB initialized with collection: {self.collection_name}")  
              
#         except ImportError:  
#             logger.warning("ChromaDB not installed, falling back to in-memory store")  
#             self.backend = "memory"  
#         except Exception as e:  
#             logger.error(f"ChromaDB initialization failed: {e}, falling back to in-memory")  
#             self.backend = "memory"  
      
#     def _init_faiss(self):  
#         """Initialize FAISS backend."""  
#         try:  
#             import faiss  
              
#             self._faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine sim  
#             logger.info("FAISS index initialized")  
              
#         except ImportError:  
#             logger.warning("FAISS not installed, falling back to in-memory store")  
#             self.backend = "memory"  
#         except Exception as e:  
#             logger.error(f"FAISS initialization failed: {e}, falling back to in-memory")  
#             self.backend = "memory"  
      
#     def add_documents(  
#         self,  
#         documents: List[Dict[str, Any]],  
#         embeddings: List[np.ndarray]  
#     ) -> List[str]:  
#         """  
#         Add documents with their embeddings to the store.  
          
#         Args:  
#             documents: List of document dicts with 'content' and 'metadata'  
#             embeddings: List of embedding vectors  
              
#         Returns:  
#             List of document IDs  
#         """  
#         if len(documents) != len(embeddings):  
#             raise ValueError(f"Document count ({len(documents)}) != embedding count ({len(embeddings)})")  
          
#         with self._lock:  
#             doc_ids = []  
              
#             for doc_dict, embedding in zip(documents, embeddings):  
#                 # Standardize document format  
#                 doc = Document(  
#                     content=doc_dict.get("content", ""),  
#                     metadata=doc_dict.get("metadata", {}),  
#                     doc_id=doc_dict.get("doc_id"),  
#                     embedding=np.array(embedding)  
#                 )  
                  
#                 # Check for duplicates  
#                 if doc.doc_id in self._id_to_index:  
#                     logger.debug(f"Document {doc.doc_id} already exists, skipping")  
#                     doc_ids.append(doc.doc_id)  
#                     continue  
                  
#                 # Add to storage  
#                 idx = len(self._documents)  
#                 self._documents.append(doc)  
#                 self._id_to_index[doc.doc_id] = idx  
#                 doc_ids.append(doc.doc_id)  
              
#             # Rebuild embedding matrix  
#             self._rebuild_embedding_matrix()  
              
#             # Add to backend  
#             if self.backend == "chroma" and self._collection:  
#                 self._add_to_chroma(documents, embeddings, doc_ids)  
#             elif self.backend == "faiss":  
#                 self._add_to_faiss(embeddings)  
              
#             logger.info(f"Added {len(doc_ids)} documents to vector store")  
#             return doc_ids  
      
#     def _rebuild_embedding_matrix(self):  
#         """Rebuild the embedding matrix from documents."""  
#         if not self._documents:  
#             self._embeddings = None  
#             return  
          
#         embeddings = [doc.embedding for doc in self._documents if doc.embedding is not None]  
#         if embeddings:  
#             self._embeddings = np.vstack(embeddings)  
      
#     def _add_to_chroma(  
#         self,  
#         documents: List[Dict[str, Any]],  
#         embeddings: List[np.ndarray],  
#         doc_ids: List[str]  
#     ):  
#         """Add documents to ChromaDB."""  
#         try:  
#             self._collection.add(  
#                 ids=doc_ids,  
#                 embeddings=[emb.tolist() for emb in embeddings],  
#                 documents=[doc.get("content", "") for doc in documents],  
#                 metadatas=[doc.get("metadata", {}) for doc in documents]  
#             )  
#         except Exception as e:  
#             logger.error(f"Failed to add to ChromaDB: {e}")  
      
#     def _add_to_faiss(self, embeddings: List[np.ndarray]):  
#         """Add embeddings to FAISS index."""  
#         try:  
#             import faiss  
#             embeddings_array = np.vstack(embeddings).astype('float32')  
#             # Normalize for cosine similarity  
#             faiss.normalize_L2(embeddings_array)  
#             self._faiss_index.add(embeddings_array)  
#         except Exception as e:  
#             logger.error(f"Failed to add to FAISS: {e}")  
      
#     def search(  
#         self,  
#         query_embedding: np.ndarray,  
#         top_k: int = 5,  
#         filter_metadata: Optional[Dict[str, Any]] = None  
#     ) -> List[Dict[str, Any]]:  
#         """  
#         Search for similar documents.  
          
#         Args:  
#             query_embedding: Query embedding vector  
#             top_k: Number of results to return  
#             filter_metadata: Optional metadata filters  
              
#         Returns:  
#             List of documents with scores in standardized format:  
#             [{"content": str, "metadata": dict, "score": float, "doc_id": str}, ...]  
#         """  
#         with self._lock:  
#             if self.backend == "chroma" and self._collection:  
#                 return self._search_chroma(query_embedding, top_k, filter_metadata)  
#             elif self.backend == "faiss":  
#                 return self._search_faiss(query_embedding, top_k, filter_metadata)  
#             else:  
#                 return self._search_memory(query_embedding, top_k, filter_metadata)  
      
#     def _search_memory(  
#         self,  
#         query_embedding: np.ndarray,  
#         top_k: int,  
#         filter_metadata: Optional[Dict[str, Any]] = None  
#     ) -> List[Dict[str, Any]]:  
#         """Search using in-memory numpy operations."""  
#         if self._embeddings is None or len(self._documents) == 0:  
#             return []  
          
#         # Normalize query  
#         query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)  
          
#         # Compute similarities  
#         similarities = np.dot(self._embeddings, query_norm)  
          
#         # Apply metadata filter if provided  
#         valid_indices = list(range(len(self._documents)))  
#         if filter_metadata:  
#             valid_indices = [  
#                 i for i in valid_indices  
#                 if self._matches_filter(self._documents[i].metadata, filter_metadata)  
#             ]  
          
#         # Get top-k from valid indices  
#         valid_similarities = [(i, similarities[i]) for i in valid_indices]  
#         valid_similarities.sort(key=lambda x: x[1], reverse=True)  
#         top_results = valid_similarities[:top_k]  
          
#         # Build results  
#         results = []  
#         for idx, score in top_results:  
#             doc = self._documents[idx]  
#             results.append({  
#                 "content": doc.content,  
#                 "metadata": doc.metadata,  
#                 "score": float(score),  
#                 "doc_id": doc.doc_id  
#             })  
          
#         return results  
      
#     def _search_chroma(  
#         self,  
#         query_embedding: np.ndarray,  
#         top_k: int,  
#         filter_metadata: Optional[Dict[str, Any]] = None  
#     ) -> List[Dict[str, Any]]:  
#         """Search using ChromaDB."""  
#         try:  
#             where_filter = filter_metadata if filter_metadata else None  
              
#             results = self._collection.query(  
#                 query_embeddings=[query_embedding.tolist()],  
#                 n_results=top_k,  
#                 where=where_filter  
#             )  
              
#             documents = []  
#             for i in range(len(results["ids"][0])):  
#                 documents.append({  
#                     "content": results["documents"][0][i] if results["documents"] else "",  
#                     "metadata": results["metadatas"][0][i] if results["metadatas"] else {},  
#                     "score": 1 - results["distances"][0][i] if results["distances"] else 0.0,  
#                     "doc_id": results["ids"][0][i]  
#                 })  
              
#             return documents  
              
#         except Exception as e:  
#             logger.error(f"ChromaDB search failed: {e}")  
#             return self._search_memory(query_embedding, top_k, filter_metadata)  
      
#     def _search_faiss(  
#         self,  
#         query_embedding: np.ndarray,  
#         top_k: int,  
#         filter_metadata: Optional[Dict[str, Any]] = None  
#     ) -> List[Dict[str, Any]]:  
#         """Search using FAISS."""  
#         try:  
#             import faiss  
              
#             query = query_embedding.reshape(1, -1).astype('float32')  
#             faiss.normalize_L2(query)  
              
#             scores, indices = self._faiss_index.search(query, min(top_k * 2, len(self._documents)))  
              
#             results = []  
#             for score, idx in zip(scores[0], indices[0]):  
#                 if idx < 0 or idx >= len(self._documents):  
#                     continue  
                  
#                 doc = self._documents[idx]  
                  
#                 # Apply metadata filter  
#                 if filter_metadata and not self._matches_filter(doc.metadata, filter_metadata):  
#                     continue  
                  
#                 results.append({  
#                     "content": doc.content,  
#                     "metadata": doc.metadata,  
#                     "score": float(score),  
#                     "doc_id": doc.doc_id  
#                 })  
                  
#                 if len(results) >= top_k:  
#                     break  
              
#             return results  
              
#         except Exception as e:  
#             logger.error(f"FAISS search failed: {e}")  
#             return self._search_memory(query_embedding, top_k, filter_metadata)  
      
#     def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:  
#         """Check if metadata matches filter criteria."""  
#         for key, value in filter_dict.items():  
#             if key not in metadata:  
#                 return False  
#             if metadata[key] != value:  
#                 return False  
#         return True  
      
#     def delete(self, doc_ids: List[str]) -> int:  
#         """  
#         Delete documents by ID.  
          
#         Args:  
#             doc_ids: List of document IDs to delete  
              
#         Returns:  
#             Number of documents deleted  
#         """  
#         with self._lock:  
#             deleted = 0  
              
#             for doc_id in doc_ids:  
#                 if doc_id in self._id_to_index:  
#                     idx = self._id_to_index[doc_id]  
#                     del self._id_to_index[doc_id]  
#                     self._documents[idx] = None  # Mark as deleted  
#                     deleted += 1  
              
#             # Compact the storage  
#             self._documents = [d for d in self._documents if d is not None]  
#             self._id_to_index = {doc.doc_id: i for i, doc in enumerate(self._documents)}  
#             self._rebuild_embedding_matrix()  
              
#             # Delete from backend  
#             if self.backend == "chroma" and self._collection:  
#                 try:  
#                     self._collection.delete(ids=doc_ids)  
#                 except Exception as e:  
#                     logger.error(f"Failed to delete from ChromaDB: {e}")  
              
#             logger.info(f"Deleted {deleted} documents")  
#             return deleted  
      
#     def clear(self):  
#         """Clear all documents from the store."""  
#         with self._lock:  
#             self._documents = []  
#             self._embeddings = None  
#             self._id_to_index = {}  
              
#             if self.backend == "chroma" and self._client:  
#                 try:  
#                     self._client.delete_collection(self.collection_name)  
#                     self._collection = self._client.create_collection(  
#                         name=self.collection_name,  
#                         metadata={"hnsw:space": "cosine"}  
#                     )  
#                 except Exception as e:  
#                     logger.error(f"Failed to clear ChromaDB: {e}")  
              
#             elif self.backend == "faiss":  
#                 self._init_faiss()  
              
#             logger.info("Vector store cleared")  
      
#     def get_stats(self) -> Dict[str, Any]:  
#         """Get statistics about the vector store."""  
#         with self._lock:  
#             return {  
#                 "backend": self.backend,  
#                 "collection_name": self.collection_name,  
#                 "document_count": len(self._documents),  
#                 "dimension": self.dimension  
#             }  
        















"""  
src/retrieval/vector_store.py  
-----------------------------  
Vector store implementation with ChromaDB persistence and duplicate detection.  
"""  
  
import logging  
import hashlib  
import json  
import os  
from pathlib import Path  
from typing import List, Dict, Any, Optional, Set  
from dataclasses import dataclass, field  
from datetime import datetime  
  
import numpy as np  
  
logger = logging.getLogger(__name__)  
  
  
@dataclass  
class Document:  
    """A document with content and metadata."""  
    content: str  
    metadata: Dict[str, Any] = field(default_factory=dict)  
    embedding: Optional[List[float]] = None  
      
    @property  
    def id(self) -> str:  
        """Generate unique ID based on content and key metadata."""  
        # Create hash from content + filename + page + chunk_id  
        hash_input = self.content  
        if self.metadata:  
            hash_input += str(self.metadata.get("filename", ""))  
            hash_input += str(self.metadata.get("page_number", ""))  
            hash_input += str(self.metadata.get("chunk_id", ""))  
        return hashlib.md5(hash_input.encode()).hexdigest()  
  
  
@dataclass   
class SearchResult:  
    """A search result with score."""  
    content: str  
    metadata: Dict[str, Any]  
    score: float  
    document_id: str = ""  
  
  
class VectorStore:  
    """  
    Vector store with ChromaDB backend, persistence, and duplicate detection.  
      
    Features:  
    - Persistent storage (survives restarts)  
    - Document deduplication  
    - Embedding caching  
    - Metadata filtering  
    """  
      
    def __init__(self, config=None, embedder=None):  
        """  
        Initialize the vector store.  
          
        Args:  
            config: Configuration object  
            embedder: Embedder instance for generating embeddings  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self.embedder = embedder  
          
        # Persistence settings  
        self.persist_directory = Path(  
            getattr(self.config, "CHROMA_PERSIST_DIR", "./chroma_db")  
        )  
        self.collection_name = getattr(self.config, "CHROMA_COLLECTION", "rag_documents")  
          
        # Document tracking  
        self._indexed_documents: Set[str] = set()  # Track indexed doc IDs  
        self._document_registry_path = self.persist_directory / "document_registry.json"  
          
        # Initialize ChromaDB  
        self._client = None  
        self._collection = None  
        self._initialize_chromadb()  
          
        # Load document registry  
        self._load_document_registry()  
          
        logger.info(  
            f"VectorStore initialized: {len(self._indexed_documents)} documents tracked, "  
            f"persist_dir={self.persist_directory}"  
        )  
      
    def _initialize_chromadb(self):  
        """Initialize ChromaDB with the new API."""  
        try:  
            import chromadb  
            from chromadb.config import Settings  
              
            # Ensure directory exists  
            self.persist_directory.mkdir(parents=True, exist_ok=True)  
              
            # Try new PersistentClient API first (ChromaDB >= 0.4.0)  
            try:  
                self._client = chromadb.PersistentClient(  
                    path=str(self.persist_directory)  
                )  
                logger.info(f"ChromaDB PersistentClient initialized at {self.persist_directory}")  
                  
            except (TypeError, AttributeError):  
                # Fall back to older API (ChromaDB < 0.4.0)  
                logger.warning("PersistentClient not available, trying legacy API")  
                try:  
                    self._client = chromadb.Client(Settings(  
                        chroma_db_impl="duckdb+parquet",  
                        persist_directory=str(self.persist_directory),  
                        anonymized_telemetry=False  
                    ))  
                    logger.info("ChromaDB legacy client initialized")  
                except Exception as e:  
                    logger.warning(f"Legacy API failed: {e}, using in-memory")  
                    self._client = chromadb.Client()  
              
            # Get or create collection  
            self._collection = self._client.get_or_create_collection(  
                name=self.collection_name,  
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity  
            )  
              
            # Log collection stats  
            count = self._collection.count()  
            logger.info(f"Collection '{self.collection_name}' has {count} vectors")  
              
        except ImportError:  
            logger.error("ChromaDB not installed. Install with: pip install chromadb")  
            raise  
        except Exception as e:  
            logger.error(f"ChromaDB initialization failed: {e}")  
            # Create in-memory fallback  
            self._create_in_memory_fallback()  
      
    def _create_in_memory_fallback(self):  
        """Create in-memory fallback when ChromaDB fails."""  
        logger.warning("Using in-memory vector store (data will not persist)")  
          
        try:  
            import chromadb  
            self._client = chromadb.Client()  
            self._collection = self._client.get_or_create_collection(  
                name=self.collection_name,  
                metadata={"hnsw:space": "cosine"}  
            )  
        except Exception as e:  
            logger.error(f"Even in-memory ChromaDB failed: {e}")  
            # Ultimate fallback: simple list-based store  
            self._collection = None  
            self._documents: List[Document] = []  
      
    def _load_document_registry(self):  
        """Load the document registry from disk."""  
        try:  
            if self._document_registry_path.exists():  
                with open(self._document_registry_path, "r", encoding="utf-8") as f:  
                    data = json.load(f)  
                    self._indexed_documents = set(data.get("document_ids", []))  
                    logger.debug(f"Loaded {len(self._indexed_documents)} document IDs from registry")  
        except Exception as e:  
            logger.warning(f"Failed to load document registry: {e}")  
            self._indexed_documents = set()  
      
    def _save_document_registry(self):  
        """Save the document registry to disk."""  
        try:  
            self.persist_directory.mkdir(parents=True, exist_ok=True)  
              
            data = {  
                "document_ids": list(self._indexed_documents),  
                "updated_at": datetime.now().isoformat(),  
                "count": len(self._indexed_documents)  
            }  
              
            with open(self._document_registry_path, "w", encoding="utf-8") as f:  
                json.dump(data, f, indent=2)  
                  
            logger.debug(f"Saved {len(self._indexed_documents)} document IDs to registry")  
        except Exception as e:  
            logger.warning(f"Failed to save document registry: {e}")  
      
    def _get_document_fingerprint(self, filename: str) -> str:  
        """  
        Generate a fingerprint for a document based on filename.  
          
        This is used to track which files have been indexed.  
        """  
        return hashlib.md5(filename.encode()).hexdigest()  
      
    def is_document_indexed(self, filename: str) -> bool:  
        """  
        Check if a document has already been indexed.  
          
        Args:  
            filename: The document filename  
              
        Returns:  
            True if document is already indexed  
        """  
        fingerprint = self._get_document_fingerprint(filename)  
        return fingerprint in self._indexed_documents  
      
    def get_indexed_documents(self) -> List[str]:  
        """  
        Get list of indexed document fingerprints.  
          
        Returns:  
            List of document fingerprints  
        """  
        return list(self._indexed_documents)  
      
    def add_documents(  
        self,  
        documents: List[Dict[str, Any]],  
        skip_duplicates: bool = True  
    ) -> Dict[str, Any]:  
        """  
        Add documents to the vector store.  
          
        Args:  
            documents: List of document dicts with 'content' and 'metadata'  
            skip_duplicates: If True, skip documents from already-indexed files  
              
        Returns:  
            Dict with status and statistics  
        """  
        if not documents:  
            return {  
                "status": "success",  
                "added": 0,  
                "skipped": 0,  
                "message": "No documents to add"  
            }  
          
        # Check for duplicate documents by filename  
        if skip_duplicates:  
            # Get unique filenames from incoming documents  
            filenames = set()  
            for doc in documents:  
                filename = doc.get("metadata", {}).get("filename", "")  
                if filename:  
                    filenames.add(filename)  
              
            # Check which are already indexed  
            already_indexed = []  
            for filename in filenames:  
                if self.is_document_indexed(filename):  
                    already_indexed.append(filename)  
              
            if already_indexed:  
                logger.info(  
                    f"Skipping {len(already_indexed)} already-indexed document(s): "  
                    f"{already_indexed}"  
                )  
                  
                # Filter out documents from already-indexed files  
                documents = [  
                    doc for doc in documents  
                    if doc.get("metadata", {}).get("filename", "") not in already_indexed  
                ]  
                  
                if not documents:  
                    return {  
                        "status": "success",  
                        "added": 0,  
                        "skipped": len(already_indexed),  
                        "message": "All documents already indexed",  
                        "skipped_files": already_indexed  
                    }  
          
        # Convert to Document objects  
        docs = []  
        for doc in documents:  
            docs.append(Document(  
                content=doc.get("content", ""),  
                metadata=doc.get("metadata", {})  
            ))  
          
        # Generate embeddings  
        if self.embedder is None:  
            from src.embedding.embedder import Embedder  
            self.embedder = Embedder(self.config)  
          
        logger.info(f"Generating embeddings for {len(docs)} documents...")  
          
        texts = [doc.content for doc in docs]  
        embeddings = self.embedder.embed_batch(texts)  
          
        # Assign embeddings to documents  
        for doc, embedding in zip(docs, embeddings):  
            doc.embedding = embedding  
          
        # Add to ChromaDB  
        added_count = self._add_to_chromadb(docs)  
          
        # Update document registry  
        new_filenames = set()  
        for doc in docs:  
            filename = doc.metadata.get("filename", "")  
            if filename:  
                new_filenames.add(filename)  
          
        for filename in new_filenames:  
            fingerprint = self._get_document_fingerprint(filename)  
            self._indexed_documents.add(fingerprint)  
          
        # Save registry  
        self._save_document_registry()  
          
        logger.info(f"Added {added_count} documents to vector store")  
          
        return {  
            "status": "success",  
            "added": added_count,  
            "skipped": 0,  
            "message": f"Added {added_count} documents",  
            "new_files": list(new_filenames)  
        }  
      
    def _add_to_chromadb(self, documents: List[Document]) -> int:  
        """  
        Add documents to ChromaDB collection.  
          
        Args:  
            documents: List of Document objects with embeddings  
              
        Returns:  
            Number of documents added  
        """  
        if self._collection is None:  
            # Fallback to simple list storage  
            self._documents.extend(documents)  
            return len(documents)  
          
        # Prepare data for ChromaDB  
        ids = []  
        embeddings = []  
        metadatas = []  
        contents = []  
          
        for doc in documents:  
            if doc.embedding is None:  
                logger.warning(f"Document missing embedding, skipping")  
                continue  
              
            doc_id = doc.id  
              
            # Check if ID already exists (shouldn't happen with our dedup, but safety check)  
            try:  
                existing = self._collection.get(ids=[doc_id])  
                if existing and existing.get("ids"):  
                    logger.debug(f"Document {doc_id[:8]} already in collection, skipping")  
                    continue  
            except Exception:  
                pass  # ID doesn't exist, which is expected  
              
            ids.append(doc_id)  
            embeddings.append(doc.embedding)  
            contents.append(doc.content)  
              
            # Clean metadata for ChromaDB (must be flat dict with simple types)  
            clean_metadata = {}  
            for key, value in doc.metadata.items():  
                if isinstance(value, (str, int, float, bool)):  
                    clean_metadata[key] = value  
                elif value is None:  
                    clean_metadata[key] = ""  
                else:  
                    clean_metadata[key] = str(value)  
            metadatas.append(clean_metadata)  
          
        if not ids:  
            return 0  
          
        # Add to collection in batches  
        batch_size = 100  
        added = 0  
          
        for i in range(0, len(ids), batch_size):  
            batch_ids = ids[i:i + batch_size]  
            batch_embeddings = embeddings[i:i + batch_size]  
            batch_metadatas = metadatas[i:i + batch_size]  
            batch_contents = contents[i:i + batch_size]  
              
            try:  
                self._collection.add(  
                    ids=batch_ids,  
                    embeddings=batch_embeddings,  
                    metadatas=batch_metadatas,  
                    documents=batch_contents  
                )  
                added += len(batch_ids)  
            except Exception as e:  
                logger.error(f"Failed to add batch to ChromaDB: {e}")  
          
        return added  
      
    def search(  
        self,  
        query: str,  
        top_k: int = 10,  
        filter_metadata: Optional[Dict[str, Any]] = None  
    ) -> List[SearchResult]:  
        """  
        Search for similar documents.  
          
        Args:  
            query: Search query  
            top_k: Number of results to return  
            filter_metadata: Optional metadata filter  
              
        Returns:  
            List of SearchResult objects  
        """  
        if self.embedder is None:  
            from src.embedding.embedder import Embedder  
            self.embedder = Embedder(self.config)  
          
        # Generate query embedding  
        query_embedding = self.embedder.embed(query)  
          
        return self.search_by_vector(  
            query_embedding=query_embedding,  
            top_k=top_k,  
            filter_metadata=filter_metadata  
        )  
      
    def search_by_vector(  
        self,  
        query_embedding: List[float],  
        top_k: int = 10,  
        filter_metadata: Optional[Dict[str, Any]] = None  
    ) -> List[SearchResult]:  
        """  
        Search by embedding vector.  
          
        Args:  
            query_embedding: Query embedding vector  
            top_k: Number of results to return  
            filter_metadata: Optional metadata filter  
              
        Returns:  
            List of SearchResult objects  
        """  
        if self._collection is None:  
            return self._search_fallback(query_embedding, top_k)  
          
        try:  
            # Build where clause for filtering  
            where = None  
            if filter_metadata:  
                where = {}  
                for key, value in filter_metadata.items():  
                    where[key] = value  
              
            # Query ChromaDB  
            results = self._collection.query(  
                query_embeddings=[query_embedding],  
                n_results=top_k,  
                where=where,  
                include=["documents", "metadatas", "distances"]  
            )  
              
            # Convert to SearchResult objects  
            search_results = []  
              
            if results and results.get("documents"):  
                documents = results["documents"][0]  
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)  
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)  
                ids = results["ids"][0] if results.get("ids") else [""] * len(documents)  
                  
                for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):  
                    # Convert distance to similarity score  
                    # ChromaDB returns L2 distance by default, or cosine distance if configured  
                    # For cosine distance: similarity = 1 - distance  
                    score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)  
                      
                    search_results.append(SearchResult(  
                        content=doc,  
                        metadata=metadata or {},  
                        score=score,  
                        document_id=doc_id  
                    ))  
              
            return search_results  
              
        except Exception as e:  
            logger.error(f"ChromaDB search failed: {e}")  
            return []  
      
    def _search_fallback(  
        self,  
        query_embedding: List[float],  
        top_k: int  
    ) -> List[SearchResult]:  
        """  
        Fallback search using simple cosine similarity.  
          
        Args:  
            query_embedding: Query embedding vector  
            top_k: Number of results  
              
        Returns:  
            List of SearchResult objects  
        """  
        if not hasattr(self, "_documents") or not self._documents:  
            return []  
          
        # Calculate cosine similarity for all documents  
        query_vec = np.array(query_embedding)  
        query_norm = np.linalg.norm(query_vec)  
          
        if query_norm == 0:  
            return []  
          
        results = []  
        for doc in self._documents:  
            if doc.embedding is None:  
                continue  
              
            doc_vec = np.array(doc.embedding)  
            doc_norm = np.linalg.norm(doc_vec)  
              
            if doc_norm == 0:  
                continue  
              
            similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)  
              
            results.append(SearchResult(  
                content=doc.content,  
                metadata=doc.metadata,  
                score=float(similarity),  
                document_id=doc.id  
            ))  
          
        # Sort by score descending  
        results.sort(key=lambda x: x.score, reverse=True)  
          
        return results[:top_k]  
      
    def delete_document(self, filename: str) -> Dict[str, Any]:  
        """  
        Delete all chunks from a document.  
          
        Args:  
            filename: The document filename to delete  
              
        Returns:  
            Dict with status and count of deleted items  
        """  
        if self._collection is None:  
            return {"status": "error", "message": "No collection available"}  
          
        try:  
            # Find all documents with this filename  
            results = self._collection.get(  
                where={"filename": filename},  
                include=["metadatas"]  
            )  
              
            if not results or not results.get("ids"):  
                return {  
                    "status": "success",  
                    "deleted": 0,  
                    "message": f"No documents found for {filename}"  
                }  
              
            # Delete the documents  
            ids_to_delete = results["ids"]  
            self._collection.delete(ids=ids_to_delete)  
              
            # Remove from registry  
            fingerprint = self._get_document_fingerprint(filename)  
            self._indexed_documents.discard(fingerprint)  
            self._save_document_registry()  
              
            logger.info(f"Deleted {len(ids_to_delete)} chunks for {filename}")  
              
            return {  
                "status": "success",  
                "deleted": len(ids_to_delete),  
                "message": f"Deleted {len(ids_to_delete)} chunks"  
            }  
              
        except Exception as e:  
            logger.error(f"Delete failed: {e}")  
            return {"status": "error", "message": str(e)}  
      
    def clear(self) -> Dict[str, Any]:  
        """  
        Clear all documents from the vector store.  
          
        Returns:  
            Dict with status  
        """  
        try:  
            if self._client is not None:  
                # Delete and recreate collection  
                self._client.delete_collection(self.collection_name)  
                self._collection = self._client.create_collection(  
                    name=self.collection_name,  
                    metadata={"hnsw:space": "cosine"}  
                )  
              
            # Clear registry  
            self._indexed_documents.clear()  
            self._save_document_registry()  
              
            # Clear fallback storage  
            if hasattr(self, "_documents"):  
                self._documents.clear()  
              
            logger.info("Vector store cleared")  
              
            return {"status": "success", "message": "Vector store cleared"}  
              
        except Exception as e:  
            logger.error(f"Clear failed: {e}")  
            return {"status": "error", "message": str(e)}  
      
    def get_stats(self) -> Dict[str, Any]:  
        """  
        Get vector store statistics.  
          
        Returns:  
            Dict with statistics  
        """  
        stats = {  
            "indexed_documents": len(self._indexed_documents),  
            "persist_directory": str(self.persist_directory),  
            "collection_name": self.collection_name  
        }  
          
        if self._collection is not None:  
            try:  
                stats["total_vectors"] = self._collection.count()  
            except Exception:  
                stats["total_vectors"] = "unknown"  
        elif hasattr(self, "_documents"):  
            stats["total_vectors"] = len(self._documents)  
        else:  
            stats["total_vectors"] = 0  
          
        return stats  