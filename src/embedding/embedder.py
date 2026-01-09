# """  
# src/embedding/embedder.py  
# -------------------------  
# Text embedding module using sentence-transformers.  
# Provides singleton-like access to prevent multiple model loads.  
# """  
  
# import logging  
# from typing import List, Union, Optional  
# import threading  
  
# import numpy as np  
  
# logger = logging.getLogger(__name__)  
  
# # Global embedder instance for sharing across components  
# _global_embedder: Optional["Embedder"] = None  
# _embedder_lock = threading.Lock()  
  
  
# def get_shared_embedder(config=None) -> "Embedder":  
#     """  
#     Get or create a shared embedder instance.  
#     Thread-safe singleton pattern to prevent multiple model loads.  
#     """  
#     global _global_embedder  
      
#     if _global_embedder is None:  
#         with _embedder_lock:  
#             # Double-check locking  
#             if _global_embedder is None:  
#                 _global_embedder = Embedder(config)  
      
#     return _global_embedder  
  
  
# class Embedder:  
#     """  
#     Text embedder using HuggingFace sentence-transformers.  
      
#     Supports:  
#     - Single text embedding  
#     - Batch embedding  
#     - Configurable models  
#     - Dimension detection  
#     """  
      
#     def __init__(self, config=None):  
#         """  
#         Initialize the embedder.  
          
#         Args:  
#             config: Configuration object with embedding settings  
#         """  
#         # Import here to avoid circular imports  
#         from config.config import Config  
          
#         self.config = config or Config()  
#         self.model_name = getattr(self.config, "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")  
#         self.device = getattr(self.config, "EMBEDDING_DEVICE", "cpu")  
#         self.normalize = getattr(self.config, "NORMALIZE_EMBEDDINGS", True)  
          
#         self.model = None  
#         self.dimension = None  
#         self._initialized = False  
#         self._init_lock = threading.Lock()  
      
#     def _ensure_initialized(self):  
#         """Lazy initialization of the model."""  
#         if self._initialized:  
#             return  
          
#         with self._init_lock:  
#             if self._initialized:  
#                 return  
              
#             try:  
#                 from sentence_transformers import SentenceTransformer  
                  
#                 logger.info(f"Loading embedding model: {self.model_name}")  
#                 self.model = SentenceTransformer(  
#                     self.model_name,  
#                     device=self.device  
#                 )  
#                 self.dimension = self.model.get_sentence_embedding_dimension()  
#                 self._initialized = True  
#                 logger.info(f"Embedding model loaded. Dimension: {self.dimension}")  
                  
#             except Exception as e:  
#                 logger.error(f"Failed to load embedding model: {e}")  
#                 raise RuntimeError(f"Failed to initialize embedder: {e}")  
      
#     def embed(self, text: Union[str, List[str]]) -> np.ndarray:  
#         """  
#         Generate embeddings for text(s).  
          
#         Args:  
#             text: Single string or list of strings to embed  
              
#         Returns:  
#             numpy array of embeddings (1D for single text, 2D for batch)  
#         """  
#         self._ensure_initialized()  
          
#         if isinstance(text, str):  
#             texts = [text]  
#             single_input = True  
#         else:  
#             texts = text  
#             single_input = False  
          
#         # Filter empty strings  
#         valid_texts = []  
#         valid_indices = []  
#         for i, t in enumerate(texts):  
#             if t and t.strip():  
#                 valid_texts.append(t.strip())  
#                 valid_indices.append(i)  
          
#         if not valid_texts:  
#             logger.warning("No valid texts to embed")  
#             if single_input:  
#                 return np.zeros(self.dimension)  
#             return np.zeros((len(texts), self.dimension))  
          
#         try:  
#             embeddings = self.model.encode(  
#                 valid_texts,  
#                 normalize_embeddings=self.normalize,  
#                 show_progress_bar=False  
#             )  
              
#             # Handle case where some inputs were empty  
#             if len(valid_texts) < len(texts):  
#                 full_embeddings = np.zeros((len(texts), self.dimension))  
#                 for i, idx in enumerate(valid_indices):  
#                     full_embeddings[idx] = embeddings[i]  
#                 embeddings = full_embeddings  
              
#             if single_input:  
#                 return embeddings[0]  
              
#             return embeddings  
              
#         except Exception as e:  
#             logger.error(f"Embedding generation failed: {e}")  
#             if single_input:  
#                 return np.zeros(self.dimension)  
#             return np.zeros((len(texts), self.dimension))  
      
#     def embed_query(self, query: str) -> np.ndarray:  
#         """  
#         Embed a query string.  
          
#         Some models use different prefixes for queries vs documents.  
          
#         Args:  
#             query: Query string to embed  
              
#         Returns:  
#             Query embedding as numpy array  
#         """  
#         # BGE models benefit from query prefix  
#         if "bge" in self.model_name.lower():  
#             query = f"Represent this sentence for searching relevant passages: {query}"  
          
#         return self.embed(query)  
      
#     def embed_documents(self, documents: List[str]) -> np.ndarray:  
#         """  
#         Embed a list of documents.  
          
#         Args:  
#             documents: List of document strings  
              
#         Returns:  
#             2D numpy array of document embeddings  
#         """  
#         return self.embed(documents)  
      
#     def get_dimension(self) -> int:  
#         """Get the embedding dimension."""  
#         self._ensure_initialized()  
#         return self.dimension  
      
#     def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:  
#         """  
#         Compute cosine similarity between two embeddings.  
          
#         Args:  
#             embedding1: First embedding  
#             embedding2: Second embedding  
              
#         Returns:  
#             Cosine similarity score  
#         """  
#         if self.normalize:  
#             # Already normalized, just dot product  
#             return float(np.dot(embedding1, embedding2))  
          
#         # Compute cosine similarity  
#         norm1 = np.linalg.norm(embedding1)  
#         norm2 = np.linalg.norm(embedding2)  
          
#         if norm1 == 0 or norm2 == 0:  
#             return 0.0  
          
#         return float(np.dot(embedding1, embedding2) / (norm1 * norm2))  


"""  
src/embedding/embedder.py  
-------------------------  
Embedding generation using sentence-transformers with batch support and caching.  
"""  
  
import logging  
import hashlib  
import json  
import os  
from pathlib import Path  
from typing import List, Union, Optional, Dict, Any  
from functools import lru_cache  
  
import numpy as np  
  
logger = logging.getLogger(__name__)  
  
  
class Embedder:  
    """  
    Embedder using sentence-transformers for generating text embeddings.  
      
    Features:  
    - Single and batch embedding  
    - Embedding caching (optional)  
    - Configurable model  
    - GPU/CPU support  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the embedder.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
          
        # Model settings  
        self.model_name = getattr(  
            self.config,   
            "EMBEDDING_MODEL",   
            "BAAI/bge-large-en-v1.5"  
        )  
        self.device = getattr(self.config, "EMBEDDING_DEVICE", None)  
          
        # Caching settings  
        self.cache_enabled = getattr(self.config, "EMBEDDING_CACHE_ENABLED", True)  
        self.cache_dir = Path(getattr(self.config, "EMBEDDING_CACHE_DIR", ".cache/embeddings"))  
          
        # Batch settings  
        self.batch_size = getattr(self.config, "EMBEDDING_BATCH_SIZE", 32)  
          
        # Initialize model  
        self._model = None  
        self._embedding_dim = None  
          
        # In-memory cache for current session  
        self._session_cache: Dict[str, List[float]] = {}  
          
        logger.info(f"Embedder initialized with model: {self.model_name}")  
      
    @property  
    def model(self):  
        """Lazy load the embedding model."""  
        if self._model is None:  
            self._load_model()  
        return self._model  
      
    @property  
    def embedding_dim(self) -> int:  
        """Get the embedding dimension."""  
        if self._embedding_dim is None:  
            # Generate a test embedding to get dimension  
            test_embedding = self.embed("test")  
            self._embedding_dim = len(test_embedding)  
        return self._embedding_dim  
      
    def _load_model(self):  
        """Load the sentence-transformer model."""  
        try:  
            from sentence_transformers import SentenceTransformer  
              
            logger.info(f"Loading embedding model: {self.model_name}")  
              
            # Determine device  
            if self.device is None:  
                import torch  
                self.device = "cuda" if torch.cuda.is_available() else "cpu"  
              
            self._model = SentenceTransformer(  
                self.model_name,  
                device=self.device  
            )  
              
            logger.info(f"Embedding model loaded on {self.device}")  
              
        except ImportError:  
            logger.error("sentence-transformers not installed")  
            logger.error("Install with: pip install sentence-transformers")  
            raise  
        except Exception as e:  
            logger.error(f"Failed to load embedding model: {e}")  
            raise  
      
    def _get_cache_key(self, text: str) -> str:  
        """  
        Generate a cache key for text.  
          
        Args:  
            text: Input text  
              
        Returns:  
            Cache key (MD5 hash)  
        """  
        # Include model name in cache key  
        cache_input = f"{self.model_name}:{text}"  
        return hashlib.md5(cache_input.encode()).hexdigest()  
      
    def _get_from_cache(self, text: str) -> Optional[List[float]]:  
        """  
        Get embedding from cache.  
          
        Args:  
            text: Input text  
              
        Returns:  
            Cached embedding or None  
        """  
        if not self.cache_enabled:  
            return None  
          
        cache_key = self._get_cache_key(text)  
          
        # Check session cache first  
        if cache_key in self._session_cache:  
            return self._session_cache[cache_key]  
          
        # Check disk cache  
        cache_file = self.cache_dir / f"{cache_key}.npy"  
        if cache_file.exists():  
            try:  
                embedding = np.load(cache_file).tolist()  
                self._session_cache[cache_key] = embedding  
                return embedding  
            except Exception as e:  
                logger.warning(f"Failed to load cached embedding: {e}")  
          
        return None  
      
    def _save_to_cache(self, text: str, embedding: List[float]):  
        """  
        Save embedding to cache.  
          
        Args:  
            text: Input text  
            embedding: Embedding vector  
        """  
        if not self.cache_enabled:  
            return  
          
        cache_key = self._get_cache_key(text)  
          
        # Save to session cache  
        self._session_cache[cache_key] = embedding  
          
        # Save to disk cache  
        try:  
            self.cache_dir.mkdir(parents=True, exist_ok=True)  
            cache_file = self.cache_dir / f"{cache_key}.npy"  
            np.save(cache_file, np.array(embedding))  
        except Exception as e:  
            logger.warning(f"Failed to save embedding to cache: {e}")  
      
    def embed(self, text: str) -> List[float]:  
        """  
        Generate embedding for a single text.  
          
        Args:  
            text: Input text  
              
        Returns:  
            Embedding vector as list of floats  
        """  
        if not text or not text.strip():  
            logger.warning("Empty text provided for embedding")  
            # Return zero vector of appropriate dimension  
            if self._embedding_dim:  
                return [0.0] * self._embedding_dim  
            # Default dimension for BGE models  
            return [0.0] * 1024  
          
        text = text.strip()  
          
        # Check cache  
        cached = self._get_from_cache(text)  
        if cached is not None:  
            return cached  
          
        # Generate embedding  
        try:  
            embedding = self.model.encode(  
                text,  
                convert_to_numpy=True,  
                normalize_embeddings=True,  
                show_progress_bar=False  
            )  
              
            embedding_list = embedding.tolist()  
              
            # Cache the result  
            self._save_to_cache(text, embedding_list)  
              
            return embedding_list  
              
        except Exception as e:  
            logger.error(f"Embedding generation failed: {e}")  
            raise  
      
    def embed_batch(self, texts: List[str]) -> List[List[float]]:  
        """  
        Generate embeddings for multiple texts.  
          
        Args:  
            texts: List of input texts  
              
        Returns:  
            List of embedding vectors  
        """  
        if not texts:  
            return []  
          
        # Separate cached and uncached texts  
        results = [None] * len(texts)  
        uncached_indices = []  
        uncached_texts = []  
          
        for i, text in enumerate(texts):  
            if not text or not text.strip():  
                # Empty text - use zero vector  
                if self._embedding_dim:  
                    results[i] = [0.0] * self._embedding_dim  
                else:  
                    results[i] = [0.0] * 1024  
                continue  
              
            text = text.strip()  
              
            # Check cache  
            cached = self._get_from_cache(text)  
            if cached is not None:  
                results[i] = cached  
            else:  
                uncached_indices.append(i)  
                uncached_texts.append(text)  
          
        # Generate embeddings for uncached texts  
        if uncached_texts:  
            logger.info(  
                f"Generating embeddings for {len(uncached_texts)} texts "  
                f"({len(texts) - len(uncached_texts)} from cache)"  
            )  
              
            try:  
                # Process in batches  
                all_embeddings = []  
                  
                for batch_start in range(0, len(uncached_texts), self.batch_size):  
                    batch_end = min(batch_start + self.batch_size, len(uncached_texts))  
                    batch_texts = uncached_texts[batch_start:batch_end]  
                      
                    batch_embeddings = self.model.encode(  
                        batch_texts,  
                        convert_to_numpy=True,  
                        normalize_embeddings=True,  
                        show_progress_bar=len(uncached_texts) > 100,  
                        batch_size=self.batch_size  
                    )  
                      
                    all_embeddings.extend(batch_embeddings.tolist())  
                      
                    # Log progress for large batches  
                    if len(uncached_texts) > 100:  
                        progress = min(batch_end, len(uncached_texts))  
                        logger.info(f"Embedded {progress}/{len(uncached_texts)} texts")  
                  
                # Place embeddings in results and cache them  
                for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):  
                    embedding = all_embeddings[i]  
                    results[idx] = embedding  
                    self._save_to_cache(text, embedding)  
                      
            except Exception as e:  
                logger.error(f"Batch embedding failed: {e}")  
                raise  
          
        # Update embedding dimension if not set  
        if self._embedding_dim is None and results and results[0]:  
            self._embedding_dim = len(results[0])  
          
        return results  
      
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  
        """  
        Generate embeddings for a list of documents.  
          
        Args:  
            documents: List of document dicts with 'content' field  
              
        Returns:  
            Documents with 'embedding' field added  
        """  
        # Extract texts  
        texts = [doc.get("content", "") for doc in documents]  
          
        # Generate embeddings  
        embeddings = self.embed_batch(texts)  
          
        # Add embeddings to documents  
        for doc, embedding in zip(documents, embeddings):  
            doc["embedding"] = embedding  
          
        return documents  
      
    def similarity(self, text1: str, text2: str) -> float:  
        """  
        Calculate cosine similarity between two texts.  
          
        Args:  
            text1: First text  
            text2: Second text  
              
        Returns:  
            Cosine similarity score (0 to 1)  
        """  
        emb1 = np.array(self.embed(text1))  
        emb2 = np.array(self.embed(text2))  
          
        # Cosine similarity  
        dot_product = np.dot(emb1, emb2)  
        norm1 = np.linalg.norm(emb1)  
        norm2 = np.linalg.norm(emb2)  
          
        if norm1 == 0 or norm2 == 0:  
            return 0.0  
          
        return float(dot_product / (norm1 * norm2))  
      
    def similarity_batch(  
        self,   
        query: str,   
        documents: List[str]  
    ) -> List[float]:  
        """  
        Calculate similarity between a query and multiple documents.  
          
        Args:  
            query: Query text  
            documents: List of document texts  
              
        Returns:  
            List of similarity scores  
        """  
        query_embedding = np.array(self.embed(query))  
        doc_embeddings = np.array(self.embed_batch(documents))  
          
        # Calculate cosine similarities  
        query_norm = np.linalg.norm(query_embedding)  
        if query_norm == 0:  
            return [0.0] * len(documents)  
          
        similarities = []  
        for doc_emb in doc_embeddings:  
            doc_norm = np.linalg.norm(doc_emb)  
            if doc_norm == 0:  
                similarities.append(0.0)  
            else:  
                sim = np.dot(query_embedding, doc_emb) / (query_norm * doc_norm)  
                similarities.append(float(sim))  
          
        return similarities  
      
    def clear_cache(self):  
        """Clear all cached embeddings."""  
        # Clear session cache  
        self._session_cache.clear()  
          
        # Clear disk cache  
        if self.cache_dir.exists():  
            import shutil  
            try:  
                shutil.rmtree(self.cache_dir)  
                logger.info("Embedding cache cleared")  
            except Exception as e:  
                logger.warning(f"Failed to clear embedding cache: {e}")  
      
    def get_cache_stats(self) -> Dict[str, Any]:  
        """  
        Get cache statistics.  
          
        Returns:  
            Dict with cache stats  
        """  
        stats = {  
            "session_cache_size": len(self._session_cache),  
            "cache_enabled": self.cache_enabled,  
            "cache_dir": str(self.cache_dir)  
        }  
          
        # Count disk cache files  
        if self.cache_dir.exists():  
            cache_files = list(self.cache_dir.glob("*.npy"))  
            stats["disk_cache_size"] = len(cache_files)  
              
            # Calculate total size  
            total_size = sum(f.stat().st_size for f in cache_files)  
            stats["disk_cache_bytes"] = total_size  
            stats["disk_cache_mb"] = round(total_size / (1024 * 1024), 2)  
        else:  
            stats["disk_cache_size"] = 0  
            stats["disk_cache_bytes"] = 0  
            stats["disk_cache_mb"] = 0  
          
        return stats  