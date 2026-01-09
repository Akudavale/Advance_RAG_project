# """  
# src/retrieval/reranker.py  
# -------------------------  
# Document reranking using cross-encoder models.  
# """  
  
# import logging  
# from typing import List, Dict, Any, Optional  
# import threading  
  
# logger = logging.getLogger(__name__)  
  
  
# class Reranker:  
#     """  
#     Reranks retrieved documents using a cross-encoder model.  
      
#     Cross-encoders provide more accurate relevance scores than  
#     bi-encoders but are slower (O(n) vs O(1) per query-doc pair).  
#     """  
      
#     def __init__(self, config=None):  
#         """  
#         Initialize the reranker.  
          
#         Args:  
#             config: Configuration object  
#         """  
#         from config.config import Config  
          
#         self.config = config or Config()  
#         self.model_name = getattr(self.config, "RERANKER_MODEL", "BAAI/bge-reranker-large")  
#         self.device = getattr(self.config, "RERANKER_DEVICE", "cpu")  
#         self.default_top_k = getattr(self.config, "TOP_K_RERANK", 5)  
          
#         self.model = None  
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
#                 from sentence_transformers import CrossEncoder  
                  
#                 logger.info(f"Loading reranker model: {self.model_name}")  
#                 self.model = CrossEncoder(  
#                     self.model_name,  
#                     device=self.device,  
#                     max_length=512  
#                 )  
#                 self._initialized = True  
#                 logger.info("Reranker model loaded successfully")  
                  
#             except Exception as e:  
#                 logger.error(f"Failed to load reranker model: {e}")  
#                 raise RuntimeError(f"Failed to initialize reranker: {e}")  
      
#     def rerank(  
#         self,  
#         query: str,  
#         documents: List[Dict[str, Any]],  
#         top_k: Optional[int] = None  
#     ) -> List[Dict[str, Any]]:  
#         """  
#         Rerank documents based on relevance to query.  
          
#         Args:  
#             query: User query  
#             documents: List of documents with 'content' key  
#             top_k: Number of top documents to return  
              
#         Returns:  
#             Reranked documents with updated scores  
#         """  
#         if not documents:  
#             return []  
          
#         if not query or not query.strip():  
#             logger.warning("Empty query provided to reranker")  
#             return documents[:top_k] if top_k else documents  
          
#         top_k = top_k or self.default_top_k  
          
#         self._ensure_initialized()  
          
#         try:  
#             # Prepare query-document pairs  
#             pairs = []  
#             valid_indices = []  
              
#             for i, doc in enumerate(documents):  
#                 # Support both 'content' and 'text' keys  
#                 content = doc.get("content") or doc.get("text", "")  
#                 if content and content.strip():  
#                     pairs.append([query, content])  
#                     valid_indices.append(i)  
              
#             if not pairs:  
#                 logger.warning("No valid documents to rerank")  
#                 return documents[:top_k]  
              
#             # Get reranking scores  
#             scores = self.model.predict(pairs)  
              
#             # Combine with original documents  
#             scored_docs = []  
#             for idx, (orig_idx, score) in enumerate(zip(valid_indices, scores)):  
#                 doc = documents[orig_idx].copy()  
#                 doc["rerank_score"] = float(score)  
#                 # Preserve original retrieval score if present  
#                 if "score" in doc:  
#                     doc["retrieval_score"] = doc["score"]  
#                 doc["score"] = float(score)  # Use rerank score as primary  
#                 scored_docs.append(doc)  
              
#             # Sort by rerank score (descending)  
#             scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)  
              
#             # Return top-k  
#             result = scored_docs[:top_k]  
              
#             logger.debug(f"Reranked {len(documents)} docs, returning top {len(result)}")  
#             return result  
              
#         except Exception as e:  
#             logger.error(f"Reranking failed: {e}")  
#             # Fallback: return original documents  
#             return documents[:top_k]  
      
#     def score_pair(self, query: str, document: str) -> float:  
#         """  
#         Score a single query-document pair.  
          
#         Args:  
#             query: Query string  
#             document: Document string  
              
#         Returns:  
#             Relevance score  
#         """  
#         self._ensure_initialized()  
          
#         try:  
#             score = self.model.predict([[query, document]])[0]  
#             return float(score)  
#         except Exception as e:  
#             logger.error(f"Pair scoring failed: {e}")  
#             return 0.0  

"""  
src/retrieval/reranker.py  
-------------------------  
Reranker for improving search result relevance.  
"""  
  
import logging  
from typing import List, Dict, Any, Union, Optional  
from dataclasses import dataclass  
  
logger = logging.getLogger(__name__)  
  
  
@dataclass  
class SearchResult:  
    """A search result with score."""  
    content: str  
    metadata: Dict[str, Any]  
    score: float  
    document_id: str = ""  
  
  
class Reranker:  
    """  
    Reranker using cross-encoder models for improved relevance ranking.  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the reranker.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
          
        # Model settings  
        self.model_name = getattr(  
            self.config,  
            "RERANKER_MODEL",  
            "BAAI/bge-reranker-base"  
        )  
        self.device = getattr(self.config, "RERANKER_DEVICE", None)  
        self.use_cross_encoder = getattr(self.config, "USE_CROSS_ENCODER", True)  
          
        # Lazy load model  
        self._model = None  
        self._model_loaded = False  
        self._model_load_attempted = False  
          
        logger.info(f"Reranker initialized with model: {self.model_name}")  
      
    @property  
    def model(self):  
        """Lazy load the reranker model."""  
        if not self._model_load_attempted:  
            self._load_model()  
        return self._model  
      
    def _load_model(self):  
        """Load the cross-encoder model."""  
        self._model_load_attempted = True  
          
        if not self.use_cross_encoder:  
            logger.info("Cross-encoder disabled, using embedding similarity")  
            return  
          
        try:  
            from sentence_transformers import CrossEncoder  
            import torch  
              
            # Determine device  
            if self.device is None:  
                self.device = "cuda" if torch.cuda.is_available() else "cpu"  
              
            logger.info(f"Loading reranker model: {self.model_name}")  
              
            self._model = CrossEncoder(  
                self.model_name,  
                device=self.device,  
                max_length=512  
            )  
              
            self._model_loaded = True  
            logger.info(f"Reranker model loaded successfully on {self.device}")  
              
        except ImportError as e:  
            logger.warning(  
                f"CrossEncoder import failed: {e}. "  
                "Falling back to original scores."  
            )  
            self._model_loaded = False  
        except Exception as e:  
            logger.warning(f"Failed to load reranker model: {e}. Using fallback.")  
            self._model_loaded = False  
      
    def rerank(  
        self,  
        query: str,  
        results: List[Union[SearchResult, Dict[str, Any]]],  
        top_k: Optional[int] = None  
    ) -> List[SearchResult]:  
        """  
        Rerank search results based on relevance to query.  
          
        Args:  
            query: The search query  
            results: List of SearchResult objects or dicts  
            top_k: Number of top results to return (None = return all)  
              
        Returns:  
            Reranked list of SearchResult objects  
        """  
        if not results:  
            return []  
          
        # Convert to SearchResult objects  
        search_results = self._convert_to_search_results(results)  
          
        if not query or not query.strip():  
            logger.warning("Empty query for reranking")  
            return search_results[:top_k] if top_k else search_results  
          
        # Try cross-encoder reranking  
        if self.use_cross_encoder:  
            # Ensure model is loaded  
            _ = self.model  
              
            if self._model_loaded and self._model is not None:  
                reranked = self._rerank_cross_encoder(query, search_results, top_k)  
                if reranked:  
                    return reranked  
          
        # Fallback: return original results sorted by score  
        logger.info("Using original retrieval scores (cross-encoder not available)")  
        search_results.sort(key=lambda x: x.score, reverse=True)  
          
        if top_k:  
            return search_results[:top_k]  
        return search_results  
      
    def _convert_to_search_results(  
        self,   
        results: List[Union[SearchResult, Dict[str, Any]]]  
    ) -> List[SearchResult]:  
        """Convert results to SearchResult objects."""  
        converted = []  
        for r in results:  
            if isinstance(r, SearchResult):  
                converted.append(r)  
            elif isinstance(r, dict):  
                converted.append(SearchResult(  
                    content=r.get("content", ""),  
                    metadata=r.get("metadata", {}),  
                    score=r.get("score", 0.0),  
                    document_id=r.get("document_id", "")  
                ))  
            else:  
                try:  
                    converted.append(SearchResult(  
                        content=getattr(r, "content", ""),  
                        metadata=getattr(r, "metadata", {}),  
                        score=getattr(r, "score", 0.0),  
                        document_id=getattr(r, "document_id", "")  
                    ))  
                except Exception as e:  
                    logger.warning(f"Could not convert result: {e}")  
        return converted  
      
    def _rerank_cross_encoder(  
        self,  
        query: str,  
        results: List[SearchResult],  
        top_k: Optional[int]  
    ) -> List[SearchResult]:  
        """Rerank using cross-encoder model."""  
        try:  
            import numpy as np  
              
            # Prepare query-document pairs  
            pairs = [(query, r.content) for r in results]  
              
            # Get cross-encoder scores  
            logger.debug(f"Reranking {len(pairs)} documents with cross-encoder")  
            raw_scores = self._model.predict(pairs, show_progress_bar=False)  
              
            logger.debug(f"Raw reranker scores: min={min(raw_scores):.3f}, max={max(raw_scores):.3f}")  
              
            # Normalize scores to 0-1 range  
            # Use min-max normalization instead of sigmoid for better spread  
            scores_array = np.array(raw_scores)  
              
            if len(scores_array) > 1:  
                min_score = scores_array.min()  
                max_score = scores_array.max()  
                  
                if max_score - min_score > 0.001:  # Avoid division by zero  
                    normalized_scores = (scores_array - min_score) / (max_score - min_score)  
                else:  
                    # All scores are similar, use sigmoid  
                    normalized_scores = 1 / (1 + np.exp(-scores_array))  
            else:  
                # Single result, use sigmoid  
                normalized_scores = 1 / (1 + np.exp(-scores_array))  
              
            # Create new results with updated scores  
            reranked = []  
            for result, raw_score, norm_score in zip(results, raw_scores, normalized_scores):  
                reranked.append(SearchResult(  
                    content=result.content,  
                    metadata=result.metadata,  
                    score=float(norm_score),  
                    document_id=result.document_id  
                ))  
              
            # Sort by score descending  
            reranked.sort(key=lambda x: x.score, reverse=True)  
              
            logger.info(  
                f"Reranked {len(results)} results. "  
                f"Top score: {reranked[0].score:.3f}, Bottom score: {reranked[-1].score:.3f}"  
            )  
              
            if top_k:  
                return reranked[:top_k]  
            return reranked  
              
        except Exception as e:  
            logger.error(f"Cross-encoder reranking failed: {e}")  
            return []  