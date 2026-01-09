"""  
src/memory/memory_retriever.py  
------------------------------  
Semantic memory retrieval for finding relevant past context.  
"""  
  
import logging  
from typing import Dict, List, Any, Optional  
import numpy as np  
  
logger = logging.getLogger(__name__)  
  
  
class MemoryRetriever:  
    """  
    Retrieves semantically relevant messages from conversation history.  
      
    Uses the shared embedder instance to avoid loading multiple models.  
    """  
      
    def __init__(self, config=None, embedder=None):  
        """  
        Initialize the memory retriever.  
          
        Args:  
            config: Configuration object  
            embedder: Shared embedder instance (recommended)  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self._embedder = embedder  
        self._message_embeddings: Dict[str, Dict[int, np.ndarray]] = {}  
      
    @property  
    def embedder(self):  
        """Get or create embedder (lazy loading)."""  
        if self._embedder is None:  
            from src.embedding.embedder import get_shared_embedder  
            self._embedder = get_shared_embedder(self.config)  
        return self._embedder  
      
    def index_message(  
        self,  
        conversation_id: str,  
        message_index: int,  
        content: str  
    ):  
        """  
        Index a message for semantic retrieval.  
          
        Args:  
            conversation_id: Conversation ID  
            message_index: Index of message in conversation  
            content: Message content  
        """  
        if not content or not content.strip():  
            return  
          
        if conversation_id not in self._message_embeddings:  
            self._message_embeddings[conversation_id] = {}  
          
        embedding = self.embedder.embed(content)  
        self._message_embeddings[conversation_id][message_index] = embedding  
      
    def retrieve_relevant(  
        self,  
        conversation_id: str,  
        query: str,  
        messages: List[Dict[str, Any]],  
        top_k: int = 5,  
        min_similarity: float = 0.5  
    ) -> List[Dict[str, Any]]:  
        """  
        Retrieve messages most relevant to a query.  
          
        Args:  
            conversation_id: Conversation ID  
            query: Query to find relevant messages for  
            messages: List of all messages in conversation  
            top_k: Maximum messages to return  
            min_similarity: Minimum similarity threshold  
              
        Returns:  
            List of relevant messages with similarity scores  
        """  
        if not query or not messages:  
            return []  
          
        # Get query embedding  
        query_embedding = self.embedder.embed_query(query)  
          
        # Get or compute message embeddings  
        conv_embeddings = self._message_embeddings.get(conversation_id, {})  
          
        # Compute similarities  
        similarities = []  
        for i, msg in enumerate(messages):  
            content = msg.get("content", "")  
            if not content:  
                continue  
              
            # Get cached embedding or compute  
            if i in conv_embeddings:  
                msg_embedding = conv_embeddings[i]  
            else:  
                msg_embedding = self.embedder.embed(content)  
                if conversation_id not in self._message_embeddings:  
                    self._message_embeddings[conversation_id] = {}  
                self._message_embeddings[conversation_id][i] = msg_embedding  
              
            # Compute similarity  
            sim = self.embedder.similarity(query_embedding, msg_embedding)  
              
            if sim >= min_similarity:  
                similarities.append((i, sim, msg))  
          
        # Sort by similarity and return top-k  
        similarities.sort(key=lambda x: x[1], reverse=True)  
          
        results = []  
        for idx, sim, msg in similarities[:top_k]:  
            result = msg.copy()  
            result["similarity_score"] = float(sim)  
            result["message_index"] = idx  
            results.append(result)  
          
        return results  
      
    def clear_conversation_embeddings(self, conversation_id: str):  
        """Clear cached embeddings for a conversation."""  
        if conversation_id in self._message_embeddings:  
            del self._message_embeddings[conversation_id]  