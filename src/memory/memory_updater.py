"""  
src/memory/memory_updater.py  
----------------------------  
Memory update and maintenance utilities.  
"""  
  
import logging  
from typing import Dict, List, Any, Optional  
from datetime import datetime  
  
logger = logging.getLogger(__name__)  
  
  
class MemoryUpdater:  
    """  
    Updates and maintains conversation memory.  
      
    Responsibilities:  
    - Adding new interactions  
    - Updating importance scores  
    - Triggering summarization  
    - Memory cleanup  
    """  
      
    def __init__(self, config=None, conversation_memory=None, memory_retriever=None):  
        """  
        Initialize the memory updater.  
          
        Args:  
            config: Configuration object  
            conversation_memory: ConversationMemory instance  
            memory_retriever: MemoryRetriever instance  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self._conversation_memory = conversation_memory  
        self._memory_retriever = memory_retriever  
      
    @property  
    def conversation_memory(self):  
        """Get or create conversation memory."""  
        if self._conversation_memory is None:  
            from src.memory.conversation_memory import ConversationMemory  
            self._conversation_memory = ConversationMemory(self.config)  
        return self._conversation_memory  
      
    @property  
    def memory_retriever(self):  
        """Get or create memory retriever."""  
        if self._memory_retriever is None:  
            from src.memory.memory_retriever import MemoryRetriever  
            self._memory_retriever = MemoryRetriever(self.config)  
        return self._memory_retriever  
      
    def add_interaction(  
        self,  
        conversation_id: str,  
        user_message: str,  
        assistant_message: str,  
        metadata: Optional[Dict[str, Any]] = None  
    ) -> Dict[str, Any]:  
        """  
        Add a complete interaction (user + assistant) to memory.  
          
        Args:  
            conversation_id: Conversation ID  
            user_message: User's message  
            assistant_message: Assistant's response  
            metadata: Optional metadata for the interaction  
              
        Returns:  
            Status dictionary  
        """  
        try:  
            # Add user message  
            user_msg = self.conversation_memory.add_message(  
                conversation_id=conversation_id,  
                role="user",  
                content=user_message,  
                metadata=metadata or {}  
            )  
              
            # Add assistant message  
            assistant_msg = self.conversation_memory.add_message(  
                conversation_id=conversation_id,  
                role="assistant",  
                content=assistant_message,  
                metadata=metadata or {}  
            )  
              
            if not user_msg or not assistant_msg:  
                return {  
                    "status": "error",  
                    "message": f"Conversation not found: {conversation_id}"  
                }  
              
            # Index messages for semantic retrieval  
            conv = self.conversation_memory.get_conversation(conversation_id)  
            if conv:  
                user_idx = len(conv.messages) - 2  
                assistant_idx = len(conv.messages) - 1  
                  
                self.memory_retriever.index_message(  
                    conversation_id, user_idx, user_message  
                )  
                self.memory_retriever.index_message(  
                    conversation_id, assistant_idx, assistant_message  
                )  
              
            return {  
                "status": "success",  
                "conversation_id": conversation_id,  
                "messages_added": 2  
            }  
              
        except Exception as e:  
            logger.error(f"Failed to add interaction: {e}")  
            return {  
                "status": "error",  
                "message": str(e)  
            }  
      
    def update_message_importance(  
        self,  
        conversation_id: str,  
        message_index: int,  
        importance_score: float  
    ) -> bool:  
        """  
        Update the importance score of a message.  
          
        Args:  
            conversation_id: Conversation ID  
            message_index: Index of message  
            importance_score: New importance score (0-1)  
              
        Returns:  
            Success status  
        """  
        try:  
            importance_score = max(0.0, min(1.0, importance_score))  
            self.conversation_memory.update_importance(  
                conversation_id, message_index, importance_score  
            )  
            return True  
        except Exception as e:  
            logger.error(f"Failed to update importance: {e}")  
            return False  
      
    def auto_update_importance(  
        self,  
        conversation_id: str,  
        query: str  
    ):  
        """  
        Automatically update importance scores based on query relevance.  
          
        Messages that are relevant to recent queries get higher importance.  
          
        Args:  
            conversation_id: Conversation ID  
            query: Recent query  
        """  
        conv = self.conversation_memory.get_conversation(conversation_id)  
        if not conv:  
            return  
          
        # Get relevant messages  
        messages = [m.to_dict() for m in conv.messages]  
        relevant = self.memory_retriever.retrieve_relevant(  
            conversation_id=conversation_id,  
            query=query,  
            messages=messages,  
            top_k=10,  
            min_similarity=0.6  
        )  
          
        # Boost importance of relevant messages  
        for msg in relevant:  
            idx = msg.get("message_index")  
            if idx is not None:  
                current = conv.messages[idx].importance_score  
                boost = msg.get("similarity_score", 0.5) * 0.2  
                new_score = min(1.0, current + boost)  
                conv.messages[idx].importance_score = new_score  
      
    def cleanup_old_conversations(  
        self,  
        max_age_hours: int = 24,  
        max_conversations: int = 100  
    ) -> int:  
        """  
        Clean up old conversations.  
          
        Args:  
            max_age_hours: Maximum age in hours  
            max_conversations: Maximum conversations to keep  
              
        Returns:  
            Number of conversations deleted  
        """  
        # This would need access to internal storage  
        # Placeholder for now  
        logger.info(f"Cleanup requested: max_age={max_age_hours}h, max_count={max_conversations}")  
        return 0  