"""
src/persistence/persistent_memory.py
------------------------------------
Persistent memory layer that bridges PostgreSQL (durable) and Redis (cache).

Provides:
- Transparent caching with PostgreSQL fallback
- Automatic synchronization between layers
- Token-aware context building with Redis caching
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from src.database.connection import get_database
from src.database.operations import (
    ConversationRepository, MessageRepository, DocumentRepository
)
from src.cache import get_redis_client
from src.memory.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class PersistentMemory:
    """
    Unified memory management layer combining PostgreSQL + Redis.
    
    Architecture:
    - PostgreSQL: Source of truth (persistent storage)
    - Redis: Fast access layer (active conversation cache)
    
    Workflow:
    1. New message: Save to PostgreSQL first (durable), then update Redis
    2. Retrieve: Check Redis first (fast), fallback to PostgreSQL if needed
    3. On startup: Rebuild Redis cache from PostgreSQL as needed
    """
    
    def __init__(self, config=None):
        """
        Initialize persistent memory.
        
        Args:
            config: Configuration object
        """
        from config.config import Config
        
        self.config = config or Config()
        self.token_counter = TokenCounter()
        
        # Initialize database connection
        self.db = None
        if self.config.ENABLE_POSTGRES:
            try:
                self.db = get_database(self.config.DB_URL)
                logger.info("✓ PostgreSQL connected for persistent storage")
            except Exception as e:
                logger.error(f"✗ PostgreSQL connection failed: {e}")
                logger.warning("Falling back to in-memory storage")
        
        # Initialize Redis cache
        self.redis = None
        if self.config.ENABLE_REDIS:
            try:
                self.redis = get_redis_client(
                    host=self.config.REDIS_HOST,
                    port=self.config.REDIS_PORT
                )
                logger.info("✓ Redis connected for caching")
            except Exception as e:
                logger.warning(f"Redis not available: {e}. Using fallback (PostgreSQL only).")
        
        # Fallback in-memory storage (if PostgreSQL unavailable)
        self._conversations_fallback: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_token_limit = self.config.MAX_TOKEN_LIMIT
        self.max_recent_turns = self.config.MAX_RECENT_TURNS
        self.summary_threshold = self.config.SUMMARY_THRESHOLD_TOKENS
    
    # ==========================================
    # Conversation Management
    # ==========================================
    
    def create_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        pdf_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation."""
        try:
            if self.db:
                session = self.db.get_session()
                try:
                    repo = ConversationRepository(session)
                    repo.create(conversation_id, user_id, pdf_filename, metadata)
                finally:
                    session.close()
                logger.info(f"Created conversation: {conversation_id}")
                return conversation_id
        except Exception as e:
            logger.error(f"Error creating conversation in PostgreSQL: {e}")
        
        # Fallback
        self._conversations_fallback[conversation_id] = {
            "id": conversation_id,
            "user_id": user_id,
            "pdf_filename": pdf_filename,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "metadata": metadata or {}
        }
        return conversation_id
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_documents: Optional[List[str]] = None,
        importance_score: float = 0.5
    ) -> str:
        """
        Add a message to a conversation.
        
        Workflow:
        1. Save to PostgreSQL (source of truth)
        2. Update Redis cache with recent messages
        """
        # Token counting
        token_count = len(content.split())  # Simple approximation
        
        try:
            if self.db:
                session = self.db.get_session()
                try:
                    repo = MessageRepository(session)
                    msg = repo.create(
                        conversation_id,
                        role,
                        content,
                        metadata=metadata,
                        source_documents=source_documents,
                        token_count=token_count,
                        importance_score=importance_score
                    )
                    message_id = msg.message_id
                finally:
                    session.close()
                
                # Update Redis cache
                if self.redis:
                    self._update_redis_recent_messages(conversation_id)
                
                logger.debug(f"Message added to {conversation_id}")
                return message_id
        
        except Exception as e:
            logger.error(f"Error adding message to PostgreSQL: {e}")
        
        # Fallback: in-memory storage
        if conversation_id not in self._conversations_fallback:
            self._conversations_fallback[conversation_id] = {
                "id": conversation_id,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
        
        import uuid
        message_id = str(uuid.uuid4())
        msg_dict = {
            "message_id": message_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "source_documents": source_documents or [],
            "token_count": token_count,
            "importance_score": importance_score
        }
        self._conversations_fallback[conversation_id]["messages"].append(msg_dict)
        return message_id
    
    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10,
        from_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages for a conversation.
        
        Workflow:
        1. If from_cache and Redis available: Check Redis first (fast)
        2. If miss or from_cache=False: Fetch from PostgreSQL
        3. Convert to chat format
        """
        # Try Redis cache first
        if from_cache and self.redis:
            cached = self.redis.lrange(
                f"recent_messages:{conversation_id}",
                0,
                limit - 1,
                json_deserialize=True
            )
            if cached:
                logger.debug(f"Redis HIT: Recent messages for {conversation_id}")
                return cached
        
        # Fallback to PostgreSQL
        if self.db:
            try:
                session = self.db.get_session()
                try:
                    repo = MessageRepository(session)
                    db_messages = repo.list_recent(conversation_id, limit)
                finally:
                    session.close()
                
                messages = [msg.to_dict() for msg in db_messages]
                
                # Populate Redis for next time
                if self.redis:
                    for msg in messages:
                        self.redis.lpush(
                            f"recent_messages:{conversation_id}",
                            msg
                        )
                    self.redis.ltrim(
                        f"recent_messages:{conversation_id}",
                        0,
                        limit - 1
                    )
                
                logger.debug(f"PostgreSQL: Fetched {len(messages)} messages for {conversation_id}")
                return messages
            
            except Exception as e:
                logger.error(f"Error fetching messages from PostgreSQL: {e}")
        
        # Fallback to in-memory
        if conversation_id in self._conversations_fallback:
            messages = self._conversations_fallback[conversation_id].get("messages", [])
            return messages[-limit:]
        
        return []
    
    def get_conversation_history(
        self,
        conversation_id: str,
        max_turns: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history with token limiting.
        
        Args:
            conversation_id: Conversation ID
            max_turns: Maximum number of turns to include
            max_tokens: Maximum tokens to include
            
        Returns:
            List of messages (most recent first)
        """
        max_turns = max_turns or self.config.MAX_HISTORY_TURNS
        max_tokens = max_tokens or self.max_token_limit
        
        # Get all messages
        all_messages = self.get_recent_messages(
            conversation_id,
            limit=max_turns * 2  # Get extra to account for tokenization
        )
        
        # Filter by token limit  
        result = []
        token_count = 0
        
        for msg in reversed(all_messages):  # Start from oldest
            msg_tokens = msg.get("token_count", len(msg.get("content", "").split()))
            if token_count + msg_tokens > max_tokens:
                break
            result.append(msg)
            token_count += msg_tokens
        
        # Return in chronological order
        return list(reversed(result))
    
    def _update_redis_recent_messages(self, conversation_id: str):
        """Update Redis cache with recent messages."""
        if not self.redis or not self.db:
            return
        
        try:
            session = self.db.get_session()
            try:
                repo = MessageRepository(session)
                recent = repo.list_recent(conversation_id, self.max_recent_turns)
            finally:
                session.close()
            
            # Clear and repopulate
            self.redis.delete(f"recent_messages:{conversation_id}")
            for msg in recent:
                self.redis.lpush(f"recent_messages:{conversation_id}", msg.to_dict())
            
            # Set expiration
            self.redis.redis_client.expire(
                self.redis._make_key(f"recent_messages:{conversation_id}"),
                self.config.REDIS_DEFAULT_TTL
            )
            
        except Exception as e:
            logger.error(f"Error updating Redis recent messages: {e}")
    
    # ==========================================
    # Document Tracking
    # ==========================================
    
    def track_document(
        self,
        conversation_id: str,
        filename: str,
        file_hash: str
    ) -> bool:
        """Track a document in a conversation."""
        try:
            if self.db:
                session = self.db.get_session()
                try:
                    # Ensure document exists
                    doc_repo = DocumentRepository(session)
                    doc = doc_repo.get_by_filename(filename)
                    if not doc:
                        import uuid
                        doc_repo.create(str(uuid.uuid4()), filename, file_hash)
                    
                    # Update conversation documents
                    conv_repo = ConversationRepository(session)
                    conv = conv_repo.get(conversation_id)
                    if conv and filename not in (conv.document_ids or []):
                        if conv.document_ids is None:
                            conv.document_ids = []
                        conv.document_ids.append(filename)
                        session.commit()
                finally:
                    session.close()
                return True
        except Exception as e:
            logger.error(f"Error tracking document: {e}")
        
        return False
    
    # ==========================================
    # Query Utilities
    # ==========================================
    
    def search_messages(
        self,
        conversation_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search messages in a conversation by keyword.
        
        Note: This is simple keyword search. For advanced semantic search,
        use pgvector (future enhancement).
        """
        try:
            if self.db:
                session = self.db.get_session()
                try:
                    repo = MessageRepository(session)
                    all_messages = repo.list_by_conversation(conversation_id)
                finally:
                    session.close()
                
                # Simple keyword search
                query_lower = query.lower()
                matches = [
                    msg.to_dict() for msg in all_messages
                    if query_lower in msg.content.lower()
                ][:limit]
                
                return matches
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
        
        return []
    
    def get_conversation_summary(
        self,
        conversation_id: str
    ) -> Optional[str]:
        """Get conversation summary."""
        try:
            if self.db:
                session = self.db.get_session()
                try:
                    repo = ConversationRepository(session)
                    conv = repo.get(conversation_id)
                    return conv.summary if conv else None
                finally:
                    session.close()
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
        
        return None
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all messages."""
        try:
            if self.db:
                session = self.db.get_session()
                try:
                    repo = ConversationRepository(session)
                    return repo.delete(conversation_id)
                finally:
                    session.close()
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
        
        # Fallback
        if conversation_id in self._conversations_fallback:
            del self._conversations_fallback[conversation_id]
            return True
        
        return False


# Singleton instance
_persistent_memory = None


def get_persistent_memory(config=None) -> PersistentMemory:
    """Get or create the global PersistentMemory instance."""
    global _persistent_memory
    if _persistent_memory is None:
        _persistent_memory = PersistentMemory(config)
    return _persistent_memory
