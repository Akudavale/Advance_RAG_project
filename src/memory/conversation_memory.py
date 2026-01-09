"""  
src/memory/conversation_memory.py  
---------------------------------  
Conversation memory management with token-aware context building.  
"""  
  
import logging  
from typing import Dict, List, Any, Optional  
from datetime import datetime  
from dataclasses import dataclass, field  
import uuid  
  
from src.memory.token_counter import TokenCounter  
  
logger = logging.getLogger(__name__)  
  
  
@dataclass  
class Message:  
    """A single conversation message."""  
    role: str  # 'user' or 'assistant'  
    content: str  
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())  
    metadata: Dict[str, Any] = field(default_factory=dict)  
    importance_score: float = 0.5  
      
    def to_dict(self) -> Dict[str, Any]:  
        return {  
            "role": self.role,  
            "content": self.content,  
            "timestamp": self.timestamp,  
            "metadata": self.metadata,  
            "importance_score": self.importance_score  
        }  
  
  
@dataclass  
class Conversation:  
    """A conversation with messages and metadata."""  
    conversation_id: str  
    messages: List[Message] = field(default_factory=list)  
    summary: str = ""  
    metadata: Dict[str, Any] = field(default_factory=dict)  
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())  
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())  
      
    def add_message(self, role: str, content: str, **kwargs) -> Message:  
        """Add a message to the conversation."""  
        message = Message(role=role, content=content, **kwargs)  
        self.messages.append(message)  
        self.updated_at = datetime.now().isoformat()  
        return message  
      
    def to_dict(self) -> Dict[str, Any]:  
        return {  
            "conversation_id": self.conversation_id,  
            "messages": [m.to_dict() for m in self.messages],  
            "summary": self.summary,  
            "metadata": self.metadata,  
            "created_at": self.created_at,  
            "updated_at": self.updated_at  
        }  
  
  
class ConversationMemory:  
    """  
    Manages conversation memory with token-aware context building.  
      
    Features:  
    - Multiple conversation support  
    - Token-limited context retrieval  
    - Conversation summarization  
    - Important message tracking  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize conversation memory.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self.token_counter = TokenCounter()  
          
        # Storage  
        self._conversations: Dict[str, Conversation] = {}  
          
        # Limits  
        self.max_token_limit = getattr(self.config, "MAX_TOKEN_LIMIT", 4000)  
        self.max_turns = getattr(self.config, "MAX_CONVERSATION_TURNS", 50)  
        self.summary_threshold = getattr(self.config, "SUMMARY_THRESHOLD_TOKENS", 2000)  
          
        logger.info("ConversationMemory initialized")  
      
    def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> str:  
        """  
        Create a new conversation.  
          
        Args:  
            metadata: Optional metadata for the conversation  
              
        Returns:  
            Conversation ID  
        """  
        conv_id = str(uuid.uuid4())  
        self._conversations[conv_id] = Conversation(  
            conversation_id=conv_id,  
            metadata=metadata or {}  
        )  
        logger.info(f"Created conversation: {conv_id}")  
        return conv_id  
      
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:  
        """Get a conversation by ID."""  
        return self._conversations.get(conversation_id)  
      
    def add_message(  
        self,  
        conversation_id: str,  
        role: str,  
        content: str,  
        importance_score: float = 0.5,  
        metadata: Optional[Dict[str, Any]] = None  
    ) -> Optional[Message]:  
        """  
        Add a message to a conversation.  
          
        Args:  
            conversation_id: Conversation ID  
            role: Message role ('user' or 'assistant')  
            content: Message content  
            importance_score: Importance score (0-1)  
            metadata: Optional message metadata  
              
        Returns:  
            Added message or None if conversation not found  
        """  
        conv = self._conversations.get(conversation_id)  
        if not conv:  
            logger.warning(f"Conversation not found: {conversation_id}")  
            return None  
          
        message = conv.add_message(  
            role=role,  
            content=content,  
            importance_score=importance_score,  
            metadata=metadata or {}  
        )  
          
        # Check if summarization is needed  
        self._maybe_summarize(conversation_id)  
          
        return message  
      
    def get_context_for_query(  
        self,  
        conversation_id: str,  
        query: str,  
        max_tokens: Optional[int] = None  
    ) -> Dict[str, Any]:  
        """  
        Get optimized context for a query.  
          
        Args:  
            conversation_id: Conversation ID  
            query: Current user query  
            max_tokens: Maximum tokens for context  
              
        Returns:  
            Context dictionary with summary, recent_messages, relevant_messages  
        """  
        max_tokens = max_tokens or self.max_token_limit  
          
        conv = self._conversations.get(conversation_id)  
        if not conv:  
            return {  
                "summary": "",  
                "recent_messages": [],  
                "relevant_messages": []  
            }  
          
        # Calculate query tokens  
        query_tokens = self.token_counter.count_tokens(query)  
        available_tokens = max_tokens - query_tokens - 100  # Reserve overhead  
          
        # Build context  
        context = {  
            "summary": conv.summary,  
            "recent_messages": [],  
            "relevant_messages": []  
        }  
          
        # Allocate tokens  
        summary_tokens = self.token_counter.count_tokens(conv.summary)  
        remaining_tokens = available_tokens - summary_tokens  
          
        if remaining_tokens <= 0:  
            # Trim summary if needed  
            context["summary"] = self.token_counter.trim_to_token_limit(  
                conv.summary,  
                available_tokens  
            )  
            return context  
          
        # Add recent messages (most recent first, working backwards)  
        recent_budget = int(remaining_tokens * 0.7)  
        current_tokens = 0  
          
        for msg in reversed(conv.messages[-20:]):  # Last 20 messages max  
            msg_tokens = self.token_counter.count_tokens(msg.content)  
            if current_tokens + msg_tokens <= recent_budget:  
                context["recent_messages"].insert(0, msg.to_dict())  
                current_tokens += msg_tokens  
            else:  
                break  
          
        # Add important messages not already included  
        important_budget = remaining_tokens - current_tokens  
        recent_contents = {m["content"] for m in context["recent_messages"]}  
          
        important_msgs = sorted(  
            [m for m in conv.messages if m.content not in recent_contents],  
            key=lambda x: x.importance_score,  
            reverse=True  
        )  
          
        important_tokens = 0  
        for msg in important_msgs:  
            msg_tokens = self.token_counter.count_tokens(msg.content)  
            if important_tokens + msg_tokens <= important_budget:  
                context["relevant_messages"].append(msg.to_dict())  
                important_tokens += msg_tokens  
            else:  
                break  
          
        return context  
      
    def get_conversation_history(  
        self,  
        conversation_id: str,  
        max_messages: Optional[int] = None  
    ) -> Dict[str, Any]:  
        """  
        Get conversation history.  
          
        Args:  
            conversation_id: Conversation ID  
            max_messages: Maximum messages to return  
              
        Returns:  
            Conversation history  
        """  
        conv = self._conversations.get(conversation_id)  
        if not conv:  
            return {  
                "status": "error",  
                "message": f"Conversation not found: {conversation_id}"  
            }  
          
        messages = conv.messages  
        if max_messages:  
            messages = messages[-max_messages:]  
          
        return {  
            "status": "success",  
            "conversation_id": conversation_id,  
            "messages": [m.to_dict() for m in messages],  
            "summary": conv.summary,  
            "metadata": conv.metadata,  
            "message_count": len(conv.messages)  
        }  
      
    def _maybe_summarize(self, conversation_id: str):  
        """Check if summarization is needed and trigger it."""  
        conv = self._conversations.get(conversation_id)  
        if not conv:  
            return  
          
        # Count tokens in messages  
        total_tokens = sum(  
            self.token_counter.count_tokens(m.content)  
            for m in conv.messages  
        )  
          
        # Summarize if over threshold  
        if total_tokens > self.summary_threshold:  
            self._summarize_conversation(conversation_id)  
      
    def _summarize_conversation(self, conversation_id: str):  
        """  
        Summarize older messages in a conversation.  
          
        Note: This is a placeholder. In production, you would use  
        an LLM to generate the summary.  
        """  
        conv = self._conversations.get(conversation_id)  
        if not conv or len(conv.messages) < 10:  
            return  
          
        # Keep recent messages, summarize older ones  
        keep_count = 5  
        to_summarize = conv.messages[:-keep_count]  
          
        # Simple extractive summary (placeholder)  
        key_points = []  
        for msg in to_summarize:  
            if msg.importance_score > 0.7:  
                key_points.append(f"{msg.role}: {msg.content[:100]}...")  
          
        if key_points:  
            new_summary = conv.summary + "\n\nKey points:\n" + "\n".join(key_points[-5:])  
            conv.summary = new_summary[-2000:]  # Limit summary length  
          
        # Remove summarized messages  
        conv.messages = conv.messages[-keep_count:]  
          
        logger.info(f"Summarized conversation {conversation_id}")  
      
    def update_importance(  
        self,  
        conversation_id: str,  
        message_index: int,  
        importance_score: float  
    ):  
        """Update importance score for a message."""  
        conv = self._conversations.get(conversation_id)  
        if conv and 0 <= message_index < len(conv.messages):  
            conv.messages[message_index].importance_score = importance_score  
      
    def clear_conversation(self, conversation_id: str) -> bool:  
        """Clear a conversation's messages but keep metadata."""  
        conv = self._conversations.get(conversation_id)  
        if conv:  
            conv.messages = []  
            conv.summary = ""  
            conv.updated_at = datetime.now().isoformat()  
            return True  
        return False  
      
    def delete_conversation(self, conversation_id: str) -> bool:  
        """Delete a conversation entirely."""  
        if conversation_id in self._conversations:  
            del self._conversations[conversation_id]  
            return True  
        return False  