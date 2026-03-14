"""
src/database/operations.py
--------------------------
Database operations for conversations, messages, and documents.

Provides high-level operations for storing/retrieving data.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from src.database.models import (
    Conversation, Message, Document, ConversationDocument, UserFeedback
)

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository for Conversation operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        pdf_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation."""
        conv = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            pdf_filename=pdf_filename,
            custom_metadata=metadata or {}
        )
        self.session.add(conv)
        self.session.commit()
        logger.info(f"Created conversation: {conversation_id}")
        return conv
    
    def get(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID."""
        return self.session.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
    
    def list_by_user(self, user_id: str, limit: int = 20) -> List[Conversation]:
        """Get recent conversations for a user."""
        return self.session.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(desc(Conversation.updated_at)).limit(limit).all()
    
    def list_by_pdf(self, pdf_filename: str) -> List[Conversation]:
        """Get conversations for a specific PDF."""
        return self.session.query(Conversation).filter(
            Conversation.pdf_filename == pdf_filename
        ).all()
    
    def update_summary(self, conversation_id: str, summary: str) -> Optional[Conversation]:
        """Update conversation summary."""
        conv = self.get(conversation_id)
        if conv:
            conv.summary = summary
            conv.updated_at = datetime.utcnow()
            self.session.commit()
            logger.info(f"Updated summary for conversation: {conversation_id}")
        return conv
    
    def update_metadata(
        self,
        conversation_id: str,
        metadata: Dict[str, Any],
        merge: bool = True
    ) -> Optional[Conversation]:
        """Update conversation metadata."""
        conv = self.get(conversation_id)
        if conv:
            if merge:
                conv.metadata.update(metadata)
            else:
                conv.metadata = metadata
            conv.updated_at = datetime.utcnow()
            self.session.commit()
            logger.info(f"Updated metadata for conversation: {conversation_id}")
        return conv
    
    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation and all associated messages."""
        conv = self.get(conversation_id)
        if conv:
            self.session.delete(conv)
            self.session.commit()
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False


class MessageRepository:
    """Repository for Message operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_documents: Optional[List[str]] = None,
        token_count: int = 0,
        importance_score: float = 0.5
    ) -> Message:
        """Create a new message."""
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            custom_metadata=metadata or {},
            source_documents=source_documents or [],
            token_count=token_count,
            importance_score=importance_score
        )
        self.session.add(msg)
        
        # Update conversation updated_at
        conv = self.session.query(Conversation).filter(
            Conversation.conversation_id == conversation_id
        ).first()
        if conv:
            conv.updated_at = datetime.utcnow()
        
        self.session.commit()
        logger.debug(f"Created message in conversation: {conversation_id}")
        return msg
    
    def get(self, message_id: str) -> Optional[Message]:
        """Get message by ID."""
        return self.session.query(Message).filter(
            Message.message_id == message_id
        ).first()
    
    def list_by_conversation(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation."""
        return self.session.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp).offset(offset).limit(limit).all()
    
    def list_recent(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> List[Message]:
        """Get most recent messages for a conversation."""
        return self.session.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(desc(Message.timestamp)).limit(limit).all()
    
    def list_by_role(
        self,
        conversation_id: str,
        role: str
    ) -> List[Message]:
        """Get messages by role (user or assistant)."""
        return self.session.query(Message).filter(
            and_(
                Message.conversation_id == conversation_id,
                Message.role == role
            )
        ).order_by(Message.timestamp).all()
    
    def list_by_date_range(
        self,
        conversation_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Message]:
        """Get messages within a date range."""
        return self.session.query(Message).filter(
            and_(
                Message.conversation_id == conversation_id,
                Message.timestamp >= start_date,
                Message.timestamp <= end_date
            )
        ).order_by(Message.timestamp).all()
    
    def update_importance(self, message_id: str, score: float) -> Optional[Message]:
        """Update importance score of a message."""
        msg = self.get(message_id)
        if msg:
            msg.importance_score = max(0.0, min(1.0, score))  # Clamp 0-1
            self.session.commit()
        return msg
    
    def delete(self, message_id: str) -> bool:
        """Delete a message."""
        msg = self.get(message_id)
        if msg:
            self.session.delete(msg)
            self.session.commit()
            return True
        return False


class DocumentRepository:
    """Repository for Document operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        document_id: str,
        filename: str,
        file_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Create a new document record."""
        doc = Document(
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            custom_metadata=metadata or {}
        )
        self.session.add(doc)
        self.session.commit()
        logger.info(f"Created document record: {filename}")
        return doc
    
    def get(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.session.query(Document).filter(
            Document.document_id == document_id
        ).first()
    
    def get_by_filename(self, filename: str) -> Optional[Document]:
        """Get document by filename."""
        return self.session.query(Document).filter(
            Document.filename == filename
        ).first()
    
    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get document by file hash (for deduplication)."""
        return self.session.query(Document).filter(
            Document.file_hash == file_hash
        ).first()
    
    def list_all(self) -> List[Document]:
        """Get all documents."""
        return self.session.query(Document).order_by(
            desc(Document.indexed_at)
        ).all()
    
    def update_chunk_count(self, document_id: str, count: int) -> Optional[Document]:
        """Update chunk count for a document."""
        doc = self.get(document_id)
        if doc:
            doc.chunk_count = count
            self.session.commit()
        return doc
    
    def delete(self, document_id: str) -> bool:
        """Delete a document."""
        doc = self.get(document_id)
        if doc:
            self.session.delete(doc)
            self.session.commit()
            return True
        return False


class UserFeedbackRepository:
    """Repository for UserFeedback operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self,
        conversation_id: str,
        rating: int,
        feedback_text: str = "",
        message_id: Optional[str] = None
    ) -> UserFeedback:
        """Create user feedback."""
        feedback = UserFeedback(
            conversation_id=conversation_id,
            message_id=message_id,
            rating=max(1, min(5, rating)),  # Clamp 1-5
            feedback_text=feedback_text
        )
        self.session.add(feedback)
        self.session.commit()
        logger.info(f"Created feedback for conversation: {conversation_id}")
        return feedback
    
    def list_by_conversation(
        self,
        conversation_id: str
    ) -> List[UserFeedback]:
        """Get all feedback for a conversation."""
        return self.session.query(UserFeedback).filter(
            UserFeedback.conversation_id == conversation_id
        ).order_by(UserFeedback.created_at).all()
    
    def get_average_rating(self, conversation_id: str) -> float:
        """Get average rating for a conversation."""
        feedback_list = self.list_by_conversation(conversation_id)
        if not feedback_list:
            return 0.0
        return sum(f.rating for f in feedback_list) / len(feedback_list)
