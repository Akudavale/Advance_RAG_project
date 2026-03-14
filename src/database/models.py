"""
src/database/models.py
---------------------
SQLAlchemy ORM models for PostgreSQL persistence.

Models:
- Conversation: Stores conversation metadata and summary
- Message: Stores individual messages with metadata
- Document: Tracks indexed documents
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Text, DateTime, Integer, Float, ForeignKey, 
    JSON, Boolean, ARRAY, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import uuid

Base = declarative_base()


class Conversation(Base):
    """
    Conversation model for storing conversation metadata.
    
    Attributes:
        conversation_id: Unique conversation identifier (UUID)
        user_id: Optional user identifier
        pdf_filename: Associated PDF filename (optional)
        created_at: Timestamp when conversation was created
        updated_at: Timestamp when conversation was last updated
        custom_metadata: JSON metadata (e.g., settings, preferences)
        summary: Optional conversation summary
        messages: Relationship to Message model
        document_ids: Array of associated document IDs
    """
    __tablename__ = "conversations"
    
    conversation_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=True, index=True)
    pdf_filename = Column(String(512), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    custom_metadata = Column(JSON, default={})
    summary = Column(Text, default="")
    document_ids = Column(ARRAY(String), default=[])  # Array of document filenames
    
    # Relationships
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "pdf_filename": self.pdf_filename,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "custom_metadata": self.custom_metadata,
            "summary": self.summary,
            "document_ids": self.document_ids,
            "message_count": len(self.messages)
        }


class Message(Base):
    """
    Message model for storing individual messages.
    
    Attributes:
        message_id: Unique message identifier (UUID)
        conversation_id: Foreign key to Conversation
        role: Message role ('user' or 'assistant')
        content: Message content (text)
        timestamp: When message was created
        custom_metadata: JSON metadata (e.g., source docs, model name)
        importance_score: Importance score (0-1) for memory retention
        source_documents: Array of associated document filenames
        token_count: Number of tokens in message
    """
    __tablename__ = "messages"
    
    message_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.conversation_id"), index=True)
    role = Column(String(32), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    custom_metadata = Column(JSON, default={})
    importance_score = Column(Float, default=0.5)
    source_documents = Column(ARRAY(String), default=[])
    token_count = Column(Integer, default=0)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def to_dict(self):
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "custom_metadata": self.custom_metadata,
            "importance_score": self.importance_score,
            "source_documents": self.source_documents,
            "token_count": self.token_count
        }


class Document(Base):
    """
    Document model for tracking indexed documents.
    
    Attributes:
        document_id: Unique document identifier (filename hash)
        filename: Original filename
        file_hash: MD5 hash of file content
        chunk_count: Number of chunks created
        indexed_at: When document was indexed
        custom_metadata: JSON metadata (size, pages, etc.)
    """
    __tablename__ = "documents"
    
    document_id = Column(String(255), primary_key=True)
    filename = Column(String(512), nullable=False, index=True, unique=True)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    chunk_count = Column(Integer, default=0)
    indexed_at = Column(DateTime, default=datetime.utcnow)
    custom_metadata = Column(JSON, default={})
    
    def to_dict(self):
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_hash": self.file_hash,
            "chunk_count": self.chunk_count,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "custom_metadata": self.custom_metadata
        }


class ConversationDocument(Base):
    """
    Association table for conversations and documents (many-to-many).
    
    Tracks which documents are used in which conversations.
    """
    __tablename__ = "conversation_documents"
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.conversation_id"), index=True)
    document_id = Column(String(255), ForeignKey("documents.document_id"), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "document_id": self.document_id,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class UserFeedback(Base):
    """
    Model for storing user feedback on responses.
    
    Attributes:
        feedback_id: Unique feedback identifier
        conversation_id: Associated conversation
        message_id: Specific message being rated
        rating: Numerical rating (1-5)
        feedback_text: Optional text feedback
        created_at: When feedback was submitted
    """
    __tablename__ = "user_feedback"
    
    feedback_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.conversation_id"), index=True)
    message_id = Column(String(36), ForeignKey("messages.message_id"), nullable=True)
    rating = Column(Integer, nullable=False)  # 1-5
    feedback_text = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            "feedback_id": self.feedback_id,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
