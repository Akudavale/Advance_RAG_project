"""
src/database/__init__.py
"""

from src.database.models import (
    Conversation, Message, Document, ConversationDocument, UserFeedback, Base
)
from src.database.connection import DatabaseConnection, get_database, init_database
from src.database.operations import (
    ConversationRepository, MessageRepository, DocumentRepository, UserFeedbackRepository
)

__all__ = [
    # Models
    "Conversation",
    "Message",
    "Document",
    "ConversationDocument",
    "UserFeedback",
    "Base",
    # Connection
    "DatabaseConnection",
    "get_database",
    "init_database",
    # Repositories
    "ConversationRepository",
    "MessageRepository",
    "DocumentRepository",
    "UserFeedbackRepository",
]
