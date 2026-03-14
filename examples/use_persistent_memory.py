#!/usr/bin/env python
"""
examples/use_persistent_memory.py
---------------------------------
Example demonstrating how to use the new PostgreSQL + Redis persistent memory layer.

This example shows:
1. Setting up database and cache connections
2. Creating conversations
3. Adding and retrieving messages
4. Integrating with RAGOrchestrator
5. Querying conversation history
"""

import sys
from pathlib import Path
import uuid
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from src.persistence import get_persistent_memory
from src.orchestrator import RAGOrchestrator


def example_basic_memory_operations():
    """Basic memory operations with PostgreSQL + Redis."""
    
    print("\n" + "="*60)
    print("Example 1: Basic Memory Operations")
    print("="*60)
    
    # Initialize
    config = Config()
    memory = get_persistent_memory(config)
    
    # Create a conversation
    conversation_id = str(uuid.uuid4())
    print(f"\n1. Creating conversation: {conversation_id}")
    
    memory.create_conversation(
        conversation_id=conversation_id,
        user_id="user_001",
        pdf_filename="example.pdf",
        metadata={
            "source": "user_upload",
            "project": "demo"
        }
    )
    print("   ✓ Conversation created")
    
    # Add messages
    print("\n2. Adding messages...")
    
    messages = [
        ("user", "What is machine learning?"),
        ("assistant", "Machine learning is a subset of AI that enables systems to learn from data. Key types include: supervised learning, unsupervised learning, and reinforcement learning."),
        ("user", "Can you give me examples?"),
        ("assistant", "Sure! Examples include: Classification (spam detection), Regression (price prediction), Clustering (customer segmentation), and Recommendation systems."),
    ]
    
    for role, content in messages:
        memory.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            importance_score=0.8 if role == "user" else 0.7
        )
        print(f"   ✓ Added {role} message: {content[:50]}...")
    
    # Retrieve recent messages
    print("\n3. Retrieving recent messages (from Redis cache)...")
    recent = memory.get_recent_messages(conversation_id, limit=2)
    
    for msg in recent:
        print(f"\n   [{msg['role'].upper()}]")
        print(f"   {msg['content'][:80]}...")
        print(f"   Timestamp: {msg['timestamp']}")
    
    # Get full conversation history
    print("\n4. Getting conversation history (with token limiting)...")
    history = memory.get_conversation_history(
        conversation_id=conversation_id,
        max_tokens=1000
    )
    
    print(f"   Total messages in history: {len(history)}")
    print(f"   Total tokens: ~{sum(m['token_count'] for m in history)}")
    
    # Search messages
    print("\n5. Searching messages by keyword...")
    results = memory.search_messages(
        conversation_id=conversation_id,
        query="learning",
        limit=5
    )
    
    print(f"   Found {len(results)} messages containing 'learning':")
    for msg in results:
        print(f"   - {msg['content'][:60]}...")
    
    return conversation_id


def example_with_rag_orchestrator():
    """Integrate persistent memory with RAGOrchestrator."""
    
    print("\n" + "="*60)
    print("Example 2: Integration with RAGOrchestrator")
    print("="*60)
    
    config = Config()
    
    # Initialize RAG system
    print("\n1. Initializing RAG orchestrator...")
    rag = RAGOrchestrator(config)
    print("   ✓ Orchestrator initialized")
    
    # Initialize persistent memory
    print("\n2. Initializing persistent memory...")
    memory = get_persistent_memory(config)
    print("   ✓ Persistent memory initialized")
    
    # Create conversation
    conversation_id = str(uuid.uuid4())
    print(f"\n3. Creating conversation: {conversation_id[:8]}...")
    
    memory.create_conversation(
        conversation_id=conversation_id,
        user_id="user_rag_001",
        pdf_filename="document.pdf"
    )
    
    # Add initial message
    memory.add_message(
        conversation_id=conversation_id,
        role="user",
        content="Please analyze this PDF document",
        importance_score=0.9
    )
    print("   ✓ Initial message added")
    
    # Simulate processing (add assistant response)
    memory.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content="I've analyzed the document. Here are the key findings...",
        source_documents=["document.pdf"],
        importance_score=0.85
    )
    print("   ✓ Response added")
    
    # Get conversation history for prompt context
    print("\n4. Retrieving history for prompt context...")
    history = memory.get_conversation_history(conversation_id)
    
    print(f"   Retrieved {len(history)} messages")
    print("   History ready for prompt augmentation")
    
    return conversation_id


def example_multi_document_conversation():
    """Manage multi-document conversations."""
    
    print("\n" + "="*60)
    print("Example 3: Multi-Document Conversation")
    print("="*60)
    
    config = Config()
    memory = get_persistent_memory(config)
    
    # Create conversation
    conversation_id = str(uuid.uuid4())
    print(f"\n1. Creating multi-document conversation...")
    
    memory.create_conversation(
        conversation_id=conversation_id,
        user_id="user_multi_doc",
        metadata={"docs_count": 3}
    )
    
    # Track multiple documents
    documents = [
        ("report_2024.pdf", "Q1 2024 Business Report"),
        ("financial_analysis.pdf", "Financial Analysis"),
        ("market_trends.pdf", "Market Trends Analysis")
    ]
    
    print("\n2. Tracking documents...")
    for filename, description in documents:
        memory.track_document(
            conversation_id=conversation_id,
            filename=filename,
            file_hash=f"hash_{filename[:10]}"
        )
        print(f"   ✓ Tracked: {filename}")
    
    # Add messages referencing different documents
    print("\n3. Adding messages with multi-document references...")
    
    queries = [
        ("user", "Compare Q1 financials across all documents", ["report_2024.pdf", "financial_analysis.pdf"]),
        ("assistant", "Q1 showed 15% growth in revenue and 12% in margins", ["report_2024.pdf", "financial_analysis.pdf"]),
        ("user", "How does this align with market trends?", ["market_trends.pdf"]),
        ("assistant", "Our performance exceeds market averages by 8-12%", ["report_2024.pdf", "market_trends.pdf"]),
    ]
    
    for role, content, source_docs in queries:
        memory.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            source_documents=source_docs,
            importance_score=0.85 if role == "user" else 0.75
        )
        print(f"   ✓ Added {role} message")
    
    # Retrieve with context
    print("\n4. Retrieving conversation with document context...")
    history = memory.get_recent_messages(conversation_id, limit=10)
    
    print(f"   Total messages: {len(history)}")
    print("   Document references in conversation:")
    
    unique_docs = set()
    for msg in history:
        for doc in msg.get("source_documents", []):
            unique_docs.add(doc)
    
    for doc in sorted(unique_docs):
        print(f"   - {doc}")
    
    return conversation_id


def example_conversation_analytics():
    """Demonstrate conversation analytics and queries."""
    
    print("\n" + "="*60)
    print("Example 4: Conversation Analytics & Queries")
    print("="*60)
    
    config = Config()
    memory = get_persistent_memory(config)
    
    # Create multiple conversations for analytics
    print("\n1. Creating sample conversations...")
    
    conversation_ids = []
    for i in range(3):
        conv_id = str(uuid.uuid4())
        memory.create_conversation(
            conversation_id=conv_id,
            user_id=f"user_{i:03d}",
            pdf_filename=f"document_{i}.pdf"
        )
        conversation_ids.append(conv_id)
        
        # Add some messages
        for j in range(3):
            memory.add_message(
                conversation_id=conv_id,
                role="user" if j % 2 == 0 else "assistant",
                content=f"Sample message {j} in conversation {i}",
                importance_score=0.5 + (j * 0.1)
            )
    
    print(f"   ✓ Created {len(conversation_ids)} sample conversations")
    
    # Retrieve and analyze
    print("\n2. Analyzing conversations...")
    
    for conv_id in conversation_ids:
        history = memory.get_recent_messages(conv_id, limit=10)
        print(f"\n   Conversation: {conv_id[:8]}...")
        print(f"   - Message count: {len(history)}")
        print(f"   - Total tokens: ~{sum(m['token_count'] for m in history)}")
        
        # Count messages by role
        user_msgs = sum(1 for m in history if m['role'] == 'user')
        print(f"   - User messages: {user_msgs}")
        print(f"   - Assistant messages: {len(history) - user_msgs}")


def main():
    """Run all examples."""
    
    print("\n" + "="*70)
    print(" PostgreSQL + Redis Persistent Memory Examples")
    print("="*70)
    
    try:
        # Example 1: Basic operations
        conv_id_1 = example_basic_memory_operations()
        
        # Example 2: Integration with orchestrator
        conv_id_2 = example_with_rag_orchestrator()
        
        # Example 3: Multi-document
        conv_id_3 = example_multi_document_conversation()
        
        # Example 4: Analytics
        example_conversation_analytics()
        
        print("\n" + "="*70)
        print(" ✓ All examples completed successfully!")
        print("="*70)
        
        print("\n📝 Sample Conversation IDs:")
        print(f"   - Example 1: {conv_id_1}")
        print(f"   - Example 2: {conv_id_2}")
        print(f"   - Example 3: {conv_id_3}")
        
        print("\n💡 Next Steps:")
        print("   1. Check PostgreSQL tables: SELECT * FROM conversations;")
        print("   2. Check Redis keys: redis-cli KEYS 'rag:*'")
        print("   3. Query conversation history using get_conversation_history()")
        print("   4. Integrate with your RAG application")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
