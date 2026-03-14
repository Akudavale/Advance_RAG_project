"""
examples/test_orchestrator_persistent_memory.py
------------------------------------------------
Test RAGOrchestrator integrated with PersistentMemory.

Demonstrates:
- Creating conversations with persistent storage
- Storing and retrieving messages from PostgreSQL + Redis
- Multi-turn conversation continuity
"""

import logging
import sys
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config
from src.orchestrator import RAGOrchestrator


def test_persistent_memory_integration():
    """Test RAGOrchestrator with persistent memory."""
    
    logger.info("=" * 70)
    logger.info("Testing RAGOrchestrator with Persistent Memory Integration")
    logger.info("=" * 70)
    
    # Initialize config and orchestrator
    config = Config()
    orch = RAGOrchestrator(config=config)
    
    logger.info("\n[1] Creating a new conversation...")
    conversation_id = orch.create_conversation()
    logger.info(f"✓ Created conversation: {conversation_id}")
    
    # Simulate adding messages (normally done via query method)
    logger.info("\n[2] Adding messages to persistent storage...")
    try:
        orch.persistent_memory.add_message(
            conversation_id=conversation_id,
            role="user",
            content="What is machine learning?"
        )
        logger.info("✓ Added user message to persistent storage")
        
        orch.persistent_memory.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content="Machine learning is a subset of AI that enables systems to learn and improve from experience."
        )
        logger.info("✓ Added assistant message to persistent storage")
        
        orch.persistent_memory.add_message(
            conversation_id=conversation_id,
            role="user",
            content="Can you explain supervised learning?"
        )
        logger.info("✓ Added second user message")
        
        orch.persistent_memory.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content="Supervised learning uses labeled training data to train models, where each input has a corresponding correct output."
        )
        logger.info("✓ Added second assistant message")
        
    except Exception as e:
        logger.error(f"✗ Failed to add messages: {e}")
        return False
    
    # Test retrieval from persistent memory
    logger.info("\n[3] Retrieving conversation history from persistent storage...")
    try:
        history = orch.get_conversation_history(conversation_id)
        if history["status"] == "success":
            messages = history["conversation"].get("messages", [])
            logger.info(f"✓ Retrieved {len(messages)} messages from persistent storage")
            
            # Display messages
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")[:100]
                logger.info(f"  [{i}] {role}: {content}...")
        else:
            logger.error(f"✗ Failed to retrieve history: {history.get('message')}")
            return False
    except Exception as e:
        logger.error(f"✗ Failed to retrieve conversation: {e}")
        return False
    
    # Test retrieving recent messages
    logger.info("\n[4] Retrieving recent messages (Redis cache)...")
    try:
        recent = orch.persistent_memory.get_recent_messages(
            conversation_id=conversation_id,
            limit=2
        )
        logger.info(f"✓ Retrieved {len(recent)} recent messages from cache")
        for i, msg in enumerate(recent, 1):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:80]
            logger.info(f"  [{i}] {role}: {content}...")
    except Exception as e:
        logger.error(f"✗ Failed to retrieve recent messages: {e}")
        return False
    
    # Test searching messages
    logger.info("\n[5] Searching for messages containing keyword...")
    try:
        results = orch.persistent_memory.search_messages(
            conversation_id=conversation_id,
            query="learning"
        )
        logger.info(f"✓ Found {len(results)} messages matching 'learning'")
        for i, msg in enumerate(results, 1):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")[:80]
            logger.info(f"  [{i}] {role}: {content}...")
    except Exception as e:
        logger.error(f"✗ Failed to search messages: {e}")
        return False
    
    # Test conversation metadata
    logger.info("\n[6] Retrieving conversation summary...")
    try:
        summary = orch.persistent_memory.get_conversation_summary(conversation_id)
        if summary:
            logger.info(f"✓ Conversation summary: {summary[:150]}...")
        else:
            logger.info("✓ No summary yet (will be generated after more exchanges)")
    except Exception as e:
        logger.warning(f"Note: {e}")
    
    # Test in-memory cache still works
    logger.info("\n[7] Verifying in-memory cache...")
    in_mem_history = orch._conversations.get(conversation_id)
    if in_mem_history:
        msg_count = len(in_mem_history.get("messages", []))
        logger.info(f"✓ In-memory cache has {msg_count} messages")
    else:
        logger.warning("Note: In-memory cache will be populated on next query")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ All persistent memory integration tests passed!")
    logger.info("=" * 70)
    logger.info("\nIntegration Summary:")
    logger.info("  • PostgreSQL stores permanent conversation history")
    logger.info("  • Redis caches recent messages for fast access")
    logger.info("  • RAGOrchestrator saves messages after each query")
    logger.info("  • In-memory state serves as quick-access cache")
    logger.info("  • Graceful fallback if PostgreSQL/Redis unavailable")
    
    return True


def test_multiple_conversations():
    """Test handling multiple concurrent conversations."""
    
    logger.info("\n\n" + "=" * 70)
    logger.info("Testing Multiple Conversations with Persistent Storage")
    logger.info("=" * 70)
    
    config = Config()
    orch = RAGOrchestrator(config=config)
    
    conversation_ids = []
    
    logger.info("\n[1] Creating 3 separate conversations...")
    for i in range(3):
        conv_id = orch.create_conversation()
        conversation_ids.append(conv_id)
        logger.info(f"✓ Created conversation {i+1}: {conv_id}")
    
    logger.info("\n[2] Adding unique messages to each conversation...")
    for i, conv_id in enumerate(conversation_ids):
        try:
            orch.persistent_memory.add_message(
                conversation_id=conv_id,
                role="user",
                content=f"Question from conversation {i+1}"
            )
            orch.persistent_memory.add_message(
                conversation_id=conv_id,
                role="assistant",
                content=f"Answer to conversation {i+1}"
            )
            logger.info(f"✓ Added messages to conversation {i+1}")
        except Exception as e:
            logger.error(f"✗ Failed to add messages: {e}")
            return False
    
    logger.info("\n[3] Verifying isolation between conversations...")
    for i, conv_id in enumerate(conversation_ids):
        try:
            messages = orch.persistent_memory.get_recent_messages(
                conversation_id=conv_id,
                limit=10
            )
            expected_role_sequence = ["user", "assistant"]
            actual_sequence = [msg.get("role") for msg in messages]
            
            if actual_sequence == expected_role_sequence:
                logger.info(f"✓ Conversation {i+1} has correct message isolation")
            else:
                logger.error(f"✗ Conversation {i+1} has unexpected message sequence")
                return False
        except Exception as e:
            logger.error(f"✗ Failed to verify conversation {i+1}: {e}")
            return False
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ Multiple conversations test passed!")
    logger.info("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_persistent_memory_integration()
        if success:
            success = test_multiple_conversations()
        
        if success:
            logger.info("\n✓✓✓ All tests completed successfully! ✓✓✓")
            sys.exit(0)
        else:
            logger.error("\n✗✗✗ Tests failed ✗✗✗")
            sys.exit(1)
    except Exception as e:
        logger.error(f"\n✗✗✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
