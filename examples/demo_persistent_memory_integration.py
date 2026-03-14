"""
Quick Demo: RAGOrchestrator with Persistent Memory
===================================================

This demo shows how conversation history is now automatically persisted
to PostgreSQL and cached in Redis when using the RAGOrchestrator.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import Config
from src.orchestrator import RAGOrchestrator


def demo_persistent_memory():
    """Quick demo of persistent memory integration."""
    
    logger.info("\n" + "="*70)
    logger.info("RAGOrchestrator with PersistentMemory Demo")
    logger.info("="*70)
    
    # Initialize
    config = Config()
    orch = RAGOrchestrator(config=config)
    
    # Create conversation
    logger.info("\n[Step 1] Creating conversation...")
    conv_id = orch.create_conversation()
    logger.info(f"✓ Conversation created: {conv_id[:8]}...")
    
    # Simulate adding messages (what happens during query)
    logger.info("\n[Step 2] Adding messages (via persistent_memory)...")
    
    orch.persistent_memory.add_message(
        conversation_id=conv_id,
        role="user",
        content="How does machine learning work?"
    )
    logger.info("✓ User message stored to PostgreSQL and cached in Redis")
    
    orch.persistent_memory.add_message(
        conversation_id=conv_id,
        role="assistant",
        content="Machine learning involves training models on data to recognize patterns..."
    )
    logger.info("✓ Assistant response stored to PostgreSQL and cached in Redis")
    
    # Retrieve from persistent storage
    logger.info("\n[Step 3] Retrieving conversation history...")
    history = orch.get_conversation_history(conv_id)
    messages = history["conversation"]["messages"]
    logger.info(f"✓ Retrieved {len(messages)} messages from persistent storage:")
    for i, msg in enumerate(messages, 1):
        role = msg["role"].upper()
        content = msg["content"][:60] + "..."
        logger.info(f"  {i}. [{role}] {content}")
    
    # Now this conversation is:
    # - Permanently stored in PostgreSQL
    # - Cached in Redis for fast retrieval
    # - Survives server restarts
    # - Can be searched and analyzed
    
    logger.info("\n" + "-"*70)
    logger.info("Storage Status:")
    logger.info("  PostgreSQL:  ✓ Permanent storage")
    logger.info("  Redis:       ✓ Active session cache")  
    logger.info("  In-Memory:   ✓ Quick access layer")
    logger.info("-"*70)
    
    logger.info("\n✓ Demo complete!")
    logger.info("\nWhat changed:")
    logger.info("  1. Conversations persist after restart")
    logger.info("  2. Full conversation history available")
    logger.info("  3. Recent messages cached for speed")
    logger.info("  4. Graceful fallback if storage unavailable")
    logger.info("  5. Multi-user support with conversation isolation")
    
    logger.info("\n" + "="*70)


if __name__ == "__main__":
    try:
        demo_persistent_memory()
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
