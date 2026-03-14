# PostgreSQL + Redis Integration Guide

## Overview

This document explains how to set up and use PostgreSQL and Redis with the Advanced RAG project for persistent conversation storage and caching.

## Architecture

```
┌─────────────────────────────────────┐
│     RAG Application                 │
│                                     │
│  ┌──────────────────────────────┐   │
│  │   PersistentMemory           │   │
│  │  (src/persistence/)          │   │
│  └──────────────────────────────┘   │
│         ↙            ↘               │
│   PostgreSQL      Redis             │
│  (Persistent)    (Cache)            │
└─────────────────────────────────────┘

Flow:
1. Write: App → PostgreSQL (source of truth) → Redis (cache update)
2. Read: App → Redis (fast) → PostgreSQL (fallback)
```

## Prerequisites

### PostgreSQL
- PostgreSQL 12 or higher
- Connection: `postgresql://user:password@host:port/dbname`

### Redis
- Redis 6 or higher  
- Connection: `redis://host:port/db`

### Python Dependencies
- Required packages already in `requirements.txt`:
  - `sqlalchemy==2.0.47`
  - `psycopg2-binary==2.9.10`
  - `redis==5.0.1`

## Installation

### 1. Install Dependencies

```bash
# Activate virtual environment
source langvenv/Scripts/activate  # Windows

# Install new packages
pip install -r requirements.txt
```

Or install individually:

```bash
pip install psycopg2-binary redis
```

### 2. Set Up PostgreSQL

#### Option A: Docker (Recommended for Development)

```bash
docker run -d \
  --name rag-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  postgres:16

# Verify connection
docker exec rag-postgres psql -U postgres -d rag_db -c "SELECT 1"
```

#### Option B: Local Installation

```bash
# macOS
brew install postgresql
brew services start postgresql
createdb rag_db

# Ubuntu/Debian
sudo apt-get install postgresql
sudo -u postgres createdb rag_db

# Windows
# Download from https://www.postgresql.org/download/windows/
# Use pgAdmin or psql CLI
```

### 3. Set Up Redis

#### Option A: Docker (Recommended)

```bash
docker run -d \
  --name rag-redis \
  -p 6379:6379 \
  redis:7-alpine

# Verify connection
docker exec rag-redis redis-cli ping  # Should return "PONG"
```

#### Option B: Local Installation

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo service redis-server start

# Windows
# Option 1: WSL2 (Recommended)
wsl --install -d Ubuntu
# Inside WSL: sudo apt-get install redis-server

# Option 2: Native build
# Download from https://github.com/microsoftarchive/redis/releases
```

## Configuration

### Environment Variables

Create or update `.env` file in project root:

```env
# PostgreSQL Configuration
ENABLE_POSTGRES=true
DB_URL=postgresql://postgres:password@localhost:5432/rag_db
DB_ECHO=false                    # Set to true to log SQL statements
DB_POOL_DISABLED=false          # Set to true to disable connection pooling

# Redis Configuration
ENABLE_REDIS=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=                  # Leave empty if no password
REDIS_DB=0
REDIS_DEFAULT_TTL=3600          # 1 hour

# Memory Configuration
MAX_TOKEN_LIMIT=4000
MAX_CONVERSATION_TURNS=50
MAX_RECENT_TURNS=10              # Keep 10 recent messages in Redis
SUMMARY_THRESHOLD_TOKENS=2000
```

### Load Configuration

The configuration is automatically loaded from `.env` when you start the application:

```python
from config.config import Config

config = Config()
# All settings are now available:
print(config.DB_URL)
print(config.REDIS_HOST)
```

## Initialization

### 1. Automatic Initialization

```python
from src.persistence import get_persistent_memory

# This automatically initializes PostgreSQL and Redis
memory = get_persistent_memory()
```

### 2. Manual Initialization

```bash
# Initialize databases
python scripts/init_databases.py
```

This script:
- Connects to PostgreSQL and creates all tables
- Connects to Redis and verifies availability
- Logs connection details

## Usage

### Basic Operations

```python
from src.persistence import get_persistent_memory
import uuid

# Get memory instance
memory = get_persistent_memory()

# Create a conversation
conversation_id = str(uuid.uuid4())
memory.create_conversation(
    conversation_id=conversation_id,
    user_id="user123",
    pdf_filename="document.pdf",
    metadata={"source": "upload"}
)

# Add messages
memory.add_message(
    conversation_id=conversation_id,
    role="user",
    content="What is this document about?",
    importance_score=0.8
)

memory.add_message(
    conversation_id=conversation_id,
    role="assistant",
    content="This document discusses...",
    source_documents=["document.pdf"],
    importance_score=0.9
)

# Get recent messages (checks Redis first, then PostgreSQL)
messages = memory.get_recent_messages(conversation_id, limit=10)
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")

# Get conversation history with token limiting
history = memory.get_conversation_history(
    conversation_id=conversation_id,
    max_tokens=3000
)

# Search messages
results = memory.search_messages(
    conversation_id=conversation_id,
    query="important topic",
    limit=5
)

# Delete conversation
memory.delete_conversation(conversation_id)
```

### Integration with RAGOrchestrator

```python
from src.orchestrator import RAGOrchestrator
from src.persistence import get_persistent_memory

# Initialize
config = Config()
rag = RAGOrchestrator(config)

# Create conversation with persistent storage
memory = get_persistent_memory(config)
conv_id = str(uuid.uuid4())
memory.create_conversation(conv_id)

# Process document
rag.process_document(conv_id, "document.pdf")

# Query
response = rag.query(
    conversation_id=conv_id,
    query="Your question here"
)

# Get history for context
history = memory.get_conversation_history(conv_id)
```

## Data Storage Details

### PostgreSQL Tables

#### `conversations`
Stores conversation metadata:

```sql
SELECT * FROM conversations;
-- conversation_id: UUID, PRIMARY KEY
-- user_id: Optional user identifier
-- pdf_filename: Associated PDF
-- created_at, updated_at: Timestamps
-- metadata: JSON (custom fields)
-- summary: Optional conversation summary
-- document_ids: Array of document filenames
```

#### `messages`
Stores individual messages:

```sql
SELECT * FROM messages WHERE conversation_id = 'conv_123';
-- message_id: UUID, PRIMARY KEY
-- conversation_id: FK to conversations
-- role: 'user' or 'assistant'
-- content: Message text
-- timestamp: When message was created
-- metadata: JSON (model used, etc.)
-- importance_score: 0-1 (for history retention)
-- source_documents: Array of referenced documents
-- token_count: For budgeting
```

#### `documents`
Tracks indexed documents:

```sql
SELECT * FROM documents;
-- document_id: Name/hash
-- filename: Original filename
-- file_hash: MD5 (for deduplication)
-- chunk_count: How many chunks
-- indexed_at: Timestamp
-- metadata: JSON (size, pages, etc.)
```

#### `user_feedback`
Stores user ratings:

```sql
SELECT * FROM user_feedback WHERE conversation_id = 'conv_123';
-- feedback_id: UUID
-- conversation_id: Associated conversation
-- message_id: Optional specific message
-- rating: 1-5
-- feedback_text: Optional text
-- created_at: Timestamp
```

### Redis Cache Keys

```
rag:recent_messages:{conversation_id}
  → List of last 10 messages (FIFO)
  → TTL: 3600 seconds (1 hour)

rag:conversation:{conversation_id}
  → Hash of conversation metadata
  → TTL: 3600 seconds

rag:session:{user_id}
  → User session state
  → TTL: 86400 seconds (24 hours)
```

## Querying Data

### SQL Queries

```sql
-- Get recent conversations for a user
SELECT * FROM conversations 
WHERE user_id = 'user123' 
ORDER BY updated_at DESC 
LIMIT 20;

-- Find conversations for a specific PDF
SELECT * FROM conversations 
WHERE pdf_filename = 'report.pdf';

-- Get message count per conversation
SELECT conversation_id, COUNT(*) as message_count 
FROM messages 
GROUP BY conversation_id 
ORDER BY message_count DESC;

-- Find messages by keyword (full-text search)
SELECT * FROM messages 
WHERE content ILIKE '%important topic%' 
  AND conversation_id = 'conv_123';

-- Get user feedback statistics
SELECT 
  conversation_id, 
  AVG(rating) as avg_rating, 
  COUNT(*) as feedback_count 
FROM user_feedback 
GROUP BY conversation_id 
HAVING COUNT(*) > 0;

-- Get conversation timeline
SELECT role, content, timestamp FROM messages 
WHERE conversation_id = 'conv_123' 
ORDER BY timestamp ASC;
```

### Python Queries

```python
from src.database import get_database
from src.database.operations import ConversationRepository, MessageRepository

db = get_database()
session = db.get_session()

# Get conversations by user
conv_repo = ConversationRepository(session)
user_convs = conv_repo.list_by_user("user123", limit=20)

# Get messages for a conversation
msg_repo = MessageRepository(session)
messages = msg_repo.list_by_conversation("conv_123")

# List messages in date range
from datetime import datetime, timedelta
yesterday = datetime.now() - timedelta(days=1)
today = datetime.now()
recent_msgs = msg_repo.list_by_date_range(
    "conv_123", 
    yesterday, 
    today
)

session.close()
```

## Monitoring and Maintenance

### PostgreSQL Maintenance

```bash
# Backup database
pg_dump -U postgres -h localhost -d rag_db > backup.sql

# Restore database
psql -U postgres -h localhost -d rag_db < backup.sql

# Vacuum (optimize)
psql -U postgres -d rag_db -c "VACUUM ANALYZE"

# Check table sizes
psql -U postgres -d rag_db -c "
  SELECT schemaname, tablename, 
         pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) 
  FROM pg_tables 
  ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
"
```

### Redis Maintenance

```bash
# Check Redis stats
redis-cli INFO

# Get memory usage
redis-cli INFO memory

# Clear all keys
redis-cli FLUSHALL

# Get all keys
redis-cli KEYS "rag:*"

# Monitor Redis commands in real-time
redis-cli MONITOR

# Backup
redis-cli BGSAVE
# Then copy /var/lib/redis/dump.rdb
```

### Logging

```python
import logging

# Enable SQL query logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Enable Redis logging
logging.getLogger('redis').setLevel(logging.DEBUG)
```

## Troubleshooting

### PostgreSQL Connection Failed

```
Error: could not connect to server: Connection refused
```

**Solution:**
1. Verify PostgreSQL is running: `pg_isready`
2. Check connection string: `psql -U postgres -h localhost -d rag_db`
3. Check credentials in `.env`
4. Check firewall rules

### Redis Connection Failed

```
Error: Connection refused
```

**Solution:**
1. Verify Redis is running: `redis-cli ping`
2. Check host/port in `.env`
3. Check firewall
4. If disabled, set `ENABLE_REDIS=false` to use PostgreSQL only

### Out of Memory (Redis)

**Solution:**
```bash
# Check memory usage
redis-cli INFO memory

# Configure max memory policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Reduce TTL
# In .env: REDIS_DEFAULT_TTL=1800  # 30 minutes
```

### Slow Queries (PostgreSQL)

```sql
-- Check slow query log
SELECT * FROM pg_stat_statements 
ORDER BY mean_exec_time DESC LIMIT 10;

-- Create index on frequently searched columns
CREATE INDEX idx_messages_conversation_timestamp 
ON messages(conversation_id, timestamp);

CREATE INDEX idx_conversations_user_id 
ON conversations(user_id);
```

## Performance Tips

1. **Redis Caching**: Keep Redis enabled for fast access to recent conversations
2. **Connection Pooling**: Keep `DB_POOL_DISABLED=false` for better performance
3. **Message Limits**: Adjust `MAX_RECENT_TURNS` based on memory constraints
4. **TTL Management**: Lower `REDIS_DEFAULT_TTL` to reduce memory usage
5. **Batch Operations**: Use repositories for bulk operations
6. **Indexing**: Ensure PostgreSQL indices are created and up-to-date

## Migration from In-Memory Storage

If you're migrating from the old in-memory storage:

```python
from src.orchestrator import RAGOrchestrator  
from src.persistence import get_persistent_memory

old_rag = RAGOrchestrator(config)
new_memory = get_persistent_memory(config)

# Migrate conversations
for conv_id, conv_data in old_rag._conversations.items():
    new_memory.create_conversation(
        conv_id,
        user_id=conv_data.get("user_id"),
        pdf_filename=conv_data.get("pdf_filename"),
        metadata=conv_data.get("metadata", {})
    )
    
    # Migrate messages
    for msg in conv_data.get("messages", []):
        new_memory.add_message(
            conv_id,
            msg["role"],
            msg["content"],
            metadata=msg.get("metadata", {}),
            importance_score=msg.get("importance_score", 0.5)
        )

print("✓ Migration complete!")
```

## Next Steps

1. **Configure `.env`** with PostgreSQL and Redis credentials
2. **Run initialization**: `python scripts/init_databases.py`
3. **Update application code** to use `PersistentMemory`
4. **Test thoroughly** before deploying to production
5. **Set up monitoring** using PostgreSQL and Redis tools
6. **(Optional) Enable pgvector** for semantic memory search

See [STORAGE_ARCHITECTURE_GUIDE.md](STORAGE_ARCHITECTURE_GUIDE.md) for architectural details.
