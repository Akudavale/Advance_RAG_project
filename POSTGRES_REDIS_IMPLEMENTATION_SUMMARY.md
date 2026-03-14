# PostgreSQL + Redis Integration - Implementation Summary

## 🎉 What's Been Implemented

A complete **dual-layer persistent memory system** combining PostgreSQL (durable storage) and Redis (fast caching) for the Advanced RAG project.

### Components Created

#### 1. **Database Layer** (`src/database/`)
- **models.py**: SQLAlchemy ORM models
  - `Conversation`: Store conversation metadata
  - `Message`: Store individual messages
  - `Document`: Track indexed documents
  - `UserFeedback`: Store user ratings
  
- **connection.py**: PostgreSQL connection management
  - Connection pooling
  - Automatic table creation
  - Session factory
  
- **operations.py**: Repository pattern data access
  - `ConversationRepository`: CRUD for conversations
  - `MessageRepository`: CRUD for messages
  - `DocumentRepository`: CRUD for documents
  - `UserFeedbackRepository`: CRUD for feedback

#### 2. **Cache Layer** (`src/cache/`)
- **redis_client.py**: Redis client wrapper
  - Connection pooling
  - Key prefixing
  - Automatic serialization
  - Support for strings, lists, hashes
  - Error handling and logging

#### 3. **Persistence Layer** (`src/persistence/`)
- **persistent_memory.py**: Unified memory management
  - Transparent PostgreSQL + Redis integration
  - Automatic caching with PostgreSQL fallback
  - Conversation lifecycle management
  - Document tracking
  - Search capabilities

#### 4. **Configuration** (`config/config.py`)
- **Database settings**
  - `DB_URL`: PostgreSQL connection
  - `DB_ECHO`, `DB_POOL_DISABLED`: Connection options
  - `ENABLE_POSTGRES`: Enable/disable
  
- **Redis settings**
  - `REDIS_HOST`, `REDIS_PORT`: Connection
  - `REDIS_PASSWORD`, `REDIS_DB`: Auth
  - `REDIS_DEFAULT_TTL`: Cache expiration
  - `ENABLE_REDIS`: Enable/disable
  
- **Memory settings**
  - `MAX_TOKEN_LIMIT`: Context limit
  - `MAX_CONVERSATION_TURNS`: Retention count
  - `MAX_RECENT_TURNS`: Redis cache size

#### 5. **Scripts & Examples**
- **scripts/init_databases.py**: Initialize PostgreSQL and Redis
- **examples/use_persistent_memory.py**: Complete usage examples
- **.env.example**: Configuration template
- **POSTGRES_REDIS_SETUP.md**: Detailed setup guide

#### 6. **Dependencies**
- Updated `requirements.txt`:
  - `psycopg2-binary==2.9.10` (PostgreSQL driver)
  - `redis==5.0.1` (Redis client)

---

## 📋 Data Model

### PostgreSQL Tables

```
conversations
├── conversation_id (UUID, PRIMARY KEY)
├── user_id (STRING, INDEX)
├── pdf_filename (STRING, INDEX)
├── created_at (DATETIME, INDEX)
├── updated_at (DATETIME)
├── metadata (JSON)
├── summary (TEXT)
└── document_ids (ARRAY[STRING])

messages
├── message_id (UUID, PRIMARY KEY)
├── conversation_id (FK, INDEX)
├── role (STRING) -- 'user' or 'assistant'
├── content (TEXT)
├── timestamp (DATETIME, INDEX)
├── metadata (JSON)
├── importance_score (FLOAT 0-1)
├── source_documents (ARRAY[STRING])
└── token_count (INTEGER)

documents
├── document_id (STRING, PRIMARY KEY)
├── filename (STRING, UNIQUE, INDEX)
├── file_hash (STRING, UNIQUE, INDEX)
├── chunk_count (INTEGER)
├── indexed_at (DATETIME)
└── metadata (JSON)

user_feedback
├── feedback_id (UUID, PRIMARY KEY)
├── conversation_id (FK, INDEX)
├── message_id (FK)
├── rating (INTEGER 1-5)
├── feedback_text (TEXT)
└── created_at (DATETIME)
```

### Redis Keys

```
rag:recent_messages:{conversation_id}     → List of recent messages
rag:conversation:{conversation_id}         → Conversation metadata
rag:session:{user_id}                      → User session state

TTL: 3600 seconds (configurable)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Inside virtual environment
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
# PostgreSQL
ENABLE_POSTGRES=true
DB_URL=postgresql://postgres:password@localhost:5432/rag_db

# Redis
ENABLE_REDIS=true
REDIS_HOST=localhost
REDIS_PORT=6379

# Memory
MAX_TOKEN_LIMIT=4000
MAX_RECENT_TURNS=10
```

Or copy from template:

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Set Up Databases

#### Option A: Docker (Recommended)

```bash
# PostgreSQL
docker run -d --name rag-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 postgres:16

# Redis
docker run -d --name rag-redis \
  -p 6379:6379 redis:7-alpine

# Verify
docker ps
```

#### Option B: Local Installation

```bash
# macOS
brew install postgresql redis
brew services start postgresql
brew services start redis

# Ubuntu/Debian
sudo apt-get install postgresql redis-server
sudo service postgresql start
sudo service redis-server start
sudo -u postgres createdb rag_db

# Windows
# Download and install from official websites
# Or use WSL2
```

### 4. Initialize Databases

```bash
python scripts/init_databases.py
```

Output:
```
✓ PostgreSQL initialized successfully
✓ Redis initialized successfully
✓ Database initialization complete!
```

### 5. Use in Your Code

#### Basic Usage

```python
from src.persistence import get_persistent_memory
import uuid

# Initialize
memory = get_persistent_memory()

# Create conversation
conv_id = str(uuid.uuid4())
memory.create_conversation(conv_id, user_id="user123")

# Add messages
memory.add_message(conv_id, "user", "What is this document?")
memory.add_message(conv_id, "assistant", "This document discusses...")

# Retrieve
messages = memory.get_recent_messages(conv_id, limit=10)
history = memory.get_conversation_history(conv_id)

# Search
results = memory.search_messages(conv_id, "important topic")
```

#### Advanced Usage

```python
from config.config import Config
from src.persistence import get_persistent_memory

config = Config()
memory = get_persistent_memory(config)

# Create with metadata
memory.create_conversation(
    conversation_id=conv_id,
    user_id="user123",
    pdf_filename="report.pdf",
    metadata={"project": "Q1", "client": "ABC Corp"}
)

# Add with source tracking
memory.add_message(
    conversation_id=conv_id,
    role="assistant",
    content="Based on the document...",
    source_documents=["report.pdf"],
    importance_score=0.9,
    metadata={"model": "gpt-4", "temperature": 0.0}
)

# Get with token limiting
history = memory.get_conversation_history(
    conversation_id=conv_id,
    max_tokens=3000,
    max_turns=5
)

# Track documents
memory.track_document(conv_id, "report.pdf", "hash_value")

# Search with filters
results = memory.search_messages(conv_id, "revenue growth", limit=5)
```

---

## 🔄 Integration with RAGOrchestrator

Update your `main.py` or `main_conversation.py`:

```python
from config.config import Config
from src.orchestrator import RAGOrchestrator
from src.persistence import get_persistent_memory
import uuid

# Initialize
config = Config()
rag = RAGOrchestrator(config)
memory = get_persistent_memory(config)

# Create conversation
conversation_id = str(uuid.uuid4())
memory.create_conversation(conversation_id, user_id="user123")

# Process document
rag.process_document(conversation_id, "document.pdf")

# Query with context
history = memory.get_conversation_history(conversation_id)
print("Chat history:")
for msg in history:
    print(f"  {msg['role']}: {msg['content'][:100]}...")

# Add new message to persistent storage
response = rag.query(conversation_id, "Your question")
memory.add_message(
    conversation_id,
    "user",
    "Your question",
    importance_score=0.8
)
memory.add_message(
    conversation_id,
    "assistant",
    response['answer'],
    source_documents=response.get('sources', []),
    importance_score=0.85
)
```

---

## 📊 Example Script

Run the comprehensive example:

```bash
python examples/use_persistent_memory.py
```

This demonstrates:
- ✓ Creating conversations
- ✓ Adding and retrieving messages
- ✓ Working with Redis cache
- ✓ Multi-document conversations
- ✓ Conversation analytics
- ✓ Integration with RAGOrchestrator

---

## 🔍 Monitoring & Debugging

### PostgreSQL

```bash
# Connect to database
psql -U postgres -d rag_db

# View conversations
SELECT conversation_id, user_id, message_count FROM conversations;

# View messages
SELECT * FROM messages WHERE conversation_id = '...';

# View document tracking
SELECT * FROM documents;

# Backup
pg_dump -U postgres rag_db > backup.sql
```

### Redis

```bash
# Connect to Redis
redis-cli

# View keys
KEYS "rag:*"

# Get recent messages
LRANGE "rag:recent_messages:{conversation_id}" 0 -1

# Monitor in real-time
MONITOR

# Get stats
INFO memory
INFO keyspace
```

### Logging

```python
import logging

# Enable SQL logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Enable Redis logging
logging.getLogger('redis').setLevel(logging.DEBUG)
```

---

## ⚙️ Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_POSTGRES` | `true` | Enable PostgreSQL storage |
| `DB_URL` | `postgresql://...` | PostgreSQL connection string |
| `DB_ECHO` | `false` | Log SQL statements |
| `DB_POOL_DISABLED` | `false` | Disable connection pooling |
| `ENABLE_REDIS` | `true` | Enable Redis caching |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_PASSWORD` | `` | Redis password (if protected) |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_DEFAULT_TTL` | `3600` | Cache TTL in seconds |
| `MAX_TOKEN_LIMIT` | `4000` | Max tokens in context |
| `MAX_CONVERSATION_TURNS` | `50` | Max turns to preserve |
| `MAX_RECENT_TURNS` | `10` | Recent turns in Redis |
| `SUMMARY_THRESHOLD_TOKENS` | `2000` | Auto-summarize threshold |

### Fallback Behavior

If PostgreSQL is unavailable, the system falls back to in-memory storage (data lost on restart).
If Redis is unavailable, the system uses PostgreSQL directly (slower but safe).

---

## 🔧 Troubleshooting

### "Connection refused" (PostgreSQL)

```bash
# Check if PostgreSQL is running
pg_isready

# Check connection string
psql -U postgres -h localhost -d rag_db

# For Docker
docker logs rag-postgres
```

### "Connection refused" (Redis)

```bash
# Check if Redis is running
redis-cli ping

# For Docker
docker logs rag-redis

# Disable Redis in .env if not needed
ENABLE_REDIS=false
```

### Slow queries

```python
# Enable SQL logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Create index on frequently queried columns
# (automatically done during initialization)
```

### Memory/performance issues

- Lower `REDIS_DEFAULT_TTL` (cache expiration)
- Lower `MAX_RECENT_TURNS` (reduce cached size)
- Set `DB_POOL_DISABLED=true` for development
- Use query limits in `list_*` methods

---

## 📚 Documentation

- **[POSTGRES_REDIS_SETUP.md](POSTGRES_REDIS_SETUP.md)**: Complete setup and usage guide
- **[STORAGE_ARCHITECTURE_GUIDE.md](STORAGE_ARCHITECTURE_GUIDE.md)**: Architecture deep-dive
- **[.env.example](.env.example)**: Configuration template
- **[examples/use_persistent_memory.py](examples/use_persistent_memory.py)**: Code examples

---

## 🎯 Next Steps

1. **✓ Install dependencies**: `pip install -r requirements.txt`
2. **✓ Configure .env**: `cp .env.example .env` and edit
3. **✓ Set up databases**: `python scripts/init_databases.py`
4. **✓ Run examples**: `python examples/use_persistent_memory.py`
5. **→ Integrate with your app**: Use `get_persistent_memory()` in your code
6. **→ Set up monitoring**: Use PostgreSQL and Redis tools
7. **→ (Optional) Add pgvector**: For semantic memory search

---

## 📋 File Structure

```
src/
├── database/
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy ORM models
│   ├── connection.py      # DB connection management
│   └── operations.py      # Repository pattern DAOs
├── cache/
│   ├── __init__.py
│   ├── redis_client.py    # Redis client wrapper
│   └── [existing files]
└── persistence/
    ├── __init__.py
    └── persistent_memory.py # Unified memory layer

config/
└── config.py              # Updated with DB/Redis settings

scripts/
└── init_databases.py      # Database initialization

examples/
└── use_persistent_memory.py # Usage examples

.env.example               # Configuration template
POSTGRES_REDIS_SETUP.md   # Setup guide
POSTGRES_REDIS_IMPLEMENTATION_SUMMARY.md (this file)
```

---

## 🎓 Key Concepts

### Write Path
```
Application
    ↓
PersistentMemory.add_message()
    ↓
PostgreSQL (save) ← Source of Truth
    ↓
Redis (cache update)
```

### Read Path
```
Application
    ↓
PersistentMemory.get_recent_messages()
    ↓
Redis Cache Hit? → Return (fast) ✓
    ↓ No
PostgreSQL Query
    ↓
Populate Redis for next time
    ↓
Return
```

### Fallback
```
PostgreSQL Down? → Use in-memory dict
Redis Down? → Use PostgreSQL directly
Both Down? → Error (data loss risk)
```

---

## 💡 Tips

1. **Use PostgreSQL as source of truth** - Always write to PostgreSQL first
2. **Redis is a cache** - Don't rely exclusively on Redis for conversation history
3. **Test your configuration** - Run `init_databases.py` before deploying
4. **Monitor both layers** - Watch PostgreSQL and Redis metrics
5. **Backup regularly** - Use `pg_dump` for PostgreSQL
6. **Consider pgvector** - For semantic memory search (future enhancement)
7. **Start simple** - PostgreSQL only, add Redis when you need caching

---

## ✅ Testing Checklist

- [ ] Installed dependencies: `pip list | grep -E "psycopg2|redis|sqlalchemy"`
- [ ] PostgreSQL running: `pg_isready`
- [ ] Redis running: `redis-cli ping`
- [ ] Databases initialized: `python scripts/init_databases.py`
- [ ] Configuration verified: Check `.env` matches setup
- [ ] Examples running: `python examples/use_persistent_memory.py`
- [ ] Data persisted: Check PostgreSQL tables
- [ ] Cache working: Check Redis keys
- [ ] Integration tested: Use with RAGOrchestrator
- [ ] Fallback tested: Disable Redis/PostgreSQL one at a time

---

## 🆘 Support

For issues:
1. Check [POSTGRES_REDIS_SETUP.md](POSTGRES_REDIS_SETUP.md) troubleshooting section
2. Verify environment variables: `echo $DB_URL`
3. Test connections: `psql` and `redis-cli`
4. Enable SQL logging for debugging
5. Check Docker logs if using containers
6. Review error messages and log files

---

**Ready to use PostgreSQL + Redis with your RAG system!** 🚀
