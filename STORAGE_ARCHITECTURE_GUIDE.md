# Storage Architecture Guide: RAG Project Design

## 📋 Overview of the Conversation

You had a technical architecture discussion exploring **how to persist conversation memory** beyond the current in-memory Python dictionary approach. The discussion covered:

1. **PostgreSQL** - Persistent conversation storage
2. **Redis** - Fast active session cache
3. **Elasticsearch** - Search engine for document retrieval
4. **Difference between BM25 (local) vs Elasticsearch** (external service)
5. **When to use each** storage layer

---

## 🏗️ CURRENT STATE vs PROPOSED STATE

### **Current Architecture (Now)**
```
User Query
    ↓
In-Memory Dict (src/orchestrator.py._conversations)
    ├─ Exists only during process lifetime
    ├─ Lost on app restart ❌
    ├─ Cannot scale across instances ❌
    └─ Not queryable by outside tools ❌
    ↓
Response
```

**Problem**: If the app crashes or restarts, all conversation history is gone.

---

### **Proposed Architecture (Target)**
```
User Query
    ↓
┌─────────────────────────────────────┐
│  PERSISTENCE LAYER (PostgreSQL)    │  ← Source of truth
│  • Full conversation history        │     Survives restarts
│  • All messages + metadata          │     Queryable
│  • User feedback                    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  CACHE LAYER (Redis)                │  ← Fast access
│  • Last N messages (active window)  │     <1ms response
│  • Session state                    │     Temporary (by design)
│  • Context summaries                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  SEARCH LAYER (Chroma + BM25)       │  ← Document retrieval
│  • Dense vectors (Chroma)           │     Semantic search
│  • Sparse text (BM25)               │     Keyword search
│  • PDF chunks + metadata            │
└─────────────────────────────────────┘
    ↓
Response
```

---

## 🗄️ THREE STORAGE LAYERS EXPLAINED

### **1. PostgreSQL - The Permanent Notebook 📔**

**What it stores:**
```sql
conversations (
  conversation_id UUID PRIMARY KEY,
  user_id STRING,
  pdf_filename STRING,
  created_at TIMESTAMP,
  messages JSONB,         -- full message history
  metadata JSONB          -- custom fields
)

messages (
  message_id UUID PRIMARY KEY,
  conversation_id UUID FOREIGN KEY,
  role STRING ('user' | 'assistant'),
  content TEXT,
  timestamp TIMESTAMP,
  source_documents JSONB,
  token_count INT
)

optional: message_embeddings (  -- if pgvector enabled
  message_id UUID,
  embedding VECTOR(1024) -- for semantic search
)
```

**Query examples:**
```sql
-- "Get last 10 messages for this conversation"
SELECT * FROM messages 
WHERE conversation_id = 'conv_123' 
ORDER BY timestamp DESC LIMIT 10;

-- "Find conversations for this PDF"
SELECT * FROM conversations 
WHERE pdf_filename = 'report.pdf';

-- "Find messages containing 'hybrid retrieval'"
SELECT * FROM messages 
WHERE content LIKE '%hybrid retrieval%';

-- "Semantic search (with pgvector)"
SELECT * FROM messages 
ORDER BY message_embeddings <-> [query_embedding]
LIMIT 5;
```

**Advantages:**
- ✅ Survives app restarts
- ✅ Multi-instance shared state
- ✅ Queryable (SQL + full-text search)
- ✅ Audit trail retained
- ✅ Optional semantic search (pgvector)

**Disadvantages:**
- ⚠️ Slower than Redis (network + disk I/O)
- ⚠️ More operational overhead

---

### **2. Redis - The Fast Whiteboard 🏃**

**What it stores:**
```python
# Active conversation window (last N turns)
{
  "active:conv_123": {
    "messages": [last 10 messages],
    "context_summary": "...",
    "total_tokens": 2847,
    "expires_at": "2026-03-14T15:30:00Z"
  }
}

# Session state
{
  "session:user_456": {
    "current_conversation_id": "conv_123",
    "last_access": "2026-03-14T15:28:00Z",
    "preferred_settings": {...}
  }
}

# Cache
{
  "embedding_cache:chunk_id_789": [0.12, 0.34, ...]
}
```

**Example usage pattern:**
```python
# On new user query:
1. Check Redis: GET "active:conv_123"
   - If hit: return immediately (<1ms) ✅
   - If miss: fetch from PostgreSQL (~50-100ms)

2. After generating response:
   SET "active:conv_123" new_window EX 3600  # 1 hour expiry
   LPUSH "message_log:conv_123" new_message  # Add to list
```

**Advantages:**
- ✅ Very fast (<1ms access)
- ✅ Reduces repeated database queries
- ✅ Perfect for real-time chat
- ✅ Low latency

**Disadvantages:**
- ❌ Data lost on restart (by default)
- ❌ Limited persistence options
- ⚠️ Requires separate service

**Important caveat:**
```
DO NOT rely on Redis alone for conversation history.
Redis is a cache, not a database.
Always write to PostgreSQL first (source of truth),
then update Redis (working memory).
```

---

### **3. Chroma - Document Vector Store 🔍**

**What it stores:**
- PDF chunks (text)
- Embeddings (1024-dim vectors from BAAI model)
- Metadata (page, source, chunk_id)
- HNSW indices for fast similarity search

**Already in your project** - no changes needed here.

---

## 📊 COMPARISON TABLE

| Feature | PostgreSQL | Redis | Chroma | Elasticsearch |
|---------|-----------|-------|--------|---------------|
| **Persistence** | ✅ Yes | ❌ No (cache) | ✅ Yes | ✅ Yes |
| **Speed** | ⚠️ Medium | ✅ Very Fast | ✅ Fast | ✅ Fast |
| **Data Loss Risk** | ❌ None | ⚠️ Likely | ❌ None | ❌ None |
| **Query Type** | SQL + Full-text | K/V lookup | Vector similarity | Full-text + ranking |
| **Use Case** | History + metadata | Active session | Document search | Scalable search |
| **Operational** | ⚠️ Medium | ⚠️ Medium | ✅ Simple | ⚠️ Complex |

---

## 🔄 REQUEST FLOW WITH ALL LAYERS

```
User: "What's in the first PDF I uploaded yesterday?"

Step 1: Parse Query
│
├─→ Extract: pdf_filename="first.pdf", time="yesterday"
│
Step 2: Check Cache (Redis)
│
├─→ GET "active:conv_123" → HIT ✅
│   └─→ Loads: [last 10 messages, context, tokens]
│   └─→ Execution time: <1ms
│
├─→ If MISS: Fetch from PostgreSQL (slower)
│   └─→ SELECT * FROM messages WHERE conversation_id='conv_123'
│   └─→ UPDATE Redis with result
│   └─→ Execution time: 50-100ms
│
Step 3: Augment with retrieval (Chroma)
│
├─→ Query Chroma for chunks from first.pdf
│   └─→ Generate embedding of user query
│   └─→ Search vector database (HNSW)
│   └─→ Get top-5 most relevant chunks
│
Step 4: Generate Response
│
├─→ Build prompt with:
│   [Chat history from Redis] + [Retrieved chunks] + [System prompt]
│   └─→ Call LLM (GPT-4 / Gemini)
│
Step 5: Persist Result
│
├─→ Save to PostgreSQL:
│   INSERT INTO messages (conversation_id, role, content, ...)
│   └─→ New message now in permanent record
│
├─→ Update Redis:
│   LPUSH "message_log:conv_123" new_message
│   SET "active:conv_123" updated_window
│   └─→ Cache now hot for next query
│
Return: Answer + Sources
```

---

## 🎯 YOUR EXACT USE CASE

### **Three Questions to Answer**

1. **"Do I store memory in text format and query it?**"
   - **Yes**, store message text in PostgreSQL
   - Query by: metadata (conversation_id, timestamp, pdf_filename)
   - Also query by: keywords (full-text search)
   - Also query by: embeddings (semantic search with pgvector)

2. **"Can I retrieve conversation memory by natural language question?"**
   - **Option A (Keyword)**: Store text, use PostgreSQL full-text search
     - Works for exact/similar phrases
     - Weaker for meaning-based recall
   
   - **Option B (Semantic)**: Store text + embeddings in pgvector
     - Works even when user rewording is different
     - Stronger, but requires pgvector setup

3. **"Replace in-memory dict with Redis/Postgres?"**
   - **Yes, exactly**
   - PostgreSQL = permanent conversation history (replace persistence need)
   - Redis = active conversation window (replace dict for active chat)
   - Combined approach gives you: speed + durability + scalability

---

## ⚡ SIMPLE MENTAL MODEL

```
PostgreSQL = The permanent filing cabinet
             └─ Everything stored forever
             └─ Can query by who/what/when
             └─ Slow to access (relative to Redis)

Redis = The whiteboard on your desk
        └─ Current work you're doing now
        └─ Very fast access
        └─ Gets erased when you leave

Chroma = The search engine
         └─ Finds relevant documents
         └─ Returns semantically similar chunks
         └─ Already in your project

BM25 = Local keyword search
       └─ Finds chunks with matching words
       └─ Stored locally in rank_bm25 JSON
       └─ Already in your project

Elasticsearch = Scalable search engine
                └─ Replace rank_bm25 if you grow
                └─ Optional, not needed now
```

---

## 🚀 IMPLEMENTATION ROADMAP

### **Phase 1: Add PostgreSQL** (Recommended first)
- [ ] Define conversation + message schema
- [ ] Create tables using SQLAlchemy ORM
- [ ] Update `src/orchestrator.py`: write to Postgres on new message
- [ ] Load conversation from Postgres on restart
- [ ] Retain current Redis-less approach (single instance)

### **Phase 2: Add Redis** (Optional, for scaling)
- [ ] Set up Redis instance
- [ ] Cache recent message windows in Redis
- [ ] Check Redis before PostgreSQL in query flow
- [ ] Update Redis when new messages arrive
- [ ] Configure Redis expiration (TTL) for auto-cleanup

### **Phase 3: Add pgvector** (Optional, for semantic memory search)
- [ ] Install pgvector PostgreSQL extension
- [ ] Add embedding column to messages table
- [ ] Store embeddings when message is created
- [ ] Implement semantic search on past conversations

### **Phase 4: Consider Elasticsearch** (Optional, if scaling document search)
- [ ] Replace rank_bm25 layer with Elasticsearch
- [ ] Move PDF chunk indexing to Elasticsearch
- [ ] Update retrieval.py to query Elasticsearch instead of local BM25

---

## 📝 KEY TAKEAWAYS

1. **PostgreSQL replaces** the current "persistence" need
   - Your dict disappears on restart → Postgres survives
   - Data queryable by metadata and keywords

2. **Redis replaces** the current "active session dict"
   - For multi-instance apps
   - For very low-latency chat
   - Cache layer only—always write to Postgres first

3. **Your BM25** (local rank_bm25) is not Elasticsearch
   - It's a Python library, in-process
   - Stored in JSON file
   - Good for single instance, <100K chunks
   - Elasticsearch for production scale

4. **Elasticsearch is not Postgres**
   - Different purposes: search vs database
   - Elasticsearch ≈ search engine
   - PostgreSQL ≈ relational database
   - They're complementary

5. **Recommended next step**: Add PostgreSQL for conversation persistence
   - Gives you: restart safety, queryability, audit trail
   - No extra complexity vs Redis
   - Solves your biggest pain point now

---

## 🔗 MAPPING TO YOUR CODE

| Current | Location | Proposed Change |
|---------|----------|-----------------|
| `self._conversations` dict | `src/orchestrator.py` | → PostgreSQL + optional Redis |
| Conversation storage | In-memory only | → Persistent database |
| Message history | Lost on restart | → Survives restarts |
| Query by metadata | Not possible | → SQL queries |
| Multiple instances | Not supported | → Fully supported |

---

## ✅ ANSWER TO YOUR SPECIFIC QUESTION

> "I can maintain conversation persistence with PostgreSQL. I can use Redis for active conversation storage instead of storing the conversation as Python dict—yes?"

**YES. Exactly.**

That is the right architecture.

The dict stays for in-process state ONLY.
Postgres keeps the permanent record.
Redis (optional) keeps the fast access layer.

Result:
- ✅ Conversation survives app restart
- ✅ Multiple instances can share state
- ✅ Query past conversations by metadata
- ✅ Semantic search on memory (with pgvector)
- ✅ Fast access to active chat (with Redis)
