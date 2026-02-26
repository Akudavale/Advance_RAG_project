"""  
src/orchestrator.py  
-------------------  
Main RAG orchestrator with document deduplication support.  
"""  
  
import logging  
import uuid  
import os  
from typing import Dict, Any, List, Optional
from datetime import datetime  
  
logger = logging.getLogger(__name__)  
  
  
class RAGOrchestrator:  
    """  
    Main orchestrator for the RAG system.  
      
    Coordinates:  
    - Document processing and indexing (with deduplication)  
    - Query processing  
    - Conversation management  
    - Memory management  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the RAG orchestrator.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
          
        # Initialize components lazily  
        self._embedder = None  
        self._storage_engine = None
        self._langchain_rag_chain = None
        self._langchain_indexing = None
        self._langchain_retrieval = None
        self._langgraph_agent = None
        self._reranker = None  
        self._query_rewriter = None  
        self._conversation_memory = None  
          
        # Conversation storage  
        self._conversations: Dict[str, Dict[str, Any]] = {}  
          
        # Document tracking (maps conversation_id -> list of filenames)  
        self._conversation_documents: Dict[str, List[str]] = {}  
          
        logger.info("RAGOrchestrator initialized")  
      
    # -------------------------------------------  
    # Lazy initialization of components  
    # -------------------------------------------  
      
    @property  
    def embedder(self):  
        """Lazy load embedder."""  
        if self._embedder is None:  
            from src.embedding.embedder import Embedder  
            self._embedder = Embedder(self.config)  
        return self._embedder  
      
    @property
    def storage_engine(self):
        """Lazy load LangChain-native storage engine."""
        if self._storage_engine is None:
            from src.langchain_pipeline import LangChainStorageEngine

            self._storage_engine = LangChainStorageEngine(
                config=self.config,
                embedder=self.embedder,
                logger=logger,
            )
        return self._storage_engine

    @property
    def langchain_indexing(self):
        """Lazy load LangChain-based indexing engine."""
        if self._langchain_indexing is None:
            from src.langchain_pipeline import LangChainIndexingEngine

            self._langchain_indexing = LangChainIndexingEngine(self.config)
        return self._langchain_indexing
      
    @property
    def langchain_rag_chain(self):
        """Lazy load LangChain-first RAG answer chain."""
        if self._langchain_rag_chain is None:
            from src.langchain_pipeline import LangChainRAGChain

            self._langchain_rag_chain = LangChainRAGChain(self.config)
        return self._langchain_rag_chain

    @property
    def langchain_retrieval(self):
        """Lazy load LangChain-first retrieval engine."""
        if self._langchain_retrieval is None:
            from src.langchain_pipeline import LangChainRetrievalEngine

            self._langchain_retrieval = LangChainRetrievalEngine(
                storage=self.storage_engine,
                logger=logger,
            )
        return self._langchain_retrieval

    @property
    def langgraph_agent(self):
        """Lazy load LangGraph-based agent runner."""
        if self._langgraph_agent is None:
            from src.langchain_pipeline import LangGraphAgentRunner

            self._langgraph_agent = LangGraphAgentRunner(
                planner_llm=self.langchain_rag_chain.llm,
                query_rewriter=self.query_rewriter,
                retrieval_engine=self.langchain_retrieval,
                logger=logger,
            )
        return self._langgraph_agent
      
    @property  
    def reranker(self):  
        """Lazy load reranker."""  
        if self._reranker is None:  
            from src.retrieval.reranker import Reranker  
            self._reranker = Reranker(self.config)  
        return self._reranker  
      
    @property  
    def query_rewriter(self):  
        """Lazy load query rewriter."""  
        if self._query_rewriter is None:  
            from src.retrieval.query_rewriter import QueryRewriter  
            self._query_rewriter = QueryRewriter(self.config)  
        return self._query_rewriter  
      
    @property  
    def conversation_memory(self):  
        """Lazy load conversation memory."""  
        if self._conversation_memory is None:  
            from src.memory.conversation_memory import ConversationMemory  
            self._conversation_memory = ConversationMemory(self.config)  
        return self._conversation_memory  
      
    # -------------------------------------------  
    # Conversation management  
    # -------------------------------------------  
      
    def create_conversation(self) -> str:  
        """  
        Create a new conversation.  
          
        Returns:  
            Conversation ID  
        """  
        conversation_id = str(uuid.uuid4())  
          
        self._conversations[conversation_id] = {  
            "id": conversation_id,  
            "created_at": datetime.now().isoformat(),  
            "messages": [],  
            "documents": []  
        }  
          
        self._conversation_documents[conversation_id] = []  
          
        logger.info(f"Created conversation: {conversation_id}")  
        return conversation_id  

    def _ensure_conversation(self, conversation_id: Optional[str]) -> None:
        """Ensure the given conversation ID exists in in-memory state."""
        if not conversation_id:
            return

        if conversation_id in self._conversations:
            return

        self._conversations[conversation_id] = {
            "id": conversation_id,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "documents": [],
        }
        self._conversation_documents.setdefault(conversation_id, [])
        logger.info(f"Initialized conversation state: {conversation_id}")

    def _resolve_filter_filenames(
        self,
        conversation_id: Optional[str],
        filter_filenames: Optional[List[str]],
    ) -> Optional[List[str]]:
        """
        Resolve retrieval filename filters.

        Priority:
        1) explicit filter_filenames from caller
        2) conversation-associated documents when conversation_id is known
        3) None (no filtering) when no conversation context exists
        """
        if filter_filenames is not None:
            cleaned = [f.strip() for f in filter_filenames if isinstance(f, str) and f.strip()]
            # Explicit empty list should intentionally mean "no docs".
            return cleaned

        if not conversation_id:
            return None

        if conversation_id in self._conversations:
            docs = self._conversations[conversation_id].get("documents", []) or []
            return [d for d in docs if isinstance(d, str) and d.strip()]

        return []
      
    def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:  
        """  
        Get conversation history.  
          
        Args:  
            conversation_id: The conversation ID  
              
        Returns:  
            Conversation data or error  
        """  
        if conversation_id not in self._conversations:  
            return {  
                "status": "error",  
                "message": f"Conversation {conversation_id} not found"  
            }  
          
        return {  
            "status": "success",  
            "conversation": self._conversations[conversation_id]  
        }  
      
    # -------------------------------------------  
    # Document processing  
    # -------------------------------------------  
      
    def process_document(  
        self,  
        conversation_id: Optional[str],  
        file_path: str,  
        force_reprocess: bool = False  
    ) -> Dict[str, Any]:  
        """  
        Process and index a document.  
          
        Args:  
            conversation_id: Optional conversation ID to associate with  
            file_path: Path to the document  
            force_reprocess: If True, reprocess even if already indexed  
              
        Returns:  
            Dict with status and document info  
        """  
        try:  
            # Validate file  
            if not os.path.exists(file_path):  
                return {  
                    "status": "error",  
                    "message": f"File not found: {file_path}"  
                }  
              
            filename = os.path.basename(file_path)  
              
            # Check if already indexed (unless force_reprocess)  
            if not force_reprocess and self.storage_engine.is_document_indexed(filename):  
                logger.info(f"Document '{filename}' already indexed, skipping processing")  
                  
                # Still associate with conversation if provided  
                if conversation_id:
                    self._ensure_conversation(conversation_id)
                    if filename not in self._conversations[conversation_id]["documents"]:
                        self._conversations[conversation_id]["documents"].append(filename)
                  
                return {  
                    "status": "success",  
                    "document": {  
                        "filename": filename,  
                        "chunks": 0,  
                        "cached": True  
                    },  
                    "message": "Document already indexed (using cached index)"  
                }  
              
            # If force_reprocess, delete existing  
            if force_reprocess and self.storage_engine.is_document_indexed(filename):  
                logger.info(f"Force reprocessing: deleting existing index for '{filename}'")  
                self.storage_engine.delete_document(filename)  
              
            # Process document using LangChain ingestion pipeline.
            logger.info(f"Processing document with LangChain indexing: {filename}")
            lc_docs = self.langchain_indexing.load_pdf_documents(file_path)
            chunks = self.langchain_indexing.to_chunk_dicts(lc_docs)
              
            if not chunks:  
                return {  
                    "status": "error",  
                    "message": f"No content extracted from {filename}"  
                }  
              
            # Add to vector store (with deduplication)  
            result = self.storage_engine.add_documents(chunks, skip_duplicates=True)
              
            # Associate with conversation  
            if conversation_id:  
                self._ensure_conversation(conversation_id)
                  
                if filename not in self._conversations[conversation_id]["documents"]:  
                    self._conversations[conversation_id]["documents"].append(filename)  
              
            logger.info(f"Processed '{filename}': {len(chunks)} chunks, {result['added']} added to index")  
              
            return {  
                "status": "success",  
                "document": {  
                    "filename": filename,  
                    "chunks": len(chunks),  
                    "added_to_index": result["added"],  
                    "cached": result["added"] == 0 and result.get("skipped", 0) > 0  
                },  
                "message": "Document processed successfully"  
            }  
              
        except Exception as e:  
            logger.error(f"Document processing failed: {e}")  
            return {  
                "status": "error",  
                "message": str(e)  
            }  
      
    def is_document_indexed(self, filename: str) -> bool:  
        """  
        Check if a document is already indexed.  
          
        Args:  
            filename: The document filename  
              
        Returns:  
            True if indexed  
        """  
        return self.storage_engine.is_document_indexed(filename)  
      
    def delete_document(self, filename: str) -> Dict[str, Any]:  
        """  
        Delete a document from the index.  
          
        Args:  
            filename: The document filename  
              
        Returns:  
            Dict with status  
        """  
        result = self.storage_engine.delete_document(filename)  
          
        # Also clear from PDF cache  
        # Note: We don't delete PDF cache as it might be useful for reprocessing  
          
        # Remove from conversations  
        for conv_id, conv_data in self._conversations.items():  
            if filename in conv_data.get("documents", []):  
                conv_data["documents"].remove(filename)  
          
        return result  
      
    # -------------------------------------------  
    # Query processing  
    # -------------------------------------------  
      
    def query(  
        self,  
        query: str,  
        conversation_id: Optional[str] = None,  
        use_optimized_prompts: bool = True,  
        use_memory: bool = True,  
        use_reranking: bool = True,  
        use_query_rewriting: bool = False,  
        top_k: int = 10,  
        rerank_top_k: int = 5,
        method: str = "hyde",
        agentic: bool = True,
        agent_max_steps: int = 3,
        retrieval_mode: str = "hybrid",
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        hybrid_alpha: float = 0.5,
        filter_filenames: Optional[List[str]] = None,
    ) -> Dict[str, Any]:  
        """  
        Process a user query.  
          
        Args:  
            query: The user's question  
            conversation_id: Optional conversation ID for context  
            use_optimized_prompts: Whether to use prompt optimization  
            use_memory: Whether to use conversation memory  
            use_reranking: Whether to rerank results  
            use_query_rewriting: Whether to rewrite the query  
            top_k: Number of initial results to retrieve  
            rerank_top_k: Number of results after reranking  
            agentic: Whether to use the agentic loop
            agent_max_steps: Max decide-act iterations for agentic mode
            retrieval_mode: Retrieval mode: vector, bm25, hybrid
            dense_top_k: vector retrieval cutoff
            sparse_top_k: bm25 retrieval cutoff
            hybrid_alpha: weighting for vector in hybrid fusion
            filter_filenames: optional list of filenames to constrain retrieval
              
        Returns:  
            Dict with answer and sources  
        """  
        try:  
            effective_filter_filenames = self._resolve_filter_filenames(
                conversation_id=conversation_id,
                filter_filenames=filter_filenames,
            )

            if agentic:
                print("Agentic RAG...")  
                return self._query_agentic(
                    query=query,
                    conversation_id=conversation_id,
                    use_optimized_prompts=use_optimized_prompts,
                    use_memory=use_memory,
                    use_reranking=use_reranking,
                    use_query_rewriting=use_query_rewriting,
                    top_k=top_k,
                    rerank_top_k=rerank_top_k,
                    method=method,
                    agent_max_steps=agent_max_steps,
                    retrieval_mode=retrieval_mode,
                    dense_top_k=dense_top_k,
                    sparse_top_k=sparse_top_k,
                    hybrid_alpha=hybrid_alpha,
                    filter_filenames=effective_filter_filenames,
                )

            # Validate query  
            if not query or not query.strip():  
                return {  
                    "status": "error",  
                    "message": "Empty query"  
                }  
              
            query = query.strip()  
            original_query = query  
              
            # Get conversation context  
            conversation_context = []  
            if use_memory and conversation_id:  
                history = self.get_conversation_history(conversation_id)  
                if history.get("status") == "success":  
                    messages = history["conversation"].get("messages", [])  
                    # Get last few exchanges for context  
                    conversation_context = messages[-6:]  # Last 3 exchanges  
              
            # Optionally rewrite query  
            if use_query_rewriting and conversation_context:  
                try:
                    print("Rewriting query...")  
                    rewrite_result = self.query_rewriter.rewrite(  
                        query=query,  
                        context=conversation_context,
                        method=method  
                    )  
                    # print(f"Rewrite result: {rewrite_result}")  
                    if rewrite_result.get("success"):  
                        rewritten_query = rewrite_result.get("rewritten_query", query)  
                        if rewritten_query and rewritten_query != query:
                            # print(f"Query rewritten: '{query}' -> '{rewritten_query}'")  
                            logger.debug(f"Query rewritten: '{query}' -> '{rewritten_query}'")  
                            query = rewritten_query  
                    else:  
                        logger.warning(f"Query rewriting unsuccessful: {rewrite_result.get('error')}")  
                        
                except Exception as e:  
                    logger.warning(f"Query rewriting failed: {e}")
              
            # Retrieve documents
            search_results = self.langchain_retrieval.retrieve(
                query=query,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                dense_top_k=dense_top_k,
                sparse_top_k=sparse_top_k,
                hybrid_alpha=hybrid_alpha,
                filter_filenames=effective_filter_filenames,
                use_reranking=use_reranking,
                reranker=self.reranker if use_reranking else None,
                rerank_query=query,
                rerank_top_k=rerank_top_k,
            )
              
            if not search_results:  
                return {  
                    "status": "success",  
                    "answer": "I couldn't find any relevant information to answer your question. Please make sure you've uploaded relevant documents.",  
                    "sources": [],  
                    "query": original_query  
                }  
              
            # Prepare context for LLM  
            context_docs = []  
            for i, result in enumerate(search_results):  
                context_docs.append({  
                    "index": i + 1,  
                    "content": result.content,  
                    "metadata": result.metadata,  
                    "score": result.score  
                })  
              
            # Generate answer using the LangChain pipeline.
            answer = self.langchain_rag_chain.invoke(
                query=original_query,
                context_docs=context_docs,
                conversation_history=conversation_context if use_memory else None,
            )
              
            # Store in conversation  
            if conversation_id:  
                self._ensure_conversation(conversation_id)
                  
                self._conversations[conversation_id]["messages"].append({  
                    "role": "user",  
                    "content": original_query,  
                    "timestamp": datetime.now().isoformat()  
                })  
                self._conversations[conversation_id]["messages"].append({  
                    "role": "assistant",  
                    "content": answer,  
                    "timestamp": datetime.now().isoformat()  
                })  
              
            # Prepare sources for response  
            sources = []  
            for result in search_results:  
                sources.append({  
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,  
                    "metadata": result.metadata,  
                    "score": result.score  
                })  
              
            return {  
                "status": "success",  
                "answer": answer,  
                "sources": sources,  
                "query": original_query,  
                "rewritten_query": query if query != original_query else None,
                "metadata": {
                    "retrieval_mode": retrieval_mode
                }  
            }  
              
        except Exception as e:  
            logger.error(f"Query processing failed: {e}")  
            return {  
                "status": "error",  
                "message": str(e)  
            }  

    def _query_agentic(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        use_optimized_prompts: bool = True,
        use_memory: bool = True,
        use_reranking: bool = True,
        use_query_rewriting: bool = False,
        top_k: int = 10,
        rerank_top_k: int = 5,
        method: str = "hyde",
        agent_max_steps: int = 3,
        retrieval_mode: str = "hybrid",
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        hybrid_alpha: float = 0.5,
        filter_filenames: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Agentic query execution via LangGraph decide-act loop."""
        if not query or not query.strip():
            return {"status": "error", "message": "Empty query"}

        effective_filter_filenames = self._resolve_filter_filenames(
            conversation_id=conversation_id,
            filter_filenames=filter_filenames,
        )

        original_query = query.strip()

        conversation_context: List[Dict[str, Any]] = []
        if use_memory and conversation_id:
            history = self.get_conversation_history(conversation_id)
            if history.get("status") == "success":
                messages = history["conversation"].get("messages", [])
                conversation_context = messages[-6:]

        graph_out = self.langgraph_agent.run(
            query=original_query,
            conversation_context=conversation_context,
            use_query_rewriting=use_query_rewriting,
            method=method,
            top_k=top_k,
            agent_max_steps=agent_max_steps,
            retrieval_mode=retrieval_mode,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            hybrid_alpha=hybrid_alpha,
            filter_filenames=effective_filter_filenames,
            use_reranking=use_reranking,
            reranker=self.reranker if use_reranking else None,
            rerank_top_k=rerank_top_k,
        )

        context_docs: List[Dict[str, Any]] = graph_out.get("context_docs", []) or []
        trace: List[Dict[str, Any]] = graph_out.get("trace", []) or []
        rewritten_query = graph_out.get("rewritten_query")

        if not context_docs:
            return {
                "status": "success",
                "answer": "I couldn't find any relevant information to answer your question. Please make sure you've uploaded relevant documents.",
                "sources": [],
                "query": original_query,
                "rewritten_query": rewritten_query,
                "metadata": {
                    "agentic": True,
                    "agent_steps": trace,
                    "retrieval_mode": retrieval_mode,
                },
            }

        sources: List[Dict[str, Any]] = []
        answer = self.langchain_rag_chain.invoke(
            query=original_query,
            context_docs=context_docs,
            conversation_history=conversation_context if use_memory else None,
        )

        if conversation_id:
            self._ensure_conversation(conversation_id)

            self._conversations[conversation_id]["messages"].append(
                {
                    "role": "user",
                    "content": original_query,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self._conversations[conversation_id]["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        for doc in context_docs:
            content = doc["content"]
            sources.append(
                {
                    "content": content[:200] + "..." if len(content) > 200 else content,
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", 0.0),
                }
            )

        return {
            "status": "success",
            "answer": answer,
            "sources": sources,
            "query": original_query,
            "rewritten_query": rewritten_query,
            "metadata": {
                "agentic": True,
                "agent_steps": trace,
                "retrieval_mode": retrieval_mode,
            },
        }

    # -------------------------------------------  
    # Statistics and management  
    # -------------------------------------------  
      
    def get_stats(self) -> Dict[str, Any]:  
        """  
        Get system statistics.  
          
        Returns:  
            Dict with statistics  
        """  
        vector_stats = self.storage_engine.get_stats()  
          
        return {  
            "conversations": len(self._conversations),  
            "vector_store": vector_stats,  
            "bm25_chunks": self.storage_engine.bm25_size(),
            "components": {  
                "embedder": self._embedder is not None,  
                "storage_engine": self._storage_engine is not None,
                "langchain_rag_chain": self._langchain_rag_chain is not None,
                "langgraph_agent": self._langgraph_agent is not None,
                "reranker": self._reranker is not None  
            }  
        }  
      
    def clear_index(self) -> Dict[str, Any]:  
        """  
        Clear the entire vector store index.  
          
        Returns:  
            Dict with status  
        """  
        return self.storage_engine.clear()  
