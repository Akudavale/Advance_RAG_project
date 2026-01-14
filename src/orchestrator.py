# """  
# src/orchestrator.py  
# -------------------  
# Main RAG orchestrator that coordinates all components.  
# """  
  
# import logging  
# import os  
# from typing import Dict, List, Any, Optional  
# from datetime import datetime  
  
# logger = logging.getLogger(__name__)  
  
  
# class RAGOrchestrator:  
#     """  
#     Main orchestrator for the RAG pipeline.  
      
#     Coordinates:  
#     - Document processing and indexing  
#     - Query processing and retrieval  
#     - Answer generation  
#     - Conversation memory  
#     - Evaluation  
#     """  
      
#     def __init__(self, config=None):  
#         """  
#         Initialize the RAG orchestrator.  
          
#         Args:  
#             config: Configuration object  
#         """  
#         from config.config import Config  
          
#         self.config = config or Config()  
          
#         # Validate configuration  
#         validation = self.config.validate()  
#         if not validation["valid"]:  
#             logger.warning(f"Configuration issues: {validation['issues']}")  
          
#         # Initialize components (lazy loading where possible)  
#         self._embedder = None  
#         self._vector_store = None  
#         self._reranker = None  
#         self._query_rewriter = None  
#         self._llm_generator = None  
#         self._pdf_processor = None  
#         self._conversation_memory = None  
#         self._memory_retriever = None  
#         self._memory_updater = None  
#         self._prompt_optimizer = None  
#         self._evaluator = None  
          
#         # Document tracking  
#         self._document_registry: Dict[str, Dict[str, Any]] = {}  
          
#         logger.info("RAGOrchestrator initialized")  
      
#     # -------------------------------------------------------------------------  
#     # Lazy-loaded component properties  
#     # -------------------------------------------------------------------------  
      
#     @property  
#     def embedder(self):  
#         """Get shared embedder instance."""  
#         if self._embedder is None:  
#             from src.embedding.embedder import get_shared_embedder  
#             self._embedder = get_shared_embedder(self.config)  
#         return self._embedder  
      
#     @property  
#     def vector_store(self):  
#         """Get vector store instance."""  
#         if self._vector_store is None:  
#             from src.retrieval.vector_store import VectorStore  
#             self._vector_store = VectorStore(self.config)  
#         return self._vector_store  
      
#     @property  
#     def reranker(self):  
#         """Get reranker instance."""  
#         if self._reranker is None:  
#             from src.retrieval.reranker import Reranker  
#             self._reranker = Reranker(self.config)  
#         return self._reranker  
      
#     @property  
#     def query_rewriter(self):  
#         """Get query rewriter instance."""  
#         if self._query_rewriter is None:  
#             from src.retrieval.query_rewriter import QueryRewriter  
#             self._query_rewriter = QueryRewriter(self.config)  
#         return self._query_rewriter  
      
#     @property  
#     def llm_generator(self):  
#         """Get LLM generator instance."""  
#         if self._llm_generator is None:  
#             from src.answer_generator.llm_generator import LLMGenerator  
#             self._llm_generator = LLMGenerator(self.config)  
#         return self._llm_generator  
      
#     @property  
#     def pdf_processor(self):  
#         """Get PDF processor instance."""  
#         if self._pdf_processor is None:  
#             from src.data_processing.pdf_processor import PDFProcessor  
#             self._pdf_processor = PDFProcessor(self.config)  
#         return self._pdf_processor  
      
#     @property  
#     def conversation_memory(self):  
#         """Get conversation memory instance."""  
#         if self._conversation_memory is None:  
#             from src.memory.conversation_memory import ConversationMemory  
#             self._conversation_memory = ConversationMemory(self.config)  
#         return self._conversation_memory  
      
#     @property  
#     def memory_retriever(self):  
#         """Get memory retriever instance."""  
#         if self._memory_retriever is None:  
#             from src.memory.memory_retriever import MemoryRetriever  
#             self._memory_retriever = MemoryRetriever(self.config, self.embedder)  
#         return self._memory_retriever  
      
#     @property  
#     def memory_updater(self):  
#         """Get memory updater instance."""  
#         if self._memory_updater is None:  
#             from src.memory.memory_updater import MemoryUpdater  
#             self._memory_updater = MemoryUpdater(  
#                 self.config,  
#                 self.conversation_memory,  
#                 self.memory_retriever  
#             )  
#         return self._memory_updater  
      
#     @property  
#     def prompt_optimizer(self):  
#         """Get prompt optimizer instance."""  
#         if self._prompt_optimizer is None:  
#             from src.prompts.prompt_optimizer import PromptOptimizer  
#             self._prompt_optimizer = PromptOptimizer(self.config)  
#         return self._prompt_optimizer  
      
#     @property  
#     def evaluator(self):  
#         """Get evaluator instance."""  
#         if self._evaluator is None:  
#             from src.evaluation.evaluator import Evaluator  
#             self._evaluator = Evaluator(self.config)  
#         return self._evaluator  
      
#     # -------------------------------------------------------------------------  
#     # Conversation Management (API Server interface)  
#     # -------------------------------------------------------------------------  
      
#     def create_conversation(self, metadata: Optional[Dict[str, Any]] = None) -> str:  
#         """  
#         Create a new conversation.  
          
#         Args:  
#             metadata: Optional conversation metadata  
              
#         Returns:  
#             Conversation ID  
#         """  
#         return self.conversation_memory.create_conversation(metadata)  
      
#     def get_conversation_history(self, conversation_id: str) -> Dict[str, Any]:  
#         """  
#         Get conversation history.  
          
#         Args:  
#             conversation_id: Conversation ID  
              
#         Returns:  
#             Conversation history  
#         """  
#         return self.conversation_memory.get_conversation_history(conversation_id)  
      
#     # -------------------------------------------------------------------------  
#     # Document Processing  
#     # -------------------------------------------------------------------------  
      
#     def process_document(  
#         self,  
#         conversation_id: str,  
#         file_path: str  
#     ) -> Dict[str, Any]:  
#         """  
#         Process and index a document.  
          
#         Args:  
#             conversation_id: Conversation ID to associate with  
#             file_path: Path to PDF file  
              
#         Returns:  
#             Processing result  
#         """  
#         try:  
#             # Validate file exists  
#             if not os.path.exists(file_path):  
#                 return {  
#                     "status": "error",  
#                     "message": f"File not found: {file_path}"  
#                 }  
              
#             # Process PDF  
#             chunks = self.pdf_processor.process(file_path)  
              
#             if not chunks:  
#                 return {  
#                     "status": "error",  
#                     "message": "No content extracted from document"  
#                 }  
              
#             # Generate embeddings  
#             contents = [chunk["content"] for chunk in chunks]  
#             embeddings = self.embedder.embed_documents(contents)  
              
#             # Add to vector store  
#             doc_ids = self.vector_store.add_documents(chunks, embeddings)  
              
#             # Register document  
#             filename = os.path.basename(file_path)  
#             self._document_registry[filename] = {  
#                 "conversation_id": conversation_id,  
#                 "file_path": file_path,  
#                 "chunk_count": len(chunks),  
#                 "doc_ids": doc_ids,  
#                 "processed_at": datetime.now().isoformat()  
#             }  
              
#             logger.info(f"Processed document: {filename}, {len(chunks)} chunks")  
              
#             return {  
#                 "status": "success",  
#                 "document": {  
#                     "filename": filename,  
#                     "chunks": len(chunks)  
#                 },  
#                 "message": "Document processed successfully"  
#             }  
              
#         except Exception as e:  
#             logger.error(f"Document processing failed: {e}")  
#             return {  
#                 "status": "error",  
#                 "message": str(e)  
#             }  
      
#     # -------------------------------------------------------------------------  
#     # Query Processing  
#     # -------------------------------------------------------------------------  
      
#     def query(  
#         self,  
#         conversation_id: Optional[str],  
#         query: str,  
#         use_optimized_prompts: bool = True,  
#         use_memory: bool = True,  
#         use_reranking: bool = True,  
#         use_query_rewriting: bool = False,  
#         top_k: Optional[int] = None  
#     ) -> Dict[str, Any]:  
#         """  
#         Process a user query and generate an answer.  
          
#         Args:  
#             conversation_id: Conversation ID (optional)  
#             query: User query  
#             use_optimized_prompts: Whether to optimize prompts  
#             use_memory: Whether to use conversation memory  
#             use_reranking: Whether to rerank results  
#             use_query_rewriting: Whether to rewrite query  
#             top_k: Number of documents to retrieve  
              
#         Returns:  
#             Query result with answer and metadata  
#         """  
#         try:  
#             start_time = datetime.now()  
              
#             # Validate query  
#             if not query or not query.strip():  
#                 return {  
#                     "status": "error",  
#                     "message": "Empty query provided"  
#                 }  
              
#             # Create conversation if needed  
#             if not conversation_id:  
#                 conversation_id = self.create_conversation()  
              
#             # Get retrieval parameters  
#             top_k_retrieval = top_k or getattr(self.config, "TOP_K_RETRIEVAL", 20)  
#             top_k_rerank = getattr(self.config, "TOP_K_RERANK", 5)  
              
#             # Step 1: Query rewriting (optional)  
#             processed_query = query  
#             rewrite_info = None  
              
#             if use_query_rewriting:  
#                 rewrite_result = self.query_rewriter.rewrite(query)  
#                 if rewrite_result.get("success"):  
#                     processed_query = rewrite_result.get("rewritten_query", query)  
#                     rewrite_info = rewrite_result  
              
#             # Step 2: Generate query embedding  
#             query_embedding = self.embedder.embed_query(processed_query)  
              
#             # Step 3: Retrieve documents  
#             retrieved_docs = self.vector_store.search(  
#                 query_embedding=query_embedding,  
#                 top_k=top_k_retrieval  
#             )  
              
#             if not retrieved_docs:  
#                 return {  
#                     "status": "success",  
#                     "conversation_id": conversation_id,  
#                     "query": query,  
#                     "answer": "I couldn't find any relevant information in the documents. Please make sure you've uploaded a document first.",  
#                     "sources": [],  
#                     "metadata": {  
#                         "retrieval_count": 0,  
#                         "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000  
#                     }  
#                 }  
              
#             # Step 4: Rerank documents (optional)  
#             if use_reranking:  
#                 reranked_docs = self.reranker.rerank(  
#                     query=processed_query,  
#                     documents=retrieved_docs,  
#                     top_k=top_k_rerank  
#                 )  
#             else:  
#                 reranked_docs = retrieved_docs[:top_k_rerank]  
              
#             # Step 5: Get memory context (optional)  
#             memory_context = None  
#             if use_memory and conversation_id:  
#                 memory_context = self.conversation_memory.get_context_for_query(  
#                     conversation_id=conversation_id,  
#                     query=query  
#                 )  
              
#             # Step 6: Optimize prompt (optional)  
#             if use_optimized_prompts:  
#                 optimized = self.prompt_optimizer.optimize_rag_prompt(  
#                     query=query,  
#                     documents=reranked_docs,  
#                     memory_context=memory_context  
#                 )  
#                 final_docs = optimized["documents"]  
#                 final_memory = optimized.get("memory_context")  
#             else:  
#                 final_docs = reranked_docs  
#                 final_memory = memory_context  
              
#             # Step 7: Generate answer  
#             answer = self.llm_generator.generate_answer(  
#                 query=query,  
#                 documents=final_docs,  
#                 memory_context={"summary": final_memory} if isinstance(final_memory, str) else final_memory  
#             )  
              
#             # Step 8: Update memory  
#             if use_memory and conversation_id:  
#                 self.memory_updater.add_interaction(  
#                     conversation_id=conversation_id,  
#                     user_message=query,  
#                     assistant_message=answer  
#                 )  
              
#             # Build response  
#             processing_time = (datetime.now() - start_time).total_seconds() * 1000  
              
#             sources = [  
#                 {  
#                     "content": doc.get("content", "")[:200] + "...",  
#                     "metadata": doc.get("metadata", {}),  
#                     "score": doc.get("score", 0)  
#                 }  
#                 for doc in final_docs[:3]  
#             ]  
              
#             return {  
#                 "status": "success",  
#                 "conversation_id": conversation_id,  
#                 "query": query,  
#                 "answer": answer,  
#                 "sources": sources,  
#                 "metadata": {  
#                     "retrieval_count": len(retrieved_docs),  
#                     "reranked_count": len(reranked_docs),  
#                     "final_count": len(final_docs),  
#                     "query_rewritten": rewrite_info is not None,  
#                     "processing_time_ms": processing_time  
#                 }  
#             }  
              
#         except Exception as e:  
#             logger.error(f"Query processing failed: {e}", exc_info=True)  
#             return {  
#                 "status": "error",  
#                 "message": f"Query processing failed: {str(e)}"  
#             }  
      
#     # -------------------------------------------------------------------------  
#     # Advanced Query Methods  
#     # -------------------------------------------------------------------------  
      
#     def query_with_self_critique(  
#         self,  
#         conversation_id: Optional[str],  
#         query: str  
#     ) -> Dict[str, Any]:  
#         """  
#         Query with self-critique for improved accuracy.  
          
#         Args:  
#             conversation_id: Conversation ID  
#             query: User query  
              
#         Returns:  
#             Result with initial and improved answers  
#         """  
#         # First get documents  
#         if not conversation_id:  
#             conversation_id = self.create_conversation()  
          
#         query_embedding = self.embedder.embed_query(query)  
#         retrieved_docs = self.vector_store.search(query_embedding, top_k=10)  
#         reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=5)  
          
#         # Generate with self-critique  
#         result = self.llm_generator.generate_with_self_critique(  
#             query=query,  
#             documents=reranked_docs  
#         )  
          
#         return {  
#             "status": "success",  
#             "conversation_id": conversation_id,  
#             "query": query,  
#             "initial_answer": result["initial_answer"],  
#             "critique": result["critique"],  
#             "improved_answer": result["improved_answer"],  
#             "sources": [  
#                 {"content": d.get("content", "")[:200], "score": d.get("score", 0)}  
#                 for d in reranked_docs[:3]  
#             ]  
#         }  
      
#     def query_with_sources(  
#         self,  
#         conversation_id: Optional[str],  
#         query: str  
#     ) -> Dict[str, Any]:  
#         """  
#         Query with detailed source attribution.  
          
#         Args:  
#             conversation_id: Conversation ID  
#             query: User query  
              
#         Returns:  
#             Result with answer and source attributions  
#         """  
#         if not conversation_id:  
#             conversation_id = self.create_conversation()  
          
#         query_embedding = self.embedder.embed_query(query)  
#         retrieved_docs = self.vector_store.search(query_embedding, top_k=10)  
#         reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=5)  
          
#         # Generate with source attribution  
#         result = self.llm_generator.generate_with_source_attribution(  
#             query=query,  
#             documents=reranked_docs  
#         )  
          
#         return {  
#             "status": "success",  
#             "conversation_id": conversation_id,  
#             "query": query,  
#             "answer": result["answer"],  
#             "attributions": result["attributions"]  
#         }  
      
#     # -------------------------------------------------------------------------  
#     # Utility Methods  
#     # -------------------------------------------------------------------------  
      
#     def get_stats(self) -> Dict[str, Any]:  
#         """Get system statistics."""  
#         return {  
#             "vector_store": self.vector_store.get_stats(),  
#             "documents_registered": len(self._document_registry),  
#             "conversations": len(self.conversation_memory._conversations)  
#         }  
      
#     def clear_all(self):  
#         """Clear all data (use with caution)."""  
#         self.vector_store.clear()  
#         self._document_registry.clear()  
#         logger.info("All data cleared")  






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
        self._vector_store = None  
        self._pdf_processor = None  
        self._llm_generator = None  
        self._reranker = None  
        self._query_rewriter = None  
        self._conversation_memory = None  
        self._prompt_optimizer = None  
          
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
    def vector_store(self):  
        """Lazy load vector store."""  
        if self._vector_store is None:  
            from src.retrieval.vector_store import VectorStore  
            self._vector_store = VectorStore(self.config, self.embedder)  
        return self._vector_store  
      
    @property  
    def pdf_processor(self):  
        """Lazy load PDF processor."""  
        if self._pdf_processor is None:  
            from src.data_processing.pdf_processor import PDFProcessor  
            self._pdf_processor = PDFProcessor(self.config)  
        return self._pdf_processor  
      
    @property  
    def llm_generator(self):  
        """Lazy load LLM generator."""  
        if self._llm_generator is None:  
            from src.answer_generator.llm_generator import LLMGenerator  
            self._llm_generator = LLMGenerator(self.config)  
        return self._llm_generator  
      
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
      
    @property  
    def prompt_optimizer(self):  
        """Lazy load prompt optimizer."""  
        if self._prompt_optimizer is None:  
            from src.prompts.prompt_optimizer import PromptOptimizer  
            self._prompt_optimizer = PromptOptimizer(self.config)  
        return self._prompt_optimizer  
      
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
            if not force_reprocess and self.vector_store.is_document_indexed(filename):  
                logger.info(f"Document '{filename}' already indexed, skipping processing")  
                  
                # Still associate with conversation if provided  
                if conversation_id and conversation_id in self._conversations:  
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
            if force_reprocess and self.vector_store.is_document_indexed(filename):  
                logger.info(f"Force reprocessing: deleting existing index for '{filename}'")  
                self.vector_store.delete_document(filename)  
              
            # Process the PDF (uses PDF cache if available)  
            logger.info(f"Processing document: {filename}")  
            chunks = self.pdf_processor.process(file_path)  
              
            if not chunks:  
                return {  
                    "status": "error",  
                    "message": f"No content extracted from {filename}"  
                }  
              
            # Add to vector store (with deduplication)  
            result = self.vector_store.add_documents(chunks, skip_duplicates=True)  
              
            # Associate with conversation  
            if conversation_id:  
                if conversation_id not in self._conversations:  
                    self.create_conversation()  
                    # Update the ID to match  
                    self._conversations[conversation_id] = self._conversations.pop(  
                        list(self._conversations.keys())[-1]  
                    )  
                    self._conversations[conversation_id]["id"] = conversation_id  
                  
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
        return self.vector_store.is_document_indexed(filename)  
      
    def delete_document(self, filename: str) -> Dict[str, Any]:  
        """  
        Delete a document from the index.  
          
        Args:  
            filename: The document filename  
              
        Returns:  
            Dict with status  
        """  
        result = self.vector_store.delete_document(filename)  
          
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
        method: str = "hyde"
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
              
        Returns:  
            Dict with answer and sources  
        """  
        try:  
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
            search_results = self.vector_store.search(query, top_k=top_k)  
              
            if not search_results:  
                return {  
                    "status": "success",  
                    "answer": "I couldn't find any relevant information to answer your question. Please make sure you've uploaded relevant documents.",  
                    "sources": [],  
                    "query": original_query  
                }  
              
            # Optionally rerank  
            if use_reranking and len(search_results) > 1:  
                try:  
                    search_results = self.reranker.rerank(  
                        query=query,  
                        results=search_results,  
                        top_k=rerank_top_k  
                    )  
                except Exception as e:  
                    logger.warning(f"Reranking failed: {e}")  
                    search_results = search_results[:rerank_top_k]  
            else:  
                search_results = search_results[:rerank_top_k]  
              
            # Prepare context for LLM  
            context_docs = []  
            for i, result in enumerate(search_results):  
                context_docs.append({  
                    "index": i + 1,  
                    "content": result.content,  
                    "metadata": result.metadata,  
                    "score": result.score  
                })  
              
            # Build prompt  
            if use_optimized_prompts:  
                try:  
                    prompt = self.prompt_optimizer.build_prompt(  
                        query=original_query,  
                        context_docs=context_docs,  
                        conversation_history=conversation_context  
                    )  
                except Exception as e:  
                    logger.warning(f"Prompt optimization failed: {e}")  
                    prompt = self._build_simple_prompt(original_query, context_docs)  
            else:  
                prompt = self._build_simple_prompt(original_query, context_docs)  
              
            # Generate answer  
            answer = self.llm_generator.generate(prompt)  
              
            # Store in conversation  
            if conversation_id:  
                if conversation_id not in self._conversations:  
                    self.create_conversation()  
                    self._conversations[conversation_id] = self._conversations.pop(  
                        list(self._conversations.keys())[-1]  
                    )  
                    self._conversations[conversation_id]["id"] = conversation_id  
                  
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
                "rewritten_query": query if query != original_query else None  
            }  
              
        except Exception as e:  
            logger.error(f"Query processing failed: {e}")  
            return {  
                "status": "error",  
                "message": str(e)  
            }  
      
    def _build_simple_prompt(  
        self,  
        query: str,  
        context_docs: List[Dict[str, Any]]  
    ) -> str:  
        """  
        Build a simple prompt without optimization.  
          
        Args:  
            query: User query  
            context_docs: Retrieved documents  
              
        Returns:  
            Formatted prompt  
        """  
        context_text = "\n\n".join([  
            f"[Document {doc['index']}]\n{doc['content']}"  
            for doc in context_docs  
        ])  
          
        return f"""Answer the following question based on the provided documents.  
If the documents don't contain relevant information, say so.  
  
Documents:  
{context_text}  
  
Question: {query}  
  
Answer:"""  
      
    # -------------------------------------------  
    # Statistics and management  
    # -------------------------------------------  
      
    def get_stats(self) -> Dict[str, Any]:  
        """  
        Get system statistics.  
          
        Returns:  
            Dict with statistics  
        """  
        vector_stats = self.vector_store.get_stats()  
          
        return {  
            "conversations": len(self._conversations),  
            "vector_store": vector_stats,  
            "components": {  
                "embedder": self._embedder is not None,  
                "vector_store": self._vector_store is not None,  
                "pdf_processor": self._pdf_processor is not None,  
                "llm_generator": self._llm_generator is not None,  
                "reranker": self._reranker is not None  
            }  
        }  
      
    def clear_index(self) -> Dict[str, Any]:  
        """  
        Clear the entire vector store index.  
          
        Returns:  
            Dict with status  
        """  
        return self.vector_store.clear()  