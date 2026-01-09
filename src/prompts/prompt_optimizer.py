# """  
# src/prompts/prompt_optimizer.py  
# -------------------------------  
# Prompt optimization for improved RAG performance.  
# """  
  
# import logging  
# from typing import Dict, List, Any, Optional  
  
# logger = logging.getLogger(__name__)  
  
  
# class PromptOptimizer:  
#     """  
#     Optimizes prompts for better LLM responses.  
      
#     Features:  
#     - Context-aware prompt construction  
#     - Token budget management  
#     - Few-shot example injection  
#     - Instruction optimization  
#     """  
      
#     def __init__(self, config=None):  
#         """  
#         Initialize the prompt optimizer.  
          
#         Args:  
#             config: Configuration object  
#         """  
#         from config.config import Config  
#         from src.memory.token_counter import TokenCounter  
          
#         self.config = config or Config()  
#         self.token_counter = TokenCounter()  
#         self.max_context_tokens = getattr(self.config, "MAX_TOKEN_LIMIT", 4000)  
      
#     def optimize_rag_prompt(  
#         self,  
#         query: str,  
#         documents: List[Dict[str, Any]],  
#         memory_context: Optional[Dict] = None,  
#         max_tokens: Optional[int] = None  
#     ) -> Dict[str, Any]:  
#         """  
#         Optimize a RAG prompt within token budget.  
          
#         Args:  
#             query: User query  
#             documents: Retrieved documents  
#             memory_context: Conversation context  
#             max_tokens: Maximum tokens for prompt  
              
#         Returns:  
#             Optimized prompt components  
#         """  
#         max_tokens = max_tokens or self.max_context_tokens  
          
#         # Calculate token budgets  
#         query_tokens = self.token_counter.count_tokens(query)  
#         overhead = 200  # System message, formatting  
          
#         available = max_tokens - query_tokens - overhead  
          
#         # Allocate tokens  
#         memory_budget = int(available * 0.2)  
#         document_budget = available - memory_budget  
          
#         # Optimize memory context  
#         optimized_memory = ""  
#         if memory_context:  
#             optimized_memory = self._optimize_memory(memory_context, memory_budget)  
          
#         # Optimize documents  
#         optimized_docs = self._optimize_documents(documents, document_budget)  
          
#         return {  
#             "query": query,  
#             "documents": optimized_docs,  
#             "memory_context": optimized_memory,  
#             "token_usage": {  
#                 "query": query_tokens,  
#                 "memory": self.token_counter.count_tokens(optimized_memory),  
#                 "documents": sum(  
#                     self.token_counter.count_tokens(d.get("content", ""))  
#                     for d in optimized_docs  
#                 ),  
#                 "total": max_tokens  
#             }  
#         }  
      
#     def _optimize_memory(self, memory_context: Dict, max_tokens: int) -> str:  
#         """Optimize memory context to fit token budget."""  
#         parts = []  
#         current_tokens = 0  
          
#         # Add summary first (highest priority)  
#         if memory_context.get("summary"):  
#             summary = memory_context["summary"]  
#             summary_tokens = self.token_counter.count_tokens(summary)  
#             if summary_tokens <= max_tokens * 0.4:  
#                 parts.append(f"Summary: {summary}")  
#                 current_tokens += summary_tokens  
          
#         # Add recent messages  
#         remaining = max_tokens - current_tokens  
#         if memory_context.get("recent_messages") and remaining > 50:  
#             recent_parts = []  
#             for msg in reversed(memory_context["recent_messages"]):  
#                 if isinstance(msg, dict):  
#                     content = f"{msg.get('role', 'User')}: {msg.get('content', '')}"  
#                     msg_tokens = self.token_counter.count_tokens(content)  
#                     if current_tokens + msg_tokens <= max_tokens:  
#                         recent_parts.insert(0, content)  
#                         current_tokens += msg_tokens  
#                     else:  
#                         break  
              
#             if recent_parts:  
#                 parts.append("Recent:\n" + "\n".join(recent_parts))  
          
#         return "\n\n".join(parts)  
      
#     def _optimize_documents(  
#         self,  
#         documents: List[Dict[str, Any]],  
#         max_tokens: int  
#     ) -> List[Dict[str, Any]]:  
#         """Optimize documents to fit token budget."""  
#         if not documents:  
#             return []  
          
#         optimized = []  
#         current_tokens = 0  
          
#         # Sort by score if available  
#         sorted_docs = sorted(  
#             documents,  
#             key=lambda x: x.get("score", 0),  
#             reverse=True  
#         )  
          
#         for doc in sorted_docs:  
#             content = doc.get("content", "")  
#             doc_tokens = self.token_counter.count_tokens(content)  
              
#             if current_tokens + doc_tokens <= max_tokens:  
#                 optimized.append(doc)  
#                 current_tokens += doc_tokens  
#             else:  
#                 # Try to fit partial document  
#                 remaining = max_tokens - current_tokens  
#                 if remaining > 100:  
#                     trimmed_content = self.token_counter.trim_to_token_limit(  
#                         content, remaining  
#                     )  
#                     trimmed_doc = doc.copy()  
#                     trimmed_doc["content"] = trimmed_content  
#                     trimmed_doc["truncated"] = True  
#                     optimized.append(trimmed_doc)  
#                 break  
          
#         return optimized  
      
#     def create_system_prompt(  
#         self,  
#         task_type: str = "qa",  
#         include_examples: bool = False  
#     ) -> str:  
#         """  
#         Create an optimized system prompt.  
          
#         Args:  
#             task_type: Type of task (qa, summarize, extract)  
#             include_examples: Whether to include few-shot examples  
              
#         Returns:  
#             System prompt string  
#         """  
#         prompts = {  
#             "qa": """You are a precise question-answering assistant. Your role is to:  
# 1. Answer questions based ONLY on the provided context  
# 2. If the answer is not in the context, say "I don't have enough information"  
# 3. Be concise but comprehensive  
# 4. Cite specific parts of the context when relevant""",  
              
#             "summarize": """You are a summarization expert. Your role is to:  
# 1. Create clear, concise summaries of provided text  
# 2. Preserve key information and main points  
# 3. Maintain factual accuracy  
# 4. Use clear, professional language""",  
              
#             "extract": """You are an information extraction specialist. Your role is to:  
# 1. Extract specific information as requested  
# 2. Return structured data when appropriate  
# 3. Only extract information that is explicitly stated  
# 4. Indicate when requested information is not found"""  
#         }  
          
#         base_prompt = prompts.get(task_type, prompts["qa"])  
          
#         if include_examples and task_type == "qa":  
#             base_prompt += """  
  
# Example:  
# Context: "The company was founded in 2015 by Jane Smith."  
# Question: "When was the company founded?"  
# Answer: "The company was founded in 2015."  
# """  
          
#         return base_prompt  

"""  
src/prompts/prompt_optimizer.py  
-------------------------------  
Prompt optimization and template management for RAG.  
"""  
  
import logging  
from typing import List, Dict, Any, Optional  
from pathlib import Path  
  
logger = logging.getLogger(__name__)  
  
  
# Default prompt templates - IMPROVED for better answers  
DEFAULT_TEMPLATES = {  
    "rag_qa": """You are a knowledgeable AI assistant. Answer the user's question based on the provided document excerpts.  
  
IMPORTANT INSTRUCTIONS:  
1. Use ONLY information from the provided documents to answer  
2. If the documents contain relevant information, provide a comprehensive answer  
3. Quote or paraphrase specific passages when helpful  
4. If the documents don't contain enough information to fully answer, say what you CAN answer based on the documents, then note what's missing  
5. Reference document numbers when citing information (e.g., "According to Document 1...")  
  
DOCUMENTS:  
{context}  
  
USER QUESTION: {query}  
  
ANSWER:""",  
  
    "rag_qa_with_history": """You are a knowledgeable AI assistant. Answer the user's question based on the provided document excerpts and conversation context.  
  
IMPORTANT INSTRUCTIONS:  
1. Use ONLY information from the provided documents to answer  
2. Consider the conversation history for context about what the user is asking  
3. If the documents contain relevant information, provide a comprehensive answer  
4. Quote or paraphrase specific passages when helpful  
5. Reference document numbers when citing information  
  
CONVERSATION HISTORY:  
{history}  
  
DOCUMENTS:  
{context}  
  
USER QUESTION: {query}  
  
ANSWER:""",  
  
    "summarize": """Summarize the key points from the following documents:  
  
{context}  
  
SUMMARY:""",  
}  
  
  
class PromptOptimizer:  
    """  
    Prompt optimizer for RAG systems.  
    """  
      
    def __init__(self, config=None):  
        """Initialize the prompt optimizer."""  
        from config.config import Config  
          
        self.config = config or Config()  
          
        # Template settings  
        self.templates_dir = Path(getattr(  
            self.config,   
            "PROMPTS_DIR",   
            "src/prompts/templates"  
        ))  
          
        # Load templates  
        self.templates: Dict[str, str] = {}  
        self._load_templates()  
          
        # Context settings - INCREASED for better context  
        self.max_context_length = getattr(self.config, "MAX_CONTEXT_LENGTH", 8000)  
        self.max_history_turns = getattr(self.config, "MAX_HISTORY_TURNS", 5)  
        self.max_chars_per_doc = getattr(self.config, "MAX_CHARS_PER_DOC", 1500)  
          
        logger.info(f"PromptOptimizer initialized with {len(self.templates)} templates")  
      
    def _load_templates(self):  
        """Load prompt templates from files and defaults."""  
        self.templates = DEFAULT_TEMPLATES.copy()  
          
        if self.templates_dir.exists():  
            for template_file in self.templates_dir.glob("*.txt"):  
                try:  
                    template_name = template_file.stem  
                    template_content = template_file.read_text(encoding="utf-8")  
                    self.templates[template_name] = template_content  
                    logger.debug(f"Loaded template: {template_name}")  
                except Exception as e:  
                    logger.warning(f"Failed to load template {template_file}: {e}")  
      
    def build_prompt(  
        self,  
        query: str,  
        context_docs: List[Dict[str, Any]],  
        conversation_history: Optional[List[Dict[str, Any]]] = None,  
        template_name: str = "rag_qa",  
        **kwargs  
    ) -> str:  
        """  
        Build an optimized prompt for the LLM.  
          
        Args:  
            query: User's question  
            context_docs: Retrieved documents with 'content' and optional 'metadata'  
            conversation_history: Optional list of previous messages  
            template_name: Name of template to use  
            **kwargs: Additional template variables  
              
        Returns:  
            Formatted prompt string  
        """  
        # Select template  
        if conversation_history and len(conversation_history) > 0:  
            template_name = "rag_qa_with_history"  
          
        template = self.templates.get(template_name, DEFAULT_TEMPLATES["rag_qa"])  
          
        # Format context - IMPROVED to show more content  
        context = self._format_context(context_docs)  
          
        # Format history  
        history = ""  
        if conversation_history:  
            history = self._format_history(conversation_history)  
          
        # Build prompt  
        try:  
            prompt = template.format(  
                query=query,  
                context=context,  
                history=history,  
                **kwargs  
            )  
        except KeyError as e:  
            logger.warning(f"Template key error: {e}, using simple format")  
            prompt = f"Documents:\n{context}\n\nQuestion: {query}\n\nAnswer:"  
          
        logger.debug(f"Built prompt with {len(context)} chars of context")  
          
        return prompt  
      
    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:  
        """  
        Format context documents into a string.  
          
        IMPROVED: Shows more content per document for better answers.  
        """  
        if not context_docs:  
            return "No relevant documents found."  
          
        formatted_parts = []  
          
        for doc in context_docs:  
            # Get document index  
            index = doc.get("index", len(formatted_parts) + 1)  
              
            # Get content - SHOW MORE CONTENT  
            content = doc.get("content", "")  
            if len(content) > self.max_chars_per_doc:  
                content = content[:self.max_chars_per_doc] + "..."  
              
            # Get metadata  
            metadata = doc.get("metadata", {})  
            page = metadata.get("page_number", "")  
            filename = metadata.get("filename", "")  
            score = doc.get("score", 0)  
              
            # Format document reference  
            header_parts = [f"--- Document {index} ---"]  
            if filename:  
                header_parts.append(f"Source: {filename}")  
            if page:  
                header_parts.append(f"Page: {page}")  
            if score:  
                header_parts.append(f"Relevance: {score:.2f}")  
              
            header = " | ".join(header_parts)  
              
            formatted_parts.append(f"{header}\n{content}")  
          
        return "\n\n".join(formatted_parts)  
      
    def _format_history(self, history: List[Dict[str, Any]]) -> str:  
        """Format conversation history into a string."""  
        if not history:  
            return "No previous conversation."  
          
        recent_history = history[-self.max_history_turns * 2:]  
          
        formatted_parts = []  
        for msg in recent_history:  
            role = msg.get("role", "user").upper()  
            content = msg.get("content", "")  
              
            if len(content) > 300:  
                content = content[:300] + "..."  
              
            formatted_parts.append(f"{role}: {content}")  
          
        return "\n".join(formatted_parts)  
      
    def get_template(self, name: str) -> Optional[str]:  
        """Get a template by name."""  
        return self.templates.get(name)  
      
    def add_template(self, name: str, template: str):  
        """Add or update a template."""  
        self.templates[name] = template  
        logger.info(f"Added template: {name}")  
      
    def list_templates(self) -> List[str]:  
        """List available template names."""  
        return list(self.templates.keys())  