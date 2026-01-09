# """  
# src/answer_generator/llm_generator.py  
# -------------------------------------  
# LLM-based answer generation with multiple strategies.  
# """  
  
# import logging  
# import json  
# import re  
# import time  
# import hashlib  
# from typing import List, Dict, Any, Optional, Callable, AsyncIterator  
# from pathlib import Path  
# from functools import lru_cache  
  
# from jinja2 import Template  
  
# logger = logging.getLogger(__name__)  
  
  
# class LLMGenerator:  
#     """  
#     Advanced LLM-based answer generator.  
      
#     Features:  
#     - Primary and fallback models  
#     - Response caching  
#     - Self-critique and improvement  
#     - Source attribution  
#     - Streaming support  
#     - Jinja2 prompt templates  
#     """  
      
#     def __init__(self, config=None):  
#         """  
#         Initialize the LLM generator.  
          
#         Args:  
#             config: Configuration object  
#         """  
#         from config.config import Config  
#         from src.memory.token_counter import TokenCounter  
          
#         self.config = config or Config()  
#         self.token_counter = TokenCounter()  
          
#         # Load prompt templates  
#         self.templates = self._load_prompt_templates()  
          
#         # Models (lazy initialized)  
#         self._primary_model = None  
#         self._fallback_model = None  
#         self._initialized = False  
          
#         # Response cache  
#         self._response_cache: Dict[str, Dict[str, Any]] = {}  
#         self.cache_enabled = getattr(self.config, "RESPONSE_CACHE_ENABLED", True)  
#         self.cache_ttl = getattr(self.config, "RESPONSE_CACHE_TTL_SECONDS", 3600)  
#         self.cache_max_size = getattr(self.config, "RESPONSE_CACHE_MAX_SIZE", 100)  
      
#     def _load_prompt_templates(self) -> Dict[str, Template]:  
#         """Load Jinja2 prompt templates."""  
#         templates = {}  
#         template_dir = Path(getattr(self.config, "PROMPT_TEMPLATE_DIR", "config/prompts"))  
          
#         # Standard answer template  
#         standard_path = template_dir / "standard_answer.j2"  
#         if standard_path.exists():  
#             templates["standard"] = Template(standard_path.read_text())  
#         else:  
#             templates["standard"] = Template(self._default_standard_template())  
          
#         # Self-critique template  
#         critique_path = template_dir / "self_critique.j2"  
#         if critique_path.exists():  
#             templates["critique"] = Template(critique_path.read_text())  
#         else:  
#             templates["critique"] = Template(self._default_critique_template())  
          
#         # Source attribution template  
#         attribution_path = template_dir / "source_attribution.j2"  
#         if attribution_path.exists():  
#             templates["attribution"] = Template(attribution_path.read_text())  
#         else:  
#             templates["attribution"] = Template(self._default_attribution_template())  
          
#         return templates  
      
#     def _default_standard_template(self) -> str:  
#         return """You are a factual assistant specialized in answering based solely on retrieved document content.  
# Do not use prior knowledge or assumptions. If the answer is not in the provided text, say 'The information is not available in the provided context.'  
  
# {% if memory_context %}  
# CONVERSATION CONTEXT:  
# {{ memory_context }}  
# {% endif %}  
  
# USER QUESTION: {{ query }}  
  
# RELEVANT INFORMATION FROM DOCUMENTS:  
# {% for doc in documents %}  
# --- Document {{ loop.index }} ---  
# {{ doc.content }}  
# {% endfor %}  
  
# {% if reasoning %}  
# REASONING:  
# {{ reasoning }}  
# {% endif %}  
  
# Please provide a comprehensive and accurate answer to the user's question based ONLY on the information provided above."""  
      
#     def _default_critique_template(self) -> str:  
#         return """You are a critical evaluator checking for answer quality and factual accuracy.  
  
# USER QUESTION: {{ query }}  
  
# RELEVANT INFORMATION FROM DOCUMENTS:  
# {% for doc in documents %}  
# {{ doc.content }}  
# {% endfor %}  
  
# INITIAL ANSWER:  
# {{ initial_answer }}  
  
# Please evaluate this answer:  
# 1. Is the answer fully supported by the provided documents?  
# 2. Does it contain any information not found in the documents?  
# 3. Did the answer miss any important information?  
# 4. How could the answer be improved?"""  
      
#     def _default_attribution_template(self) -> str:  
#         return """Analyze which parts of the answer are supported by which source documents.  
  
# USER QUESTION: {{ query }}  
  
# SOURCE DOCUMENTS:  
# {% for doc in documents %}  
# [{{ doc.source_id }}]: {{ doc.content }}  
# {% endfor %}  
  
# ANSWER TO ANALYZE:  
# {{ answer }}  
  
# Return a JSON object with attributions for each claim."""  
      
#     def _ensure_initialized(self):  
#         """Lazy initialization of LLM models."""  
#         if self._initialized:  
#             return  
          
#         try:  
#             from langchain_openai import AzureChatOpenAI  
              
#             azure_config = self.config.get_azure_openai_config()  
              
#             if not all([azure_config.get("azure_endpoint"), azure_config.get("api_key")]):  
#                 logger.error("Azure OpenAI configuration incomplete")  
#                 self._initialized = True  
#                 return  
              
#             self._primary_model = AzureChatOpenAI(  
#                 azure_endpoint=azure_config["azure_endpoint"],  
#                 api_key=azure_config["api_key"],  
#                 api_version=azure_config["api_version"],  
#                 azure_deployment=azure_config["azure_deployment"],  
#                 temperature=getattr(self.config, "PRIMARY_TEMPERATURE", 0.2),  
#                 max_tokens=getattr(self.config, "PRIMARY_MAX_TOKENS", 2000),  
#                 request_timeout=getattr(self.config, "REQUEST_TIMEOUT", 60),  
#                 max_retries=getattr(self.config, "MAX_RETRIES", 2)  
#             )  
              
#             logger.info("Primary LLM model initialized")  
#             self._initialized = True  
              
#         except Exception as e:  
#             logger.error(f"Failed to initialize LLM: {e}")  
#             self._initialized = True  
      
#     @property  
#     def primary_model(self):  
#         """Get primary model (lazy loading)."""  
#         self._ensure_initialized()  
#         return self._primary_model  
      
#     def _get_fallback_model(self):  
#         """Get or create fallback model."""  
#         if self._fallback_model is not None:  
#             return self._fallback_model  
          
#         try:  
#             from langchain_openai import AzureChatOpenAI  
              
#             azure_config = self.config.get_azure_openai_config()  
              
#             self._fallback_model = AzureChatOpenAI(  
#                 azure_endpoint=azure_config["azure_endpoint"],  
#                 api_key=azure_config["api_key"],  
#                 api_version=azure_config["api_version"],  
#                 azure_deployment=azure_config["azure_deployment"],  
#                 temperature=getattr(self.config, "FALLBACK_TEMPERATURE", 0.3),  
#                 max_tokens=getattr(self.config, "FALLBACK_MAX_TOKENS", 1500)  
#             )  
              
#             return self._fallback_model  
              
#         except Exception as e:  
#             logger.error(f"Failed to initialize fallback model: {e}")  
#             return None  
      
#     def _prepare_memory_context(self, memory_context: Optional[Dict]) -> str:  
#         """Format memory context into a string."""  
#         if not memory_context:  
#             return ""  
          
#         parts = []  
          
#         if memory_context.get("summary"):  
#             parts.append(f"Conversation summary: {memory_context['summary']}")  
          
#         if memory_context.get("recent_messages"):  
#             parts.append("Recent conversation:")  
#             for msg in memory_context["recent_messages"]:  
#                 if isinstance(msg, dict):  
#                     role = msg.get("role", "").capitalize()  
#                     content = msg.get("content", "")  
#                     if role and content:  
#                         parts.append(f"  {role}: {content}")  
          
#         if memory_context.get("relevant_messages"):  
#             parts.append("Relevant past context:")  
#             for msg in memory_context["relevant_messages"]:  
#                 if isinstance(msg, dict):  
#                     role = msg.get("role", "").capitalize()  
#                     content = msg.get("content", "")  
#                     if role and content:  
#                         parts.append(f"  {role}: {content}")  
          
#         return "\n".join(parts)  
      
#     def _get_cache_key(  
#         self,  
#         query: str,  
#         documents: List[Dict[str, Any]],  
#         memory_context: Optional[Dict] = None  
#     ) -> str:  
#         """Generate cache key for query and context."""  
#         doc_content = "|".join([doc.get("content", "")[:100] for doc in documents[:3]])  
#         memory_str = str(memory_context)[:100] if memory_context else ""  
#         key_str = f"{query}|{doc_content}|{memory_str}"  
#         return hashlib.md5(key_str.encode()).hexdigest()  
      
#     def _check_cache(self, cache_key: str) -> Optional[str]:  
#         """Check if response is cached and not expired."""  
#         if not self.cache_enabled:  
#             return None  
          
#         entry = self._response_cache.get(cache_key)  
#         if not entry:  
#             return None  
          
#         if time.time() - entry.get("timestamp", 0) > self.cache_ttl:  
#             del self._response_cache[cache_key]  
#             return None  
          
#         return entry.get("response")  
      
#     def _update_cache(self, cache_key: str, response: str):  
#         """Add response to cache."""  
#         if not self.cache_enabled:  
#             return  
          
#         self._response_cache[cache_key] = {  
#             "response": response,  
#             "timestamp": time.time()  
#         }  
          
#         # Prune if too large  
#         if len(self._response_cache) > self.cache_max_size:  
#             sorted_keys = sorted(  
#                 self._response_cache.keys(),  
#                 key=lambda k: self._response_cache[k]["timestamp"]  
#             )  
#             for old_key in sorted_keys[:-self.cache_max_size]:  
#                 del self._response_cache[old_key]  
      
#     def generate_answer(  
#         self,  
#         query: str,  
#         documents: List[Dict[str, Any]],  
#         memory_context: Optional[Dict] = None,  
#         reasoning: Optional[Dict] = None,  
#         temperature: Optional[float] = None,  
#         max_tokens: Optional[int] = None,  
#         use_cache: bool = True  
#     ) -> str:  
#         """  
#         Generate an answer using LLM.  
          
#         Args:  
#             query: User query  
#             documents: Retrieved documents  
#             memory_context: Conversation memory context  
#             reasoning: Optional reasoning steps  
#             temperature: Override temperature  
#             max_tokens: Override max tokens  
#             use_cache: Whether to use caching  
              
#         Returns:  
#             Generated answer  
#         """  
#         # Validate inputs  
#         if not query or not query.strip():  
#             return "I don't see a question to answer. Please provide a question."  
          
#         if not documents:  
#             return "I don't have any relevant information to answer your question. Could you please provide more context or rephrase your question?"  
          
#         # Check cache  
#         if use_cache:  
#             cache_key = self._get_cache_key(query, documents, memory_context)  
#             cached = self._check_cache(cache_key)  
#             if cached:  
#                 logger.debug("Using cached response")  
#                 return cached  
          
#         # Prepare context  
#         formatted_memory = self._prepare_memory_context(memory_context)  
          
#         reasoning_str = ""  
#         if reasoning:  
#             if isinstance(reasoning, dict):  
#                 reasoning_str = reasoning.get("reasoning", "")  
#             elif isinstance(reasoning, str):  
#                 reasoning_str = reasoning  
          
#         # Render prompt  
#         try:  
#             prompt_text = self.templates["standard"].render(  
#                 query=query,  
#                 documents=documents,  
#                 memory_context=formatted_memory,  
#                 reasoning=reasoning_str  
#             )  
#         except Exception as e:  
#             logger.error(f"Template rendering failed: {e}")  
#             prompt_text = f"Question: {query}\n\nDocuments: {documents}\n\nAnswer:"  
          
#         # Generate response  
#         try:  
#             from langchain_core.messages import SystemMessage, HumanMessage  
              
#             if not self.primary_model:  
#                 return "I apologize, but the language model is not available."  
              
#             messages = [  
#                 SystemMessage(content=(  
#                     "You are a factual assistant that answers based on provided documents. "  
#                     "Be comprehensive yet concise."  
#                 )),  
#                 HumanMessage(content=prompt_text)  
#             ]  
              
#             kwargs = {}  
#             if temperature is not None:  
#                 kwargs["temperature"] = temperature  
#             if max_tokens is not None:  
#                 kwargs["max_tokens"] = max_tokens  
              
#             response = self.primary_model.invoke(messages, **kwargs)  
#             answer = response.content  
              
#         except Exception as e:  
#             logger.error(f"Primary model failed: {e}")  
              
#             # Try fallback  
#             try:  
#                 fallback = self._get_fallback_model()  
#                 if fallback:  
#                     response = fallback.invoke(messages)  
#                     answer = response.content  
#                 else:  
#                     answer = f"I apologize, but I encountered an error: {str(e)}"  
#             except Exception as e2:  
#                 logger.error(f"Fallback model failed: {e2}")  
#                 answer = "I apologize, but I encountered an error. Please try again later."  
          
#         # Update cache  
#         if use_cache:  
#             self._update_cache(cache_key, answer)  
          
#         return answer  
      
#     async def generate_answer_streaming(  
#         self,  
#         query: str,  
#         documents: List[Dict[str, Any]],  
#         memory_context: Optional[Dict] = None,  
#         reasoning: Optional[Dict] = None  
#     ) -> AsyncIterator[str]:  
#         """  
#         Generate a streaming answer.  
          
#         Args:  
#             query: User query  
#             documents: Retrieved documents  
#             memory_context: Memory context  
#             reasoning: Reasoning steps  
              
#         Yields:  
#             Answer tokens  
#         """  
#         if not query or not query.strip():  
#             yield "I don't see a question to answer."  
#             return  
          
#         if not documents:  
#             yield "I don't have any relevant information to answer your question."  
#             return  
          
#         # Prepare prompt  
#         formatted_memory = self._prepare_memory_context(memory_context)  
          
#         reasoning_str = ""  
#         if reasoning and isinstance(reasoning, dict):  
#             reasoning_str = reasoning.get("reasoning", "")  
          
#         try:  
#             prompt_text = self.templates["standard"].render(  
#                 query=query,  
#                 documents=documents,  
#                 memory_context=formatted_memory,  
#                 reasoning=reasoning_str  
#             )  
#         except Exception as e:  
#             logger.error(f"Template rendering failed: {e}")  
#             prompt_text = f"Question: {query}\n\nAnswer:"  
          
#         try:  
#             from langchain_core.messages import SystemMessage, HumanMessage  
              
#             if not self.primary_model:  
#                 yield "Language model not available."  
#                 return  
              
#             messages = [  
#                 SystemMessage(content="You are a factual assistant."),  
#                 HumanMessage(content=prompt_text)  
#             ]  
              
#             async for chunk in self.primary_model.astream(messages):  
#                 if chunk.content:  
#                     yield chunk.content  
                      
#         except Exception as e:  
#             logger.error(f"Streaming failed: {e}")  
#             yield f"\nError: {str(e)}"  
      
#     def generate_with_self_critique(  
#         self,  
#         query: str,  
#         documents: List[Dict[str, Any]],  
#         memory_context: Optional[Dict] = None,  
#         max_iterations: int = 2  
#     ) -> Dict[str, str]:  
#         """  
#         Generate answer with self-critique for improved accuracy.  
          
#         Args:  
#             query: User query  
#             documents: Retrieved documents  
#             memory_context: Memory context  
#             max_iterations: Maximum improvement iterations  
              
#         Returns:  
#             Dictionary with initial_answer, critique, improved_answer  
#         """  
#         # Generate initial answer  
#         initial_answer = self.generate_answer(  
#             query=query,  
#             documents=documents,  
#             memory_context=memory_context,  
#             use_cache=False  
#         )  
          
#         if not initial_answer or "error" in initial_answer.lower():  
#             return {  
#                 "initial_answer": initial_answer,  
#                 "critique": "Skipped due to error",  
#                 "improved_answer": initial_answer  
#             }  
          
#         try:  
#             from langchain_core.messages import SystemMessage, HumanMessage  
              
#             # Generate critique  
#             critique_prompt = self.templates["critique"].render(  
#                 query=query,  
#                 documents=documents,  
#                 initial_answer=initial_answer  
#             )  
              
#             critique_response = self.primary_model.invoke([  
#                 SystemMessage(content="You are a critical evaluator."),  
#                 HumanMessage(content=critique_prompt)  
#             ])  
#             critique = critique_response.content  
              
#             # Generate improved answer  
#             improvement_prompt = f"""  
# Based on this critique, provide an improved answer.  
  
# Original question: {query}  
  
# Initial answer: {initial_answer}  
  
# Critique: {critique}  
  
# Improved answer:"""  
              
#             improved_response = self.primary_model.invoke([  
#                 SystemMessage(content="You improve answers based on critique."),  
#                 HumanMessage(content=improvement_prompt)  
#             ])  
#             improved_answer = improved_response.content  
              
#             return {  
#                 "initial_answer": initial_answer,  
#                 "critique": critique,  
#                 "improved_answer": improved_answer  
#             }  
              
#         except Exception as e:  
#             logger.error(f"Self-critique failed: {e}")  
#             return {  
#                 "initial_answer": initial_answer,  
#                 "critique": f"Error: {str(e)}",  
#                 "improved_answer": initial_answer  
#             }  
      
#     def generate_with_source_attribution(  
#         self,  
#         query: str,  
#         documents: List[Dict[str, Any]],  
#         answer: Optional[str] = None  
#     ) -> Dict[str, Any]:  
#         """  
#         Generate answer with source attributions.  
          
#         Args:  
#             query: User query  
#             documents: Retrieved documents with source_id  
#             answer: Pre-generated answer (optional)  
              
#         Returns:  
#             Dictionary with answer and attributions  
#         """  
#         if not answer:  
#             answer = self.generate_answer(query, documents)  
          
#         # Ensure documents have source IDs  
#         docs_with_ids = []  
#         for i, doc in enumerate(documents):  
#             doc_copy = doc.copy()  
#             if "source_id" not in doc_copy:  
#                 metadata = doc_copy.get("metadata", {})  
#                 if metadata.get("filename") and metadata.get("chunk_id") is not None:  
#                     doc_copy["source_id"] = f"{metadata['filename']}_{metadata['chunk_id']}"  
#                 else:  
#                     doc_copy["source_id"] = f"doc_{i+1}"  
#             docs_with_ids.append(doc_copy)  
          
#         try:  
#             from langchain_core.messages import SystemMessage, HumanMessage  
              
#             attribution_prompt = self.templates["attribution"].render(  
#                 query=query,  
#                 documents=docs_with_ids,  
#                 answer=answer  
#             )  
              
#             response = self.primary_model.invoke([  
#                 SystemMessage(content="You analyze source attributions. Return valid JSON."),  
#                 HumanMessage(content=attribution_prompt)  
#             ])  
              
#             # Parse JSON from response  
#             content = response.content  
#             json_match = re.search(r'\{.*\}', content, re.DOTALL)  
#             if json_match:  
#                 attributions = json.loads(json_match.group())  
#             else:  
#                 attributions = {"error": "No JSON found", "raw": content}  
                  
#         except Exception as e:  
#             logger.error(f"Attribution failed: {e}")  
#             attributions = {"error": str(e)}  
          
#         return {  
#             "answer": answer,  
#             "attributions": attributions  
#         }  
      
#     @lru_cache(maxsize=32)  
#     def summarize_document(self, content: str, max_length: int = 200) -> str:  
#         """  
#         Summarize document content.  
          
#         Args:  
#             content: Document content  
#             max_length: Maximum summary length in words  
              
#         Returns:  
#             Summary  
#         """  
#         if not content or len(content.split()) <= max_length:  
#             return content  
          
#         try:  
#             from langchain_core.messages import SystemMessage, HumanMessage  
              
#             response = self.primary_model.invoke([  
#                 SystemMessage(content="Create concise document summaries."),  
#                 HumanMessage(content=f"Summarize in {max_length} words or less:\n\n{content}")  
#             ])  
              
#             summary = response.content  
#             words = summary.split()  
#             if len(words) > max_length:  
#                 summary = " ".join(words[:max_length]) + "..."  
              
#             return summary  
              
#         except Exception as e:  
#             logger.error(f"Summarization failed: {e}")  
#             return content[:max_length * 5] + "..."  


"""  
src/answer_generator/llm_generator.py  
-------------------------------------  
Multi-provider LLM generator supporting Azure OpenAI, Gemini, and OpenAI.  
"""  
  
import logging  
from typing import List, Dict, Any, Optional, Generator  
from abc import ABC, abstractmethod  
  
logger = logging.getLogger(__name__)  
  
  
class BaseLLM(ABC):  
    """Abstract base class for LLM providers."""  
      
    @abstractmethod  
    def generate(self, prompt: str) -> str:  
        """Generate a response for the given prompt."""  
        pass  
      
    @abstractmethod  
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:  
        """Generate a response using a list of messages."""  
        pass  
      
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:  
        """Generate a streaming response (optional)."""  
        yield self.generate(prompt)  
  
  
class AzureOpenAILLM(BaseLLM):  
    """Azure OpenAI LLM implementation."""  
      
    def __init__(self, config: Dict[str, Any]):  
        """  
        Initialize Azure OpenAI LLM.  
          
        Args:  
            config: Configuration dict with Azure settings  
        """  
        self.api_key = config.get("api_key")  
        self.azure_endpoint = config.get("azure_endpoint")  
        self.api_version = config.get("api_version", "2024-02-15-preview")  
        self.deployment_name = config.get("azure_deployment")  
        self.temperature = config.get("temperature", 0.0)  
        self.max_tokens = config.get("max_tokens", 2000)  
          
        self._llm = None  
          
        logger.info(f"AzureOpenAILLM initialized with deployment: {self.deployment_name}")  
      
    @property  
    def llm(self):  
        """Lazy load the LLM."""  
        if self._llm is None:  
            self._load_llm()  
        return self._llm  
      
    def _load_llm(self):  
        """Load the Azure OpenAI LLM."""  
        try:  
            from langchain_openai import AzureChatOpenAI  
              
            self._llm = AzureChatOpenAI(  
                azure_endpoint=self.azure_endpoint,  
                api_key=self.api_key,  
                api_version=self.api_version,  
                azure_deployment=self.deployment_name,  
                temperature=self.temperature,  
                max_tokens=self.max_tokens  
            )  
              
            logger.info("Azure OpenAI LLM loaded successfully")  
              
        except ImportError:  
            logger.error("langchain-openai not installed. Install with: pip install langchain-openai")  
            raise  
      
    def generate(self, prompt: str) -> str:  
        """Generate a response."""  
        from langchain_core.messages import HumanMessage  
          
        messages = [HumanMessage(content=prompt)]  
        response = self.llm.invoke(messages)  
          
        return response.content if hasattr(response, "content") else str(response)  
      
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:  
        """Generate a response using messages."""  
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  
          
        lc_messages = []  
        for msg in messages:  
            role = msg.get("role", "user").lower()  
            content = msg.get("content", "")  
              
            if role == "system":  
                lc_messages.append(SystemMessage(content=content))  
            elif role in ("assistant", "ai"):  
                lc_messages.append(AIMessage(content=content))  
            else:  
                lc_messages.append(HumanMessage(content=content))  
          
        response = self.llm.invoke(lc_messages)  
        return response.content if hasattr(response, "content") else str(response)  
      
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:  
        """Generate streaming response."""  
        from langchain_core.messages import HumanMessage  
          
        messages = [HumanMessage(content=prompt)]  
          
        for chunk in self.llm.stream(messages):  
            if hasattr(chunk, "content"):  
                yield chunk.content  
  
  
class GeminiLLM(BaseLLM):  
    """  
    Google Gemini LLM implementation.  
      
    Supports both:  
    - New: google-genai package (recommended)  
    - Legacy: google-generativeai package (deprecated)  
    """  
      
    def __init__(self, config: Dict[str, Any]):  
        """  
        Initialize Gemini LLM.  
          
        Args:  
            config: Configuration dict with Gemini settings  
        """  
        self.api_key = config.get("api_key")  
        self.model_name = config.get("model_name", "gemini-2.5-flash")  
        self.temperature = config.get("temperature", 0.0)  
        self.max_tokens = config.get("max_tokens", 2000)  
          
        self._client = None  
        self._model = None  
        self._use_new_sdk = True  # Try new SDK first  
          
        logger.info(f"GeminiLLM initialized with model: {self.model_name}")  
      
    @property  
    def model(self):  
        """Lazy load the model."""  
        if self._model is None and self._client is None:  
            self._load_model()  
        return self._model or self._client  
      
    def _load_model(self):  
        """Load the Gemini model using the appropriate SDK."""  
        # Try new google-genai package first  
        try:  
            self._load_new_sdk()  
            self._use_new_sdk = True  
            logger.info("Using new google-genai SDK")  
            return  
        except ImportError:  
            logger.info("google-genai not found, trying legacy SDK")  
        except Exception as e:  
            logger.warning(f"New SDK failed: {e}, trying legacy SDK")  
          
        # Fall back to legacy google-generativeai package  
        try:  
            self._load_legacy_sdk()  
            self._use_new_sdk = False  
            logger.info("Using legacy google-generativeai SDK (deprecated)")  
            return  
        except ImportError:  
            logger.error(  
                "Neither google-genai nor google-generativeai is installed.\n"  
                "Install with: pip install google-genai"  
            )  
            raise  
      
    def _load_new_sdk(self):  
        """Load using the new google-genai package."""  
        from google import genai  
        from google.genai import types  
          
        # Create client  
        self._client = genai.Client(api_key=self.api_key)  
          
        # Store config for generation  
        self._generation_config = types.GenerateContentConfig(  
            temperature=self.temperature,  
            max_output_tokens=self.max_tokens,  
        )  
          
        logger.info(f"Gemini client initialized with model: {self.model_name}")  
      
    def _load_legacy_sdk(self):  
        """Load using the legacy google-generativeai package."""  
        import warnings  
          
        # Suppress the deprecation warning since we're handling it  
        with warnings.catch_warnings():  
            warnings.filterwarnings("ignore", category=FutureWarning)  
            import google.generativeai as genai  
          
        # Configure API key  
        genai.configure(api_key=self.api_key)  
          
        # Create model with generation config  
        generation_config = genai.GenerationConfig(  
            temperature=self.temperature,  
            max_output_tokens=self.max_tokens,  
        )  
          
        self._model = genai.GenerativeModel(  
            model_name=self.model_name,  
            generation_config=generation_config  
        )  
          
        logger.info(f"Gemini model loaded (legacy): {self.model_name}")  
      
    def generate(self, prompt: str) -> str:  
        """Generate a response."""  
        # Ensure model is loaded  
        _ = self.model  
          
        try:  
            if self._use_new_sdk:  
                return self._generate_new_sdk(prompt)  
            else:  
                return self._generate_legacy_sdk(prompt)  
        except Exception as e:  
            logger.error(f"Gemini generation failed: {e}")  
            raise  
      
    def _generate_new_sdk(self, prompt: str) -> str:  
        """Generate using new SDK."""  
        from google.genai import types  
          
        response = self._client.models.generate_content(  
            model=self.model_name,  
            contents=prompt,  
            config=self._generation_config  
        )  
          
        # Extract text from response  
        if hasattr(response, "text"):  
            return response.text  
        elif hasattr(response, "candidates") and response.candidates:  
            candidate = response.candidates[0]  
            if hasattr(candidate, "content") and candidate.content.parts:  
                return candidate.content.parts[0].text  
          
        return str(response)  
      
    def _generate_legacy_sdk(self, prompt: str) -> str:  
        """Generate using legacy SDK."""  
        response = self._model.generate_content(prompt)  
          
        if hasattr(response, "text"):  
            return response.text  
        elif hasattr(response, "parts"):  
            return "".join(part.text for part in response.parts)  
          
        return str(response)  
      
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:  
        """Generate a response using messages."""  
        # Ensure model is loaded  
        _ = self.model  
          
        try:  
            if self._use_new_sdk:  
                return self._generate_with_messages_new_sdk(messages)  
            else:  
                return self._generate_with_messages_legacy_sdk(messages)  
        except Exception as e:  
            logger.error(f"Gemini chat generation failed: {e}")  
            # Fallback to simple generation  
            full_prompt = self._messages_to_prompt(messages)  
            return self.generate(full_prompt)  
      
    def _generate_with_messages_new_sdk(self, messages: List[Dict[str, str]]) -> str:  
        """Generate with messages using new SDK."""  
        from google.genai import types  
          
        # Convert messages to Gemini format  
        contents = []  
        for msg in messages:  
            role = msg.get("role", "user").lower()  
            content = msg.get("content", "")  
              
            # Map roles  
            if role in ("user", "human"):  
                gemini_role = "user"  
            elif role in ("assistant", "ai", "model"):  
                gemini_role = "model"  
            else:  
                gemini_role = "user"  
              
            contents.append(types.Content(  
                role=gemini_role,  
                parts=[types.Part(text=content)]  
            ))  
          
        response = self._client.models.generate_content(  
            model=self.model_name,  
            contents=contents,  
            config=self._generation_config  
        )  
          
        if hasattr(response, "text"):  
            return response.text  
        elif hasattr(response, "candidates") and response.candidates:  
            candidate = response.candidates[0]  
            if hasattr(candidate, "content") and candidate.content.parts:  
                return candidate.content.parts[0].text  
          
        return str(response)  
      
    def _generate_with_messages_legacy_sdk(self, messages: List[Dict[str, str]]) -> str:  
        """Generate with messages using legacy SDK."""  
        # Convert to chat format  
        history = []  
        for msg in messages[:-1]:  
            role = msg.get("role", "user").lower()  
            content = msg.get("content", "")  
              
            if role in ("user", "human"):  
                history.append({"role": "user", "parts": [content]})  
            elif role in ("assistant", "ai", "model"):  
                history.append({"role": "model", "parts": [content]})  
          
        # Start chat with history  
        chat = self._model.start_chat(history=history)  
          
        # Send last message  
        if messages:  
            last_content = messages[-1].get("content", "")  
            response = chat.send_message(last_content)  
            return response.text if hasattr(response, "text") else str(response)  
          
        return ""  
      
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:  
        """Convert messages to a single prompt string."""  
        parts = []  
        for msg in messages:  
            role = msg.get("role", "user").upper()  
            content = msg.get("content", "")  
            parts.append(f"{role}: {content}")  
        return "\n\n".join(parts)  
      
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:  
        """Generate streaming response."""  
        # Ensure model is loaded  
        _ = self.model  
          
        try:  
            if self._use_new_sdk:  
                yield from self._generate_stream_new_sdk(prompt)  
            else:  
                yield from self._generate_stream_legacy_sdk(prompt)  
        except Exception as e:  
            logger.error(f"Gemini streaming failed: {e}")  
            yield self.generate(prompt)  
      
    def _generate_stream_new_sdk(self, prompt: str) -> Generator[str, None, None]:  
        """Stream using new SDK."""  
        response = self._client.models.generate_content_stream(  
            model=self.model_name,  
            contents=prompt,  
            config=self._generation_config  
        )  
          
        for chunk in response:  
            if hasattr(chunk, "text"):  
                yield chunk.text  
            elif hasattr(chunk, "candidates") and chunk.candidates:  
                candidate = chunk.candidates[0]  
                if hasattr(candidate, "content") and candidate.content.parts:  
                    yield candidate.content.parts[0].text  
      
    def _generate_stream_legacy_sdk(self, prompt: str) -> Generator[str, None, None]:  
        """Stream using legacy SDK."""  
        response = self._model.generate_content(prompt, stream=True)  
          
        for chunk in response:  
            if hasattr(chunk, "text"):  
                yield chunk.text  
            elif hasattr(chunk, "parts"):  
                for part in chunk.parts:  
                    yield part.text  
  
  
class OpenAILLM(BaseLLM):  
    """OpenAI LLM implementation (direct, not Azure)."""  
      
    def __init__(self, config: Dict[str, Any]):  
        """  
        Initialize OpenAI LLM.  
          
        Args:  
            config: Configuration dict with OpenAI settings  
        """  
        self.api_key = config.get("api_key")  
        self.model_name = config.get("model_name", "gpt-4")  
        self.temperature = config.get("temperature", 0.0)  
        self.max_tokens = config.get("max_tokens", 2000)  
          
        self._llm = None  
          
        logger.info(f"OpenAILLM initialized with model: {self.model_name}")  
      
    @property  
    def llm(self):  
        """Lazy load the LLM."""  
        if self._llm is None:  
            self._load_llm()  
        return self._llm  
      
    def _load_llm(self):  
        """Load the OpenAI LLM."""  
        try:  
            from langchain_openai import ChatOpenAI  
              
            self._llm = ChatOpenAI(  
                api_key=self.api_key,  
                model=self.model_name,  
                temperature=self.temperature,  
                max_tokens=self.max_tokens  
            )  
              
            logger.info("OpenAI LLM loaded successfully")  
              
        except ImportError:  
            logger.error("langchain-openai not installed. Install with: pip install langchain-openai")  
            raise  
      
    def generate(self, prompt: str) -> str:  
        """Generate a response."""  
        from langchain_core.messages import HumanMessage  
          
        messages = [HumanMessage(content=prompt)]  
        response = self.llm.invoke(messages)  
          
        return response.content if hasattr(response, "content") else str(response)  
      
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:  
        """Generate a response using messages."""  
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  
          
        lc_messages = []  
        for msg in messages:  
            role = msg.get("role", "user").lower()  
            content = msg.get("content", "")  
              
            if role == "system":  
                lc_messages.append(SystemMessage(content=content))  
            elif role in ("assistant", "ai"):  
                lc_messages.append(AIMessage(content=content))  
            else:  
                lc_messages.append(HumanMessage(content=content))  
          
        response = self.llm.invoke(lc_messages)  
        return response.content if hasattr(response, "content") else str(response)  
  
  
class LLMGenerator:  
    """  
    Multi-provider LLM generator.  
      
    Factory class that creates the appropriate LLM based on configuration.  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the LLM generator.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
          
        # Get LLM configuration  
        self.llm_config = self.config.get_llm_config()  
        self.provider = self.llm_config.get("provider", "azure")  
          
        # Lazy load LLM  
        self._llm: Optional[BaseLLM] = None  
          
        logger.info(f"LLMGenerator initialized with provider: {self.provider}")  
      
    @property  
    def llm(self) -> BaseLLM:  
        """Lazy load the appropriate LLM."""  
        if self._llm is None:  
            self._llm = self._create_llm()  
        return self._llm  
      
    def _create_llm(self) -> BaseLLM:  
        """Create the appropriate LLM based on provider."""  
        provider = self.provider  
          
        if provider == "azure":  
            return AzureOpenAILLM(self.llm_config)  
        elif provider == "gemini":  
            return GeminiLLM(self.llm_config)  
        elif provider == "openai":  
            return OpenAILLM(self.llm_config)  
        else:  
            raise ValueError(  
                f"Unknown LLM provider: {provider}. "  
                "Supported providers: azure, gemini, openai"  
            )  
      
    def generate(self, prompt: str) -> str:  
        """  
        Generate a response for the given prompt.  
          
        Args:  
            prompt: The input prompt  
              
        Returns:  
            Generated response text  
        """  
        if not prompt or not prompt.strip():  
            logger.warning("Empty prompt provided")  
            return ""  
          
        try:  
            return self.llm.generate(prompt)  
        except Exception as e:  
            logger.error(f"LLM generation failed: {e}")  
            raise  
      
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:  
        """  
        Generate a response using a list of messages.  
          
        Args:  
            messages: List of message dicts with 'role' and 'content'  
              
        Returns:  
            Generated response text  
        """  
        if not messages:  
            logger.warning("Empty messages list provided")  
            return ""  
          
        try:  
            return self.llm.generate_with_messages(messages)  
        except Exception as e:  
            logger.error(f"LLM generation with messages failed: {e}")  
            raise  
      
    def generate_stream(self, prompt: str) -> Generator[str, None, None]:  
        """  
        Generate a streaming response.  
          
        Args:  
            prompt: The input prompt  
              
        Yields:  
            Response chunks  
        """  
        if not prompt or not prompt.strip():  
            return  
          
        try:  
            yield from self.llm.generate_stream(prompt)  
        except Exception as e:  
            logger.error(f"LLM streaming failed: {e}")  
            raise  
      
    def count_tokens(self, text: str) -> int:  
        """  
        Count the number of tokens in text.  
          
        Args:  
            text: Input text  
              
        Returns:  
            Token count  
        """  
        try:  
            import tiktoken  
            encoding = tiktoken.get_encoding("cl100k_base")  
            return len(encoding.encode(text))  
        except ImportError:  
            # Rough estimate  
            return len(text) // 4  
      
    def get_model_info(self) -> Dict[str, Any]:  
        """  
        Get information about the configured model.  
          
        Returns:  
            Dict with model information  
        """  
        info = {  
            "provider": self.provider,  
            "loaded": self._llm is not None,  
        }  
          
        if self.provider == "azure":  
            info["deployment"] = self.llm_config.get("azure_deployment")  
            info["endpoint"] = self.llm_config.get("azure_endpoint")  
        elif self.provider == "gemini":  
            info["model"] = self.llm_config.get("model_name")  
        elif self.provider == "openai":  
            info["model"] = self.llm_config.get("model_name")  
          
        info["temperature"] = self.llm_config.get("temperature")  
        info["max_tokens"] = self.llm_config.get("max_tokens")  
          
        return info  