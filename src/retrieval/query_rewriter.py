"""  
src/retrieval/query_rewriter.py  
-------------------------------  
Query rewriting and expansion using LLM.  
"""  
  
import logging  
import json  
import re  
from typing import Dict, Any, Optional, List  
  
logger = logging.getLogger(__name__)  
  
  
class QueryRewriter:  
    """  
    Rewrites and expands queries for better retrieval.  
      
    Techniques:  
    - Query expansion with synonyms  
    - Hypothetical document generation (HyDE)  
    - Multi-query generation  
    - Query decomposition  
    """  
      
    def __init__(self, config=None):  
        """  
        Initialize the query rewriter.  
          
        Args:  
            config: Configuration object  
        """  
        from config.config import Config  
          
        self.config = config or Config()  
        self.llm = None  
        self._initialized = False  
      
    def _ensure_initialized(self):  
        """Lazy initialization of the LLM."""  
        if self._initialized:  
            return  
          
        try:  
            from langchain_openai import AzureChatOpenAI  
              
            azure_config = self.config.get_azure_openai_config()  
              
            if not all([azure_config.get("azure_endpoint"), azure_config.get("api_key")]):  
                logger.warning("Azure OpenAI not configured, query rewriting disabled")  
                self._initialized = True  
                return  
              
            self.llm = AzureChatOpenAI(  
                azure_endpoint=azure_config["azure_endpoint"],  
                api_key=azure_config["api_key"],  
                api_version=azure_config["api_version"],  
                azure_deployment=azure_config["azure_deployment"],  
                temperature=0.3,  
                max_tokens=500  
            )  
              
            self._initialized = True  
            logger.info("Query rewriter initialized")  
              
        except Exception as e:  
            logger.error(f"Failed to initialize query rewriter: {e}")  
            self._initialized = True  # Mark as initialized to prevent retries  
      
    def rewrite(  
        self,  
        query: str,  
        context: Optional[str] = None,  
        method: str = "expand"  
    ) -> Dict[str, Any]:  
        """  
        Rewrite a query for better retrieval.  
          
        Args:  
            query: Original user query  
            context: Optional conversation context  
            method: Rewriting method ('expand', 'hyde', 'multi', 'decompose')  
              
        Returns:  
            Dictionary with rewritten query and metadata  
        """  
        if not query or not query.strip():  
            return {  
                "original_query": query,  
                "rewritten_query": query,  
                "method": method,  
                "success": False,  
                "error": "Empty query"  
            }  
          
        self._ensure_initialized()  
          
        if not self.llm:  
            return {  
                "original_query": query,  
                "rewritten_query": query,  
                "method": method,  
                "success": False,  
                "error": "LLM not available"  
            }  
          
        try:  
            if method == "expand":  
                return self._expand_query(query, context)  
            elif method == "hyde":  
                return self._hyde_query(query, context)  
            elif method == "multi":  
                return self._multi_query(query, context)  
            elif method == "decompose":  
                return self._decompose_query(query, context)  
            else:  
                return self._expand_query(query, context)  
                  
        except Exception as e:  
            logger.error(f"Query rewriting failed: {e}")  
            return {  
                "original_query": query,  
                "rewritten_query": query,  
                "method": method,  
                "success": False,  
                "error": str(e)  
            }  
      
    def _extract_content(self, response) -> str:  
        """Safely extract string content from LLM response."""  
        try:  
            content = response.content  
              
            # If content is already a string, return it  
            if isinstance(content, str):  
                return content.strip()  
              
            # If content is a dict, try to extract text  
            if isinstance(content, dict):  
                # Common keys where text might be stored  
                for key in ['text', 'content', 'message', 'output']:  
                    if key in content and isinstance(content[key], str):  
                        return content[key].strip()  
                # If no known key, convert to string  
                return str(content)  
              
            # If content is a list (e.g., message chunks)  
            if isinstance(content, list):  
                text_parts = []  
                for item in content:  
                    if isinstance(item, str):  
                        text_parts.append(item)  
                    elif isinstance(item, dict) and 'text' in item:  
                        text_parts.append(item['text'])  
                return ''.join(text_parts).strip()  
              
            # Fallback: convert to string  
            return str(content).strip()  
              
        except Exception as e:  
            logger.error(f"Failed to extract content from response: {e}")  
            return ""  
  
    def _expand_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:  
        """Expand query with related terms and concepts."""  
        from langchain_core.messages import SystemMessage, HumanMessage  
  
        system_prompt = """You are a query expansion expert. Your task is to rewrite the user's query   
                        to improve document retrieval. Add relevant synonyms, related concepts, and clarifying terms.  
                        Keep the core intent but make it more comprehensive.  
                        Return ONLY the expanded query, nothing else."""  
  
        user_prompt = f"Original query: {query}"  
        if context:  
            user_prompt += f"\n\nConversation context: {context}"  
  
        response = self.llm.invoke([  
            SystemMessage(content=system_prompt),  
            HumanMessage(content=user_prompt)  
        ])  
  
        # ✅ FIX: Use helper method instead of direct .strip()  
        rewritten = self._extract_content(response)  
  
        return {  
            "original_query": query,  
            "rewritten_query": rewritten if rewritten else query,  
            "method": "expand",  
            "success": bool(rewritten)  
        }  
  
    def _hyde_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:  
        """Hypothetical Document Embeddings (HyDE)."""  
        from langchain_core.messages import SystemMessage, HumanMessage  
  
        system_prompt = """You are an expert at generating hypothetical document passages.  
                        Given a question, generate a short passage (2-3 sentences) that would answer the question.  
                        This passage will be used to find similar real documents.  
                        Generate ONLY the hypothetical passage, nothing else."""  
  
        user_prompt = f"Question: {query}"  
        if context:  
            user_prompt += f"\n\nContext: {context}"  
  
        response = self.llm.invoke([  
            SystemMessage(content=system_prompt),  
            HumanMessage(content=user_prompt)  
        ])  
  
        # ✅ FIX: Use helper method  
        hypothetical_doc = self._extract_content(response)  
  
        return {  
            "original_query": query,  
            "rewritten_query": hypothetical_doc if hypothetical_doc else query,  
            "hypothetical_document": hypothetical_doc,  
            "method": "hyde",  
            "success": bool(hypothetical_doc)  
        }  
  
    def _multi_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:  
        """Generate multiple query variations."""  
        from langchain_core.messages import SystemMessage, HumanMessage  
  
        system_prompt = """You are a query generation expert. Generate 3 different versions of the   
                            user's question that capture the same intent but use different wording.  
                            Return a JSON array of strings with the 3 query variations.  
                            Example: ["query 1", "query 2", "query 3"]"""  
  
        user_prompt = f"Original question: {query}"  
        if context:  
            user_prompt += f"\n\nContext: {context}"  
  
        response = self.llm.invoke([  
            SystemMessage(content=system_prompt),  
            HumanMessage(content=user_prompt)  
        ])  
  
        # ✅ FIX: Use helper method  
        content = self._extract_content(response)  
  
        # Parse JSON response  
        try:  
            # ✅ FIX: Correct regex for JSON array  
            json_match = re.search(r'$.*$', content, re.DOTALL)  
            if json_match:  
                queries = json.loads(json_match.group())  
            else:  
                queries = [query]  
        except json.JSONDecodeError:  
            queries = [query]  
  
        rewritten = queries[0] if queries else query  
  
        return {  
            "original_query": query,  
            "rewritten_query": rewritten,  
            "query_variations": queries,  
            "method": "multi",  
            "success": True  
        }  
  
    def _decompose_query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:  
        """Decompose complex query into sub-queries."""  
        from langchain_core.messages import SystemMessage, HumanMessage  
  
        system_prompt = """You are a query decomposition expert. Break down the user's complex   
                        question into simpler sub-questions that can be answered independently.  
                        Return a JSON object with:  
                        - "main_query": simplified main question  
                        - "sub_queries": array of sub-questions  
                        Example: {"main_query": "...", "sub_queries": ["...", "..."]}"""  
  
        user_prompt = f"Complex question: {query}"  
        if context:  
            user_prompt += f"\n\nContext: {context}"  
  
        response = self.llm.invoke([  
            SystemMessage(content=system_prompt),  
            HumanMessage(content=user_prompt)  
        ])  
  
        # ✅ FIX: Use helper method  
        content = self._extract_content(response)  
  
        # Parse JSON response  
        try:  
            json_match = re.search(r'\{.*\}', content, re.DOTALL)  
            if json_match:  
                result = json.loads(json_match.group())  
                main_query = result.get("main_query", query)  
                sub_queries = result.get("sub_queries", [])  
            else:  
                main_query = query  
                sub_queries = []  
        except json.JSONDecodeError:  
            main_query = query  
            sub_queries = []  
  
        return {  
            "original_query": query,  
            "rewritten_query": main_query,  
            "sub_queries": sub_queries,  
            "method": "decompose",  
            "success": True  
        }  