"""  
src/memory/token_counter.py  
---------------------------  
Token counting and management utilities.  
"""  
  
import logging  
from typing import Dict, List, Any, Optional  
  
logger = logging.getLogger(__name__)  
  
  
class TokenCounter:  
    """  
    Utility for counting and managing token usage.  
      
    Uses tiktoken for accurate GPT token counting with  
    fallback to word-based estimation.  
    """  
      
    def __init__(self, model: str = "gpt-4"):  
        """  
        Initialize the token counter.  
          
        Args:  
            model: Model name for tokenizer selection  
        """  
        self.model = model  
        self.tokenizer = None  
        self._initialize_tokenizer()  
      
    def _initialize_tokenizer(self):  
        """Initialize the tokenizer."""  
        try:  
            import tiktoken  
              
            # Try to get encoding for specific model  
            try:  
                self.tokenizer = tiktoken.encoding_for_model(self.model)  
            except KeyError:  
                # Fall back to cl100k_base (GPT-4 / GPT-3.5)  
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  
              
            logger.debug(f"Tiktoken initialized for model: {self.model}")  
              
        except ImportError:  
            logger.warning("tiktoken not available, using word-based estimation")  
            self.tokenizer = None  
      
    def count_tokens(self, text: str) -> int:  
        """  
        Count tokens in a text string.  
          
        Args:  
            text: Text to count tokens for  
              
        Returns:  
            Token count  
        """  
        if not text:  
            return 0  
          
        if self.tokenizer:  
            try:  
                return len(self.tokenizer.encode(text))  
            except Exception as e:  
                logger.debug(f"Tokenization failed, using fallback: {e}")  
          
        # Fallback: approximate 1 token per 4 characters  
        return max(1, len(text) // 4)  
      
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:  
        """  
        Count tokens in a list of chat messages.  
          
        Args:  
            messages: List of message dicts with 'role' and 'content'  
              
        Returns:  
            Total token count including overhead  
        """  
        total = 0  
          
        for message in messages:  
            # Message overhead (role, formatting)  
            total += 4  
              
            for key, value in message.items():  
                if isinstance(value, str):  
                    total += self.count_tokens(value)  
                    total += 1  # Key overhead  
          
        # Conversation overhead  
        total += 2  
          
        return total  
      
    def trim_to_token_limit(  
        self,  
        text: str,  
        max_tokens: int,  
        from_end: bool = False  
    ) -> str:  
        """  
        Trim text to fit within token limit.  
          
        Args:  
            text: Text to trim  
            max_tokens: Maximum allowed tokens  
            from_end: If True, keep end of text; otherwise keep beginning  
              
        Returns:  
            Trimmed text  
        """  
        if not text or max_tokens <= 0:  
            return ""  
          
        current_tokens = self.count_tokens(text)  
        if current_tokens <= max_tokens:  
            return text  
          
        if self.tokenizer:  
            try:  
                tokens = self.tokenizer.encode(text)  
                if from_end:  
                    kept_tokens = tokens[-max_tokens:]  
                else:  
                    kept_tokens = tokens[:max_tokens]  
                return self.tokenizer.decode(kept_tokens)  
            except Exception:  
                pass  
          
        # Fallback: character-based trimming  
        char_limit = max_tokens * 4  # Approximate  
        if from_end:  
            return "..." + text[-char_limit:]  
        else:  
            return text[:char_limit] + "..."  
      
    def get_memory_token_usage(self, memory_digest: Dict[str, Any]) -> Dict[str, int]:  
        """  
        Calculate token usage of a memory digest.  
          
        Args:  
            memory_digest: Memory digest with summary, recent, important sections  
              
        Returns:  
            Token usage by section  
        """  
        usage = {  
            "summary": 0,  
            "recent": 0,  
            "important": 0,  
            "total": 0  
        }  
          
        # Summary tokens  
        if memory_digest.get("summary"):  
            usage["summary"] = self.count_tokens(memory_digest["summary"])  
          
        # Recent messages tokens  
        for item in memory_digest.get("recent", []):  
            if isinstance(item, dict):  
                usage["recent"] += self.count_tokens(item.get("content", ""))  
          
        # Important messages tokens  
        for item in memory_digest.get("important", []):  
            if isinstance(item, dict):  
                usage["important"] += self.count_tokens(item.get("content", ""))  
          
        usage["total"] = usage["summary"] + usage["recent"] + usage["important"]  
          
        return usage  
      
    def optimize_memory_for_context(  
        self,  
        memory_digest: Dict[str, Any],  
        max_tokens: int,  
        query_tokens: int = 0  
    ) -> Dict[str, Any]:  
        """  
        Optimize memory digest to fit within token budget.  
          
        Priority allocation:  
        - Summary: 40%  
        - Recent: 35%  
        - Important: 25%  
          
        Args:  
            memory_digest: Memory digest to optimize  
            max_tokens: Maximum total tokens  
            query_tokens: Tokens already used by query  
              
        Returns:  
            Optimized memory digest  
        """  
        # Reserve overhead  
        overhead = 200  
        available = max(0, max_tokens - query_tokens - overhead)  
          
        usage = self.get_memory_token_usage(memory_digest)  
          
        # Already within limit  
        if usage["total"] <= available:  
            return memory_digest  
          
        optimized = memory_digest.copy()  
          
        # Allocate tokens  
        summary_budget = int(available * 0.4)  
        recent_budget = int(available * 0.35)  
        important_budget = available - summary_budget - recent_budget  
          
        # Trim summary  
        if optimized.get("summary") and usage["summary"] > summary_budget:  
            optimized["summary"] = self.trim_to_token_limit(  
                optimized["summary"],  
                summary_budget  
            )  
          
        # Trim recent (keep most recent)  
        if optimized.get("recent") and usage["recent"] > recent_budget:  
            optimized["recent"] = self._trim_message_list(  
                optimized["recent"],  
                recent_budget,  
                sort_key="timestamp",  
                reverse=True  
            )  
          
        # Trim important (keep highest importance)  
        if optimized.get("important") and usage["important"] > important_budget:  
            optimized["important"] = self._trim_message_list(  
                optimized["important"],  
                important_budget,  
                sort_key="importance_score",  
                reverse=True  
            )  
          
        return optimized  
      
    def _trim_message_list(  
        self,  
        messages: List[Dict[str, Any]],  
        max_tokens: int,  
        sort_key: str,  
        reverse: bool = True  
    ) -> List[Dict[str, Any]]:  
        """Trim a list of messages to fit token budget."""  
        if not messages:  
            return []  
          
        # Sort by priority  
        sorted_msgs = sorted(  
            messages,  
            key=lambda x: x.get(sort_key, 0),  
            reverse=reverse  
        )  
          
        result = []  
        current_tokens = 0  
          
        for msg in sorted_msgs:  
            msg_tokens = self.count_tokens(msg.get("content", ""))  
            if current_tokens + msg_tokens <= max_tokens:  
                result.append(msg)  
                current_tokens += msg_tokens  
            else:  
                # Try to fit partial message  
                remaining = max_tokens - current_tokens  
                if remaining > 50:  # Minimum useful content  
                    trimmed_msg = msg.copy()  
                    trimmed_msg["content"] = self.trim_to_token_limit(  
                        msg.get("content", ""),  
                        remaining  
                    )  
                    result.append(trimmed_msg)  
                break  
          
        return result  