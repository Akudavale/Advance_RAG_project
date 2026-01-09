"""  
config/config.py  
----------------  
Configuration management for RAG system with multi-LLM support.  
"""  
  
import os  
import logging  
from pathlib import Path  
from typing import Dict, Any, Optional, Literal  
from dotenv import load_dotenv  
  
# Load environment variables  
load_dotenv()  
  
logger = logging.getLogger(__name__)  
  
  
class Config:  
    """  
    Configuration class for RAG system.  
      
    Supports:  
    - Azure OpenAI  
    - Google Gemini  
    - Local models (future)  
    """  
      
    # -------------------------------------------  
    # LLM Provider Selection  
    # -------------------------------------------  
    LLM_PROVIDER: Literal["azure", "gemini", "openai"] = os.getenv("LLM_PROVIDER", "azure").lower()  
      
    # -------------------------------------------  
    # Azure OpenAI Configuration  
    # -------------------------------------------  
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")  
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")  
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")  
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")  
      
    # -------------------------------------------  
    # Google Gemini Configuration  
    # -------------------------------------------  
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")  
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-05-20")  
      
    # -------------------------------------------  
    # OpenAI Configuration (optional, for direct OpenAI)  
    # -------------------------------------------  
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")  
    OPENAI_MODEL_NAME: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4")  
      
    # -------------------------------------------  
    # Embedding Configuration  
    # -------------------------------------------  
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")  
    EMBEDDING_DEVICE: Optional[str] = os.getenv("EMBEDDING_DEVICE", None)  
    EMBEDDING_CACHE_ENABLED: bool = os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"  
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR", ".cache/embeddings")  
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))  
      
    # -------------------------------------------  
    # Reranker Configuration  
    # -------------------------------------------  
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")  
    RERANKER_DEVICE: Optional[str] = os.getenv("RERANKER_DEVICE", None)  
    USE_CROSS_ENCODER: bool = os.getenv("USE_CROSS_ENCODER", "true").lower() == "true"  
      
    # -------------------------------------------  
    # Vector Store Configuration  
    # -------------------------------------------  
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")  
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_documents")  
      
    # -------------------------------------------  
    # PDF Processing Configuration  
    # -------------------------------------------  
    PDF_CACHE_DIR: str = os.getenv("PDF_CACHE_DIR", ".cache/pdf")  
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))  
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))  
      
    # -------------------------------------------  
    # LLM Generation Configuration  
    # -------------------------------------------  
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))  
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))  
      
    # -------------------------------------------  
    # Prompt Configuration  
    # -------------------------------------------  
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))  
    MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "5"))  
    MAX_CHARS_PER_DOC: int = int(os.getenv("MAX_CHARS_PER_DOC", "1500"))  
    PROMPTS_DIR: str = os.getenv("PROMPTS_DIR", "src/prompts/templates")  
      
    def __init__(self):  
        """Initialize configuration."""  
        self._validate_on_init()  
      
    def _validate_on_init(self):  
        """Validate configuration on initialization."""  
        # Auto-detect provider if not set  
        if not self.LLM_PROVIDER or self.LLM_PROVIDER == "auto":  
            self.LLM_PROVIDER = self._detect_provider()  
            logger.info(f"Auto-detected LLM provider: {self.LLM_PROVIDER}")  
      
    def _detect_provider(self) -> str:  
        """Auto-detect which LLM provider to use based on available keys."""  
        if self.AZURE_OPENAI_API_KEY and self.AZURE_OPENAI_ENDPOINT:  
            return "azure"  
        elif self.GEMINI_API_KEY:  
            return "gemini"  
        elif self.OPENAI_API_KEY:  
            return "openai"  
        else:  
            logger.warning("No LLM API keys found!")  
            return "none"  
      
    def get_llm_provider(self) -> str:  
        """Get the configured LLM provider."""  
        return self.LLM_PROVIDER  
      
    def get_azure_openai_config(self) -> Dict[str, Any]:  
        """Get Azure OpenAI configuration."""  
        return {  
            "api_key": self.AZURE_OPENAI_API_KEY,  
            "azure_endpoint": self.AZURE_OPENAI_ENDPOINT,  
            "api_version": self.AZURE_OPENAI_API_VERSION,  
            "azure_deployment": self.AZURE_OPENAI_DEPLOYMENT_NAME,  
        }  
      
    def get_gemini_config(self) -> Dict[str, Any]:  
        """Get Google Gemini configuration."""  
        return {  
            "api_key": self.GEMINI_API_KEY,  
            "model_name": self.GEMINI_MODEL_NAME,  
        }  
      
    def get_openai_config(self) -> Dict[str, Any]:  
        """Get OpenAI configuration."""  
        return {  
            "api_key": self.OPENAI_API_KEY,  
            "model_name": self.OPENAI_MODEL_NAME,  
        }  
      
    def get_llm_config(self) -> Dict[str, Any]:  
        """Get configuration for the active LLM provider."""  
        provider = self.get_llm_provider()  
          
        if provider == "azure":  
            return {  
                "provider": "azure",  
                **self.get_azure_openai_config(),  
                "temperature": self.LLM_TEMPERATURE,  
                "max_tokens": self.LLM_MAX_TOKENS,  
            }  
        elif provider == "gemini":  
            return {  
                "provider": "gemini",  
                **self.get_gemini_config(),  
                "temperature": self.LLM_TEMPERATURE,  
                "max_tokens": self.LLM_MAX_TOKENS,  
            }  
        elif provider == "openai":  
            return {  
                "provider": "openai",  
                **self.get_openai_config(),  
                "temperature": self.LLM_TEMPERATURE,  
                "max_tokens": self.LLM_MAX_TOKENS,  
            }  
        else:  
            return {"provider": "none"}  
      
    def validate(self) -> Dict[str, Any]:  
        """  
        Validate the configuration.  
          
        Returns:  
            Dict with 'valid' boolean and 'issues' list  
        """  
        issues = []  
          
        provider = self.get_llm_provider()  
          
        if provider == "azure":  
            if not self.AZURE_OPENAI_API_KEY:  
                issues.append("AZURE_OPENAI_API_KEY not set")  
            if not self.AZURE_OPENAI_ENDPOINT:  
                issues.append("AZURE_OPENAI_ENDPOINT not set")  
            if not self.AZURE_OPENAI_DEPLOYMENT_NAME:  
                issues.append("AZURE_OPENAI_DEPLOYMENT_NAME not set")  
        elif provider == "gemini":  
            if not self.GEMINI_API_KEY:  
                issues.append("GEMINI_API_KEY not set")  
        elif provider == "openai":  
            if not self.OPENAI_API_KEY:  
                issues.append("OPENAI_API_KEY not set")  
        elif provider == "none":  
            issues.append("No LLM provider configured. Set LLM_PROVIDER and API keys.")  
          
        return {  
            "valid": len(issues) == 0,  
            "issues": issues,  
            "provider": provider  
        }  
      
    def __repr__(self) -> str:  
        """String representation of config."""  
        return (  
            f"Config(provider={self.LLM_PROVIDER}, "  
            f"embedding={self.EMBEDDING_MODEL}, "  
            f"reranker={self.RERANKER_MODEL})"  
        )  