"""  
config/config.py  
----------------  
Centralized configuration for the RAG system.  
All environment variables are standardized here.  
"""  
  
import os  
from typing import Optional  
from dotenv import load_dotenv  
  
load_dotenv()  
  
  
class Config:  
    """Centralized configuration for RAG system."""  
      
    # -------------------------------------------------------------------------  
    # PDF Processing  
    # -------------------------------------------------------------------------  
    CHUNK_SIZE: int = 1000  
    CHUNK_OVERLAP: int = 150  
      
    # -------------------------------------------------------------------------  
    # Embedding Configuration  
    # -------------------------------------------------------------------------  
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"  
    EMBEDDING_DIMENSION: int = 1024  # Must match the model  
    EMBEDDING_DEVICE: str = "cpu"  # Options: "cpu", "cuda", "mps"  
    NORMALIZE_EMBEDDINGS: bool = True  
      
    # -------------------------------------------------------------------------  
    # Vector Database  
    # -------------------------------------------------------------------------  
    VECTOR_DB: str = "chroma"  # Options: "chroma", "qdrant", "faiss"  
    COLLECTION_NAME: str = "pdf_documents"  
    CHROMA_PERSIST_DIR: str = "./chroma_db"  
      
    # -------------------------------------------------------------------------  
    # Retrieval  
    # -------------------------------------------------------------------------  
    TOP_K_RETRIEVAL: int = 20  
    HYBRID_ALPHA: float = 0.5  # Weight between sparse (BM25) and dense  
      
    # -------------------------------------------------------------------------  
    # Re-ranking  
    # -------------------------------------------------------------------------  
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"  
    TOP_K_RERANK: int = 5  
    RERANKER_DEVICE: str = "cpu"  
      
    # -------------------------------------------------------------------------  
    # Azure OpenAI Configuration (STANDARDIZED)  
    # -------------------------------------------------------------------------  
    # Primary environment variable names  
    AZURE_OPENAI_API_KEY: Optional[str] = (  
        os.getenv("AZURE_OPENAI_API_KEY") or  
        os.getenv("azure_openai_key") or  
        os.getenv("openai_api_key")  
    )  
      
    AZURE_OPENAI_ENDPOINT: Optional[str] = (  
        os.getenv("AZURE_OPENAI_ENDPOINT") or  
        os.getenv("azure_openai_endpoint") or  
        os.getenv("openai_api_base")  
    )  
      
    AZURE_OPENAI_API_VERSION: str = (  
        os.getenv("AZURE_OPENAI_API_VERSION") or  
        os.getenv("api_version") or  
        os.getenv("openai_api_version") or  
        "2024-02-15-preview"  
    )  
      
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = (  
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or  
        os.getenv("deployment_name") or  
        "gpt-4o"  
    )  
      
    # LLM Parameters  
    PRIMARY_TEMPERATURE: float = 0.2  
    PRIMARY_MAX_TOKENS: int = 2000  
    FALLBACK_TEMPERATURE: float = 0.3  
    FALLBACK_MAX_TOKENS: int = 1500  
    REQUEST_TIMEOUT: int = 60  
    MAX_RETRIES: int = 2  
      
    # -------------------------------------------------------------------------  
    # Memory Configuration  
    # -------------------------------------------------------------------------  
    MEMORY_KEY: str = "chat_history"  
    MAX_TOKEN_LIMIT: int = 4000  
    MAX_CONVERSATION_TURNS: int = 50  
    SUMMARY_THRESHOLD_TOKENS: int = 2000  
      
    # -------------------------------------------------------------------------  
    # Caching  
    # -------------------------------------------------------------------------  
    RESPONSE_CACHE_ENABLED: bool = True  
    RESPONSE_CACHE_TTL_SECONDS: int = 3600  
    RESPONSE_CACHE_MAX_SIZE: int = 100  
    PDF_CACHE_ENABLED: bool = True  
    PDF_CACHE_DIR: str = "./.cache/pdf"  
      
    # -------------------------------------------------------------------------  
    # Prompt Templates  
    # -------------------------------------------------------------------------  
    PROMPT_TEMPLATE_DIR: str = "config/prompts"  
      
    # -------------------------------------------------------------------------  
    # Logging  
    # -------------------------------------------------------------------------  
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  
    LOG_DIR: str = "logs"  
      
    # -------------------------------------------------------------------------  
    # UI Configuration  
    # -------------------------------------------------------------------------  
    UI_TITLE: str = "Advanced RAG PDF Chat"  
    UI_DESCRIPTION: str = "Upload a PDF and chat with its contents using advanced RAG techniques."  
      
    # -------------------------------------------------------------------------  
    # Validation  
    # -------------------------------------------------------------------------  
    @classmethod  
    def validate(cls) -> dict:  
        """Validate configuration and return status."""  
        issues = []  
          
        if not cls.AZURE_OPENAI_API_KEY:  
            issues.append("AZURE_OPENAI_API_KEY is not set")  
          
        if not cls.AZURE_OPENAI_ENDPOINT:  
            issues.append("AZURE_OPENAI_ENDPOINT is not set")  
          
        if not cls.AZURE_OPENAI_DEPLOYMENT_NAME:  
            issues.append("AZURE_OPENAI_DEPLOYMENT_NAME is not set")  
          
        return {  
            "valid": len(issues) == 0,  
            "issues": issues  
        }  
      
    @classmethod  
    def get_azure_openai_config(cls) -> dict:  
        """Get Azure OpenAI configuration as a dictionary."""  
        return {  
            "azure_endpoint": cls.AZURE_OPENAI_ENDPOINT,  
            "api_key": cls.AZURE_OPENAI_API_KEY,  
            "api_version": cls.AZURE_OPENAI_API_VERSION,  
            "azure_deployment": cls.AZURE_OPENAI_DEPLOYMENT_NAME,  
        }  