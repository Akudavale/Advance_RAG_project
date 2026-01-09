"""  
src/api/server.py  
-----------------  
FastAPI server for the RAG system with better error handling.  
"""  
  
import os  
import sys  
import logging  
import tempfile  
import shutil  
from typing import Optional  
from contextlib import asynccontextmanager  
  
# Setup path FIRST  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  
  
# Load environment variables  
try:  
    from dotenv import load_dotenv  
    load_dotenv()  
except ImportError:  
    pass  
  
# Configure logging  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
    handlers=[logging.StreamHandler(sys.stdout)]  
)  
logger = logging.getLogger("rag_api")  
  
# Import FastAPI  
try:  
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks  
    from fastapi.middleware.cors import CORSMiddleware  
    from pydantic import BaseModel, Field, ConfigDict  
    import uvicorn  
except ImportError as e:  
    logger.error(f"Missing FastAPI dependencies: {e}")  
    logger.error("Install with: pip install fastapi uvicorn python-multipart")  
    sys.exit(1)  
  
# Global orchestrator (initialized lazily)  
orchestrator = None  
  
def get_orchestrator():  
    """Get or create the RAG orchestrator."""  
    global orchestrator  
    if orchestrator is None:  
        try:  
            from config.config import Config  
            from src.orchestrator import RAGOrchestrator  
              
            config = Config()  
            orchestrator = RAGOrchestrator(config)  
            logger.info("RAG orchestrator initialized")  
        except Exception as e:  
            logger.error(f"Failed to initialize orchestrator: {e}")  
            raise  
    return orchestrator  
  
  
# Lifespan context manager (replaces on_event)  
@asynccontextmanager  
async def lifespan(app: FastAPI):  
    """Startup and shutdown events."""  
    # Startup  
    logger.info("Starting RAG API server...")  
    try:  
        get_orchestrator()  
        logger.info("RAG system ready")  
    except Exception as e:  
        logger.error(f"Startup failed: {e}")  
        # Don't raise - let the server start anyway for health checks  
      
    yield  
      
    # Shutdown  
    logger.info("Shutting down RAG API server...")  
  
  
# Create FastAPI app  
app = FastAPI(  
    title="RAG API",  
    description="API for RAG system with PDF chat capabilities",  
    version="1.0.0",  
    lifespan=lifespan  
)  
  
# Add CORS  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=["*"],  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  
  
  
# ---------------------------------------------------  
# Request Models  
# ---------------------------------------------------  
class QueryRequest(BaseModel):  
    query: str = Field(..., min_length=1)  
    conversation_id: Optional[str] = None  
    use_optimized_prompts: bool = True  
    use_memory: bool = True  
    use_reranking: bool = True  
    use_query_rewriting: bool = False  
      
    model_config = ConfigDict(  
        json_schema_extra={  
            "example": {  
                "query": "What is this document about?",  
                "conversation_id": None  
            }  
        }  
    )  
  
  
# ---------------------------------------------------  
# Endpoints  
# ---------------------------------------------------  
@app.get("/")  
def root():  
    """Root endpoint."""  
    return {  
        "name": "RAG API",  
        "version": "1.0.0",  
        "docs": "/docs"  
    }  
  
  
@app.get("/health")  
def health_check():  
    """Health check endpoint."""  
    return {"status": "healthy"}  
  
  
@app.get("/ready")  
def readiness_check():  
    """Readiness check - verifies orchestrator is initialized."""  
    try:  
        orch = get_orchestrator()  
        return {  
            "status": "ready",  
            "orchestrator": "initialized"  
        }  
    except Exception as e:  
        raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")  
  
  
@app.post("/conversation")  
def create_conversation():  
    """Create a new conversation."""  
    try:  
        orch = get_orchestrator()  
        conversation_id = orch.create_conversation()  
        return {  
            "status": "success",  
            "conversation_id": conversation_id  
        }  
    except Exception as e:  
        logger.error(f"Create conversation error: {e}")  
        raise HTTPException(status_code=500, detail=str(e))  
  
  
@app.get("/conversation/{conversation_id}")  
def get_conversation(conversation_id: str):  
    """Get conversation history."""  
    try:  
        orch = get_orchestrator()  
        result = orch.get_conversation_history(conversation_id)  
          
        if result.get("status") == "error":  
            raise HTTPException(status_code=404, detail=result["message"])  
          
        return result  
    except HTTPException:  
        raise  
    except Exception as e:  
        logger.error(f"Get conversation error: {e}")  
        raise HTTPException(status_code=500, detail=str(e))  
  
  
@app.post("/upload_pdf")  
async def upload_pdf(  
    background_tasks: BackgroundTasks,  
    file: UploadFile = File(...),  
    conversation_id: Optional[str] = Form(None)  
):  
    """Upload and process a PDF document."""  
    try:  
        orch = get_orchestrator()  
          
        # Create conversation if needed  
        if not conversation_id:  
            conversation_id = orch.create_conversation()  
          
        # Validate file  
        if not file.filename:  
            raise HTTPException(status_code=400, detail="No filename provided")  
          
        if not file.filename.lower().endswith(".pdf"):  
            raise HTTPException(status_code=400, detail="File must be a PDF")  
          
        # Save to temp file  
        temp_dir = tempfile.mkdtemp()  
        temp_path = os.path.join(temp_dir, file.filename)  
          
        try:  
            with open(temp_path, "wb") as buffer:  
                content = await file.read()  
                buffer.write(content)  
              
            # Process document  
            result = orch.process_document(conversation_id, temp_path)  
              
            if result.get("status") == "error":  
                raise HTTPException(status_code=400, detail=result["message"])  
              
            return {  
                "status": "success",  
                "conversation_id": conversation_id,  
                "document": result.get("document", {}),  
                "message": "PDF processed successfully"  
            }  
              
        finally:  
            # Cleanup temp files  
            background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)  
              
    except HTTPException:  
        raise  
    except Exception as e:  
        logger.error(f"Upload PDF error: {e}")  
        raise HTTPException(status_code=500, detail=str(e))  
  
  
@app.post("/query")  
def query(request: QueryRequest):  
    """Process a user query."""  
    try:  
        orch = get_orchestrator()  
          
        result = orch.query(  
            conversation_id=request.conversation_id,  
            query=request.query,  
            use_optimized_prompts=request.use_optimized_prompts,  
            use_memory=request.use_memory,  
            use_reranking=request.use_reranking,  
            use_query_rewriting=request.use_query_rewriting  
        )  
          
        if result.get("status") == "error":  
            raise HTTPException(status_code=400, detail=result["message"])  
          
        return result  
          
    except HTTPException:  
        raise  
    except Exception as e:  
        logger.error(f"Query error: {e}")  
        raise HTTPException(status_code=500, detail=str(e))  
  
  
@app.get("/stats")  
def get_stats():  
    """Get system statistics."""  
    try:  
        orch = get_orchestrator()  
        return orch.get_stats()  
    except Exception as e:  
        logger.error(f"Stats error: {e}")  
        raise HTTPException(status_code=500, detail=str(e))  
  
  
# ---------------------------------------------------  
# Main entry point  
# ---------------------------------------------------  
if __name__ == "__main__":  
    port = int(os.getenv("PORT", 8000))  
    host = os.getenv("HOST", "127.0.0.1")  
      
    print(f"Starting server on http://{host}:{port}")  
    print(f"API docs: http://{host}:{port}/docs")  
      
    uvicorn.run(  
        app,  
        host=host,  
        port=port,  
        log_level="info"  
    )  