from src.orchestrator import RAGOrchestrator  
from config.config import Config  
  
# Initialize  
config = Config()  
rag = RAGOrchestrator(config)  
  
# Create a conversation  
conversation_id = rag.create_conversation()  
  
# Process a PDF  
result = rag.process_document(conversation_id, "Abhishek_Master_Thesis.pdf")  
print(f"Processed: {result}")  
  
# Query  
response = rag.query(  
    conversation_id=conversation_id,  
    query="What are the main topics discussed in this document?",  
    use_reranking=True,  
    use_memory=True  
)  
  
print(f"Answer: {response['answer']}")  
print(f"Sources: {response['sources']}")  