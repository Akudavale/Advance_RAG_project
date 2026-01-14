#!/usr/bin/env python3  
"""  
main.py  
-------  
RAG system with multi-LLM support (Azure OpenAI, Gemini, OpenAI).  
"""  
  
import os  
import sys  
import time  
import logging  
  
# Enable logging  
logging.basicConfig(  
    level=logging.INFO,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  
)  
  
# Ensure project root is in path  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  
  
from src.orchestrator import RAGOrchestrator  
from config.config import Config  
  
  
def main(call_config: dict):  
    print("=" * 60)  
    print("RAG System with Multi-LLM Support")  
    print("=" * 60)  
      
    print("\nInitializing RAG system...")  
    config = Config()  
      
    # Validate configuration  
    validation = config.validate()  
    if not validation["valid"]:  
        print(f"Configuration issues: {validation['issues']}")  
        print("Please check your .env file.")  
        return  
      
    # Show provider info  
    print(f"\n✓ LLM Provider: {validation['provider'].upper()}")  
      
    llm_config = config.get_llm_config()  
    if validation['provider'] == 'azure':  
        print(f"  Deployment: {llm_config.get('azure_deployment')}")  
    elif validation['provider'] == 'gemini':  
        print(f"  Model: {llm_config.get('model_name')}")  
    elif validation['provider'] == 'openai':  
        print(f"  Model: {llm_config.get('model_name')}")  
      
    # Create orchestrator  
    rag = RAGOrchestrator(config)  
    print("✓ RAG system initialized successfully!")  
      
    # Show current stats  
    stats = rag.get_stats()  
    print(f"\nCurrent stats:")  
    print(f"  - Indexed documents: {stats['vector_store'].get('indexed_documents', 0)}")  
    print(f"  - Total vectors: {stats['vector_store'].get('total_vectors', 0)}")  
      
    # Create a conversation  
    conversation_id = rag.create_conversation()  
    print(f"\nCreated conversation: {conversation_id[:8]}...")  
      
    # Process a PDF  
    pdf_path = call_config.get("PDF_PATH")  
      
    if pdf_path and os.path.exists(pdf_path):  
        print(f"\nProcessing {pdf_path}...")  
        start_time = time.time()  
          
        result = rag.process_document(conversation_id, pdf_path)  
          
        elapsed = time.time() - start_time  
          
        if result["status"] == "success":  
            doc_info = result["document"]  
              
            if doc_info.get("cached"):  
                print(f"✓ Document already indexed (loaded from cache)")  
                print(f"  Time: {elapsed:.2f}s")  
            else:  
                print(f"✓ Processed: {doc_info['chunks']} chunks")  
                print(f"  Added to index: {doc_info.get('added_to_index', doc_info['chunks'])}")  
                print(f"  Time: {elapsed:.2f}s")  
              
            stats = rag.get_stats()  
            print(f"\nUpdated stats:")  
            print(f"  - Indexed documents: {stats['vector_store'].get('indexed_documents', 0)}")  
            print(f"  - Total vectors: {stats['vector_store'].get('total_vectors', 0)}")  
        else:  
            print(f"✗ Error: {result['message']}")  
            return  
    elif pdf_path:  
        print(f"File not found: {pdf_path}")  
        return  
    else:  
        print("Skipping PDF upload.")  
        stats = rag.get_stats()  
        if stats['vector_store'].get('total_vectors', 0) == 0:  
            print("Note: No documents indexed. Queries won't return results.")  
      
    # Interactive query loop  
    print("\n" + "=" * 60)  
    print("RAG Chat Interface")  
    print(f"Using: {validation['provider'].upper()}")  
    print("Commands: 'quit', 'stats', 'clear', 'reprocess', 'switch'")  
    print("=" * 60 + "\n")  
      
    while True:  
        try:  
            query = input("You: ").strip()  
              
            if not query:  
                continue  
              
            if query.lower() == 'quit':  
                print("Goodbye!")  
                break  
              
            if query.lower() == 'stats':  
                stats = rag.get_stats()  
                model_info = rag.llm_generator.get_model_info()  
                print(f"\nStats:")  
                print(f"  - Provider: {model_info.get('provider')}")  
                print(f"  - Indexed documents: {stats['vector_store'].get('indexed_documents', 0)}")  
                print(f"  - Total vectors: {stats['vector_store'].get('total_vectors', 0)}")  
                print(f"  - Conversations: {stats['conversations']}")  
                print()  
                continue  
              
            if query.lower() == 'clear':  
                confirm = input("Clear all indexed documents? (yes/no): ").strip().lower()  
                if confirm == 'yes':  
                    result = rag.clear_index()  
                    print(f"Index cleared: {result}")  
                continue  
              
            if query.lower() == 'reprocess':  
                if pdf_path:  
                    print(f"Reprocessing {pdf_path} (force)...")  
                    result = rag.process_document(  
                        conversation_id,   
                        pdf_path,   
                        force_reprocess=True  
                    )  
                    print(f"Result: {result}")  
                else:  
                    print("No PDF path set.")  
                continue  
              
            if query.lower() == 'switch':  
                print("\nAvailable providers:")  
                print("  1. azure  - Azure OpenAI")  
                print("  2. gemini - Google Gemini")  
                print("  3. openai - OpenAI")  
                choice = input("Enter provider name: ").strip().lower()  
                if choice in ('azure', 'gemini', 'openai'):  
                    print(f"\nTo switch providers, update LLM_PROVIDER in .env to '{choice}'")  
                    print("Then restart the application.")  
                continue  
              
            # Process query  
            print("Thinking...")  
            start_time = time.time()  
              
            response = rag.query(  
                conversation_id=conversation_id,  
                query=query,  
                use_reranking=True,  
                use_memory=True,
                use_query_rewriting=call_config.get("use_query_rewriting"), 
                top_k=call_config.get("top_k",10),  
                rerank_top_k=call_config.get("rerank_top_k",5),
                method=call_config.get("method", "hyde")
            )  
              
            elapsed = time.time() - start_time  
              
            if response["status"] == "success":  
                print(f"\nAssistant: {response['answer']}")  
                print(f"\n[{elapsed:.2f}s, {len(response.get('sources', []))} sources]")  
                  
                # if response.get("sources"):  
                #     show_sources = input("Show sources? (y/n): ").strip().lower()  
                #     if show_sources == 'y':  
                #         print("\nSources:")  
                #         for i, source in enumerate(response["sources"], 1):  
                #             score = source.get("score", 0)  
                #             page = source.get("metadata", {}).get("page_number", "?")  
                #             content_preview = source.get("content", "")[:150]  
                #             print(f"\n  [{i}] Page {page} (score: {score:.3f})")  
                #             print(f"      {content_preview}...")  
                print()  
            else:  
                print(f"Error: {response['message']}\n")  
                  
        except KeyboardInterrupt:  
            print("\nGoodbye!")  
            break  
        except Exception as e:  
            print(f"Error: {e}\n")  
            import traceback  
            traceback.print_exc()  
  
  
if __name__ == "__main__": 

    call_config={
        "PDF_PATH": "Abhishek_Master_Thesis_draft_1.pdf",
        "top_k": 20, # number of top documents to retrieve
        "rerank_top_k": 10, # number of top documents to re-rank
        "use_query_rewriting": True,  # Enable query re-writing
        "method": "expand" # query re-write method: "hyde" , "expand", "multi", "decompose" 
    }
    main(call_config)