#!/usr/bin/env python3  
"""  
main.py  
-------  
Example script to run the RAG system with caching demonstration.  
"""  
  
import os  
import sys  
import time  
  
# Ensure project root is in path  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  
  
from src.orchestrator import RAGOrchestrator  
from config.config import Config  
  
  
def main(PDF_PATH: str):  
    # Initialize  
    print("=" * 60)  
    print("RAG System with Caching")  
    print("=" * 60)  
      
    print("\nInitializing RAG system...")  
    config = Config()  
      
    # Validate configuration  
    validation = config.validate()  
    if not validation["valid"]:  
        print(f"Configuration issues: {validation['issues']}")  
        print("Please check your environment variables.")  
        return  
      
    # Create orchestrator  
    rag = RAGOrchestrator(config)  
    print("RAG system initialized successfully!")  
      
    # Show current stats  
    stats = rag.get_stats()  
    print(f"\nCurrent stats:")  
    print(f"  - Indexed documents: {stats['vector_store'].get('indexed_documents', 0)}")  
    print(f"  - Total vectors: {stats['vector_store'].get('total_vectors', 0)}")  
      
    # Create a conversation  
    conversation_id = rag.create_conversation()  
    print(f"\nCreated conversation: {conversation_id[:8]}...")  
      
    # Process a PDF  
    pdf_path = PDF_PATH  
      
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
              
            # Show updated stats  
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
          
        # Check if there are existing documents  
        stats = rag.get_stats()  
        if stats['vector_store'].get('total_vectors', 0) == 0:  
            print("Note: No documents indexed. Queries won't return results.")  
      
    # Interactive query loop  
    print("\n" + "=" * 60)  
    print("RAG Chat Interface")  
    print("Commands: 'quit', 'stats', 'clear', 'reprocess'")  
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
                print(f"\nStats:")  
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
                    print("No PDF path set. Enter a path first.")  
                continue  
              
            # Process query  
            print("Thinking...")  
            start_time = time.time()  
              
            response = rag.query(  
                conversation_id=conversation_id,  
                query=query,  
                use_reranking=True,  
                use_memory=True  
            )  
              
            elapsed = time.time() - start_time  
              
            if response["status"] == "success":  
                print(f"\nAssistant: {response['answer']}")  
                print(f"\n[{elapsed:.2f}s, {len(response.get('sources', []))} sources]")  
                  
                if response.get("sources"):  
                    show_sources = input("Show sources? (y/n): ").strip().lower()  
                    if show_sources == 'y':  
                        print("\nSources:")  
                        for i, source in enumerate(response["sources"], 1):  
                            score = source.get("score", 0)  
                            page = source.get("metadata", {}).get("page_number", "?")  
                            content_preview = source.get("content", "")[:100]  
                            print(f"  [{i}] Page {page} (score: {score:.3f})")  
                            print(f"      {content_preview}...")  
                print()  
            else:  
                print(f"Error: {response['message']}\n")  
                  
        except KeyboardInterrupt:  
            print("\nGoodbye!")  
            break  
        except Exception as e:  
            print(f"Error: {e}\n")  
  
  
if __name__ == "__main__": 
    PDF_PATH = "Abhishek_Master_Thesis.pdf" 
    main(PDF_PATH)  