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


def _normalize_pdf_paths(call_config: dict):
    """Normalize PDF input config to a list of file paths."""
    pdf_paths = call_config.get("PDF_PATHS")

    if isinstance(pdf_paths, str):
        pdf_paths = [p.strip() for p in pdf_paths.split(",") if p.strip()]
    elif not isinstance(pdf_paths, list):
        pdf_paths = []

    # Backward-compatible single path support
    if not pdf_paths and call_config.get("PDF_PATH"):
        pdf_paths = [call_config.get("PDF_PATH")]

    return pdf_paths


def _process_pdf_batch(rag, conversation_id: str, pdf_paths):
    """Process a list of PDFs and print per-file status."""
    if not pdf_paths:
        return

    print(f"\nProcessing {len(pdf_paths)} PDF(s)...")

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"X File not found: {pdf_path}")
            continue

        print(f"\nProcessing {pdf_path}...")
        start_time = time.time()
        result = rag.process_document(conversation_id, pdf_path)
        elapsed = time.time() - start_time

        if result.get("status") == "success":
            doc_info = result.get("document", {})

            if doc_info.get("cached"):
                print("OK Document already indexed (loaded from cache)")
                print(f"  Time: {elapsed:.2f}s")
            else:
                print(f"OK Processed: {doc_info.get('chunks', 0)} chunks")
                print(f"  Added to index: {doc_info.get('added_to_index', doc_info.get('chunks', 0))}")
                print(f"  Time: {elapsed:.2f}s")
        else:
            print(f"X Error: {result.get('message', 'Unknown error')}")


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
    print(f"\nLLM Provider: {validation['provider'].upper()}")

    llm_config = config.get_llm_config()
    if validation['provider'] == 'azure':
        print(f"  Deployment: {llm_config.get('azure_deployment')}")
    elif validation['provider'] == 'gemini':
        print(f"  Model: {llm_config.get('model_name')}")
    elif validation['provider'] == 'openai':
        print(f"  Model: {llm_config.get('model_name')}")

    # Create orchestrator
    rag = RAGOrchestrator(config)
    print("OK RAG system initialized successfully!")

    # Show current stats
    stats = rag.get_stats()
    print("\nCurrent stats:")
    print(f"  - Indexed documents: {stats['vector_store'].get('indexed_documents', 0)}")
    print(f"  - Total vectors: {stats['vector_store'].get('total_vectors', 0)}")

    # Create a conversation
    conversation_id = rag.create_conversation()
    print(f"\nCreated conversation: {conversation_id[:8]}...")

    # Process one or more PDFs
    pdf_paths = _normalize_pdf_paths(call_config)

    if pdf_paths:
        _process_pdf_batch(rag, conversation_id, pdf_paths)
        stats = rag.get_stats()
        print("\nUpdated stats:")
        print(f"  - Indexed documents: {stats['vector_store'].get('indexed_documents', 0)}")
        print(f"  - Total vectors: {stats['vector_store'].get('total_vectors', 0)}")
    else:
        print("Skipping PDF upload.")
        stats = rag.get_stats()
        if stats['vector_store'].get('total_vectors', 0) == 0:
            print("Note: No documents indexed. Queries won't return results.")

    # Interactive query loop
    print("\n" + "=" * 60)
    print("RAG Chat Interface")
    print(f"Using: {validation['provider'].upper()}")
    print("Commands: 'quit', 'stats', 'clear', 'reprocess', 'addpdf', 'switch'")
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
                llm_cfg = rag.config.get_llm_config()
                provider = llm_cfg.get("provider", "unknown")
                model_name = (
                    llm_cfg.get("azure_deployment")
                    or llm_cfg.get("model_name")
                    or "unknown"
                )
                print("\nStats:")
                print(f"  - Provider: {provider}")
                print(f"  - Model/Deployment: {model_name}")
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
                if pdf_paths:
                    for pdf_path in pdf_paths:
                        if not os.path.exists(pdf_path):
                            print(f"File not found: {pdf_path}")
                            continue
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

            if query.lower().startswith('addpdf'):
                raw = query[len('addpdf'):].strip()
                if not raw:
                    raw = input("Enter PDF path(s), comma-separated: ").strip()

                new_paths = [p.strip() for p in raw.split(",") if p.strip()]
                if not new_paths:
                    print("No valid PDF path provided.")
                    continue

                _process_pdf_batch(rag, conversation_id, new_paths)
                for p in new_paths:
                    if p not in pdf_paths:
                        pdf_paths.append(p)
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
                top_k=call_config.get("top_k", 10),
                rerank_top_k=call_config.get("rerank_top_k", 5),
                method=call_config.get("method", "hyde"),
                retrieval_mode=call_config.get("retrieval_mode", "hybrid"),
                dense_top_k=call_config.get("dense_top_k"),
                sparse_top_k=call_config.get("sparse_top_k"),
                hybrid_alpha=call_config.get("hybrid_alpha", 0.5),
            )

            elapsed = time.time() - start_time

            if response["status"] == "success":
                print(f"\nAssistant: {response['answer']}")
                print(f"\n[{elapsed:.2f}s, {len(response.get('sources', []))} sources]")
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

    call_config = {
        "PDF_PATHS": [
            "Abhishek_Master_Thesis_draft_1.pdf",
            # "resume.pdf"
            r"C:\\Users\\I012606\\Desktop\\Thesis\\papers\\AD in unstructed environment.pdf"
        ],
        "top_k": 20,  # number of top documents to retrieve
        "rerank_top_k": 10,  # number of top documents to re-rank
        "use_query_rewriting": True,  # Enable query re-writing
        "method": "expand",  # query re-write method: "hyde" , "expand", "multi", "decompose"
        "retrieval_mode": "hybrid",  # "vector", "bm25", "hybrid"
        "dense_top_k": 20,
        "sparse_top_k": 20,
        "hybrid_alpha": 0.5,
    }
    main(call_config)
