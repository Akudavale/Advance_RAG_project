#!/usr/bin/env python3  
"""  
Evaluate one PDF against a JSON QA set and export results to JSON.  
Reads configuration and questions from an external JSON file.  
"""  
  
import json  
import sys  
from pathlib import Path  
from statistics import mean  
from typing import Any, Dict, List  
  
from config.config import Config  
from src.orchestrator import RAGOrchestrator  
from src.evaluation import (
    conciseness_score,
    load_json_eval_config,
    normalize_text,
    run_ragas_metrics,
    token_f1,
    validate_qa_set,
)
  
  
# ============================================================  
# CONFIG: Path to the input JSON file  
# ============================================================  
  
INPUT_JSON_PATH = r"evaluation_config.json"  # <-- Set this to your JSON file path  
  
# Optional: Override via command line argument  
# Usage: python script.py path/to/config.json  
  
  
# ============================================================  
# MAIN  
# ============================================================  
  
def run_evaluation(json_path: str) -> None:  
    """Run evaluation using the specified JSON configuration file."""  
      
    print(f"Loading configuration from: {json_path}")  
      
    # Load and parse JSON  
    parsed = load_json_eval_config(json_path)  
    eval_config = parsed["config"]  
    qa_rows = validate_qa_set(parsed["qa_set"])  
  
    # Extract configuration values  
    pdf_path = eval_config["input_pdf_path"]  
    out_json_path = eval_config["output_json_path"]  
    agentic = eval_config["agentic"]  
    top_k = eval_config["top_k"]  
    rerank_top_k = eval_config["rerank_top_k"]  
    retrieval_mode = eval_config["retrieval_mode"]  
    use_query_rewriting = eval_config["use_query_rewriting"]  
    force_reprocess = eval_config["force_reprocess"]  
  
    print(f"PDF Path: {pdf_path}")  
    print(f"Output Path: {out_json_path}")  
    print(f"Questions to evaluate: {len(qa_rows)}")  
    print(f"Config: agentic={agentic}, top_k={top_k}, rerank_top_k={rerank_top_k}, "  
          f"retrieval_mode={retrieval_mode}, use_query_rewriting={use_query_rewriting}")  
  
    # Initialize RAG components  
    cfg = Config()  
    rag = RAGOrchestrator(cfg)  
    conversation_id = rag.create_conversation()  
  
    # Process document  
    print(f"\nProcessing document: {pdf_path}")  
    process_result = rag.process_document(  
        conversation_id=conversation_id,  
        file_path=pdf_path,  
        force_reprocess=force_reprocess,  
    )  
  
    if process_result.get("status") != "success":  
        raise RuntimeError(f"Document processing failed: {process_result}")  
  
    print("Document processed successfully.\n")  
  
    # Initialize results storage  
    results: List[Dict[str, Any]] = []  
  
    em_scores: List[float] = []  
    f1_scores: List[float] = []  
    source_counts: List[int] = []  
    relevance_scores: List[float] = []  
    faithfulness_scores: List[float] = []  
    completeness_scores: List[float] = []  
    conciseness_scores: List[float] = []  
  
    ragas_samples: List[Dict[str, Any]] = []
    ragas_result_indices: List[int] = []

    # Process each question  
    for idx, row in enumerate(qa_rows, 1):  
        qid = row["id"]  
        question = row["question"]  
        reference = row["ground_truth"]  
  
        print(f"[{idx}/{len(qa_rows)}] Evaluating question: {qid}")  
  
        response = rag.query(  
            query=question,  
            conversation_id=conversation_id,  
            use_optimized_prompts=True,  
            use_memory=False,  
            use_reranking=True,  
            use_query_rewriting=use_query_rewriting,  
            top_k=top_k,  
            rerank_top_k=rerank_top_k,  
            method="hyde",  
            agentic=agentic,  
            retrieval_mode=retrieval_mode,  
        )  
  
        if response.get("status") != "success":  
            results.append(  
                {  
                    "id": qid,  
                    "question": question,  
                    "ground_truth": reference,  
                    "answer_type": row.get("answer_type"),  
                    "difficulty": row.get("difficulty"),  
                    "evidence_topics": row.get("evidence_topics", []),  
                    "status": "error",  
                    "error": response.get("message", "unknown error"),  
                }  
            )  
            print(f"  -> Error: {response.get('message', 'unknown error')}")  
            continue  
  
        prediction = response.get("answer", "")  
        sources = response.get("sources", []) or []  
  
        conciseness = conciseness_score(prediction)  
  
        em = 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0  
        f1 = token_f1(prediction, reference)  
  
        conciseness_scores.append(conciseness)  
        em_scores.append(em)  
        f1_scores.append(f1)  
        source_counts.append(len(sources))  

        ragas_samples.append(
            {
                "question": question,
                "answer": prediction,
                "contexts": [s.get("content", "") for s in sources if s.get("content")],
                "ground_truth": reference,
            }
        )

        ragas_result_indices.append(len(results))

        results.append(  
            {  
                "id": qid,  
                "question": question,  
                "ground_truth": reference,  
                "predicted_answer": prediction,  
                "answer_type": row.get("answer_type"),  
                "difficulty": row.get("difficulty"),  
                "evidence_topics": row.get("evidence_topics", []),  
                "status": "success",  
                "exact_match": em,  
                "token_f1": f1,  
                "relevance": None,
                "faithfulness": None,
                "completeness": None,
                "conciseness": conciseness,  
                "metric_details": {  
                    "relevance": {"method": "ragas"},
                    "faithfulness": {"method": "ragas"},
                    "completeness": {"method": "ragas"},
                    "conciseness": {"method": "heuristic_word_count"},
                },  
                "source_count": len(sources),  
                "sources": sources,  
                "response_metadata": response.get("metadata", {}),  
            }  
        )  

        print(f"  -> Success: queued for RAGAS scoring, f1={f1:.2f}")

    if ragas_samples:
        print("\nRunning RAGAS metrics (faithfulness, relevance, completeness)...")
        ragas_scores = run_ragas_metrics(ragas_samples, cfg)
        for i, metric_row in enumerate(ragas_scores):
            if i >= len(ragas_result_indices):
                break
            result_idx = ragas_result_indices[i]
            results[result_idx]["faithfulness"] = metric_row["faithfulness"]
            results[result_idx]["relevance"] = metric_row["relevance"]
            results[result_idx]["completeness"] = metric_row["completeness"]

            faithfulness_scores.append(metric_row["faithfulness"])
            relevance_scores.append(metric_row["relevance"])
            completeness_scores.append(metric_row["completeness"])
  
    # Build summary  
    summary = {  
        "total_questions": len(qa_rows),  
        "answered_successfully": len([r for r in results if r.get("status") == "success"]),  
        "relevance_avg": mean(relevance_scores) if relevance_scores else None,  
        "faithfulness_avg": mean(faithfulness_scores) if faithfulness_scores else None,  
        "completeness_avg": mean(completeness_scores) if completeness_scores else None,  
        "conciseness_avg": mean(conciseness_scores) if conciseness_scores else None,  
        "exact_match_avg": mean(em_scores) if em_scores else None,  
        "token_f1_avg": mean(f1_scores) if f1_scores else None,  
        "avg_source_count": mean(source_counts) if source_counts else 0.0,  
    }  
  
    # Build output  
    output = {  
        "pdf": pdf_path,  
        "config": {  
            "agentic": agentic,  
            "top_k": top_k,  
            "rerank_top_k": rerank_top_k,  
            "retrieval_mode": retrieval_mode,  
            "use_query_rewriting": use_query_rewriting,  
            "force_reprocess": force_reprocess,  
        },  
        "qa_set": qa_rows,  
        "document_processing": process_result,  
        "summary": summary,  
        "results": results,  
    }  
  
    # Ensure output directory exists  
    out_path = Path(out_json_path)  
    out_path.parent.mkdir(parents=True, exist_ok=True)  
  
    # Write results  
    with open(out_json_path, "w", encoding="utf-8") as f:  
        json.dump(output, f, ensure_ascii=False, indent=2)  
  
    print(f"\n{'='*60}")  
    print(f"Evaluation complete!")  
    print(f"Results written to: {out_json_path}")  
    print(f"\nSummary:")  
    print(f"  Total questions: {summary['total_questions']}")  
    print(f"  Answered successfully: {summary['answered_successfully']}")  
    if summary['relevance_avg'] is not None:  
        print(f"  Avg Relevance: {summary['relevance_avg']:.3f}")  
        print(f"  Avg Faithfulness: {summary['faithfulness_avg']:.3f}")  
        print(f"  Avg Completeness: {summary['completeness_avg']:.3f}")  
        print(f"  Avg Conciseness: {summary['conciseness_avg']:.3f}")  
        print(f"  Avg Token F1: {summary['token_f1_avg']:.3f}")  
  
  
def main() -> None:  
    # Determine JSON path: command line arg or default  
    if len(sys.argv) > 1:  
        json_path = sys.argv[1]  
    else:  
        json_path = INPUT_JSON_PATH  
  
    if not Path(json_path).exists():  
        print(f"Error: JSON file not found: {json_path}")  
        print(f"Usage: python {sys.argv[0]} <path_to_config.json>")  
        sys.exit(1)  
  
    run_evaluation(json_path)  
  
  
if __name__ == "__main__":  
    main()  
