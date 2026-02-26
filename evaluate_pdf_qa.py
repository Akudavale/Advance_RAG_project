#!/usr/bin/env python3  
"""  
Evaluate one PDF against a JSON QA set and export results to JSON.  
Reads configuration and questions from an external JSON file.  
"""  
  
import json  
import re  
import sys  
from pathlib import Path  
from statistics import mean  
from typing import Any, Dict, List  
  
from config.config import Config  
from src.evaluation.evaluator import Evaluator  
from src.orchestrator import RAGOrchestrator  
  
  
# ============================================================  
# CONFIG: Path to the input JSON file  
# ============================================================  
  
INPUT_JSON_PATH = r"evaluation_config.json"  # <-- Set this to your JSON file path  
  
# Optional: Override via command line argument  
# Usage: python script.py path/to/config.json  
  
  
# ============================================================  
# HELPERS  
# ============================================================  
  
def _normalize_text(text: str) -> str:  
    return " ".join((text or "").strip().lower().split())  
  
  
def _token_f1(prediction: str, reference: str) -> float:  
    p_tokens = _normalize_text(prediction).split()  
    r_tokens = _normalize_text(reference).split()  
  
    if not p_tokens and not r_tokens:  
        return 1.0  
    if not p_tokens or not r_tokens:  
        return 0.0  
  
    p_set = set(p_tokens)  
    r_set = set(r_tokens)  
    overlap = len(p_set & r_set)  
  
    if overlap == 0:  
        return 0.0  
  
    precision = overlap / len(p_set)  
    recall = overlap / len(r_set)  
    return (2 * precision * recall) / (precision + recall)  
  
  
def _conciseness_score(answer: str) -> float:  
    """Heuristic conciseness score in [0, 1]."""  
    words = re.findall(r"\w+", answer or "")  
    word_count = len(words)  
  
    if word_count == 0:  
        return 0.0  
    if word_count <= 25:  
        return 0.9  
    if word_count <= 60:  
        return 0.8  
    if word_count <= 120:  
        return 0.6  
    if word_count <= 220:  
        return 0.4  
    return 0.2  
  
  
def _load_json_config(json_path: str) -> Dict[str, Any]:  
    """  
    Load and parse the JSON configuration file.  
      
    Expected format:  
    [  
        {  
            "input_pdf_path": "...",  
            "output_json_path": "...",  
            "agentic": true,  
            "top_k": 10,  
            "rerank_top_k": 5,  
            "retrieval_mode": "hybrid",  
            "use_query_rewriting": true,  
            "force_reprocess": true  # optional, defaults to True  
        },  
        { "id": "q1", "question": "...", "ground_truth": "...", ... },  
        { "id": "q2", "question": "...", "ground_truth": "...", ... },  
        ...  
    ]  
      
    Returns:  
        Dict with 'config' and 'qa_set' keys  
    """  
    with open(json_path, "r", encoding="utf-8") as f:  
        data = json.load(f)  
  
    if not isinstance(data, list) or len(data) < 2:  
        raise ValueError(  
            "JSON must be a list with at least 2 elements: "  
            "[config_object, question1, question2, ...]"  
        )  
  
    # First element is the configuration  
    config_obj = data[0]  
      
    # Validate required config fields  
    required_config_fields = ["input_pdf_path", "output_json_path"]  
    for field in required_config_fields:  
        if field not in config_obj:  
            raise ValueError(f"Config object missing required field: '{field}'")  
  
    # Extract config with defaults  
    config = {  
        "input_pdf_path": config_obj["input_pdf_path"],  
        "output_json_path": config_obj["output_json_path"],  
        "agentic": config_obj.get("agentic", True),  
        "top_k": config_obj.get("top_k", 10),  
        "rerank_top_k": config_obj.get("rerank_top_k", 5),  
        "retrieval_mode": config_obj.get("retrieval_mode", "hybrid"),  
        "use_query_rewriting": config_obj.get("use_query_rewriting", True),  
        "force_reprocess": config_obj.get("force_reprocess", True),  
    }  
  
    # Remaining elements are QA items  
    qa_set = data[1:]  
  
    return {"config": config, "qa_set": qa_set}  
  
  
def _validate_qa_set(qa_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  
    """  
    Ensures the QA set follows the expected JSON format.  
    Required keys:  
      - id  
      - question  
      - ground_truth  
    Optional keys:  
      - answer_type  
      - difficulty  
      - evidence_topics  
    """  
    validated_rows: List[Dict[str, Any]] = []  
  
    if not isinstance(qa_set, list):  
        raise ValueError("QA set must be a list of QA objects")  
  
    for i, item in enumerate(qa_set):  
        if not isinstance(item, dict):  
            raise ValueError(f"QA set item at index {i} is not a dictionary")  
  
        qid = str(item.get("id", f"q{i+1}")).strip()  
        question = str(item.get("question", "")).strip()  
        ground_truth = str(item.get("ground_truth", "")).strip()  
  
        if not question:  
            raise ValueError(f"QA set item at index {i} is missing 'question'")  
        if not ground_truth:  
            raise ValueError(f"QA set item at index {i} is missing 'ground_truth'")  
  
        validated_rows.append(  
            {  
                "id": qid,  
                "question": question,  
                "ground_truth": ground_truth,  
                "answer_type": item.get("answer_type"),  
                "difficulty": item.get("difficulty"),  
                "evidence_topics": item.get("evidence_topics", []),  
            }  
        )  
  
    return validated_rows  
  
  
# ============================================================  
# MAIN  
# ============================================================  
  
def run_evaluation(json_path: str) -> None:  
    """Run evaluation using the specified JSON configuration file."""  
      
    print(f"Loading configuration from: {json_path}")  
      
    # Load and parse JSON  
    parsed = _load_json_config(json_path)  
    eval_config = parsed["config"]  
    qa_rows = _validate_qa_set(parsed["qa_set"])  
  
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
    evaluator = Evaluator(cfg)  
  
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
  
        eval_results = evaluator.evaluate(  
            query=question,  
            answer=prediction,  
            context=sources,  
            reference_answer=reference,  
        )  
  
        relevance = float(eval_results["relevance"].score)  
        faithfulness = float(eval_results["faithfulness"].score)  
        completeness = float(eval_results["completeness"].score)  
        conciseness = _conciseness_score(prediction)  
  
        em = 1.0 if _normalize_text(prediction) == _normalize_text(reference) else 0.0  
        f1 = _token_f1(prediction, reference)  
  
        relevance_scores.append(relevance)  
        faithfulness_scores.append(faithfulness)  
        completeness_scores.append(completeness)  
        conciseness_scores.append(conciseness)  
        em_scores.append(em)  
        f1_scores.append(f1)  
        source_counts.append(len(sources))  
  
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
                "relevance": relevance,  
                "faithfulness": faithfulness,  
                "completeness": completeness,  
                "conciseness": conciseness,  
                "metric_details": {  
                    "relevance": eval_results["relevance"].details,  
                    "faithfulness": eval_results["faithfulness"].details,  
                    "completeness": eval_results["completeness"].details,  
                    "conciseness": {"method": "heuristic_word_count"},  
                },  
                "source_count": len(sources),  
                "sources": sources,  
                "response_metadata": response.get("metadata", {}),  
            }  
        )  
  
        print(f"  -> Success: relevance={relevance:.2f}, faithfulness={faithfulness:.2f}, "  
              f"completeness={completeness:.2f}, f1={f1:.2f}")  
  
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