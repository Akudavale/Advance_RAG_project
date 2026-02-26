"""
Shared evaluation helpers for JSON config parsing and metric computation.
"""

import json
import re
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    """Normalize text for lexical matching metrics."""
    return " ".join((text or "").strip().lower().split())


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score."""
    p_tokens = normalize_text(prediction).split()
    r_tokens = normalize_text(reference).split()

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


def conciseness_score(answer: str) -> float:
    """Heuristic conciseness score in [0, 1] based on answer length."""
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


def load_json_eval_config(json_path: str) -> Dict[str, Any]:
    """
    Load evaluation config JSON with format:
    [config_object, qa_item_1, qa_item_2, ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(
            "JSON must be a list with at least 2 elements: "
            "[config_object, question1, question2, ...]"
        )

    config_obj = data[0]

    required_config_fields = ["input_pdf_path", "output_json_path"]
    for field in required_config_fields:
        if field not in config_obj:
            raise ValueError(f"Config object missing required field: '{field}'")

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

    return {"config": config, "qa_set": data[1:]}


def validate_qa_set(qa_set: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate QA rows used by evaluator."""
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


def _build_ragas_runtime(config) -> Dict[str, Any]:
    """Build explicit RAGAS llm/embeddings runtime from project config."""
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except Exception as e:
        raise RuntimeError(
            "RAGAS wrappers are unavailable. Please verify ragas installation."
        ) from e

    # Use project LLM provider (Azure/OpenAI/Gemini) instead of RAGAS default OpenAI client.
    from src.langchain_pipeline.models import build_chat_model

    lc_llm = build_chat_model(config)
    ragas_llm = LangchainLLMWrapper(lc_llm)

    # Use local HF embeddings for metric steps that require embeddings.
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        device = config.EMBEDDING_DEVICE if config.EMBEDDING_DEVICE else "cpu"
        lc_embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize embeddings for RAGAS. "
            "Ensure sentence-transformers/langchain-community are installed."
        ) from e

    ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
    return {"llm": ragas_llm, "embeddings": ragas_embeddings}


def run_ragas_metrics(samples: List[Dict[str, Any]], config) -> List[Dict[str, float]]:
    """
    Run RAGAS metrics for faithfulness, relevance, and completeness.

    Each sample must include keys: question, answer, contexts, ground_truth.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas import metrics as ragas_metrics
    except Exception as e:
        raise RuntimeError(
            "RAGAS is not available. Install compatible versions (e.g., ragas, datasets). "
            f"Import error: {e}"
        ) from e

    def resolve_metric(candidates: List[str]):
        for name in candidates:
            if not hasattr(ragas_metrics, name):
                continue
            metric_obj = getattr(ragas_metrics, name)
            if isinstance(metric_obj, type):
                try:
                    return metric_obj()
                except Exception:
                    continue
            return metric_obj
        return None

    faithfulness_metric = resolve_metric(["Faithfulness", "faithfulness"])
    relevance_metric = resolve_metric(
        ["ResponseRelevancy", "AnswerRelevancy", "response_relevancy", "answer_relevancy"]
    )
    completeness_metric = resolve_metric(
        ["LLMContextRecall", "ContextRecall", "context_recall", "context_entity_recall"]
    )

    selected_metrics = [
        m for m in [faithfulness_metric, relevance_metric, completeness_metric] if m is not None
    ]
    if not selected_metrics:
        raise RuntimeError(
            "Could not resolve RAGAS metrics for faithfulness/relevance/completeness."
        )

    dataset = Dataset.from_list(
        [
            {
                "question": s["question"],
                "answer": s["answer"],
                "contexts": s["contexts"],
                "ground_truth": s["ground_truth"],
            }
            for s in samples
        ]
    )

    runtime = _build_ragas_runtime(config)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=selected_metrics,
        llm=runtime["llm"],
        embeddings=runtime["embeddings"],
    )
    rows = ragas_result.to_pandas().to_dict(orient="records")

    def pick_score(row: Dict[str, Any], keys: List[str]) -> float:
        for key in keys:
            if key in row and row[key] is not None:
                try:
                    return float(row[key])
                except Exception:
                    continue
        return 0.0

    normalized: List[Dict[str, float]] = []
    for row in rows:
        normalized.append(
            {
                "faithfulness": pick_score(row, ["faithfulness", "Faithfulness"]),
                "relevance": pick_score(
                    row,
                    [
                        "response_relevancy",
                        "answer_relevancy",
                        "ResponseRelevancy",
                        "AnswerRelevancy",
                    ],
                ),
                "completeness": pick_score(
                    row,
                    [
                        "context_recall",
                        "llm_context_recall",
                        "context_entity_recall",
                        "ContextRecall",
                        "LLMContextRecall",
                    ],
                ),
            }
        )
    return normalized
