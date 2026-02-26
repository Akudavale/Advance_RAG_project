"""
Adapters for converting internal retrieval results into prompt-ready context.
"""

from typing import Any, Dict, List


def format_context_docs(context_docs: List[Dict[str, Any]], max_chars_per_doc: int = 1500) -> str:
    """Render retrieved docs into a compact, citation-friendly context block."""
    if not context_docs:
        return "No relevant documents found."

    rendered: List[str] = []
    for i, doc in enumerate(context_docs, start=1):
        index = doc.get("index", i)
        content = (doc.get("content", "") or "").strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."

        metadata = doc.get("metadata", {}) or {}
        filename = metadata.get("filename", "")
        page = metadata.get("page_number", "")
        modality = metadata.get("modality", "")
        score = doc.get("score", 0.0)

        header_bits = [f"Document {index}"]
        if filename:
            header_bits.append(f"source={filename}")
        if page != "":
            header_bits.append(f"page={page}")
        if modality:
            header_bits.append(f"modality={modality}")
        if score:
            try:
                header_bits.append(f"score={float(score):.3f}")
            except Exception:
                pass

        header = " | ".join(header_bits)
        rendered.append(f"[{header}]\n{content}")

    return "\n\n".join(rendered)


def format_history(messages: List[Dict[str, Any]], max_turns: int = 5, max_chars_per_msg: int = 300) -> str:
    """Render recent conversation history for history-aware prompts."""
    if not messages:
        return "No previous conversation."

    clipped = messages[-(max_turns * 2) :]
    parts: List[str] = []
    for msg in clipped:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "..."
        parts.append(f"{role}: {content}")

    return "\n".join(parts)

