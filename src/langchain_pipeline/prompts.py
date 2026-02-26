"""
LangChain prompt templates for RAG.
"""

from langchain_core.prompts import ChatPromptTemplate


def build_rag_prompt(with_history: bool = False) -> ChatPromptTemplate:
    """Create base RAG prompt template."""
    if with_history:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a factual assistant. Answer only from provided documents.\n"
                        "If information is missing, say what is missing instead of inventing.\n"
                        "When possible, cite document numbers like 'Document 2'."
                    ),
                ),
                (
                    "human",
                    (
                        "Conversation history:\n{history}\n\n"
                        "Documents:\n{context}\n\n"
                        "Question: {query}\n\n"
                        "Answer:"
                    ),
                ),
            ]
        )

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a factual assistant. Answer only from provided documents.\n"
                    "If information is missing, say what is missing instead of inventing.\n"
                    "When possible, cite document numbers like 'Document 2'."
                ),
            ),
            (
                "human",
                "Documents:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            ),
        ]
    )

