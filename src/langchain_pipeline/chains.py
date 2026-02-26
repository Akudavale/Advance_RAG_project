"""
LangChain runnable chains for RAG answer generation.
"""

from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

from src.langchain_pipeline.documents import format_context_docs, format_history
from src.langchain_pipeline.models import build_chat_model
from src.langchain_pipeline.prompts import build_rag_prompt


class LangChainRAGChain:
    """Primary LangChain QA chain for retrieved-context answering."""

    def __init__(self, config=None):
        from config.config import Config

        self.config = config or Config()
        self._llm = None
        self._qa_chain = None
        self._qa_chain_with_history = None

        self.max_chars_per_doc = int(getattr(self.config, "MAX_CHARS_PER_DOC", 1500))
        self.max_history_turns = int(getattr(self.config, "MAX_HISTORY_TURNS", 5))

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_chat_model(self.config)
        return self._llm

    @property
    def qa_chain(self):
        if self._qa_chain is None:
            prompt = build_rag_prompt(with_history=False)
            self._qa_chain = prompt | self.llm | StrOutputParser()
        return self._qa_chain

    @property
    def qa_chain_with_history(self):
        if self._qa_chain_with_history is None:
            prompt = build_rag_prompt(with_history=True)
            self._qa_chain_with_history = prompt | self.llm | StrOutputParser()
        return self._qa_chain_with_history

    def invoke(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate answer from retrieved docs and optional history."""
        context = format_context_docs(context_docs, max_chars_per_doc=self.max_chars_per_doc)

        if conversation_history:
            history = format_history(
                conversation_history,
                max_turns=self.max_history_turns,
            )
            return self.qa_chain_with_history.invoke(
                {"query": query, "context": context, "history": history}
            )

        return self.qa_chain.invoke({"query": query, "context": context})

