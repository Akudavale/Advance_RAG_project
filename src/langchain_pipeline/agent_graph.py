"""
LangGraph-based agentic loop for RAG.
"""

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage


class AgentState(TypedDict):
    original_query: str
    working_query: str
    rewritten_query: Optional[str]
    conversation_context: List[Dict[str, Any]]
    context_docs: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    current_step: int
    max_steps: int
    decision: Dict[str, Any]
    done: bool
    run_cfg: Dict[str, Any]


class LangGraphAgentRunner:
    """Graph runner for decide -> (rewrite|retrieve|answer)."""

    def __init__(
        self,
        planner_llm,
        query_rewriter,
        retrieval_engine,
        logger=None,
    ):
        self.planner_llm = planner_llm
        self.query_rewriter = query_rewriter
        self.retrieval_engine = retrieval_engine
        self.logger = logger
        self._graph = self._build_graph()

    def run(
        self,
        *,
        query: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None,
        use_query_rewriting: bool,
        method: str,
        top_k: int,
        agent_max_steps: int,
        retrieval_mode: str,
        dense_top_k: Optional[int],
        sparse_top_k: Optional[int],
        hybrid_alpha: float,
        filter_filenames: Optional[List[str]],
        use_reranking: bool,
        reranker: Optional[Any],
        rerank_top_k: int,
    ) -> Dict[str, Any]:
        run_cfg = {
            "use_query_rewriting": use_query_rewriting,
            "default_method": method,
            "default_top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "dense_top_k": dense_top_k,
            "sparse_top_k": sparse_top_k,
            "hybrid_alpha": hybrid_alpha,
            "filter_filenames": filter_filenames,
            "use_reranking": use_reranking,
            "reranker": reranker,
            "rerank_top_k": rerank_top_k,
        }

        initial: AgentState = {
            "original_query": query,
            "working_query": query,
            "rewritten_query": None,
            "conversation_context": conversation_context or [],
            "context_docs": [],
            "trace": [],
            "current_step": 1,
            "max_steps": max(1, int(agent_max_steps)),
            "decision": {},
            "done": False,
            "run_cfg": run_cfg,
        }

        final = self._graph.invoke(initial)
        return {
            "rewritten_query": final.get("rewritten_query"),
            "context_docs": final.get("context_docs", []),
            "trace": final.get("trace", []),
            "working_query": final.get("working_query"),
        }

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("decide", self._decide_node)
        graph.add_node("rewrite", self._rewrite_node)
        graph.add_node("retrieve", self._retrieve_node)

        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {
                "rewrite": "rewrite",
                "retrieve": "retrieve",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "rewrite",
            self._route_after_action,
            {
                "decide": "decide",
                "end": END,
            },
        )
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_action,
            {
                "decide": "decide",
                "end": END,
            },
        )
        return graph.compile()

    def _route_after_decide(self, state: AgentState) -> str:
        if state.get("done"):
            return "end"
        action = str((state.get("decision") or {}).get("action", "retrieve")).lower()
        run_cfg = state["run_cfg"]
        if action == "rewrite" and run_cfg["use_query_rewriting"]:
            return "rewrite"
        if action == "answer":
            return "end"
        return "retrieve"

    def _route_after_action(self, state: AgentState) -> str:
        return "end" if state.get("done") else "decide"

    def _decide_node(self, state: AgentState) -> AgentState:
        run_cfg = state["run_cfg"]
        step = int(state.get("current_step", 1))
        max_steps = int(state.get("max_steps", 1))
        has_context = bool(state.get("context_docs"))

        if step > max_steps:
            state["done"] = True
            return state

        if step >= max_steps and has_context:
            state["decision"] = {"action": "answer"}
            state["trace"] = state.get("trace", []) + [
                {"step": step, "action": "answer", "reason": "sufficient_context"}
            ]
            state["done"] = True
            return state

        if not has_context:
            state["decision"] = {
                "action": "rewrite" if run_cfg["use_query_rewriting"] and step == 1 else "retrieve",
                "method": run_cfg["default_method"],
                "top_k": run_cfg["default_top_k"],
            }
            return state

        planner_prompt = (
            "You are controlling a RAG agent. Choose one next action.\n"
            "Return ONLY valid JSON with keys: action, method, top_k, reason.\n"
            "Allowed action values: rewrite, retrieve, answer.\n"
            "Allowed method values: expand, hyde, multi, decompose.\n"
            f"Original query: {state['original_query']}\n"
            f"Current query: {state['working_query']}\n"
            f"Has retrieved context: {has_context}\n"
            f"Step: {step}/{max_steps}\n"
            "Rules:\n"
            "1) If context already exists and appears sufficient, choose answer.\n"
            "2) Choose retrieve if more evidence is needed.\n"
            "3) Choose rewrite only if retrieval is likely improved by reframing.\n"
            "4) Keep top_k between 3 and 20.\n"
        )

        decision = {
            "action": "answer" if has_context else "retrieve",
            "method": run_cfg["default_method"],
            "top_k": run_cfg["default_top_k"],
        }
        try:
            response = self.planner_llm.invoke([HumanMessage(content=planner_prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
            parsed = self._extract_json_object(raw)
            if parsed:
                action = str(parsed.get("action", "retrieve")).lower()
                if action not in {"rewrite", "retrieve", "answer"}:
                    action = "retrieve"
                method = str(parsed.get("method", run_cfg["default_method"])).lower()
                if method not in {"expand", "hyde", "multi", "decompose"}:
                    method = run_cfg["default_method"]
                top_k = parsed.get("top_k", run_cfg["default_top_k"])
                try:
                    top_k = int(top_k)
                except Exception:
                    top_k = run_cfg["default_top_k"]
                decision = {
                    "action": action,
                    "method": method,
                    "top_k": max(3, min(top_k, 20)),
                    "reason": parsed.get("reason", ""),
                }
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Agent planner failed: {e}")

        if decision.get("action") == "answer" and has_context:
            state["trace"] = state.get("trace", []) + [
                {"step": step, "action": "answer", "reason": "sufficient_context"}
            ]
            state["done"] = True

        state["decision"] = decision
        return state

    def _rewrite_node(self, state: AgentState) -> AgentState:
        run_cfg = state["run_cfg"]
        step = int(state.get("current_step", 1))
        decision = state.get("decision", {}) or {}
        rewrite_method = str(decision.get("method", run_cfg["default_method"])).strip().lower()
        rewrite_method = rewrite_method or run_cfg["default_method"]

        success = False
        try:
            rewrite_result = self.query_rewriter.rewrite(
                query=state["working_query"],
                context=state.get("conversation_context") or [],
                method=rewrite_method,
            )
            next_query = (rewrite_result.get("rewritten_query") or state["working_query"]).strip()
            if next_query and next_query != state["working_query"]:
                state["rewritten_query"] = next_query
                state["working_query"] = next_query
            success = bool(rewrite_result.get("success"))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Agent rewrite failed: {e}")

        state["trace"] = state.get("trace", []) + [
            {
                "step": step,
                "action": "rewrite",
                "method": rewrite_method,
                "success": success,
            }
        ]
        state["current_step"] = step + 1
        if state["current_step"] > state["max_steps"]:
            state["done"] = True
        return state

    def _retrieve_node(self, state: AgentState) -> AgentState:
        run_cfg = state["run_cfg"]
        step = int(state.get("current_step", 1))
        decision = state.get("decision", {}) or {}

        retrieve_top_k = decision.get("top_k", run_cfg["default_top_k"])
        try:
            retrieve_top_k = int(retrieve_top_k)
        except Exception:
            retrieve_top_k = run_cfg["default_top_k"]
        retrieve_top_k = max(1, min(retrieve_top_k, 50))

        search_results = self.retrieval_engine.retrieve(
            query=state["working_query"],
            top_k=retrieve_top_k,
            retrieval_mode=run_cfg["retrieval_mode"],
            dense_top_k=run_cfg["dense_top_k"],
            sparse_top_k=run_cfg["sparse_top_k"],
            hybrid_alpha=run_cfg["hybrid_alpha"],
            filter_filenames=run_cfg["filter_filenames"],
            use_reranking=run_cfg["use_reranking"],
            reranker=run_cfg["reranker"],
            rerank_query=state["working_query"],
            rerank_top_k=run_cfg["rerank_top_k"],
        )

        state["context_docs"] = [
            {
                "index": i + 1,
                "content": result.content,
                "metadata": result.metadata,
                "score": result.score,
            }
            for i, result in enumerate(search_results)
        ]
        state["trace"] = state.get("trace", []) + [
            {
                "step": step,
                "action": "retrieve",
                "query_used": state["working_query"],
                "retrieved": len(state["context_docs"]),
            }
        ]

        state["current_step"] = step + 1
        if state["current_step"] > state["max_steps"]:
            state["done"] = True
        return state

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}

        text = text.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
