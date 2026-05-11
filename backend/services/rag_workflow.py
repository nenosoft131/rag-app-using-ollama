from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
import uuid
import asyncio
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from services.ollama_service import OllamaService
from services.vector_service import VectorService
from stats import stats


def _log(msg: str):
    print(msg)
    stats.add_log(msg)


class RAGState(TypedDict, total=False):
    query: str
    model: str
    session_id: str
    route: str                  # "retrieval" | "general"
    context: List[str]          # raw retrieved docs
    filtered_context: List[str] # docs that passed relevance grading
    sources: List[Any]
    response: str
    hallucination_check: str    # "grounded" | "not_grounded"
    retrieval_retry: int        # how many times grader has sent us back to retrieve
    generation_retry: int       # how many times hallucination checker has regenerated


class RAGWorkflow:
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.ollama_service = OllamaService()
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)

        g = StateGraph(RAGState)

        g.add_node("router",              self._router_agent)
        g.add_node("retrieve",            self._retrieval_agent)
        g.add_node("grader",              self._relevance_grader_agent)
        g.add_node("generate",            self._generator_agent)
        g.add_node("hallucination_check", self._hallucination_grader_agent)

        g.set_entry_point("router")

        # Router: retrieval path or direct generation
        g.add_conditional_edges(
            "router",
            lambda s: s.get("route", "retrieval"),
            {"retrieval": "retrieve", "general": "generate"},
        )

        g.add_edge("retrieve", "grader")

        # Grader: good docs → generate | no docs + retries left → re-retrieve | give up → generate
        g.add_conditional_edges(
            "grader",
            self._grader_route,
            {"generate": "generate", "retry": "retrieve"},
        )

        g.add_edge("generate", "hallucination_check")

        # Hallucination checker: grounded → end | not grounded + retries left → regenerate
        g.add_conditional_edges(
            "hallucination_check",
            self._hallucination_route,
            {"end": END, "regenerate": "generate"},
        )

        return g.compile(checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # Agent 1 — Router
    # ------------------------------------------------------------------
    def _router_agent(self, state: RAGState) -> RAGState:
        _log("[Router] Classifying query...")

        if self.vector_service.get_vector_size() == 0:
            _log("[Router] No documents loaded → general")
            state["route"] = "general"
            stats.inc("general_route")
            return state

        prompt = (
            "Classify this question with a single word — either 'retrieval' or 'general'.\n\n"
            "- 'retrieval': the answer likely requires looking up an uploaded document\n"
            "- 'general': a general knowledge or conversational question\n\n"
            f"Question: {state['query']}"
        )
        reply = asyncio.run(
            self.ollama_service.chat(
                prompt=prompt,
                system_prompt="You are a query router. Reply with exactly one word.",
                model=state.get("model", "llama2"),
            )
        )
        route = "retrieval" if "retrieval" in reply.lower() else "general"
        _log(f"[Router] → {route}")
        state["route"] = route
        stats.inc("retrieval_route" if route == "retrieval" else "general_route")
        return state

    # ------------------------------------------------------------------
    # Agent 2 — Retrieval
    # ------------------------------------------------------------------
    def _retrieval_agent(self, state: RAGState) -> RAGState:
        _log("[Retrieval] Fetching documents...")
        try:
            result = self.vector_service.search(state["query"], k=4)
            state["context"] = result.get("context", [])
            state["sources"] = result
            _log(f"[Retrieval] Found {len(state['context'])} chunks")
        except Exception as e:
            _log(f"[Retrieval] Error: {e}")
            state["context"] = []
            state["sources"] = []
        return state

    # ------------------------------------------------------------------
    # Agent 3 — Relevance Grader
    # ------------------------------------------------------------------
    def _relevance_grader_agent(self, state: RAGState) -> RAGState:
        _log("[Grader] Scoring document relevance...")
        query = state["query"]
        docs = state.get("context", [])

        if not docs:
            state["filtered_context"] = []
            state["retrieval_retry"] = state.get("retrieval_retry", 0) + 1
            stats.inc("grader_retries")
            return state

        filtered = []
        for doc in docs:
            prompt = (
                "Is this document relevant to answering the question? "
                "Reply only 'yes' or 'no'.\n\n"
                f"Question: {query}\n"
                f"Document: {doc[:600]}"
            )
            reply = asyncio.run(
                self.ollama_service.chat(
                    prompt=prompt,
                    system_prompt="You are a relevance grader. Reply only yes or no.",
                    model=state.get("model", "llama2"),
                )
            )
            if "yes" in reply.lower():
                filtered.append(doc)

        _log(f"[Grader] {len(filtered)}/{len(docs)} docs passed")
        state["filtered_context"] = filtered

        if not filtered:
            state["retrieval_retry"] = state.get("retrieval_retry", 0) + 1
            stats.inc("grader_retries")

        return state

    def _grader_route(self, state: RAGState) -> str:
        if state.get("filtered_context"):
            return "generate"
        if state.get("retrieval_retry", 0) <= 1:
            _log("[Grader] No relevant docs — retrying retrieval")
            return "retry"
        _log("[Grader] Still no relevant docs after retry — generating without context")
        return "generate"

    # ------------------------------------------------------------------
    # Agent 4 — Generator
    # ------------------------------------------------------------------
    def _generator_agent(self, state: RAGState) -> RAGState:
        _log("[Generator] Generating response...")
        context_docs = state.get("filtered_context") or state.get("context", [])
        context_text = "\n\n".join(context_docs) if context_docs else "No relevant context found."

        system_prompt = (
            "You are a helpful assistant that answers questions based on provided context. "
            "Use only information from the context. If the context is insufficient, say so."
        )
        full_prompt = f"Context:\n{context_text}\n\nQuestion: {state['query']}"

        try:
            response = asyncio.run(
                self.ollama_service.chat(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    model=state.get("model", "llama2"),
                )
            )
            state["response"] = response
        except Exception as e:
            _log(f"[Generator] Error: {e}")
            state["response"] = f"Error generating response: {str(e)}"

        return state

    # ------------------------------------------------------------------
    # Agent 5 — Hallucination Grader
    # ------------------------------------------------------------------
    def _hallucination_grader_agent(self, state: RAGState) -> RAGState:
        _log("[Hallucination Grader] Checking answer grounding...")
        context_docs = state.get("filtered_context") or state.get("context", [])

        if not context_docs:
            state["hallucination_check"] = "grounded"
            return state

        context_text = "\n\n".join(context_docs[:2])[:800]
        prompt = (
            "Is the answer below fully supported by the context? "
            "Reply only 'yes' or 'no'.\n\n"
            f"Context: {context_text}\n\n"
            f"Answer: {state.get('response', '')}"
        )
        reply = asyncio.run(
            self.ollama_service.chat(
                prompt=prompt,
                system_prompt="You are a hallucination grader. Reply only yes or no.",
                model=state.get("model", "llama2"),
            )
        )
        check = "grounded" if "yes" in reply.lower() else "not_grounded"
        _log(f"[Hallucination Grader] → {check}")
        state["hallucination_check"] = check

        if check == "not_grounded":
            state["generation_retry"] = state.get("generation_retry", 0) + 1
            stats.inc("hallucination_retries")

        return state

    def _hallucination_route(self, state: RAGState) -> str:
        if state.get("hallucination_check") == "not_grounded" and state.get("generation_retry", 0) <= 1:
            _log("[Hallucination Grader] Not grounded — regenerating")
            return "regenerate"
        return "end"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def process_message(
        self, message: str, session_id: str = None, model: str = "llama2"
    ) -> Dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        initial_state: RAGState = {
            "query": message,
            "session_id": sid,
            "model": model,
            "retrieval_retry": 0,
            "generation_retry": 0,
        }

        try:
            config = {"configurable": {"thread_id": sid}}
            final_state = self.workflow.invoke(initial_state, config=config)

            return {
                "response": final_state.get("response", "No response generated."),
                "session_id": final_state.get("session_id", sid),
                "sources": final_state.get("sources", []),
            }
        except Exception as e:
            print(f"[Workflow] Error: {e}")
            return {
                "response": f"Error processing message: {str(e)}",
                "session_id": sid,
                "sources": [],
            }
