"""
Self-healing RAG graph built with LangGraph.

Topology:
    START → retrieve → generate → critique
                                      │
                       ┌──────────────┴──────────────┐
                   grounded                      not_grounded
                       │                              │
               final_answer → END             rewrite_query
                                                      │
                                         ┌────────────┴────────────┐
                                  retry < MAX              retry > MAX
                                         │                          │
                                      retrieve               (give_up)
                                                        final_answer → END
"""

import os
from typing import Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from ingest import get_vector_store

load_dotenv()

MAX_RETRIES = 2


# ── State ────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    query: str
    retrieved_docs: list[str]
    answer: str
    critique: str
    verdict: Literal["grounded", "not_grounded", "give_up"]
    retry_count: int
    final_answer: str


# ── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> dict:
    print(f"\n[retrieve] query='{state['query']}'")
    store = get_vector_store()
    results = store.similarity_search(state["query"], k=4)
    docs = [doc.page_content for doc in results]
    print(f"[retrieve] {len(docs)} chunk(s) returned")
    for i, d in enumerate(docs, 1):
        print(f"  chunk {i}: {d[:120].replace(chr(10), ' ')}…")
    return {"retrieved_docs": docs}


def generate(state: RAGState) -> dict:
    print(f"\n[generate] question='{state['question']}'")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context = "\n\n---\n\n".join(state["retrieved_docs"])

    messages = [
        SystemMessage(content=(
            "You are a precise assistant. Answer the user's question using ONLY "
            "the information in the provided context. Do not add any facts not "
            "present in the context. If the context is insufficient, say so explicitly."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()
    print(f"[generate] answer (first 200 chars): {answer[:200]}…")
    return {"answer": answer}


def critique(state: RAGState) -> dict:
    print(f"\n[critique] evaluating grounding …")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context = "\n\n---\n\n".join(state["retrieved_docs"])

    messages = [
        SystemMessage(content=(
            "You are a strict fact-checker. Decide whether every claim in the "
            "given answer is directly supported by the provided context.\n"
            "Respond with EXACTLY two lines:\n"
            "Line 1: GROUNDED  or  NOT_GROUNDED\n"
            "Line 2: One sentence explaining your verdict."
        )),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Answer to evaluate:\n{state['answer']}"
        )),
    ]

    response = llm.invoke(messages)
    critique_text = response.content.strip()
    lines = critique_text.split("\n", 1)
    first_line = lines[0].strip().upper()

    verdict: Literal["grounded", "not_grounded"] = (
        "not_grounded" if "NOT_GROUNDED" in first_line else "grounded"
    )

    reason = lines[1].strip() if len(lines) > 1 else "(no reason given)"
    print(f"[critique] verdict={verdict}")
    print(f"[critique] reason: {reason}")
    return {"critique": critique_text, "verdict": verdict}


def rewrite_query(state: RAGState) -> dict:
    retry_count = state.get("retry_count", 0) + 1
    print(f"\n[rewrite_query] attempt #{retry_count} / {MAX_RETRIES}")

    if retry_count > MAX_RETRIES:
        print(f"[rewrite_query] max retries exceeded → giving up")
        return {"retry_count": retry_count, "verdict": "give_up"}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content=(
            "You are a search-query optimizer. Given an original question and "
            "critic feedback explaining why the previous retrieval was unhelpful, "
            "rewrite the search query so it is more likely to retrieve relevant "
            "documents. Return ONLY the rewritten query — no preamble, no quotes."
        )),
        HumanMessage(content=(
            f"Original question: {state['question']}\n"
            f"Previous query: {state['query']}\n"
            f"Critic feedback: {state['critique']}"
        )),
    ]

    response = llm.invoke(messages)
    new_query = response.content.strip()
    print(f"[rewrite_query] new query='{new_query}'")
    return {"query": new_query, "retry_count": retry_count, "verdict": "not_grounded"}


def final_answer(state: RAGState) -> dict:
    print(f"\n[final_answer] verdict={state.get('verdict')}")

    if state.get("verdict") == "give_up":
        answer = (
            "I don't have enough information in the available documents to give "
            "a reliable answer to your question. Please try rephrasing it or "
            "consult additional sources."
        )
    else:
        answer = state["answer"]

    print(f"[final_answer] → {answer[:200]}…")
    return {"final_answer": answer}


# ── Routing ──────────────────────────────────────────────────────────────────

def route_after_critique(state: RAGState) -> str:
    return "final_answer" if state["verdict"] == "grounded" else "rewrite_query"


def route_after_rewrite(state: RAGState) -> str:
    return "final_answer" if state["verdict"] == "give_up" else "retrieve"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    wf = StateGraph(RAGState)

    wf.add_node("retrieve", retrieve)
    wf.add_node("generate", generate)
    wf.add_node("critique", critique)
    wf.add_node("rewrite_query", rewrite_query)
    wf.add_node("final_answer", final_answer)

    wf.add_edge(START, "retrieve")
    wf.add_edge("retrieve", "generate")
    wf.add_edge("generate", "critique")

    wf.add_conditional_edges(
        "critique",
        route_after_critique,
        {"final_answer": "final_answer", "rewrite_query": "rewrite_query"},
    )

    wf.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {"retrieve": "retrieve", "final_answer": "final_answer"},
    )

    wf.add_edge("final_answer", END)

    return wf.compile()


graph = build_graph()
