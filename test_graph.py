"""
Mock-based verification of the self-healing RAG graph topology.
No real API calls — stubs ChromaDB and ChatOpenAI.

Tests three paths:
  Path A  — GROUNDED on first try            (1 retrieve-generate-critique cycle)
  Path B  — NOT_GROUNDED once, then GROUNDED (retry loop exercises rewrite_query)
  Path C  — NOT_GROUNDED 3 times             (give_up path)
"""

import sys
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage

# ── helpers ──────────────────────────────────────────────────────────────────

def make_fake_doc(content: str):
    doc = MagicMock()
    doc.page_content = content
    return doc

FAKE_DOCS = [
    make_fake_doc("Acme Corp refunds are processed within 5-7 business days."),
    make_fake_doc("Returns are accepted within 30 days of purchase."),
    make_fake_doc("Digital products are non-refundable once activated."),
    make_fake_doc("Defective items receive free replacement shipping."),
]

def fake_vector_store():
    store = MagicMock()
    store.similarity_search.return_value = FAKE_DOCS
    return store


def make_llm_sequence(*responses: str):
    """Return a ChatOpenAI mock that cycles through the given string responses."""
    llm = MagicMock()
    llm.invoke.side_effect = [AIMessage(content=r) for r in responses]
    return llm


DIVIDER = "─" * 64

# ── path A: grounded on first try ────────────────────────────────────────────

def test_path_a():
    print(f"\n{DIVIDER}")
    print("PATH A  —  GROUNDED on first try")
    print(DIVIDER)

    # generate → answer; critique → GROUNDED
    llm_calls = make_llm_sequence(
        "Refunds are processed within 5-7 business days.",        # generate
        "GROUNDED\nEvery claim is directly supported by the context.",  # critique
    )

    with patch("graph.get_vector_store", side_effect=fake_vector_store), \
         patch("graph.ChatOpenAI", return_value=llm_calls):

        from graph import build_graph
        g = build_graph()
        result = g.invoke({
            "question": "How long do refunds take?",
            "query":    "How long do refunds take?",
            "retrieved_docs": [],
            "answer": "",
            "critique": "",
            "verdict": "not_grounded",
            "retry_count": 0,
            "final_answer": "",
        })

    assert result["verdict"] == "grounded", f"Expected grounded, got {result['verdict']}"
    assert result["retry_count"] == 0
    assert "5-7 business days" in result["final_answer"]
    print(f"\n  verdict      : {result['verdict']}")
    print(f"  retry_count  : {result['retry_count']}")
    print(f"  final_answer : {result['final_answer']}")
    print("\n  ✅  PASS")


# ── path B: not-grounded once, then grounded ────────────────────────────────

def test_path_b():
    print(f"\n{DIVIDER}")
    print("PATH B  —  NOT_GROUNDED once → rewrite → GROUNDED")
    print(DIVIDER)

    llm_calls = make_llm_sequence(
        # cycle 1
        "Refunds happen instantly.",                                      # generate (wrong answer)
        "NOT_GROUNDED\nThe claim 'instantly' is not in the context.",    # critique
        "refund processing time business days",                          # rewrite_query
        # cycle 2
        "Refunds are processed within 5-7 business days.",               # generate
        "GROUNDED\nThe timeframe is directly stated in the context.",    # critique
    )

    with patch("graph.get_vector_store", side_effect=fake_vector_store), \
         patch("graph.ChatOpenAI", return_value=llm_calls):

        from graph import build_graph
        g = build_graph()
        result = g.invoke({
            "question": "How long does a refund take?",
            "query":    "How long does a refund take?",
            "retrieved_docs": [],
            "answer": "",
            "critique": "",
            "verdict": "not_grounded",
            "retry_count": 0,
            "final_answer": "",
        })

    assert result["verdict"] == "grounded", f"Expected grounded, got {result['verdict']}"
    assert result["retry_count"] == 1
    assert "5-7 business days" in result["final_answer"]
    print(f"\n  verdict      : {result['verdict']}")
    print(f"  retry_count  : {result['retry_count']}")
    print(f"  final_answer : {result['final_answer']}")
    print("\n  ✅  PASS")


# ── path C: give_up after max retries ───────────────────────────────────────

def test_path_c():
    print(f"\n{DIVIDER}")
    print("PATH C  —  NOT_GROUNDED × 3 → give_up")
    print(DIVIDER)

    not_grounded_critique = "NOT_GROUNDED\nThe answer is not supported by the context."

    llm_calls = make_llm_sequence(
        # cycle 1
        "The moon is made of cheese.",      # generate
        not_grounded_critique,              # critique
        "refund lunar policy",              # rewrite_query (retry 1)
        # cycle 2
        "Refunds come via carrier pigeon.",  # generate
        not_grounded_critique,              # critique
        "refund pigeon avian",              # rewrite_query (retry 2)
        # cycle 3  — rewrite_query hits MAX_RETRIES, no LLM call for rewrite
        "Refunds are free forever.",        # generate
        not_grounded_critique,              # critique
        # rewrite_query node sees retry_count > MAX_RETRIES, sets give_up — no LLM call
    )

    with patch("graph.get_vector_store", side_effect=fake_vector_store), \
         patch("graph.ChatOpenAI", return_value=llm_calls):

        from graph import build_graph
        g = build_graph()
        result = g.invoke({
            "question": "What is the refund policy on Mars?",
            "query":    "What is the refund policy on Mars?",
            "retrieved_docs": [],
            "answer": "",
            "critique": "",
            "verdict": "not_grounded",
            "retry_count": 0,
            "final_answer": "",
        })

    assert result["verdict"] == "give_up", f"Expected give_up, got {result['verdict']}"
    assert result["retry_count"] == 3
    assert "don't have enough information" in result["final_answer"]
    print(f"\n  verdict      : {result['verdict']}")
    print(f"  retry_count  : {result['retry_count']}")
    print(f"  final_answer : {result['final_answer']}")
    print("\n  ✅  PASS")


# ── runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0
    for fn in [test_path_a, test_path_b, test_path_c]:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"\n  ❌  FAIL — {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{DIVIDER}")
    print(f"Results: {passed} passed, {failed} failed")
    print(DIVIDER)
    sys.exit(0 if failed == 0 else 1)
