"""
CLI entrypoint for the self-healing RAG pipeline.

Usage:
    # Single question
    python main.py "What is the refund policy?"

    # Interactive REPL (no args)
    python main.py
"""

import sys
from dotenv import load_dotenv
from graph import graph

load_dotenv()

DIVIDER = "=" * 64


def ask(question: str) -> str:
    print(f"\n{DIVIDER}")
    print(f"Question: {question}")
    print(DIVIDER)

    initial_state = {
        "question": question,
        "query": question,
        "retrieved_docs": [],
        "answer": "",
        "critique": "",
        "verdict": "not_grounded",
        "retry_count": 0,
        "final_answer": "",
    }

    result = graph.invoke(initial_state)
    return result["final_answer"]


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer = ask(question)
        print(f"\n{DIVIDER}")
        print(f"Answer:\n{answer}")
        print(DIVIDER)
        return

    # Interactive REPL
    print("Self-Healing RAG Pipeline  —  interactive mode")
    print("Type  'quit'  or  'exit'  to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        answer = ask(question)
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
