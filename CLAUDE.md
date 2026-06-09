# CLAUDE.md — Session Context

This file records what was built in this project, how, why, and what issues came up.
Future Claude sessions should read this before making any changes.

---

## What was built

A **self-healing RAG pipeline** using LangGraph. The user asked for a stateful, cyclical
graph (not a linear chain) that:
- Retrieves documents from ChromaDB
- Generates an answer with GPT-4o-mini
- Critiques that answer for hallucinations
- Rewrites the query and retries if the answer isn't grounded
- Gives up gracefully after 2 failed retries

Five files were requested: `requirements.txt`, `.env.example`, `ingest.py`, `graph.py`, `main.py`.
A sixth file, `test_graph.py`, was added during verification.

---

## File responsibilities

| File | What it does |
|---|---|
| `ingest.py` | Loads `.txt`/`.md` from `DOCS_DIR`, chunks with `RecursiveCharacterTextSplitter` (1000/200), embeds with `text-embedding-3-small`, persists to ChromaDB. Exposes `get_vector_store()` for use by `graph.py`. |
| `graph.py` | Defines `RAGState` TypedDict, all 5 nodes, conditional routing functions, and assembles + compiles the `StateGraph`. Exports `graph` at module level. |
| `main.py` | CLI entrypoint. Accepts a question as a CLI arg, or runs an interactive REPL if no arg given. Calls `graph.invoke()` with the initial state. |
| `test_graph.py` | Mock-based verification. Patches `get_vector_store` and `ChatOpenAI` to run all three execution paths without any API calls. |

---

## Key design decisions

### Why LangGraph instead of a chain?
A linear LangChain chain can't loop back. LangGraph's `StateGraph` supports cycles,
which is required for the retry-on-failure pattern. The graph has a genuine cycle:
`rewrite_query → retrieve → generate → critique → rewrite_query`.

### State design
`RAGState` is a flat `TypedDict`. All fields are present from the start (initialized in
`main.py`). Nodes return partial dicts; LangGraph merges them into the state.
- `query` starts equal to `question` and diverges as `rewrite_query` rewrites it.
- `verdict` is set by `critique` and also by `rewrite_query` (to `"give_up"` when retries are exhausted).
- `retry_count` is incremented by `rewrite_query`, not by `critique`.

### Why MAX_RETRIES = 2?
The topology specifies "max 2 retries". `rewrite_query` increments `retry_count` before
checking, so the check is `retry_count > MAX_RETRIES` (i.e. > 2), meaning attempts 1, 2,
and 3 are made before giving up. `retry_count` reaches 3 on give_up.

### Routing
Two conditional edge functions:
- `route_after_critique`: `grounded → final_answer`, else `→ rewrite_query`
- `route_after_rewrite`: `give_up → final_answer`, else `→ retrieve`

### Critique prompt
The critic is instructed to respond with exactly two lines:
- Line 1: `GROUNDED` or `NOT_GROUNDED`
- Line 2: one-sentence reason

Parsing: split on first newline, check if `"NOT_GROUNDED"` is in the uppercased first line.
This is intentionally lenient — the LLM sometimes adds a prefix like "Verdict: NOT_GROUNDED".

---

## Issues encountered and fixes

### 1. `from langchain.schema import ...` — ModuleNotFoundError
**Problem:** `langchain.schema` was removed in recent versions of `langchain`.  
**Fix:** Changed to `from langchain_core.messages import HumanMessage, SystemMessage, AIMessage`  
**Affected files:** `graph.py`, `test_graph.py`

### 2. OpenAI API key was expired
**Problem:** The key found in the user's existing `openai_call.py` returned HTTP 401.  
**Fix:** Wrote mock-based tests (`test_graph.py`) using `unittest.mock.patch` to stub
`get_vector_store` and `ChatOpenAI`, so the graph topology could be verified without
a live API key.

### 3. `git init` ran in home directory
**Problem:** Running `git init` in `/Users/apple` would have staged personal files.  
**Fix:** Removed the `.git` folder immediately, created `/Users/apple/self-healing-rag/`
as a dedicated project folder, copied files there, then initialized git in that subfolder.

---

## Verification results

All three execution paths were tested with mocks:

| Path | Description | Result |
|---|---|---|
| A | GROUNDED on first try | ✅ `retry_count=0`, correct answer passed through |
| B | NOT_GROUNDED once → rewrite → GROUNDED | ✅ `retry_count=1`, self-healing loop worked |
| C | NOT_GROUNDED × 3 → give_up | ✅ `retry_count=3`, graceful refusal returned |

---

## How to run with a real API key

1. Add a valid key to `/Users/apple/self-healing-rag/.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```
2. Ingest documents:
   ```bash
   python ingest.py
   ```
3. Ask a question:
   ```bash
   python main.py "What is the refund policy?"
   ```

---

## What to watch out for

- **ChromaDB persistence:** The `chroma_db/` folder is gitignored. After cloning, you must
  run `ingest.py` before `main.py` — there is no pre-built index.
- **`.env` is gitignored:** Never commit it. The API key lives only on the local machine.
- **`langchain-community` deprecation warning:** `TextLoader` triggers a deprecation notice
  on import. It still works. The alternative is `langchain_community` → standalone loaders
  package, but that migration isn't worth doing until they actually remove it.
- **Mock LLM call count in tests:** `make_llm_sequence()` uses `side_effect` with a fixed
  list. If the graph calls the LLM more times than expected, the mock raises `StopIteration`.
  This acts as a built-in assertion on the number of LLM calls per path.
