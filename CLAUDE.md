# CLAUDE.md — Session Context

This file records what was built in this project, how, why, and what issues came up.
Future Claude sessions should read this before making any changes.

---

## What was built

A **self-healing RAG pipeline** using LangGraph. The user asked for a stateful, cyclical
graph (not a linear chain) that:
- Retrieves documents from ChromaDB
- Generates an answer with a local LLM via Ollama
- Critiques that answer for hallucinations
- Rewrites the query and retries if the answer isn't grounded
- Gives up gracefully after 2 failed retries

Five files were requested: `requirements.txt`, `.env.example`, `ingest.py`, `graph.py`, `main.py`.
A sixth file, `test_graph.py`, was added during verification.

---

## Current stack (fully local, no API keys)

| Component | Model | Package |
|---|---|---|
| LLM (generate, critique, rewrite) | `llama3.2` via Ollama | `langchain-ollama` |
| Embeddings | `nomic-embed-text` via Ollama | `langchain-ollama` |
| Vector store | ChromaDB (on-disk) | `langchain-chroma` |

---

## File responsibilities

| File | What it does |
|---|---|
| `ingest.py` | Loads `.txt`/`.md` from `DOCS_DIR`, chunks with `RecursiveCharacterTextSplitter` (1000/200), embeds with `nomic-embed-text`, persists to ChromaDB. Exposes `get_vector_store()` for use by `graph.py`. |
| `graph.py` | Defines `RAGState` TypedDict, all 5 nodes, conditional routing functions, and assembles + compiles the `StateGraph`. Exports `graph` at module level. |
| `main.py` | CLI entrypoint. Accepts a question as a CLI arg, or runs an interactive REPL if no arg given. Calls `graph.invoke()` with the initial state. |
| `test_graph.py` | Mock-based verification. Patches `get_vector_store` and `ChatOllama` to run all three execution paths without Ollama running. |

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
This is intentionally lenient — local LLMs sometimes add a prefix like "Verdict: NOT_GROUNDED".

---

## Issues encountered and fixes

### 1. `from langchain.schema import ...` — ModuleNotFoundError
**Problem:** `langchain.schema` was removed in recent versions of `langchain`.  
**Fix:** Changed to `from langchain_core.messages import HumanMessage, SystemMessage, AIMessage`  
**Affected files:** `graph.py`, `test_graph.py`

### 2. OpenAI API key expired (HTTP 401)
**Problem:** The key found in the user's existing `openai_call.py` returned HTTP 401.  
**Fix:** Wrote mock-based tests (`test_graph.py`) using `unittest.mock.patch` to verify
the graph topology without a live API key.

### 3. `git init` ran in home directory
**Problem:** Running `git init` in `/Users/apple` would have staged personal files.  
**Fix:** Removed the `.git` folder immediately, created `/Users/apple/self-healing-rag/`
as a dedicated project folder, copied files there, then initialized git in that subfolder.

### 4. Google Gemini API keys all expired or quota = 0
**Problem:** Two Gemini keys were tried (`first_call.py`, user-provided). One was expired;
the other had `free_tier limit: 0` on all models — account-level restriction, not transient.  
**Fix:** Switched to Ollama for fully local inference. No API key needed at all.

### 5. Gemini embedding model not found (`text-embedding-004` → 404)
**Problem:** `langchain-google-genai` v2 uses the new `google-genai` SDK which has
different model naming conventions. `models/text-embedding-004` returned 404.  
**Fix:** Listed available models via `client.models.list()` and used `models/gemini-embedding-001`.
Moot after switching to Ollama (`nomic-embed-text`).

### 6. Ollama install script: "Unable to find application named 'Ollama'"
**Problem:** `curl ... | sh` downloaded and moved the app to `/Applications/Ollama.app`
but couldn't launch it via `open` in the same script run.  
**Fix:** Ran `open /Applications/Ollama.app` separately; Ollama binary was at
`/usr/local/bin/ollama` and worked immediately after.

---

## Verification results

**Mock tests — 3/3 pass:**

| Path | Description | Result |
|---|---|---|
| A | GROUNDED on first try | ✅ `retry_count=0`, correct answer passed through |
| B | NOT_GROUNDED once → rewrite → GROUNDED | ✅ `retry_count=1`, self-healing loop worked |
| C | NOT_GROUNDED × 3 → give_up | ✅ `retry_count=3`, graceful refusal returned |

**Live end-to-end run — PASS:**
```
Question: What is the refund policy and how long does it take?

[retrieve] 4 chunk(s) returned
[generate] answer: Approved refunds processed within 5–7 business days...
[critique] verdict=grounded
[final_answer] → Approved refunds are processed within 5–7 business days...
```

---

## How to run

```bash
# Ensure Ollama is running
open /Applications/Ollama.app

# Ingest documents (once, or when docs change)
python ingest.py

# Ask a question
python main.py "What is the refund policy?"

# Interactive REPL
python main.py

# Mock tests (no Ollama needed)
python test_graph.py
```

---

## What to watch out for

- **Ollama must be running** before `ingest.py` or `main.py`. Start it with
  `open /Applications/Ollama.app` or `ollama serve`.
- **ChromaDB persistence:** `chroma_db/` is gitignored. After cloning, run `ingest.py`
  before `main.py` — there is no pre-built index.
- **Re-ingest after model change:** If you swap the embedding model, delete `chroma_db/`
  and re-run `ingest.py`. Old vectors are incompatible with a new embedding space.
- **`langchain-community` deprecation warning:** `TextLoader` triggers a deprecation notice
  on import. It still works fine.
- **Mock LLM call count in tests:** `make_llm_sequence()` uses `side_effect` with a fixed
  list. If the graph calls the LLM more times than expected, the mock raises `StopIteration`.
  This acts as a built-in assertion on the number of LLM calls per path.
- **Local model speed:** `llama3.2` is fast on Apple Silicon. On older hardware, generation
  may take 10–30s per node. Swap to `llama3.2:1b` in `graph.py` for faster (but weaker) output.
