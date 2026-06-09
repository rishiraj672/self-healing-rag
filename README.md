# Self-Healing RAG Pipeline

A stateful, cyclical Retrieval-Augmented Generation pipeline built with **LangGraph**. Unlike a linear chain, this system critiques its own answers and retries with a rewritten query when the answer isn't grounded in the source documents.

## How it works

```
START → retrieve → generate → critique
                                  │
                   ┌──────────────┴──────────────┐
               grounded                      not_grounded
                   │                              │
           final_answer → END             rewrite_query
                                                  │
                                     ┌────────────┴────────────┐
                                retry ≤ 2                  retry > 2
                                     │                          │
                                  retrieve               give_up →
                                                      final_answer → END
```

| Node | Role |
|---|---|
| `retrieve` | Queries ChromaDB for top-4 chunks matching the current query |
| `generate` | GPT-4o-mini answers using **only** the retrieved context |
| `critique` | A second GPT-4o-mini call checks every claim; emits `GROUNDED` or `NOT_GROUNDED` |
| `rewrite_query` | Rewrites the search query using original question + critic feedback; tracks retries |
| `final_answer` | Passes through grounded answer, or returns a graceful refusal on give-up |

**Max retries:** 2 (configurable via `MAX_RETRIES` in `graph.py`)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Add your documents
Drop `.txt` or `.md` files into the `docs/` folder (any depth).

### 4. Ingest documents
```bash
python ingest.py
# or point to a different folder:
python ingest.py ./my-docs
```

This chunks the documents with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap), embeds them with `text-embedding-3-small`, and persists the index to ChromaDB.

---

## Usage

**Single question:**
```bash
python main.py "What is the refund policy?"
```

**Interactive REPL:**
```bash
python main.py
```

**Run mock tests (no API key needed):**
```bash
python test_graph.py
```

---

## Project structure

```
.
├── ingest.py          # Document loading, chunking, embedding → ChromaDB
├── graph.py           # LangGraph nodes, edges, and routing logic
├── main.py            # CLI entrypoint (single question or interactive REPL)
├── test_graph.py      # Mock-based tests for all three graph paths
├── requirements.txt
├── .env.example
└── docs/              # Place your .txt / .md source documents here
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key |
| `DOCS_DIR` | `./docs` | Folder containing source documents |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores its index |

---

## Dependencies

- `langgraph` — stateful graph runtime
- `langchain-openai` — GPT-4o-mini and text-embedding-3-small
- `langchain-chroma` — ChromaDB vector store integration
- `langchain-text-splitters` — `RecursiveCharacterTextSplitter`
- `langchain-community` — `TextLoader`
- `chromadb` — on-disk vector database
- `python-dotenv` — `.env` loading
- `tiktoken` — token counting for OpenAI models
