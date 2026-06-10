# Self-Healing RAG Pipeline

A stateful, cyclical Retrieval-Augmented Generation pipeline built with **LangGraph**. Unlike a linear chain, this system critiques its own answers and retries with a rewritten query when the answer isn't grounded in the source documents.

Runs **fully locally** via [Ollama](https://ollama.com) — no API keys or internet connection required after setup.

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
| `generate` | `llama3.2` answers using **only** the retrieved context |
| `critique` | A second `llama3.2` call checks every claim; emits `GROUNDED` or `NOT_GROUNDED` |
| `rewrite_query` | Rewrites the search query using original question + critic feedback; tracks retries |
| `final_answer` | Passes through grounded answer, or returns a graceful refusal on give-up |

**Max retries:** 2 (configurable via `MAX_RETRIES` in `graph.py`)

---

## Setup

### 1. Install Ollama
Download from **https://ollama.com/download** or run:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull the required models
```bash
ollama pull llama3.2          # LLM for generate, critique, rewrite_query
ollama pull nomic-embed-text  # Embeddings for ChromaDB
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your documents
Drop `.txt` or `.md` files into the `docs/` folder (any depth).

### 5. Ingest documents
```bash
python ingest.py
# or point to a different folder:
python ingest.py ./my-docs
```

This chunks the documents with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap), embeds them with `nomic-embed-text`, and persists the index to ChromaDB.

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

**Run mock tests (Ollama not required):**
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
| `DOCS_DIR` | `./docs` | Folder containing source documents |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Where ChromaDB stores its index |

No API key required — all inference runs locally via Ollama.

---

## Dependencies

- `langgraph` — stateful graph runtime
- `langchain-ollama` — `llama3.2` (LLM) and `nomic-embed-text` (embeddings)
- `langchain-chroma` — ChromaDB vector store integration
- `langchain-text-splitters` — `RecursiveCharacterTextSplitter`
- `langchain-community` — `TextLoader`
- `chromadb` — on-disk vector database
- `python-dotenv` — `.env` loading
