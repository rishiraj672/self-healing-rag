"""
Ingest .txt / .md documents → chunk → embed → persist to ChromaDB.

Usage:
    python ingest.py                  # uses DOCS_DIR from .env
    python ingest.py ./my-docs-folder
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
COLLECTION_NAME = "rag_documents"


def get_vector_store() -> Chroma:
    """Return the persisted ChromaDB vector store (used by graph.py)."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def ingest_documents(docs_dir: str = DOCS_DIR) -> None:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    documents = []
    for pattern in ("**/*.txt", "**/*.md"):
        for file_path in sorted(docs_path.glob(pattern)):
            print(f"[ingest] Loading: {file_path}")
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    if not documents:
        print("[ingest] No .txt or .md files found — nothing to ingest.")
        return

    print(f"[ingest] Loaded {len(documents)} document(s)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] Split into {len(chunks)} chunk(s)")

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(
        f"[ingest] Persisted {len(chunks)} chunk(s) to ChromaDB "
        f"at '{CHROMA_PERSIST_DIR}' (collection: {COLLECTION_NAME})"
    )


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DOCS_DIR
    ingest_documents(target)
