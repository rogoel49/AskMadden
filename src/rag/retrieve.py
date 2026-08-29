"""Query the ChromaDB collection built by embed.py."""
from __future__ import annotations

from pathlib import Path

import chromadb

from src.rag.embed import CHROMA_DIR, COLLECTION_NAME


def query(question: str, n_results: int = 5, persist_dir: Path = CHROMA_DIR) -> list[dict]:
    """Return the n_results chunks most relevant to question, each as
    {"id", "text", "metadata", "distance"} ordered by relevance."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(query_texts=[question], n_results=n_results)
    return [
        {"id": id_, "text": text, "metadata": metadata, "distance": distance}
        for id_, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
