from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from src.db.models import Message


@dataclass
class SimilarResult:
    id: str
    similarity: float


class VectorStore:
    """ChromaDB wrapper for semantic deduplication via embedding cosine similarity."""

    def __init__(self, persist_dir: str = "data/chromadb"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="message_embeddings",
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, message_id: str, embedding: list[float], metadata: dict | None = None) -> None:
        """Upsert a single message embedding."""
        kwargs: dict = {"ids": [message_id], "embeddings": [embedding]}
        if metadata:
            kwargs["metadatas"] = [metadata]
        self.collection.upsert(**kwargs)

    def batch_upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Bulk upsert for efficiency."""
        kwargs: dict = {"ids": ids, "embeddings": embeddings}
        if metadatas:
            kwargs["metadatas"] = metadatas
        self.collection.upsert(**kwargs)

    def query_similar(
        self,
        embedding: list[float],
        n_results: int = 10,
        min_similarity: float = 0.80,
        exclude_ids: list[str] | None = None,
    ) -> list[SimilarResult]:
        """Find similar embeddings. Returns (id, similarity) pairs above threshold."""
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )

        similar: list[SimilarResult] = []
        if not results["ids"] or not results["ids"][0]:
            return similar

        ids = results["ids"][0]
        # ChromaDB returns distances; for cosine space: similarity = 1 - distance
        distances = results["distances"][0] if results["distances"] else []

        for doc_id, distance in zip(ids, distances):
            similarity = 1.0 - distance
            if exclude_ids and doc_id in exclude_ids:
                continue
            if similarity >= min_similarity:
                similar.append(SimilarResult(id=doc_id, similarity=similarity))

        return similar

    @staticmethod
    def embedding_from_bytes(data: bytes) -> list[float]:
        """Convert stored bytes back to float list for ChromaDB."""
        return np.frombuffer(data, dtype=np.float32).tolist()

    @staticmethod
    def embedding_from_message(message: Message) -> list[float] | None:
        """Extract embedding from a Message object."""
        if message.embedding_vector is None:
            return None
        return np.frombuffer(message.embedding_vector, dtype=np.float32).tolist()
