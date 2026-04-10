"""Tests for deduplication logic: MinHash, vector store, and pipeline."""

import pickle

import numpy as np
import pytest

from src.dedup.minhash import MinHashIndex
from src.dedup.vector_store import VectorStore


class TestMinHashIndex:
    """Test MinHash LSH near-duplicate detection."""

    def test_near_duplicate_detected(self):
        """Two messages with high word overlap should match."""
        idx = MinHashIndex(threshold=0.7, num_perm=128)

        text_a = (
            "Google DeepMind announces new AI model Gemma 4 "
            "with improved performance on benchmarks today"
        )
        text_b = (
            "Google DeepMind announces new AI model Gemma 4 "
            "with improved performance on benchmarks and results"
        )

        mh_a = MinHashIndex.create_minhash(text_a)
        mh_b = MinHashIndex.create_minhash(text_b)

        idx.insert("1", mh_a)
        matches = idx.query(mh_b)

        assert "1" in matches, "Near-duplicate should be detected"

    def test_unrelated_messages_not_matched(self):
        """Two completely different messages should NOT match."""
        idx = MinHashIndex(threshold=0.7, num_perm=128)

        text_a = "Google announces new AI model Gemma 4 with improved performance"
        text_b = "Bitcoin price reaches new all time high amid regulatory concerns in Europe"

        mh_a = MinHashIndex.create_minhash(text_a)
        mh_b = MinHashIndex.create_minhash(text_b)

        idx.insert("1", mh_a)
        matches = idx.query(mh_b)

        assert "1" not in matches, "Unrelated messages should not match"

    def test_serialize_deserialize_roundtrip(self):
        """MinHash should survive pickle roundtrip."""
        text = "Test message for serialization"
        mh = MinHashIndex.create_minhash(text, num_perm=128)
        data = pickle.dumps(mh)
        mh2 = MinHashIndex.deserialize(data)

        assert mh.num_perm == mh2.num_perm
        assert list(mh.hashvalues) == list(mh2.hashvalues)

    def test_duplicate_key_insert_ignored(self):
        """Inserting the same key twice should not raise."""
        idx = MinHashIndex(threshold=0.7, num_perm=128)
        mh = MinHashIndex.create_minhash("test message")
        idx.insert("1", mh)
        idx.insert("1", mh)  # should not raise
        assert "1" in idx._keys

    def test_short_text_minhash(self):
        """MinHash should handle very short text (fewer words than n-gram size)."""
        mh = MinHashIndex.create_minhash("hi", num_perm=128)
        assert mh.num_perm == 128


class TestVectorStore:
    """Test ChromaDB vector store operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary vector store."""
        return VectorStore(persist_dir=str(tmp_path / "chromadb"))

    def test_upsert_and_query(self, store):
        """Basic upsert and similarity query."""
        emb_a = np.random.randn(384).astype(np.float32).tolist()
        store.upsert("1", emb_a)

        results = store.query_similar(emb_a, n_results=5, min_similarity=0.9)
        assert len(results) >= 1
        assert results[0].id == "1"
        assert results[0].similarity > 0.9

    def test_dissimilar_not_returned(self, store):
        """Dissimilar embeddings should not be returned above threshold."""
        emb_a = np.zeros(384, dtype=np.float32).tolist()
        emb_a[0] = 1.0
        emb_b = np.zeros(384, dtype=np.float32).tolist()
        emb_b[1] = 1.0

        store.upsert("1", emb_a)
        results = store.query_similar(emb_b, n_results=5, min_similarity=0.88)

        # Orthogonal vectors should have ~0 cosine similarity
        high_sim = [r for r in results if r.similarity >= 0.88]
        assert len(high_sim) == 0

    def test_exclude_ids(self, store):
        """Excluded IDs should not appear in results."""
        emb = np.random.randn(384).astype(np.float32).tolist()
        store.upsert("1", emb)

        results = store.query_similar(emb, n_results=5, min_similarity=0.5, exclude_ids=["1"])
        ids = [r.id for r in results]
        assert "1" not in ids

    def test_batch_upsert(self, store):
        """Batch upsert should work."""
        embeddings = [np.random.randn(384).astype(np.float32).tolist() for _ in range(3)]
        store.batch_upsert(["1", "2", "3"], embeddings)

        results = store.query_similar(embeddings[0], n_results=5, min_similarity=0.0)
        assert len(results) >= 1

    def test_embedding_from_bytes(self):
        """Convert bytes to float list and back."""
        original = np.random.randn(384).astype(np.float32)
        as_bytes = original.tobytes()
        recovered = VectorStore.embedding_from_bytes(as_bytes)
        np.testing.assert_array_almost_equal(original, np.array(recovered))


class TestPrimaryElection:
    """Test cluster primary election scoring."""

    def test_original_beats_forward(self):
        """Original post should score higher than a forward."""
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        from src.pipelines.deduplicate import _compute_primary_score

        original = MagicMock()
        original.forwarded_from_channel = None
        original.content = "Short content"
        original.raw_metadata = {"views": 100}
        original.published_at = datetime.now(UTC)

        forward = MagicMock()
        forward.forwarded_from_channel = "@some_channel"
        forward.content = "Short content"
        forward.raw_metadata = {"views": 100}
        forward.published_at = datetime.now(UTC)

        assert _compute_primary_score(original) > _compute_primary_score(forward)

    def test_longer_content_scores_higher(self):
        """Longer content should score higher."""
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        from src.pipelines.deduplicate import _compute_primary_score

        short = MagicMock()
        short.forwarded_from_channel = None
        short.content = "Short"
        short.raw_metadata = {"views": 0}
        short.published_at = datetime.now(UTC)

        long_msg = MagicMock()
        long_msg.forwarded_from_channel = None
        long_msg.content = "A" * 1500
        long_msg.raw_metadata = {"views": 0}
        long_msg.published_at = datetime.now(UTC)

        assert _compute_primary_score(long_msg) > _compute_primary_score(short)
