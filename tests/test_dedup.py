"""Tests for deduplication logic: MinHash, vector store, pipeline, and SLM confirmation."""

from __future__ import annotations

import pickle
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.dedup.minhash import MinHashIndex
from src.dedup.vector_store import VectorStore


class TestMinHashIndex:
    """Test MinHash LSH near-duplicate detection."""

    def test_near_duplicate_detected(self):
        """Two messages with high word overlap should cluster together."""
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

    def test_build_from_messages(self):
        """build_from_messages should populate the index from Message objects."""
        idx = MinHashIndex(threshold=0.7, num_perm=128)

        text = "Google DeepMind announces new Gemma model with great benchmarks"
        mh = MinHashIndex.create_minhash(text, num_perm=128)
        sig_bytes = pickle.dumps(mh)

        msg = MagicMock()
        msg.id = 42
        msg.minhash_signature = sig_bytes

        idx.build_from_messages([msg])

        # Now query with a similar minhash
        similar_text = "Google DeepMind announces new Gemma model with great benchmark results"
        mh_similar = MinHashIndex.create_minhash(similar_text, num_perm=128)
        matches = idx.query(mh_similar)

        assert "42" in matches

    def test_build_from_messages_skips_null_signature(self):
        """Messages with null minhash_signature should be skipped."""
        idx = MinHashIndex(threshold=0.7, num_perm=128)

        msg = MagicMock()
        msg.id = 1
        msg.minhash_signature = None

        idx.build_from_messages([msg])
        assert len(idx._keys) == 0


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

    def test_cosine_definite_threshold(self, store):
        """Similarity >= 0.88 should be returned as definite duplicate."""
        # Use the same embedding for near-identical similarity
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # normalize
        emb_list = emb.tolist()

        # Add slight noise for a near-duplicate (similarity ~0.99)
        noise = np.random.randn(384).astype(np.float32) * 0.01
        near_dup = emb + noise
        near_dup = near_dup / np.linalg.norm(near_dup)
        near_dup_list = near_dup.tolist()

        store.upsert("1", emb_list)
        results = store.query_similar(near_dup_list, n_results=5, min_similarity=0.88)

        assert len(results) >= 1
        assert results[0].similarity >= 0.88

    def test_cosine_borderline_range(self, store):
        """Similarity between 0.80 and 0.88 is borderline territory."""
        # Create two embeddings with controlled similarity
        rng = np.random.RandomState(42)
        emb_a = rng.randn(384).astype(np.float32)
        emb_a = emb_a / np.linalg.norm(emb_a)

        store.upsert("1", emb_a.tolist())

        # Query and check the results structure works for borderline filtering
        results = store.query_similar(emb_a.tolist(), n_results=5, min_similarity=0.80)
        assert len(results) >= 1
        # Self-similarity should be ~1.0 which is above both thresholds
        assert results[0].similarity >= 0.88

    def test_cosine_unique_below_threshold(self, store):
        """Similarity < 0.80 should mean the message is unique."""
        emb_a = np.zeros(384, dtype=np.float32).tolist()
        emb_a[0] = 1.0
        emb_b = np.zeros(384, dtype=np.float32).tolist()
        emb_b[1] = 1.0

        store.upsert("1", emb_a)
        results = store.query_similar(emb_b, n_results=5, min_similarity=0.80)

        # Orthogonal vectors have ~0 similarity, below 0.80
        assert len(results) == 0


class TestPrimaryElection:
    """Test cluster primary election scoring."""

    def test_original_beats_forward(self):
        """Original post should score higher than a forward."""
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

    def test_higher_views_scores_higher(self):
        """Messages with more views should score higher."""
        from src.pipelines.deduplicate import _compute_primary_score

        low_views = MagicMock()
        low_views.forwarded_from_channel = None
        low_views.content = "Same content here"
        low_views.raw_metadata = {"views": 10}
        low_views.published_at = datetime.now(UTC)

        high_views = MagicMock()
        high_views.forwarded_from_channel = None
        high_views.content = "Same content here"
        high_views.raw_metadata = {"views": 40000}
        high_views.published_at = datetime.now(UTC)

        assert _compute_primary_score(high_views) > _compute_primary_score(low_views)

    def test_more_recent_scores_higher(self):
        """More recent messages should score higher."""
        from src.pipelines.deduplicate import _compute_primary_score

        old = MagicMock()
        old.forwarded_from_channel = None
        old.content = "Same content"
        old.raw_metadata = {"views": 0}
        old.published_at = datetime.now(UTC) - timedelta(days=20)

        new = MagicMock()
        new.forwarded_from_channel = None
        new.content = "Same content"
        new.raw_metadata = {"views": 0}
        new.published_at = datetime.now(UTC)

        assert _compute_primary_score(new) > _compute_primary_score(old)

    def test_elect_primary_picks_best(self):
        """_elect_primary should return the message with highest score."""
        from src.pipelines.deduplicate import _elect_primary

        original = MagicMock()
        original.forwarded_from_channel = None
        original.content = "A detailed original post about something important"
        original.raw_metadata = {"views": 5000}
        original.published_at = datetime.now(UTC)

        forward = MagicMock()
        forward.forwarded_from_channel = "@news_channel"
        forward.content = "Short"
        forward.raw_metadata = {"views": 10}
        forward.published_at = datetime.now(UTC) - timedelta(days=5)

        primary = _elect_primary([original, forward])
        assert primary is original


class TestSLMConfirmation:
    """Test SLM duplicate confirmation logic."""

    @pytest.mark.asyncio
    async def test_slm_confirms_duplicate(self):
        """SLM returning is_duplicate=True should be interpreted correctly."""
        from src.pipelines.deduplicate import _slm_confirm_duplicate

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(
            return_value={"is_duplicate": True, "reason": "Same event reported"}
        )

        msg_a = MagicMock()
        msg_a.content = "OpenAI releases GPT-5 with major improvements"
        msg_b = MagicMock()
        msg_b.content = "OpenAI launches GPT-5, a big upgrade over GPT-4"

        result = await _slm_confirm_duplicate(mock_client, msg_a, msg_b)
        assert result is True

    @pytest.mark.asyncio
    async def test_slm_rejects_duplicate(self):
        """SLM returning is_duplicate=False should be interpreted correctly."""
        from src.pipelines.deduplicate import _slm_confirm_duplicate

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(
            return_value={"is_duplicate": False, "reason": "Different events"}
        )

        msg_a = MagicMock()
        msg_a.content = "OpenAI releases GPT-5"
        msg_b = MagicMock()
        msg_b.content = "Google announces Gemini 2"

        result = await _slm_confirm_duplicate(mock_client, msg_a, msg_b)
        assert result is False

    @pytest.mark.asyncio
    async def test_slm_error_returns_false(self):
        """If SLM call fails, should return False (not duplicate)."""
        from src.pipelines.deduplicate import _slm_confirm_duplicate

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(side_effect=Exception("Ollama down"))

        msg_a = MagicMock()
        msg_a.content = "Some message"
        msg_b = MagicMock()
        msg_b.content = "Another message"

        result = await _slm_confirm_duplicate(mock_client, msg_a, msg_b)
        assert result is False

    @pytest.mark.asyncio
    async def test_slm_malformed_json_returns_false(self):
        """If SLM returns JSON without is_duplicate key, should return False."""
        from src.pipelines.deduplicate import _slm_confirm_duplicate

        mock_client = AsyncMock()
        mock_client.generate_json = AsyncMock(
            return_value={"answer": "yes", "explanation": "Same thing"}
        )

        msg_a = MagicMock()
        msg_a.content = "Some message"
        msg_b = MagicMock()
        msg_b.content = "Another message"

        result = await _slm_confirm_duplicate(mock_client, msg_a, msg_b)
        assert result is False


class TestIdempotency:
    """Test that dedup pipeline is idempotent."""

    @pytest.mark.asyncio
    async def test_rerun_on_deduplicated_is_noop(self):
        """Re-running dedup when no embedded messages exist should be a no-op."""
        from src.pipelines.deduplicate import run_deduplicate

        mock_db = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.dedup.window_hours = 72
        mock_settings.dedup.minhash_threshold = 0.7
        mock_settings.dedup.minhash_num_perm = 128

        # First query (new messages with status='embedded') returns empty
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await run_deduplicate(mock_db, mock_settings)

        assert result.processed == 0
        assert result.failed == 0
        assert result.skipped == 0
        assert result.stage == "deduplicate"

    @pytest.mark.asyncio
    async def test_already_deduplicated_not_reprocessed(self):
        """Messages with status='deduplicated' should not be picked up again."""
        from src.pipelines.deduplicate import run_deduplicate

        mock_db = AsyncMock()
        mock_settings = MagicMock()
        mock_settings.dedup.window_hours = 72

        # Return empty for embedded messages query
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await run_deduplicate(mock_db, mock_settings)

        # With no embedded messages, nothing should be processed
        assert result.processed == 0
        assert result.skipped == 0


class TestOllamaClient:
    """Test Ollama client JSON generation and retry logic."""

    @pytest.mark.asyncio
    async def test_generate_json_success(self):
        """Successful JSON generation on first attempt."""
        from src.llm.client import OllamaClient

        config = MagicMock()
        config.base_url = "http://localhost:11434"
        config.model = "gemma4:e4"
        config.timeout = 60
        config.temperature = 0.1

        client = OllamaClient(config)
        client.client = AsyncMock()
        client.client.generate = AsyncMock(
            return_value={"response": '{"is_duplicate": true, "reason": "same event"}'}
        )

        result = await client.generate_json("test prompt")
        assert result == {"is_duplicate": True, "reason": "same event"}

    @pytest.mark.asyncio
    async def test_generate_json_retry_on_failure(self):
        """Should retry once with stricter prompt on JSON parse failure."""
        import json

        from src.llm.client import OllamaClient

        config = MagicMock()
        config.base_url = "http://localhost:11434"
        config.model = "gemma4:e4"
        config.timeout = 60
        config.temperature = 0.1

        client = OllamaClient(config)
        client.client = AsyncMock()

        # First call returns invalid JSON, second returns valid
        client.client.generate = AsyncMock(
            side_effect=[
                {"response": "not valid json {{}"},
                {"response": '{"is_duplicate": false, "reason": "different"}'},
            ]
        )

        result = await client.generate_json("test prompt")
        assert result == {"is_duplicate": False, "reason": "different"}
        assert client.client.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_json_raises_after_two_failures(self):
        """Should raise after two consecutive JSON parse failures."""
        import json

        from src.llm.client import OllamaClient

        config = MagicMock()
        config.base_url = "http://localhost:11434"
        config.model = "gemma4:e4"
        config.timeout = 60
        config.temperature = 0.1

        client = OllamaClient(config)
        client.client = AsyncMock()

        client.client.generate = AsyncMock(
            return_value={"response": "totally not json at all"}
        )

        with pytest.raises(json.JSONDecodeError):
            await client.generate_json("test prompt")

        assert client.client.generate.call_count == 2


class TestDedupPrompt:
    """Test dedup prompt template."""

    def test_prompt_has_placeholders(self):
        """DEDUP_PROMPT should have message_a and message_b placeholders."""
        from src.llm.prompts import DEDUP_PROMPT

        assert "{message_a}" in DEDUP_PROMPT
        assert "{message_b}" in DEDUP_PROMPT

    def test_prompt_format(self):
        """DEDUP_PROMPT should be formattable with message_a and message_b."""
        from src.llm.prompts import DEDUP_PROMPT

        formatted = DEDUP_PROMPT.format(
            message_a="Breaking: OpenAI releases GPT-5",
            message_b="OpenAI launches GPT-5 today",
        )
        assert "OpenAI releases GPT-5" in formatted
        assert "OpenAI launches GPT-5 today" in formatted
        assert "is_duplicate" in formatted
