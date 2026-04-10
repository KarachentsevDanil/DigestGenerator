"""Tests for knowledge graph: entity tracking, novelty computation."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.knowledge.tracker import _cosine_similarity, compute_novelty


class TestNoveltyComputation:
    """Test novelty score calculation."""

    @pytest.mark.asyncio
    async def test_all_new_entities(self):
        """Message with all unknown entities -> novelty = 1.0."""
        db = MagicMock()
        # Mock the execute to return None (no known entities)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute = MagicMock(return_value=mock_result)

        # Make db.execute an async function
        async def mock_execute(*args, **kwargs):
            return mock_result

        db.execute = mock_execute

        msg = MagicMock()
        msg.entities_json = [
            {"name": "Gemma 4", "type": "model"},
            {"name": "OpenAI", "type": "company"},
            {"name": "GPT-5", "type": "model"},
        ]

        novelty = await compute_novelty(db, user_id=1, message=msg)
        assert novelty == 1.0

    @pytest.mark.asyncio
    async def test_all_known_entities(self):
        """Message with all recently-seen entities -> novelty = 0.0."""
        known = MagicMock()
        known.last_seen_at = datetime.now(UTC)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = known

        async def mock_execute(*args, **kwargs):
            return mock_result

        db = MagicMock()
        db.execute = mock_execute

        msg = MagicMock()
        msg.entities_json = [
            {"name": "Gemma 4", "type": "model"},
            {"name": "OpenAI", "type": "company"},
        ]

        novelty = await compute_novelty(db, user_id=1, message=msg)
        assert novelty == 0.0

    @pytest.mark.asyncio
    async def test_no_entities_neutral(self):
        """Message with no entities -> novelty = 0.5 (neutral)."""
        db = MagicMock()

        msg = MagicMock()
        msg.entities_json = None

        novelty = await compute_novelty(db, user_id=1, message=msg)
        assert novelty == 0.5

    @pytest.mark.asyncio
    async def test_empty_entities_neutral(self):
        """Message with empty entities list -> novelty = 0.5."""
        db = MagicMock()

        msg = MagicMock()
        msg.entities_json = []

        novelty = await compute_novelty(db, user_id=1, message=msg)
        assert novelty == 0.5

    @pytest.mark.asyncio
    async def test_stale_entity_partial_novelty(self):
        """Entity not seen in >30 days -> counts as 0.3 novelty."""
        stale = MagicMock()
        stale.last_seen_at = datetime.now(UTC) - timedelta(days=60)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = stale

        async def mock_execute(*args, **kwargs):
            return mock_result

        db = MagicMock()
        db.execute = mock_execute

        msg = MagicMock()
        msg.entities_json = [
            {"name": "Old Entity", "type": "technology"},
        ]

        novelty = await compute_novelty(db, user_id=1, message=msg)
        assert abs(novelty - 0.3) < 0.01

    @pytest.mark.asyncio
    async def test_mixed_entities(self):
        """Mix of new and known entities."""
        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                # First entity: unknown
                mock_result.scalar_one_or_none.return_value = None
            else:
                # Second entity: known
                known = MagicMock()
                known.last_seen_at = datetime.now(UTC)
                mock_result.scalar_one_or_none.return_value = known
            return mock_result

        db = MagicMock()
        db.execute = mock_execute

        msg = MagicMock()
        msg.entities_json = [
            {"name": "New Entity", "type": "model"},
            {"name": "Known Entity", "type": "company"},
        ]

        novelty = await compute_novelty(db, user_id=1, message=msg)
        # 1 new / 2 total = 0.5
        assert abs(novelty - 0.5) < 0.01


class TestCosineSimilarity:
    """Test embedding cosine similarity."""

    def test_identical_vectors(self):
        import numpy as np

        vec = np.random.randn(384).astype(np.float32)
        sim = _cosine_similarity(vec.tobytes(), vec.tobytes())
        assert abs(sim - 1.0) < 0.001

    def test_orthogonal_vectors(self):
        import numpy as np

        vec_a = np.zeros(384, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b = np.zeros(384, dtype=np.float32)
        vec_b[1] = 1.0
        sim = _cosine_similarity(vec_a.tobytes(), vec_b.tobytes())
        assert abs(sim) < 0.001

    def test_opposite_vectors(self):
        import numpy as np

        vec_a = np.ones(384, dtype=np.float32)
        vec_b = -np.ones(384, dtype=np.float32)
        sim = _cosine_similarity(vec_a.tobytes(), vec_b.tobytes())
        assert sim < -0.99
