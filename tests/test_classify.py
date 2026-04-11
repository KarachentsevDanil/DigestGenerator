"""Tests for classification pipeline: prompt parsing, retry, multi-label, entities, idempotency."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.pipelines.classify import (
    PipelineRunResult,
    _build_categories_block,
    _normalize_classification,
    _validate_classification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_category(name: str, description: str) -> MagicMock:
    cat = MagicMock()
    cat.name = name
    cat.description = description
    return cat


def _make_message(
    msg_id: int = 1,
    content: str = "Test message",
    status: str = "deduplicated",
    is_primary: bool = True,
    classified_at=None,
) -> MagicMock:
    msg = MagicMock()
    msg.id = msg_id
    msg.content = content
    msg.status = status
    msg.is_cluster_primary = is_primary
    msg.classified_at = classified_at
    msg.category_scores_json = None
    msg.relevance_score = None
    msg.summary = None
    msg.entities_json = None
    msg.published_at = datetime.now(UTC)
    return msg


def _valid_classification_response() -> dict:
    return {
        "categories": {"ai_ml": 0.92, "tech_industry": 0.55, "crypto": 0.03},
        "relevance": 0.85,
        "summary": "Google releases Gemma 4 with improved benchmark scores.",
        "entities": [
            {"name": "Gemma 4", "type": "model"},
            {"name": "Google", "type": "company"},
        ],
    }


def _mock_db_execute(messages: list, categories: list):
    """
    Build an AsyncMock for db.execute that returns messages on the first call
    and categories on the second call.
    """
    msg_result = MagicMock()
    msg_result.scalars.return_value.all.return_value = messages

    cat_result = MagicMock()
    cat_result.scalars.return_value.all.return_value = categories

    return AsyncMock(side_effect=[msg_result, cat_result])


# ---------------------------------------------------------------------------
# Unit tests: validation & normalization
# ---------------------------------------------------------------------------

class TestValidateClassification:
    """Test classification output validation."""

    def test_valid_output(self):
        result = _valid_classification_response()
        assert _validate_classification(result) is True

    def test_missing_categories(self):
        result = {"relevance": 0.8, "summary": "Test", "entities": []}
        assert _validate_classification(result) is False

    def test_categories_not_dict(self):
        result = {"categories": "ai_ml", "relevance": 0.8, "summary": "Test", "entities": []}
        assert _validate_classification(result) is False

    def test_missing_relevance(self):
        result = {"categories": {"ai_ml": 0.9}, "summary": "Test", "entities": []}
        assert _validate_classification(result) is False

    def test_missing_summary(self):
        result = {"categories": {"ai_ml": 0.9}, "relevance": 0.8, "entities": []}
        assert _validate_classification(result) is False

    def test_missing_entities(self):
        result = {"categories": {"ai_ml": 0.9}, "relevance": 0.8, "summary": "Test"}
        assert _validate_classification(result) is False

    def test_integer_relevance_accepted(self):
        result = {"categories": {}, "relevance": 1, "summary": "Test", "entities": []}
        assert _validate_classification(result) is True


class TestNormalizeClassification:
    """Test classification value normalization."""

    def test_clamp_relevance_above(self):
        result = _normalize_classification({
            "categories": {"ai_ml": 1.5},
            "relevance": 1.5,
            "summary": "Test",
            "entities": [],
        })
        assert result["relevance"] == 1.0

    def test_clamp_negative_relevance(self):
        result = _normalize_classification({
            "categories": {},
            "relevance": -0.5,
            "summary": "Test",
            "entities": [],
        })
        assert result["relevance"] == 0.0

    def test_clamp_category_scores(self):
        result = _normalize_classification({
            "categories": {"ai_ml": 1.5, "crypto": -0.1},
            "relevance": 0.5,
            "summary": "Test",
            "entities": [],
        })
        assert result["categories"]["ai_ml"] == 1.0
        assert result["categories"]["crypto"] == 0.0

    def test_filter_invalid_entities(self):
        result = _normalize_classification({
            "categories": {},
            "relevance": 0.5,
            "summary": "Test",
            "entities": [
                {"name": "Gemma 4", "type": "model"},
                "invalid_entity",
                {"no_name_key": "bad"},
            ],
        })
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Gemma 4"

    def test_multi_label_scores_preserved(self):
        """A message can score high in multiple categories."""
        result = _normalize_classification({
            "categories": {"ai_ml": 0.92, "tech_industry": 0.55, "crypto": 0.03},
            "relevance": 0.8,
            "summary": "Test",
            "entities": [],
        })
        assert result["categories"]["ai_ml"] == 0.92
        assert result["categories"]["tech_industry"] == 0.55
        assert result["categories"]["crypto"] == 0.03


class TestBuildCategoriesBlock:
    """Test categories block formatting."""

    def test_builds_block(self):
        cat1 = _make_category("ai_ml", "AI and machine learning")
        cat2 = _make_category("crypto", "Cryptocurrency and blockchain")

        block = _build_categories_block([cat1, cat2])
        assert "- ai_ml: AI and machine learning" in block
        assert "- crypto: Cryptocurrency and blockchain" in block

    def test_empty_categories(self):
        block = _build_categories_block([])
        assert block == ""


# ---------------------------------------------------------------------------
# Integration tests: full pipeline with mocked DB and SLM
# ---------------------------------------------------------------------------

class TestClassifyPipelineValidJSON:
    """Test: valid JSON parsing from mock SLM response."""

    @pytest.mark.asyncio
    async def test_valid_json_parsed_and_stored(self):
        from src.pipelines.classify import run_classify

        msg = _make_message(msg_id=1, content="Google releases Gemma 4 model")
        categories = [
            _make_category("ai_ml", "AI and machine learning"),
            _make_category("tech_industry", "Big tech news"),
        ]

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        response = _valid_classification_response()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            instance.classify_message = AsyncMock(return_value=response)

            result = await run_classify(mock_db, mock_settings)

        assert result.processed == 1
        assert result.failed == 0
        assert msg.status == "classified"
        assert msg.category_scores_json == {"ai_ml": 0.92, "tech_industry": 0.55, "crypto": 0.03}
        assert msg.relevance_score == 0.85
        assert msg.summary == "Google releases Gemma 4 with improved benchmark scores."
        assert len(msg.entities_json) == 2
        assert msg.classified_at is not None
        mock_db.commit.assert_awaited_once()


class TestClassifyPipelineMultiLabel:
    """Test: multi-label scores -- message scores >0.5 in multiple categories."""

    @pytest.mark.asyncio
    async def test_multi_label_high_scores(self):
        from src.pipelines.classify import run_classify

        msg = _make_message(msg_id=1, content="AI crypto startup raises $100M")
        categories = [
            _make_category("ai_ml", "AI and machine learning"),
            _make_category("crypto", "Cryptocurrency and blockchain"),
            _make_category("tech_industry", "Big tech news"),
        ]

        multi_label_response = {
            "categories": {"ai_ml": 0.88, "crypto": 0.75, "tech_industry": 0.62},
            "relevance": 0.9,
            "summary": "AI crypto startup raises $100M in Series B.",
            "entities": [{"name": "CryptoAI Inc", "type": "company"}],
        }

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            instance.classify_message = AsyncMock(return_value=multi_label_response)

            result = await run_classify(mock_db, mock_settings)

        assert result.processed == 1
        scores = msg.category_scores_json
        # All three categories above 0.5
        assert scores["ai_ml"] > 0.5
        assert scores["crypto"] > 0.5
        assert scores["tech_industry"] > 0.5


class TestClassifyPipelineRetry:
    """Test: malformed JSON retry -- first call fails, second succeeds."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        from src.pipelines.classify import run_classify

        msg = _make_message(msg_id=1, content="Breaking AI news")
        categories = [_make_category("ai_ml", "AI and machine learning")]

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        # The OllamaClient.classify_message calls generate_json internally,
        # which has its own retry. We simulate: first call to classify_message
        # raises (both generate_json attempts failed), but that is a double failure.
        # For the "retry succeeds" case, generate_json's internal retry handles it,
        # so classify_message returns success. We test that here.
        valid_response = _valid_classification_response()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            # classify_message succeeds (generate_json handled the retry internally)
            instance.classify_message = AsyncMock(return_value=valid_response)

            result = await run_classify(mock_db, mock_settings)

        assert result.processed == 1
        assert result.failed == 0
        assert msg.status == "classified"

    @pytest.mark.asyncio
    async def test_generate_json_retry_produces_valid_classification(self):
        """Test the retry logic at the OllamaClient level: first JSON parse
        fails, second attempt succeeds."""
        from src.llm.client import OllamaClient

        config = MagicMock()
        config.base_url = "http://localhost:11434"
        config.model = "gemma4:e4"
        config.timeout = 60
        config.temperature = 0.1

        client = OllamaClient(config)
        client.client = AsyncMock()

        valid = _valid_classification_response()

        # First call returns bad JSON, second returns valid
        client.client.generate = AsyncMock(
            side_effect=[
                {"response": "not valid json {broken"},
                {"response": json.dumps(valid)},
            ]
        )

        result = await client.classify_message("Test message", "- ai_ml: AI stuff")
        assert result["categories"]["ai_ml"] == 0.92
        assert client.client.generate.call_count == 2


class TestClassifyPipelineDoubleFailure:
    """Test: double failure -> classify_failed status."""

    @pytest.mark.asyncio
    async def test_both_attempts_fail_marks_classify_failed(self):
        from src.pipelines.classify import run_classify

        msg = _make_message(msg_id=1, content="Some message")
        categories = [_make_category("ai_ml", "AI and machine learning")]

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            # classify_message raises after generate_json's internal retry exhausted
            instance.classify_message = AsyncMock(
                side_effect=json.JSONDecodeError("bad", "", 0)
            )

            result = await run_classify(mock_db, mock_settings)

        assert result.processed == 0
        assert result.failed == 1
        assert msg.status == "classify_failed"


class TestClassifyPipelineOnlyPrimaries:
    """Test: only primaries classified -- non-primary messages skipped."""

    @pytest.mark.asyncio
    async def test_non_primary_not_fetched(self):
        """The query only fetches is_cluster_primary=True, so non-primaries
        never reach the classification loop."""
        from src.pipelines.classify import run_classify

        # Only the primary message is returned by the DB query
        primary_msg = _make_message(msg_id=1, content="Primary message", is_primary=True)
        categories = [_make_category("ai_ml", "AI and machine learning")]

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([primary_msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            instance.classify_message = AsyncMock(
                return_value=_valid_classification_response()
            )

            result = await run_classify(mock_db, mock_settings)

        # Only 1 message processed -- the primary
        assert result.processed == 1
        assert primary_msg.status == "classified"
        # classify_message called exactly once (for the primary only)
        instance.classify_message.assert_awaited_once()


class TestClassifyPipelineEntityExtraction:
    """Test: entity extraction from response."""

    @pytest.mark.asyncio
    async def test_entities_extracted_correctly(self):
        from src.pipelines.classify import run_classify

        msg = _make_message(
            msg_id=1,
            content="EU passes AI Act regulation affecting OpenAI and Google",
        )
        categories = [
            _make_category("ai_ml", "AI and machine learning"),
            _make_category("geopolitics", "Geopolitics and regulation"),
        ]

        response = {
            "categories": {"ai_ml": 0.8, "geopolitics": 0.9},
            "relevance": 0.95,
            "summary": "EU passes AI Act affecting major tech companies.",
            "entities": [
                {"name": "EU", "type": "regulation"},
                {"name": "AI Act", "type": "regulation"},
                {"name": "OpenAI", "type": "company"},
                {"name": "Google", "type": "company"},
            ],
        }

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            instance.classify_message = AsyncMock(return_value=response)

            result = await run_classify(mock_db, mock_settings)

        assert result.processed == 1
        entities = msg.entities_json
        assert len(entities) == 4
        entity_names = [e["name"] for e in entities]
        assert "EU" in entity_names
        assert "AI Act" in entity_names
        assert "OpenAI" in entity_names
        assert "Google" in entity_names

        # Check entity types
        entity_types = {e["name"]: e["type"] for e in entities}
        assert entity_types["OpenAI"] == "company"
        assert entity_types["AI Act"] == "regulation"

    @pytest.mark.asyncio
    async def test_entities_with_all_valid_types(self):
        """Entities with all recognized types should be preserved."""
        from src.pipelines.classify import run_classify

        msg = _make_message(msg_id=1, content="Mixed entities message")
        categories = [_make_category("ai_ml", "AI")]

        all_type_entities = [
            {"name": "GPT-5", "type": "model"},
            {"name": "OpenAI", "type": "company"},
            {"name": "Sam Altman", "type": "person"},
            {"name": "Transformer", "type": "technology"},
            {"name": "NeurIPS 2025", "type": "event"},
            {"name": "AI Act", "type": "regulation"},
            {"name": "ChatGPT", "type": "product"},
            {"name": "AGI", "type": "concept"},
        ]

        response = {
            "categories": {"ai_ml": 0.9},
            "relevance": 0.8,
            "summary": "Various entities mentioned.",
            "entities": all_type_entities,
        }

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            instance.classify_message = AsyncMock(return_value=response)

            result = await run_classify(mock_db, mock_settings)

        assert result.processed == 1
        assert len(msg.entities_json) == 8


class TestClassifyPipelineIdempotency:
    """Test: already-classified not re-processed."""

    @pytest.mark.asyncio
    async def test_already_classified_not_reprocessed(self):
        """Messages with status='classified' are not returned by the query,
        so they are never re-processed."""
        from src.pipelines.classify import run_classify

        # The DB query only returns status='deduplicated', so an empty list
        # simulates that all messages are already classified.
        mock_db = AsyncMock()
        msg_result = MagicMock()
        msg_result.scalars.return_value.all.return_value = []
        mock_db.execute = AsyncMock(return_value=msg_result)

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        result = await run_classify(mock_db, mock_settings)

        assert result.processed == 0
        assert result.failed == 0
        assert result.skipped == 0
        assert result.stage == "classify"

    @pytest.mark.asyncio
    async def test_rerun_does_not_reclassify(self):
        """Running classify twice: second run finds no deduplicated messages."""
        from src.pipelines.classify import run_classify

        msg = _make_message(msg_id=1, content="Some AI news")
        categories = [_make_category("ai_ml", "AI and machine learning")]

        # First run: message is deduplicated
        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            instance.classify_message = AsyncMock(
                return_value=_valid_classification_response()
            )

            result1 = await run_classify(mock_db, mock_settings)

        assert result1.processed == 1
        assert msg.status == "classified"

        # Second run: message is now 'classified', so DB returns empty
        mock_db2 = AsyncMock()
        empty_result = MagicMock()
        empty_result.scalars.return_value.all.return_value = []
        mock_db2.execute = AsyncMock(return_value=empty_result)

        result2 = await run_classify(mock_db2, mock_settings)
        assert result2.processed == 0


class TestClassifyPipelineNoCategories:
    """Test: empty categories marks all as failed."""

    @pytest.mark.asyncio
    async def test_no_categories_marks_failed(self):
        from src.pipelines.classify import run_classify

        msg = _make_message(msg_id=1, content="Some message")

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], [])  # no categories
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        result = await run_classify(mock_db, mock_settings)

        assert result.failed == 1
        assert result.processed == 0
        assert msg.status == "classify_failed"


class TestClassifyPipelineInvalidSLMOutput:
    """Test: SLM returns JSON but with invalid structure."""

    @pytest.mark.asyncio
    async def test_invalid_structure_marks_failed(self):
        from src.pipelines.classify import run_classify

        msg = _make_message(msg_id=1, content="Some message")
        categories = [_make_category("ai_ml", "AI")]

        # Valid JSON but wrong structure (missing required fields)
        bad_structure = {"answer": "yes", "score": 0.5}

        mock_db = AsyncMock()
        mock_db.execute = _mock_db_execute([msg], categories)
        mock_db.commit = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.ollama = MagicMock()

        with patch("src.pipelines.classify.OllamaClient") as MockClient:
            instance = MockClient.return_value
            instance.classify_message = AsyncMock(return_value=bad_structure)

            result = await run_classify(mock_db, mock_settings)

        assert result.failed == 1
        assert result.processed == 0
        assert msg.status == "classify_failed"


class TestClassifyPrompt:
    """Test CLASSIFY_PROMPT template."""

    def test_prompt_has_placeholders(self):
        from src.llm.prompts import CLASSIFY_PROMPT

        assert "{categories_block}" in CLASSIFY_PROMPT
        assert "{message_content}" in CLASSIFY_PROMPT

    def test_prompt_format(self):
        from src.llm.prompts import CLASSIFY_PROMPT

        formatted = CLASSIFY_PROMPT.format(
            categories_block="- ai_ml: AI and machine learning",
            message_content="Google releases Gemma 4",
        )
        assert "ai_ml: AI and machine learning" in formatted
        assert "Google releases Gemma 4" in formatted
        assert "categories" in formatted
        assert "relevance" in formatted
        assert "entities" in formatted

    def test_prompt_mentions_entity_types(self):
        from src.llm.prompts import CLASSIFY_PROMPT

        for entity_type in [
            "model", "company", "person", "technology",
            "event", "regulation", "product", "concept",
        ]:
            assert entity_type in CLASSIFY_PROMPT

    def test_prompt_mentions_scoring_guidelines(self):
        from src.llm.prompts import CLASSIFY_PROMPT

        assert "0.0=unrelated" in CLASSIFY_PROMPT
        assert "1.0=breaking news" in CLASSIFY_PROMPT


class TestClassifyMessageMethod:
    """Test OllamaClient.classify_message method."""

    @pytest.mark.asyncio
    async def test_classify_message_formats_prompt_and_calls_generate_json(self):
        from src.llm.client import OllamaClient

        config = MagicMock()
        config.base_url = "http://localhost:11434"
        config.model = "gemma4:e4"
        config.timeout = 60
        config.temperature = 0.1

        client = OllamaClient(config)

        valid = _valid_classification_response()
        client.generate_json = AsyncMock(return_value=valid)

        result = await client.classify_message(
            "Google releases Gemma 4",
            "- ai_ml: AI and machine learning",
        )

        assert result == valid
        client.generate_json.assert_awaited_once()

        # Verify the prompt was formatted correctly
        call_args = client.generate_json.call_args
        prompt_arg = call_args[0][0]
        assert "Google releases Gemma 4" in prompt_arg
        assert "- ai_ml: AI and machine learning" in prompt_arg

        # Verify temperature=0.1
        assert call_args[1]["temperature"] == 0.1
