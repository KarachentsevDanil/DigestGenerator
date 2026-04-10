"""Tests for classification logic: prompt parsing, validation, retry."""


from src.pipelines.classify import (
    _build_categories_block,
    _normalize_classification,
    _validate_classification,
)


class TestValidateClassification:
    """Test classification output validation."""

    def test_valid_output(self):
        result = {
            "categories": {"ai_ml": 0.92, "tech_industry": 0.55},
            "relevance": 0.8,
            "summary": "Google releases Gemma 4 model.",
            "entities": [{"name": "Gemma 4", "type": "model"}],
        }
        assert _validate_classification(result) is True

    def test_missing_categories(self):
        result = {
            "relevance": 0.8,
            "summary": "Test",
            "entities": [],
        }
        assert _validate_classification(result) is False

    def test_categories_not_dict(self):
        result = {
            "categories": "ai_ml",
            "relevance": 0.8,
            "summary": "Test",
            "entities": [],
        }
        assert _validate_classification(result) is False

    def test_missing_relevance(self):
        result = {
            "categories": {"ai_ml": 0.9},
            "summary": "Test",
            "entities": [],
        }
        assert _validate_classification(result) is False

    def test_missing_summary(self):
        result = {
            "categories": {"ai_ml": 0.9},
            "relevance": 0.8,
            "entities": [],
        }
        assert _validate_classification(result) is False

    def test_missing_entities(self):
        result = {
            "categories": {"ai_ml": 0.9},
            "relevance": 0.8,
            "summary": "Test",
        }
        assert _validate_classification(result) is False


class TestNormalizeClassification:
    """Test classification value normalization."""

    def test_clamp_relevance(self):
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

    def test_multi_label_scores(self):
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
        from unittest.mock import MagicMock

        cat1 = MagicMock()
        cat1.name = "ai_ml"
        cat1.description = "AI and machine learning"

        cat2 = MagicMock()
        cat2.name = "crypto"
        cat2.description = "Cryptocurrency and blockchain"

        block = _build_categories_block([cat1, cat2])
        assert "- ai_ml: AI and machine learning" in block
        assert "- crypto: Cryptocurrency and blockchain" in block

    def test_empty_categories(self):
        block = _build_categories_block([])
        assert block == ""
