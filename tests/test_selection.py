"""Tests for digest selection algorithm and delivery."""

from src.bot.delivery import _split_at_boundaries, escape_markdown_v2
from src.pipelines.digest import CATEGORY_EMOJIS


class TestEscapeMarkdownV2:
    """Test Telegram MarkdownV2 escaping."""

    def test_escapes_special_chars(self):
        text = "Hello_world (test) [link]"
        escaped = escape_markdown_v2(text)
        assert "\\_" in escaped
        assert "\\(" in escaped
        assert "\\)" in escaped
        assert "\\[" in escaped
        assert "\\]" in escaped

    def test_plain_text_unchanged(self):
        text = "Hello world"
        assert escape_markdown_v2(text) == "Hello world"

    def test_dots_and_dashes(self):
        text = "v2.0 - release"
        escaped = escape_markdown_v2(text)
        assert "\\." in escaped
        assert "\\-" in escaped


class TestMessageSplitting:
    """Test 4096 character limit handling."""

    def test_short_message_not_split(self):
        text = "Short message"
        chunks = _split_at_boundaries(text, max_len=4096)
        assert len(chunks) == 1
        assert chunks[0] == "Short message"

    def test_splits_at_double_newline(self):
        section1 = "A" * 2000
        section2 = "B" * 2000
        section3 = "C" * 2000
        text = f"{section1}\n\n{section2}\n\n{section3}"
        chunks = _split_at_boundaries(text, max_len=4096)
        assert len(chunks) >= 2

    def test_respects_max_len(self):
        text = "\n\n".join(["X" * 100 for _ in range(50)])
        chunks = _split_at_boundaries(text, max_len=500)
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_very_long_line_force_split(self):
        text = "A" * 10000
        chunks = _split_at_boundaries(text, max_len=4096)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_empty_chunks_filtered(self):
        text = "\n\n\n\nHello\n\n\n\n"
        chunks = _split_at_boundaries(text, max_len=4096)
        for chunk in chunks:
            assert chunk.strip()


class TestCategoryEmojis:
    """Test category emoji mapping."""

    def test_known_categories_have_emojis(self):
        assert "ai_ml" in CATEGORY_EMOJIS
        assert "crypto" in CATEGORY_EMOJIS
        assert "tech_industry" in CATEGORY_EMOJIS

    def test_emojis_are_strings(self):
        for emoji in CATEGORY_EMOJIS.values():
            assert isinstance(emoji, str)
            assert len(emoji) > 0
