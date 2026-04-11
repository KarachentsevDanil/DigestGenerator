"""Tests for digest selection algorithm and delivery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.bot.delivery import _split_at_boundaries, escape_markdown_v2
from src.config import DigestConfig, Settings
from src.db.models import (
    Base,
    Category,
    Digest,
    DigestItem,
    Message,
    Source,
    User,
    UserCategory,
)
from src.pipelines.digest import CATEGORY_EMOJIS, DigestCandidate, select_items


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def engine():
    """Create an in-memory SQLite engine for tests."""
    eng = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await eng.dispose()


@pytest_asyncio.fixture
async def db(engine):
    """Provide a transactional DB session that rolls back after each test."""
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session


def _make_settings(**overrides) -> Settings:
    """Create a Settings object with sensible defaults for testing."""
    digest_cfg = overrides.pop("digest", DigestConfig())
    return Settings(
        telegram_bot_token="test-token",
        webhook_url="https://test.example.com",
        digest=digest_cfg,
        **overrides,
    )


async def _seed_user(db: AsyncSession, **kwargs) -> User:
    defaults = dict(
        name="TestUser",
        telegram_chat_id=12345,
        timezone="UTC",
        daily_enabled=True,
        daily_hour=8,
        weekly_enabled=True,
        weekly_day=6,
        weekly_hour=9,
        daily_min_confidence=0.5,
        weekly_min_confidence=0.7,
    )
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def _seed_source(db: AsyncSession, identifier: str = "@test_channel") -> Source:
    source = Source(
        source_type="telegram",
        source_identifier=identifier,
        display_name=identifier,
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def _seed_category(
    db: AsyncSession, user: User, name: str = "tech", display: str = "Tech"
) -> Category:
    cat = Category(
        name=name,
        display_name=display,
        description=f"{display} category",
        created_by_user_id=user.id,
    )
    db.add(cat)
    await db.commit()
    await db.refresh(cat)
    return cat


async def _subscribe(
    db: AsyncSession,
    user: User,
    category: Category,
    top_k: int = 5,
    min_confidence: float = 0.5,
) -> UserCategory:
    uc = UserCategory(
        user_id=user.id,
        category_id=category.id,
        top_k=top_k,
        min_confidence=min_confidence,
    )
    db.add(uc)
    await db.commit()
    return uc


async def _seed_message(
    db: AsyncSession,
    source: Source,
    external_id: str,
    category_scores: dict | None = None,
    relevance: float = 0.5,
    published_at: datetime | None = None,
    status: str = "classified",
    is_primary: bool = True,
    summary: str | None = None,
    content_url: str | None = None,
    cluster_id: str | None = None,
) -> Message:
    msg = Message(
        source_id=source.id,
        external_id=external_id,
        content=f"Content for {external_id}",
        published_at=published_at or datetime.now(UTC),
        status=status,
        is_cluster_primary=is_primary,
        category_scores_json=category_scores,
        relevance_score=relevance,
        summary=summary or f"Summary of {external_id}",
        content_url=content_url,
        dedup_cluster_id=cluster_id,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


async def _seed_digest(
    db: AsyncSession,
    user: User,
    digest_type: str,
    message_ids: list[int],
    category_name: str = "tech",
) -> Digest:
    now = datetime.now(UTC)
    digest = Digest(
        user_id=user.id,
        digest_type=digest_type,
        window_start=now - timedelta(hours=24),
        window_end=now,
        generated_at=now,
        item_count=len(message_ids),
    )
    db.add(digest)
    await db.commit()
    await db.refresh(digest)

    for rank, mid in enumerate(message_ids):
        di = DigestItem(
            digest_id=digest.id,
            message_id=mid,
            category_name=category_name,
            rank_in_category=rank + 1,
            confidence_score=0.8,
        )
        db.add(di)
    await db.commit()
    return digest


# ---------------------------------------------------------------------------
# Selection Algorithm Tests
# ---------------------------------------------------------------------------


class TestThresholdFiltering:
    """Messages below confidence threshold should be excluded."""

    @pytest.mark.asyncio
    async def test_below_threshold_excluded(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.6)

        # Message with cat score below threshold (0.6)
        await _seed_message(
            db, source, "msg_low",
            category_scores={"tech": 0.4},
            relevance=0.8,
        )
        # Message with cat score above threshold
        await _seed_message(
            db, source, "msg_high",
            category_scores={"tech": 0.8},
            relevance=0.8,
        )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)

        assert result is not None
        items = result["Tech"]
        assert len(items) == 1
        assert items[0].message.external_id == "msg_high"


class TestRelevanceFloorFiltering:
    """Messages below relevance floor should be excluded."""

    @pytest.mark.asyncio
    async def test_daily_relevance_floor(self, db):
        """Daily floor is 0.2 by default."""
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        # relevance 0.1 < floor 0.2 -> excluded
        await _seed_message(
            db, source, "msg_low_rel",
            category_scores={"tech": 0.8},
            relevance=0.1,
        )
        # relevance 0.5 >= floor 0.2 -> included
        await _seed_message(
            db, source, "msg_ok_rel",
            category_scores={"tech": 0.8},
            relevance=0.5,
        )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)

        assert result is not None
        items = result["Tech"]
        assert len(items) == 1
        assert items[0].message.external_id == "msg_ok_rel"

    @pytest.mark.asyncio
    async def test_weekly_relevance_floor(self, db):
        """Weekly floor is 0.4 by default, higher than daily."""
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        # relevance 0.3 < weekly floor 0.4 -> excluded
        await _seed_message(
            db, source, "msg_low_weekly",
            category_scores={"tech": 0.8},
            relevance=0.3,
        )
        # relevance 0.6 >= weekly floor 0.4 -> included
        await _seed_message(
            db, source, "msg_ok_weekly",
            category_scores={"tech": 0.8},
            relevance=0.6,
        )

        settings = _make_settings()
        result = await select_items(db, user, "weekly", settings)

        assert result is not None
        items = result["Tech"]
        assert len(items) == 1
        assert items[0].message.external_id == "msg_ok_weekly"


class TestAlreadySentExclusion:
    """Daily digests exclude previously sent daily items."""

    @pytest.mark.asyncio
    async def test_daily_excludes_prev_daily(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        msg1 = await _seed_message(
            db, source, "msg_sent",
            category_scores={"tech": 0.9},
            relevance=0.8,
        )
        msg2 = await _seed_message(
            db, source, "msg_new",
            category_scores={"tech": 0.9},
            relevance=0.8,
        )

        # Record msg1 as already sent in a daily digest
        await _seed_digest(db, user, "daily", [msg1.id], category_name="tech")

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)

        assert result is not None
        items = result["Tech"]
        assert len(items) == 1
        assert items[0].message.id == msg2.id


class TestWeeklyReInclusion:
    """Weekly digest CAN re-include items from daily digests."""

    @pytest.mark.asyncio
    async def test_weekly_includes_daily_items(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        msg = await _seed_message(
            db, source, "msg_daily",
            category_scores={"tech": 0.9},
            relevance=0.8,
        )

        # This message was in a daily digest
        await _seed_digest(db, user, "daily", [msg.id], category_name="tech")

        settings = _make_settings()
        result = await select_items(db, user, "weekly", settings)

        # Weekly should still include it (only excludes prev weekly items)
        assert result is not None
        items = result["Tech"]
        assert len(items) == 1
        assert items[0].message.id == msg.id

    @pytest.mark.asyncio
    async def test_weekly_excludes_prev_weekly(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        msg = await _seed_message(
            db, source, "msg_prev_weekly",
            category_scores={"tech": 0.9},
            relevance=0.8,
        )

        # This message was in a previous weekly digest
        await _seed_digest(db, user, "weekly", [msg.id], category_name="tech")

        settings = _make_settings()
        result = await select_items(db, user, "weekly", settings)

        # Should be excluded
        assert result is None


class TestTopKLimit:
    """Top_k should limit the number of selected items per category."""

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=2, min_confidence=0.3)

        # Create 5 qualifying messages
        for i in range(5):
            await _seed_message(
                db, source, f"msg_{i}",
                category_scores={"tech": 0.8},
                relevance=0.5 + i * 0.05,
            )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)

        assert result is not None
        items = result["Tech"]
        assert len(items) == 2


class TestEmptyDigest:
    """When no items qualify, select_items returns None."""

    @pytest.mark.asyncio
    async def test_no_messages_returns_none(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.5)

        # No messages at all
        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_all_below_threshold_returns_none(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.8)

        # All messages below threshold
        await _seed_message(
            db, source, "msg_1",
            category_scores={"tech": 0.3},
            relevance=0.5,
        )
        await _seed_message(
            db, source, "msg_2",
            category_scores={"tech": 0.5},
            relevance=0.5,
        )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_subscriptions_returns_none(self, db):
        user = await _seed_user(db)
        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)
        assert result is None


class TestRankingOrder:
    """Daily and weekly digests use different ranking strategies."""

    @pytest.mark.asyncio
    async def test_daily_ranking_confidence_then_relevance(self, db):
        """Daily: category_confidence DESC, then relevance DESC."""
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        # msg_a: confidence=0.9, relevance=0.5
        await _seed_message(
            db, source, "msg_a",
            category_scores={"tech": 0.9},
            relevance=0.5,
        )
        # msg_b: confidence=0.7, relevance=0.9
        await _seed_message(
            db, source, "msg_b",
            category_scores={"tech": 0.7},
            relevance=0.9,
        )
        # msg_c: confidence=0.9, relevance=0.8  (same confidence as a, higher rel)
        await _seed_message(
            db, source, "msg_c",
            category_scores={"tech": 0.9},
            relevance=0.8,
        )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)

        assert result is not None
        items = result["Tech"]
        # Sorted by confidence DESC, then relevance DESC
        # msg_c (0.9, 0.8), msg_a (0.9, 0.5), msg_b (0.7, 0.9)
        assert items[0].message.external_id == "msg_c"
        assert items[1].message.external_id == "msg_a"
        assert items[2].message.external_id == "msg_b"

    @pytest.mark.asyncio
    async def test_weekly_ranking_composite(self, db):
        """Weekly: relevance * category_confidence composite DESC."""
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        # msg_a: confidence=0.9, relevance=0.5 -> composite = 0.45
        await _seed_message(
            db, source, "msg_a",
            category_scores={"tech": 0.9},
            relevance=0.5,
        )
        # msg_b: confidence=0.7, relevance=0.9 -> composite = 0.63
        await _seed_message(
            db, source, "msg_b",
            category_scores={"tech": 0.7},
            relevance=0.9,
        )
        # msg_c: confidence=0.8, relevance=0.7 -> composite = 0.56
        await _seed_message(
            db, source, "msg_c",
            category_scores={"tech": 0.8},
            relevance=0.7,
        )

        settings = _make_settings()
        result = await select_items(db, user, "weekly", settings)

        assert result is not None
        items = result["Tech"]
        # Sorted by composite = relevance * confidence DESC
        # msg_b (0.63), msg_c (0.56), msg_a (0.45)
        assert items[0].message.external_id == "msg_b"
        assert items[1].message.external_id == "msg_c"
        assert items[2].message.external_id == "msg_a"


class TestBasicSelection:
    """Basic selection: user with 1 category, 3 classified messages, top_k=2."""

    @pytest.mark.asyncio
    async def test_basic_top_k_selection(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=2, min_confidence=0.3)

        for i in range(3):
            await _seed_message(
                db, source, f"msg_{i}",
                category_scores={"tech": 0.8},
                relevance=0.5 + i * 0.1,
            )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)

        assert result is not None
        assert "Tech" in result
        assert len(result["Tech"]) == 2


class TestMultiCategory:
    """A message can appear in different categories for different users."""

    @pytest.mark.asyncio
    async def test_message_in_multiple_categories(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)

        cat_tech = await _seed_category(db, user, "tech", "Tech")
        cat_ai = await _seed_category(db, user, "ai_ml", "AI/ML")
        await _subscribe(db, user, cat_tech, top_k=5, min_confidence=0.3)
        await _subscribe(db, user, cat_ai, top_k=5, min_confidence=0.3)

        # Message scores in both categories
        await _seed_message(
            db, source, "msg_cross",
            category_scores={"tech": 0.8, "ai_ml": 0.9},
            relevance=0.7,
        )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)

        assert result is not None
        assert "Tech" in result
        assert "AI/ML" in result
        # Same message appears in both
        tech_ids = {i.message.id for i in result["Tech"]}
        ai_ids = {i.message.id for i in result["AI/ML"]}
        assert tech_ids & ai_ids  # intersection is non-empty


class TestWeeklyThreshold:
    """Weekly uses max(per-category, weekly_min_confidence)."""

    @pytest.mark.asyncio
    async def test_weekly_uses_max_threshold(self, db):
        user = await _seed_user(db, weekly_min_confidence=0.7)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        # User category threshold is 0.3, but user weekly min is 0.7
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        # This message has category score 0.5 -> above per-cat but below weekly min
        await _seed_message(
            db, source, "msg_mid",
            category_scores={"tech": 0.5},
            relevance=0.8,
        )
        # This message has category score 0.8 -> above both
        await _seed_message(
            db, source, "msg_high",
            category_scores={"tech": 0.8},
            relevance=0.8,
        )

        settings = _make_settings()
        result = await select_items(db, user, "weekly", settings)

        assert result is not None
        items = result["Tech"]
        assert len(items) == 1
        assert items[0].message.external_id == "msg_high"


class TestOnlyClassifiedPrimaries:
    """Only classified, primary messages are considered."""

    @pytest.mark.asyncio
    async def test_unclassified_excluded(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        await _seed_message(
            db, source, "msg_unprocessed",
            category_scores={"tech": 0.9},
            relevance=0.8,
            status="unprocessed",
        )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_primary_excluded(self, db):
        user = await _seed_user(db)
        source = await _seed_source(db)
        cat = await _seed_category(db, user, "tech", "Tech")
        await _subscribe(db, user, cat, top_k=5, min_confidence=0.3)

        await _seed_message(
            db, source, "msg_dup",
            category_scores={"tech": 0.9},
            relevance=0.8,
            is_primary=False,
        )

        settings = _make_settings()
        result = await select_items(db, user, "daily", settings)
        assert result is None


# ---------------------------------------------------------------------------
# Escape + Splitting Tests
# ---------------------------------------------------------------------------


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

    def test_digest_over_4096_splits_at_category_boundaries(self):
        """A rendered digest > 4096 chars should split at section boundaries."""
        section_a = "Category A\n" + "\n".join([f"Item {i}" for i in range(100)])
        section_b = "Category B\n" + "\n".join([f"Item {i}" for i in range(100)])
        text = f"{section_a}\n\n{section_b}"
        chunks = _split_at_boundaries(text, max_len=2000)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 2000


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
