"""Tests for narrative detection and lifecycle management."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.db.models import (
    Base,
    KnowledgeEntity,
    KnowledgeNarrative,
    Message,
    Source,
    User,
    UserEntityExposure,
)


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def narrative_data(db: AsyncSession):
    """Create entities and messages that should form a narrative."""
    user = User(name="Test", telegram_chat_id=99999)
    source = Source(source_type="telegram", source_identifier="@news")
    db.add_all([user, source])
    await db.flush()

    # Create 4 messages sharing 3+ entities about "OpenAI funding"
    shared_entities = [
        {"name": "OpenAI", "type": "ORGANIZATION"},
        {"name": "Sam Altman", "type": "PERSON"},
        {"name": "Microsoft", "type": "ORGANIZATION"},
        {"name": "Funding Round", "type": "EVENT"},
    ]

    messages = []
    for i in range(4):
        msg = Message(
            source_id=source.id,
            external_id=f"narr_msg_{i}",
            content=f"OpenAI funding news part {i}",
            published_at=datetime.now(UTC) - timedelta(days=i),
            status="knowledge_extracted",
            entities_json=shared_entities,
        )
        db.add(msg)
        messages.append(msg)

    # Create global entities for matching
    kg_entities = []
    for ent in shared_entities:
        ke = KnowledgeEntity(
            canonical_name=ent["name"].lower().replace(" ", "-"),
            display_name=ent["name"],
            entity_type=ent["type"],
            mention_count=4,
            first_seen_at=datetime.now(UTC),
            last_seen_at=datetime.now(UTC),
        )
        db.add(ke)
        kg_entities.append(ke)

    await db.flush()

    # Also create a message with different entities (should NOT form narrative)
    unrelated_msg = Message(
        source_id=source.id,
        external_id="unrelated",
        content="Weather forecast for tomorrow",
        published_at=datetime.now(UTC),
        status="knowledge_extracted",
        entities_json=[{"name": "Weather", "type": "CONCEPT"}],
    )
    db.add(unrelated_msg)
    await db.commit()

    return {
        "user": user,
        "messages": messages,
        "entities": kg_entities,
        "unrelated": unrelated_msg,
    }


class TestNarrativeDetection:
    @pytest.mark.asyncio
    async def test_detects_narrative_from_shared_entities(self, db: AsyncSession, narrative_data):
        from src.knowledge.narratives import detect_narratives

        count = await detect_narratives(db)
        assert count >= 1  # At least one narrative detected

        from sqlalchemy import select
        narratives = list(
            (await db.execute(select(KnowledgeNarrative))).scalars().all()
        )
        assert len(narratives) >= 1
        assert narratives[0].status == "active"
        assert narratives[0].message_count >= 3

    @pytest.mark.asyncio
    async def test_narrative_has_title(self, db: AsyncSession, narrative_data):
        from src.knowledge.narratives import detect_narratives

        await detect_narratives(db)

        from sqlalchemy import select
        narratives = list(
            (await db.execute(select(KnowledgeNarrative))).scalars().all()
        )
        assert len(narratives) >= 1
        assert narratives[0].title  # Title should be non-empty
        assert len(narratives[0].title) > 0

    @pytest.mark.asyncio
    async def test_narrative_entity_ids_populated(self, db: AsyncSession, narrative_data):
        from src.knowledge.narratives import detect_narratives

        await detect_narratives(db)

        from sqlalchemy import select
        narratives = list(
            (await db.execute(select(KnowledgeNarrative))).scalars().all()
        )
        assert len(narratives) >= 1
        assert narratives[0].entity_ids_json is not None
        assert len(narratives[0].entity_ids_json) >= 3


class TestNarrativeLifecycle:
    @pytest.mark.asyncio
    async def test_active_stays_active(self, db: AsyncSession):
        from src.knowledge.narratives import update_narrative_lifecycle

        narr = KnowledgeNarrative(
            title="Active Story",
            entity_ids_json=[1, 2, 3],
            first_message_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC),
            message_count=5,
            status="active",
        )
        db.add(narr)
        await db.commit()

        transitions = await update_narrative_lifecycle(db)
        assert transitions["to_stale"] == 0
        assert transitions["to_concluded"] == 0

    @pytest.mark.asyncio
    async def test_active_to_stale_after_7_days(self, db: AsyncSession):
        from src.knowledge.narratives import update_narrative_lifecycle

        narr = KnowledgeNarrative(
            title="Going Stale",
            entity_ids_json=[1, 2, 3],
            first_message_at=datetime.now(UTC) - timedelta(days=14),
            last_message_at=datetime.now(UTC) - timedelta(days=10),
            message_count=5,
            status="active",
        )
        db.add(narr)
        await db.commit()

        transitions = await update_narrative_lifecycle(db)
        assert transitions["to_stale"] == 1

        await db.refresh(narr)
        assert narr.status == "stale"

    @pytest.mark.asyncio
    async def test_stale_to_concluded_after_30_days(self, db: AsyncSession):
        from src.knowledge.narratives import update_narrative_lifecycle

        narr = KnowledgeNarrative(
            title="Old Story",
            entity_ids_json=[1, 2, 3],
            first_message_at=datetime.now(UTC) - timedelta(days=60),
            last_message_at=datetime.now(UTC) - timedelta(days=35),
            message_count=10,
            status="stale",
        )
        db.add(narr)
        await db.commit()

        transitions = await update_narrative_lifecycle(db)
        assert transitions["to_concluded"] == 1

        await db.refresh(narr)
        assert narr.status == "concluded"


class TestUserNarratives:
    @pytest.mark.asyncio
    async def test_user_follows_narrative_via_entity_overlap(self, db: AsyncSession):
        from src.knowledge.narratives import get_user_narratives

        user = User(name="Test", telegram_chat_id=77777)
        db.add(user)

        # Create entities
        entities = []
        for name in ["A", "B", "C"]:
            e = KnowledgeEntity(
                canonical_name=name.lower(), display_name=name,
                entity_type="CONCEPT", mention_count=1,
                first_seen_at=datetime.now(UTC), last_seen_at=datetime.now(UTC),
            )
            db.add(e)
            entities.append(e)
        await db.flush()

        # Expose 2 entities to user
        for e in entities[:2]:
            db.add(UserEntityExposure(
                user_id=user.id, entity_id=e.id,
                first_exposed_at=datetime.now(UTC),
                last_exposed_at=datetime.now(UTC),
            ))

        # Create narrative with those entities
        narr = KnowledgeNarrative(
            title="Story ABC",
            entity_ids_json=[e.id for e in entities],
            first_message_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC),
            message_count=5,
            status="active",
        )
        db.add(narr)
        await db.commit()

        user_narrs = await get_user_narratives(db, user.id)
        assert len(user_narrs) == 1
        assert user_narrs[0].title == "Story ABC"

    @pytest.mark.asyncio
    async def test_user_not_following_without_exposure(self, db: AsyncSession):
        from src.knowledge.narratives import get_user_narratives

        user = User(name="Empty", telegram_chat_id=88888)
        db.add(user)
        await db.flush()

        narr = KnowledgeNarrative(
            title="Unfollowed",
            entity_ids_json=[999, 998, 997],
            first_message_at=datetime.now(UTC),
            last_message_at=datetime.now(UTC),
            message_count=5,
            status="active",
        )
        db.add(narr)
        await db.commit()

        user_narrs = await get_user_narratives(db, user.id)
        assert len(user_narrs) == 0
