"""Tests for Knowledge Graph v2: entity resolution, graph building, novelty scoring."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.db.models import (
    Base,
    KnowledgeEntity,
    KnowledgeNarrative,
    KnowledgeRelation,
    Message,
    Source,
    User,
    UserEntityExposure,
    UserRelationExposure,
)


@pytest_asyncio.fixture
async def db():
    """Create an in-memory SQLite database for testing."""
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
async def sample_data(db: AsyncSession):
    """Create basic test data."""
    user = User(name="Test User", telegram_chat_id=12345)
    db.add(user)

    source = Source(source_type="telegram", source_identifier="@test_channel")
    db.add(source)
    await db.flush()

    msg = Message(
        source_id=source.id,
        external_id="msg1",
        content="Google announced Gemma 4, a new AI model that competes with Llama 4.",
        published_at=datetime.now(UTC),
        status="knowledge_extracted",
        entities_json=[
            {"name": "Gemma 4", "type": "MODEL"},
            {"name": "Google", "type": "ORGANIZATION"},
            {"name": "Llama 4", "type": "MODEL"},
        ],
        relations_json={
            "entities": [
                {"name": "Gemma 4", "type": "MODEL"},
                {"name": "Google", "type": "ORGANIZATION"},
                {"name": "Llama 4", "type": "MODEL"},
            ],
            "relations": [
                {"subject": "Gemma 4", "predicate": "CREATED_BY", "object": "Google", "confidence": 0.95},
                {"subject": "Gemma 4", "predicate": "COMPETES_WITH", "object": "Llama 4", "confidence": 0.9},
            ],
        },
    )
    db.add(msg)
    await db.commit()

    return {"user": user, "source": source, "message": msg}


# ========== Entity Resolution Tests ==========


class TestEntityResolver:
    @pytest.mark.asyncio
    async def test_get_or_create_new_entity(self, db: AsyncSession):
        from src.knowledge.entity_resolver import EntityResolver

        resolver = EntityResolver()
        with patch("src.knowledge.entity_resolver._generate_embedding", return_value=None):
            entity = await resolver.get_or_create(db, "Gemma 4", "MODEL")

        assert entity.id is not None
        assert entity.canonical_name == "gemma-4"
        assert entity.display_name == "Gemma 4"
        assert entity.entity_type == "MODEL"
        assert entity.mention_count == 1

    @pytest.mark.asyncio
    async def test_get_or_create_existing_increments_count(self, db: AsyncSession):
        from src.knowledge.entity_resolver import EntityResolver

        resolver = EntityResolver()
        with patch("src.knowledge.entity_resolver._generate_embedding", return_value=None):
            e1 = await resolver.get_or_create(db, "Gemma 4", "MODEL")
            e2 = await resolver.get_or_create(db, "Gemma 4", "MODEL")

        assert e1.id == e2.id
        assert e2.mention_count == 2

    @pytest.mark.asyncio
    async def test_resolve_exact_canonical(self, db: AsyncSession):
        from src.knowledge.entity_resolver import EntityResolver

        resolver = EntityResolver()
        with patch("src.knowledge.entity_resolver._generate_embedding", return_value=None):
            created = await resolver.get_or_create(db, "OpenAI", "ORGANIZATION")
            found = await resolver.resolve(db, "OpenAI")

        assert found is not None
        assert found.id == created.id

    @pytest.mark.asyncio
    async def test_resolve_alias_lookup(self, db: AsyncSession):
        from src.knowledge.entity_resolver import EntityResolver

        resolver = EntityResolver()
        with patch("src.knowledge.entity_resolver._generate_embedding", return_value=None):
            entity = await resolver.get_or_create(db, "World Health Organization", "ORGANIZATION")
            await resolver.add_alias(db, "WHO", entity.id)
            await db.commit()

            found = await resolver.resolve(db, "WHO")

        assert found is not None
        assert found.id == entity.id

    @pytest.mark.asyncio
    async def test_resolve_not_found(self, db: AsyncSession):
        from src.knowledge.entity_resolver import EntityResolver

        resolver = EntityResolver()
        with patch("src.knowledge.entity_resolver._generate_embedding", return_value=None):
            found = await resolver.resolve(db, "NonExistentEntity12345")

        assert found is None


# ========== Graph Builder Tests ==========


class TestGraphBuilder:
    @pytest.mark.asyncio
    async def test_merge_triples_creates_entities_and_relations(self, db: AsyncSession, sample_data):
        from src.knowledge.graph_builder import merge_triples_to_graph

        msg = sample_data["message"]
        with patch("src.knowledge.entity_resolver._generate_embedding", return_value=None):
            new_count = await merge_triples_to_graph(db, [msg])

        assert new_count == 2  # Two new relations

        # Check entities were created
        from sqlalchemy import select
        entities = (await db.execute(select(KnowledgeEntity))).scalars().all()
        assert len(entities) == 3  # Gemma 4, Google, Llama 4

        relations = (await db.execute(select(KnowledgeRelation))).scalars().all()
        assert len(relations) == 2

    @pytest.mark.asyncio
    async def test_merge_triples_idempotent(self, db: AsyncSession, sample_data):
        from src.knowledge.graph_builder import merge_triples_to_graph

        msg = sample_data["message"]
        with patch("src.knowledge.entity_resolver._generate_embedding", return_value=None):
            count1 = await merge_triples_to_graph(db, [msg])
            count2 = await merge_triples_to_graph(db, [msg])

        assert count1 == 2
        assert count2 == 0  # No new relations on second run

    @pytest.mark.asyncio
    async def test_load_graph_into_igraph(self, db: AsyncSession, sample_data):
        from src.knowledge.graph_builder import load_graph_from_db_sync, merge_triples_to_graph

        msg = sample_data["message"]
        with patch("src.knowledge.entity_resolver._generate_embedding", return_value=None):
            await merge_triples_to_graph(db, [msg])
            await db.commit()

        from sqlalchemy import select
        entities = list((await db.execute(select(KnowledgeEntity))).scalars().all())
        relations = list((await db.execute(select(KnowledgeRelation))).scalars().all())

        graph, id_map = load_graph_from_db_sync(entities, relations)

        assert graph.vcount() == 3
        assert graph.ecount() == 2

    @pytest.mark.asyncio
    async def test_detect_communities(self, db: AsyncSession):
        from src.knowledge.graph_builder import detect_communities, load_graph_from_db_sync

        # Create two disconnected clusters
        entities = []
        for i, name in enumerate(["A", "B", "C", "D", "E", "F"]):
            e = KnowledgeEntity(
                canonical_name=name.lower(), display_name=name,
                entity_type="CONCEPT", mention_count=1,
                first_seen_at=datetime.now(UTC), last_seen_at=datetime.now(UTC),
            )
            db.add(e)
            entities.append(e)
        await db.flush()

        # Cluster 1: A-B-C, Cluster 2: D-E-F
        for src, tgt in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
            db.add(KnowledgeRelation(
                source_entity_id=entities[src].id, target_entity_id=entities[tgt].id,
                relation_type="RELATED_TO", confidence=0.9,
                first_observed_at=datetime.now(UTC), last_observed_at=datetime.now(UTC),
            ))
        await db.flush()

        from sqlalchemy import select
        all_ents = list((await db.execute(select(KnowledgeEntity))).scalars().all())
        all_rels = list((await db.execute(select(KnowledgeRelation))).scalars().all())
        graph, _ = load_graph_from_db_sync(all_ents, all_rels)

        communities = detect_communities(graph)
        community_values = list(communities.values())
        # Should detect 2 communities
        assert len(set(community_values)) == 2


# ========== Novelty Tests ==========


class TestNovelty:
    @pytest.mark.asyncio
    async def test_entity_novelty_all_new(self, db: AsyncSession, sample_data):
        from src.knowledge.novelty import entity_novelty

        user = sample_data["user"]
        entities = [{"name": "Gemma 4", "type": "MODEL"}, {"name": "Google", "type": "ORG"}]

        score = await entity_novelty(db, user.id, entities)
        assert score == 1.0  # No entities in global graph, all new

    @pytest.mark.asyncio
    async def test_entity_novelty_all_known(self, db: AsyncSession, sample_data):
        from src.knowledge.novelty import entity_novelty

        user = sample_data["user"]

        # Create entities and expose them to user
        for name in ["Gemma 4", "Google"]:
            e = KnowledgeEntity(
                canonical_name=name.lower().replace(" ", "-"),
                display_name=name, entity_type="MODEL", mention_count=1,
                first_seen_at=datetime.now(UTC), last_seen_at=datetime.now(UTC),
            )
            db.add(e)
            await db.flush()
            db.add(UserEntityExposure(
                user_id=user.id, entity_id=e.id,
                first_exposed_at=datetime.now(UTC),
                last_exposed_at=datetime.now(UTC), exposure_count=1,
            ))
        await db.commit()

        entities = [{"name": "Gemma 4", "type": "MODEL"}, {"name": "Google", "type": "ORG"}]
        score = await entity_novelty(db, user.id, entities)
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_entity_novelty_stale(self, db: AsyncSession, sample_data):
        from src.knowledge.novelty import entity_novelty

        user = sample_data["user"]
        stale_time = datetime.now(UTC) - timedelta(days=45)

        e = KnowledgeEntity(
            canonical_name="gemma-4", display_name="Gemma 4",
            entity_type="MODEL", mention_count=1,
            first_seen_at=stale_time, last_seen_at=stale_time,
        )
        db.add(e)
        await db.flush()
        db.add(UserEntityExposure(
            user_id=user.id, entity_id=e.id,
            first_exposed_at=stale_time, last_exposed_at=stale_time,
            exposure_count=1,
        ))
        await db.commit()

        entities = [{"name": "Gemma 4", "type": "MODEL"}]
        score = await entity_novelty(db, user.id, entities)
        assert score == pytest.approx(0.3, abs=0.05)

    @pytest.mark.asyncio
    async def test_relation_novelty_all_new(self, db: AsyncSession, sample_data):
        from src.knowledge.novelty import relation_novelty

        user = sample_data["user"]
        relations = [
            {"subject": "Gemma 4", "predicate": "CREATED_BY", "object": "Google"},
        ]

        score = await relation_novelty(db, user.id, relations)
        assert score == 1.0  # No relations in graph

    @pytest.mark.asyncio
    async def test_evolution_novelty_detects_change(self, db: AsyncSession, sample_data):
        from src.knowledge.novelty import evolution_novelty

        user = sample_data["user"]

        # Create existing knowledge: OpenAI WORKS_AT "old office"
        e1 = KnowledgeEntity(
            canonical_name="openai", display_name="OpenAI",
            entity_type="ORGANIZATION", mention_count=1,
            first_seen_at=datetime.now(UTC), last_seen_at=datetime.now(UTC),
        )
        e2 = KnowledgeEntity(
            canonical_name="old-office", display_name="Old Office",
            entity_type="LOCATION", mention_count=1,
            first_seen_at=datetime.now(UTC), last_seen_at=datetime.now(UTC),
        )
        db.add_all([e1, e2])
        await db.flush()

        db.add(KnowledgeRelation(
            source_entity_id=e1.id, target_entity_id=e2.id,
            relation_type="HEADQUARTERED_IN", confidence=0.9,
            first_observed_at=datetime.now(UTC), last_observed_at=datetime.now(UTC),
        ))
        await db.commit()

        # New info says OpenAI HEADQUARTERED_IN "new-office" (different object)
        relations = [
            {"subject": "OpenAI", "predicate": "HEADQUARTERED_IN", "object": "New Office"},
        ]
        score = await evolution_novelty(db, user.id, relations)
        assert score > 0.0  # Should detect the change

    @pytest.mark.asyncio
    async def test_composite_novelty_weighted(self, db: AsyncSession, sample_data):
        from src.knowledge.novelty import compute_composite_novelty

        user = sample_data["user"]
        msg = sample_data["message"]

        score = await compute_composite_novelty(db, user.id, msg)
        # All entities and relations are new → high novelty
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should be fairly novel since everything is new

    @pytest.mark.asyncio
    async def test_empty_entities_neutral_score(self, db: AsyncSession, sample_data):
        from src.knowledge.novelty import entity_novelty

        score = await entity_novelty(db, 1, [])
        assert score == 0.5

    @pytest.mark.asyncio
    async def test_bridge_novelty_single_community(self, db: AsyncSession, sample_data):
        from src.knowledge.novelty import bridge_novelty

        user = sample_data["user"]
        entities = [{"name": "A"}, {"name": "B"}]
        # All in same community
        community_map = {1: 0, 2: 0}

        score = await bridge_novelty(db, user.id, entities, community_map)
        assert score == 0.0  # Same community, no bridge
