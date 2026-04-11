from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import structlog
from slugify import slugify
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import EntityAlias, KnowledgeEntity

log = structlog.get_logger()

# Cosine similarity threshold for Tier 3 embedding match
EMBEDDING_SIMILARITY_THRESHOLD = 0.85


def _generate_embedding(text: str) -> bytes | None:
    """Generate embedding for a text string using the shared sentence-transformers model."""
    try:
        from src.pipelines.scrape import generate_embedding

        return generate_embedding(text)
    except Exception:
        log.warning("embedding_generation_failed", text=text)
        return None


def _cosine_similarity(a: bytes, b: bytes) -> float:
    """Compute cosine similarity between two embedding byte arrays."""
    arr_a = np.frombuffer(a, dtype=np.float32)
    arr_b = np.frombuffer(b, dtype=np.float32)
    dot = np.dot(arr_a, arr_b)
    norm = np.linalg.norm(arr_a) * np.linalg.norm(arr_b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class EntityResolver:
    """Three-tier entity resolution: exact canonical -> alias lookup -> embedding similarity."""

    async def resolve(
        self,
        db: AsyncSession,
        entity_name: str,
        entity_type: str | None = None,
    ) -> KnowledgeEntity | None:
        """Find an existing entity or return None.

        Tier 1: Exact canonical name match (slugified).
        Tier 2: Alias table lookup.
        Tier 3: Embedding similarity (cosine >= 0.85).
        """
        canonical = slugify(entity_name, separator="-")

        # Tier 1: exact canonical match
        result = await db.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.canonical_name == canonical)
        )
        entity = result.scalar_one_or_none()
        if entity:
            log.debug("entity_resolved_tier1", name=entity_name, entity_id=entity.id)
            return entity

        # Tier 2: alias table lookup
        alias_canonical = canonical
        result = await db.execute(
            select(EntityAlias).where(EntityAlias.alias == alias_canonical)
        )
        alias_row = result.scalar_one_or_none()
        if alias_row:
            result = await db.execute(
                select(KnowledgeEntity).where(KnowledgeEntity.id == alias_row.entity_id)
            )
            entity = result.scalar_one_or_none()
            if entity:
                log.debug(
                    "entity_resolved_tier2",
                    name=entity_name,
                    alias=alias_canonical,
                    entity_id=entity.id,
                )
                return entity

        # Tier 3: embedding similarity
        name_embedding = _generate_embedding(entity_name)
        if name_embedding:
            result = await db.execute(
                select(KnowledgeEntity).where(KnowledgeEntity.embedding.isnot(None))
            )
            candidates = list(result.scalars().all())

            best_match: KnowledgeEntity | None = None
            best_sim = 0.0

            for candidate in candidates:
                sim = _cosine_similarity(name_embedding, candidate.embedding)
                if sim >= EMBEDDING_SIMILARITY_THRESHOLD and sim > best_sim:
                    best_sim = sim
                    best_match = candidate

            if best_match:
                log.debug(
                    "entity_resolved_tier3",
                    name=entity_name,
                    matched=best_match.canonical_name,
                    similarity=round(best_sim, 4),
                )
                # Auto-register this as an alias for future Tier 2 hits
                await self.add_alias(db, entity_name, best_match.id)
                return best_match

        return None

    async def get_or_create(
        self,
        db: AsyncSession,
        entity_name: str,
        entity_type: str | None = None,
        properties: dict | None = None,
    ) -> KnowledgeEntity:
        """Find an existing entity or create a new one.

        If the entity already exists, increments mention_count and updates last_seen_at.
        """
        existing = await self.resolve(db, entity_name, entity_type)
        now = datetime.now(UTC)

        if existing:
            existing.mention_count += 1
            existing.last_seen_at = now
            if properties:
                merged = existing.properties_json or {}
                merged.update(properties)
                existing.properties_json = merged
            await db.flush()
            return existing

        # Create new entity
        canonical = slugify(entity_name, separator="-")
        embedding = _generate_embedding(entity_name)

        entity = KnowledgeEntity(
            canonical_name=canonical,
            display_name=entity_name,
            entity_type=entity_type or "CONCEPT",
            properties_json=properties,
            first_seen_at=now,
            last_seen_at=now,
            mention_count=1,
            embedding=embedding,
        )
        db.add(entity)
        await db.flush()

        log.info(
            "entity_created",
            name=entity_name,
            canonical=canonical,
            entity_type=entity.entity_type,
            entity_id=entity.id,
        )
        return entity

    async def add_alias(
        self,
        db: AsyncSession,
        alias_name: str,
        entity_id: int,
    ) -> None:
        """Register a new alias for an entity. No-op if alias already exists."""
        alias_canonical = slugify(alias_name, separator="-")

        # Check if alias already exists
        result = await db.execute(
            select(EntityAlias).where(EntityAlias.alias == alias_canonical)
        )
        if result.scalar_one_or_none():
            return

        alias = EntityAlias(
            alias=alias_canonical,
            entity_id=entity_id,
        )
        db.add(alias)
        await db.flush()

        log.info("alias_registered", alias=alias_canonical, entity_id=entity_id)
