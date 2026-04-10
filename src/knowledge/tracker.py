from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import structlog
from slugify import slugify
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DigestItem, Message, UserKnowledge

log = structlog.get_logger()


def _get_embedding_for_entity(entity_name: str) -> bytes | None:
    """Generate embedding for an entity name for fuzzy matching."""
    try:
        from src.pipelines.scrape import generate_embedding

        return generate_embedding(entity_name)
    except Exception:
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


async def _find_known_entity(
    db: AsyncSession,
    user_id: int,
    entity_name: str,
    entity_embedding: bytes | None = None,
) -> UserKnowledge | None:
    """
    Find a known entity for this user.
    Primary: exact match on canonical_name.
    Fallback: embedding cosine similarity >= 0.75.
    """
    canonical = slugify(entity_name, separator="_")

    # Exact match
    result = await db.execute(
        select(UserKnowledge).where(
            UserKnowledge.user_id == user_id,
            UserKnowledge.canonical_name == canonical,
        )
    )
    known = result.scalar_one_or_none()
    if known:
        return known

    # Embedding fallback
    if entity_embedding:
        result = await db.execute(
            select(UserKnowledge).where(
                UserKnowledge.user_id == user_id,
                UserKnowledge.embedding.isnot(None),
            )
        )
        candidates = list(result.scalars().all())
        for candidate in candidates:
            sim = _cosine_similarity(entity_embedding, candidate.embedding)
            if sim >= 0.75:
                return candidate

    return None


async def update_user_knowledge(
    db: AsyncSession,
    user_id: int,
    digest_items: list[DigestItem],
) -> int:
    """
    After delivering a digest, update the user's knowledge graph.
    For each entity in delivered items:
    - New entity -> create UserKnowledge record
    - Known entity -> increment encounter_count, update last_seen_at
    Returns count of new entities added.
    """
    new_count = 0
    now = datetime.now(UTC)

    for di in digest_items:
        # Load the message
        msg_result = await db.execute(
            select(Message).where(Message.id == di.message_id)
        )
        message = msg_result.scalar_one_or_none()
        if not message or not message.entities_json:
            continue

        for entity in message.entities_json:
            name = entity.get("name", "")
            entity_type = entity.get("type")
            if not name:
                continue

            canonical = slugify(name, separator="_")
            entity_embedding = _get_embedding_for_entity(name)

            known = await _find_known_entity(
                db, user_id, name, entity_embedding
            )

            if known:
                known.encounter_count += 1
                known.last_seen_at = now
                # Update categories if not already present
                cat_name = di.category_name
                existing_cats = known.categories or []
                if cat_name not in existing_cats:
                    known.categories = [*existing_cats, cat_name]
            else:
                uk = UserKnowledge(
                    user_id=user_id,
                    entity_name=name,
                    entity_type=entity_type,
                    canonical_name=canonical,
                    first_seen_at=now,
                    last_seen_at=now,
                    encounter_count=1,
                    categories=[di.category_name],
                    embedding=entity_embedding,
                )
                db.add(uk)
                new_count += 1

    await db.commit()
    log.info(
        "knowledge_updated",
        user_id=user_id,
        new_entities=new_count,
        total_items=len(digest_items),
    )
    return new_count


async def compute_novelty(
    db: AsyncSession,
    user_id: int,
    message: Message,
) -> float:
    """
    What % of this message's entities are new to the user?
    No SLM call — pure DB lookups.
    Returns 0.0 (all known) to 1.0 (all new).
    """
    if not message.entities_json:
        return 0.5  # can't assess, neutral score

    entities = message.entities_json
    new_score = 0.0
    now = datetime.now(UTC)

    for entity in entities:
        name = entity.get("name", "")
        if not name:
            continue

        known = await _find_known_entity(db, user_id, name)

        if not known:
            new_score += 1.0
        elif known.last_seen_at:
            days_since = (now - known.last_seen_at.replace(tzinfo=UTC)).days
            if days_since > 30:
                new_score += 0.3  # stale knowledge = partial novelty

    return min(1.0, new_score / max(len(entities), 1))
