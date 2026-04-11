from __future__ import annotations

from datetime import UTC, datetime

import structlog
from slugify import slugify
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    KnowledgeEntity,
    KnowledgeRelation,
    Message,
    UserEntityExposure,
    UserRelationExposure,
)

log = structlog.get_logger()

# Novelty dimension weights
ENTITY_WEIGHT = 0.15
RELATION_WEIGHT = 0.30
BRIDGE_WEIGHT = 0.25
EVOLUTION_WEIGHT = 0.15
NARRATIVE_WEIGHT = 0.15

# Staleness thresholds (days)
ENTITY_STALE_DAYS = 30
RELATION_STALE_DAYS = 60
ENTITY_STALE_NOVELTY = 0.3
RELATION_STALE_NOVELTY = 0.5


async def entity_novelty(
    db: AsyncSession, user_id: int, entities: list[dict],
) -> float:
    """What fraction of entities are unknown to the user?

    Entities not seen in >30 days count as 0.3 novel (stale).
    """
    if not entities:
        return 0.5

    now = datetime.now(UTC)
    new_score = 0.0

    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue

        canonical = slugify(name, separator="-")

        # Look up the global entity
        ent_result = await db.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.canonical_name == canonical)
        )
        global_entity = ent_result.scalar_one_or_none()

        if not global_entity:
            new_score += 1.0
            continue

        # Check user exposure
        exp_result = await db.execute(
            select(UserEntityExposure).where(
                UserEntityExposure.user_id == user_id,
                UserEntityExposure.entity_id == global_entity.id,
            )
        )
        exposure = exp_result.scalar_one_or_none()

        if not exposure:
            new_score += 1.0
        elif exposure.last_exposed_at:
            days_since = (now - exposure.last_exposed_at.replace(tzinfo=UTC)).days
            if days_since > ENTITY_STALE_DAYS:
                new_score += ENTITY_STALE_NOVELTY

    return min(1.0, new_score / max(len(entities), 1))


async def relation_novelty(
    db: AsyncSession, user_id: int, relations: list[dict],
) -> float:
    """What fraction of relationships are unknown to the user?

    Relations not seen in >60 days count as 0.5 novel (stale).
    """
    if not relations:
        return 0.5

    now = datetime.now(UTC)
    new_score = 0.0

    for rel in relations:
        subj = rel.get("subject", "").strip()
        obj = rel.get("object", "").strip()
        predicate = rel.get("predicate", "")
        if not subj or not obj:
            continue

        subj_canonical = slugify(subj, separator="-")
        obj_canonical = slugify(obj, separator="-")

        # Find the global relation
        subj_result = await db.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.canonical_name == subj_canonical)
        )
        subj_entity = subj_result.scalar_one_or_none()

        obj_result = await db.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.canonical_name == obj_canonical)
        )
        obj_entity = obj_result.scalar_one_or_none()

        if not subj_entity or not obj_entity:
            new_score += 1.0
            continue

        rel_result = await db.execute(
            select(KnowledgeRelation).where(
                KnowledgeRelation.source_entity_id == subj_entity.id,
                KnowledgeRelation.target_entity_id == obj_entity.id,
                KnowledgeRelation.relation_type == predicate,
            )
        )
        global_rel = rel_result.scalar_one_or_none()

        if not global_rel:
            new_score += 1.0
            continue

        # Check user exposure to this relation
        exp_result = await db.execute(
            select(UserRelationExposure).where(
                UserRelationExposure.user_id == user_id,
                UserRelationExposure.relation_id == global_rel.id,
            )
        )
        exposure = exp_result.scalar_one_or_none()

        if not exposure:
            new_score += 1.0
        elif exposure.last_exposed_at:
            days_since = (now - exposure.last_exposed_at.replace(tzinfo=UTC)).days
            if days_since > RELATION_STALE_DAYS:
                new_score += RELATION_STALE_NOVELTY

    return min(1.0, new_score / max(len(relations), 1))


async def bridge_novelty(
    db: AsyncSession,
    user_id: int,
    entities: list[dict],
    community_map: dict[int, int] | None = None,
) -> float:
    """Does this message connect entities from different communities?

    Higher score if the user hasn't seen these communities connected before.
    """
    if not entities or not community_map or len(entities) < 2:
        return 0.0

    # Resolve entities to IDs and find their communities
    communities_in_message = set()
    for ent in entities:
        name = ent.get("name", "").strip()
        if not name:
            continue
        canonical = slugify(name, separator="-")
        ent_result = await db.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.canonical_name == canonical)
        )
        global_entity = ent_result.scalar_one_or_none()
        if global_entity and global_entity.id in community_map:
            communities_in_message.add(community_map[global_entity.id])

    if len(communities_in_message) <= 1:
        return 0.0

    # More communities connected = higher bridge score
    # 2 communities = 0.5, 3+ = 1.0
    return min(1.0, (len(communities_in_message) - 1) * 0.5)


async def evolution_novelty(
    db: AsyncSession, user_id: int, relations: list[dict],
) -> float:
    """Does this message update or contradict existing knowledge?

    Detects same subject+predicate with different object.
    """
    if not relations:
        return 0.0

    evolution_count = 0
    for rel in relations:
        subj = rel.get("subject", "").strip()
        predicate = rel.get("predicate", "")
        obj = rel.get("object", "").strip()
        if not subj or not predicate or not obj:
            continue

        subj_canonical = slugify(subj, separator="-")
        obj_canonical = slugify(obj, separator="-")

        # Find subject entity
        subj_result = await db.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.canonical_name == subj_canonical)
        )
        subj_entity = subj_result.scalar_one_or_none()
        if not subj_entity:
            continue

        # Check if there's an existing relation with same subject+predicate but different object
        existing = await db.execute(
            select(KnowledgeRelation).where(
                KnowledgeRelation.source_entity_id == subj_entity.id,
                KnowledgeRelation.relation_type == predicate,
            )
        )
        existing_rels = list(existing.scalars().all())

        for existing_rel in existing_rels:
            target_result = await db.execute(
                select(KnowledgeEntity).where(
                    KnowledgeEntity.id == existing_rel.target_entity_id
                )
            )
            target = target_result.scalar_one_or_none()
            if target and target.canonical_name != obj_canonical:
                # Different object for same subject+predicate = evolution
                evolution_count += 1
                break

    return min(1.0, evolution_count / max(len(relations), 1))


async def narrative_novelty(
    db: AsyncSession, user_id: int, message: Message,
) -> float:
    """Is this message part of a story the user follows but hasn't seen recently?

    Higher score for stories with stale last update.
    """
    from src.knowledge.narratives import get_user_narratives

    if not message.entities_json:
        return 0.0

    msg_entity_names = {
        slugify(e.get("name", ""), separator="-")
        for e in message.entities_json
        if e.get("name")
    }

    if not msg_entity_names:
        return 0.0

    narratives = await get_user_narratives(db, user_id)
    if not narratives:
        return 0.0

    now = datetime.now(UTC)
    best_score = 0.0

    for narrative in narratives:
        narr_entity_ids = narrative.entity_ids_json or []
        if not narr_entity_ids:
            continue

        # Check overlap between message entities and narrative entities
        narr_entities_result = await db.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.id.in_(narr_entity_ids))
        )
        narr_entity_names = {
            e.canonical_name for e in narr_entities_result.scalars().all()
        }

        overlap = msg_entity_names & narr_entity_names
        if len(overlap) < 2:
            continue

        # Score based on how stale the narrative is
        last_update = narrative.last_message_at.replace(tzinfo=UTC)
        days_since = (now - last_update).days
        if days_since <= 1:
            score = 0.1  # very recent update, low novelty
        elif days_since <= 7:
            score = 0.3 + (days_since / 7) * 0.3
        else:
            score = min(1.0, 0.6 + (days_since / 30) * 0.4)

        best_score = max(best_score, score)

    return best_score


async def compute_composite_novelty(
    db: AsyncSession,
    user_id: int,
    message: Message,
    community_map: dict[int, int] | None = None,
) -> float:
    """Compute the 5-dimensional composite novelty score (0.0 to 1.0).

    Dimensions and weights:
    - Entity novelty (0.15): % of entities unknown to user
    - Relation novelty (0.30): % of relations unknown to user
    - Bridge novelty (0.25): connects different communities?
    - Evolution novelty (0.15): updates/contradicts known facts?
    - Narrative novelty (0.15): part of a story user hasn't seen recently?
    """
    extraction = message.relations_json
    entities = extraction.get("entities", []) if extraction else (message.entities_json or [])
    relations = extraction.get("relations", []) if extraction else []

    ent_score = await entity_novelty(db, user_id, entities)
    rel_score = await relation_novelty(db, user_id, relations)
    brg_score = await bridge_novelty(db, user_id, entities, community_map)
    evo_score = await evolution_novelty(db, user_id, relations)
    nar_score = await narrative_novelty(db, user_id, message)

    composite = (
        ENTITY_WEIGHT * ent_score
        + RELATION_WEIGHT * rel_score
        + BRIDGE_WEIGHT * brg_score
        + EVOLUTION_WEIGHT * evo_score
        + NARRATIVE_WEIGHT * nar_score
    )

    log.debug(
        "novelty_computed",
        msg_id=message.id,
        entity=round(ent_score, 3),
        relation=round(rel_score, 3),
        bridge=round(brg_score, 3),
        evolution=round(evo_score, 3),
        narrative=round(nar_score, 3),
        composite=round(composite, 3),
    )

    return min(1.0, composite)
