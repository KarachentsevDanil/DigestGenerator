from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime, timedelta

import structlog
from slugify import slugify
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    KnowledgeEntity,
    KnowledgeNarrative,
    Message,
    UserEntityExposure,
)

log = structlog.get_logger()

# Narrative detection parameters
MIN_SHARED_ENTITIES = 3
MIN_MESSAGES_FOR_NARRATIVE = 3
NARRATIVE_WINDOW_DAYS = 14

# Lifecycle thresholds
STALE_DAYS = 7
CONCLUDED_DAYS = 30


async def detect_narratives(db: AsyncSession) -> int:
    """Find message clusters that form evolving stories.

    Groups messages sharing 3+ entities within a 14-day rolling window.
    Creates KnowledgeNarrative records for groups with 3+ messages.
    Returns count of new narratives detected.
    """
    now = datetime.now(UTC)
    window_start = now - timedelta(days=NARRATIVE_WINDOW_DAYS)

    # Fetch recent messages with extracted knowledge
    result = await db.execute(
        select(Message)
        .where(
            Message.status.in_(["knowledge_extracted", "classified"]),
            Message.published_at >= window_start,
            Message.entities_json.isnot(None),
        )
        .order_by(Message.published_at)
    )
    messages = list(result.scalars().all())

    if len(messages) < MIN_MESSAGES_FOR_NARRATIVE:
        return 0

    # Build entity -> message mapping
    entity_to_messages: dict[str, list[int]] = defaultdict(list)
    message_entities: dict[int, set[str]] = {}

    for msg in messages:
        entities = set()
        for ent in (msg.entities_json or []):
            name = ent.get("name", "").strip()
            if name:
                canonical = slugify(name, separator="-")
                entities.add(canonical)
                entity_to_messages[canonical].append(msg.id)
        message_entities[msg.id] = entities

    # Find message groups sharing 3+ entities (greedy clustering)
    processed_messages: set[int] = set()
    narrative_groups: list[tuple[set[str], list[int]]] = []

    for msg in messages:
        if msg.id in processed_messages:
            continue

        my_entities = message_entities.get(msg.id, set())
        if len(my_entities) < MIN_SHARED_ENTITIES:
            continue

        # Find other messages sharing 3+ entities with this one
        cluster_msgs = [msg.id]
        cluster_entities = set(my_entities)

        for other_msg in messages:
            if other_msg.id == msg.id or other_msg.id in processed_messages:
                continue
            other_entities = message_entities.get(other_msg.id, set())
            shared = my_entities & other_entities
            if len(shared) >= MIN_SHARED_ENTITIES:
                cluster_msgs.append(other_msg.id)
                cluster_entities |= other_entities

        if len(cluster_msgs) >= MIN_MESSAGES_FOR_NARRATIVE:
            narrative_groups.append((cluster_entities, cluster_msgs))
            processed_messages.update(cluster_msgs)

    # Create narratives for new groups
    new_count = 0
    for entities, msg_ids in narrative_groups:
        # Check if a narrative with similar entities already exists
        existing = await _find_matching_narrative(db, entities)
        if existing:
            # Update existing narrative
            existing.last_message_at = now
            existing.message_count = max(existing.message_count, len(msg_ids))
            existing.status = "active"
            continue

        # Resolve entity names to IDs
        entity_ids = []
        for canonical in entities:
            ent_result = await db.execute(
                select(KnowledgeEntity).where(
                    KnowledgeEntity.canonical_name == canonical
                )
            )
            ent = ent_result.scalar_one_or_none()
            if ent:
                entity_ids.append(ent.id)

        if len(entity_ids) < MIN_SHARED_ENTITIES:
            continue

        # Generate a simple title from entity names
        title = await _generate_title(db, entity_ids)

        narrative = KnowledgeNarrative(
            title=title,
            entity_ids_json=entity_ids,
            first_message_at=now - timedelta(days=NARRATIVE_WINDOW_DAYS),
            last_message_at=now,
            message_count=len(msg_ids),
            status="active",
        )
        db.add(narrative)
        new_count += 1

    await db.commit()
    log.info("narratives_detected", new=new_count, total_groups=len(narrative_groups))
    return new_count


async def _find_matching_narrative(
    db: AsyncSession, entity_canonicals: set[str],
) -> KnowledgeNarrative | None:
    """Find an existing narrative that overlaps significantly with given entities."""
    result = await db.execute(
        select(KnowledgeNarrative).where(
            KnowledgeNarrative.status.in_(["active", "stale"])
        )
    )
    narratives = list(result.scalars().all())

    for narrative in narratives:
        narr_entity_ids = narrative.entity_ids_json or []
        if not narr_entity_ids:
            continue

        # Get canonical names for narrative entities
        ent_result = await db.execute(
            select(KnowledgeEntity).where(KnowledgeEntity.id.in_(narr_entity_ids))
        )
        narr_canonicals = {e.canonical_name for e in ent_result.scalars().all()}

        overlap = entity_canonicals & narr_canonicals
        if len(overlap) >= MIN_SHARED_ENTITIES:
            return narrative

    return None


async def _generate_title(db: AsyncSession, entity_ids: list[int]) -> str:
    """Generate a narrative title from top entity names."""
    result = await db.execute(
        select(KnowledgeEntity).where(KnowledgeEntity.id.in_(entity_ids[:5]))
    )
    entities = list(result.scalars().all())
    names = [e.display_name for e in entities[:3]]
    return " / ".join(names) if names else "Untitled Narrative"


async def update_narrative_lifecycle(db: AsyncSession) -> dict[str, int]:
    """Update status of existing narratives based on age.

    - active: received message in last 7 days
    - stale: no message in 7-30 days
    - concluded: no message in 30+ days

    Returns counts of transitions.
    """
    now = datetime.now(UTC)
    stale_cutoff = now - timedelta(days=STALE_DAYS)
    concluded_cutoff = now - timedelta(days=CONCLUDED_DAYS)

    result = await db.execute(
        select(KnowledgeNarrative).where(
            KnowledgeNarrative.status.in_(["active", "stale"])
        )
    )
    narratives = list(result.scalars().all())

    transitions = {"to_stale": 0, "to_concluded": 0}

    for narrative in narratives:
        last_msg = narrative.last_message_at.replace(tzinfo=UTC)

        if last_msg < concluded_cutoff and narrative.status != "concluded":
            narrative.status = "concluded"
            transitions["to_concluded"] += 1
        elif last_msg < stale_cutoff and narrative.status == "active":
            narrative.status = "stale"
            transitions["to_stale"] += 1

    await db.commit()
    log.info("narrative_lifecycle_updated", **transitions)
    return transitions


async def get_user_narratives(
    db: AsyncSession, user_id: int,
) -> list[KnowledgeNarrative]:
    """Get narratives the user is implicitly following via entity exposure overlap."""
    # Get entity IDs the user has been exposed to
    exp_result = await db.execute(
        select(UserEntityExposure.entity_id).where(
            UserEntityExposure.user_id == user_id
        )
    )
    user_entity_ids = set(row[0] for row in exp_result.all())

    if not user_entity_ids:
        return []

    # Find active/stale narratives
    narr_result = await db.execute(
        select(KnowledgeNarrative).where(
            KnowledgeNarrative.status.in_(["active", "stale"])
        )
    )
    narratives = list(narr_result.scalars().all())

    # Filter to narratives with entity overlap
    user_narratives = []
    for narrative in narratives:
        narr_entity_ids = set(narrative.entity_ids_json or [])
        overlap = user_entity_ids & narr_entity_ids
        if len(overlap) >= 2:  # user knows at least 2 entities in the narrative
            user_narratives.append(narrative)

    return user_narratives
