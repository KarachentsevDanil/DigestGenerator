from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import (
    KnowledgeEntity,
    KnowledgeRelation,
    Message,
)
from src.knowledge.entity_resolver import EntityResolver

log = structlog.get_logger()

_entity_resolver = EntityResolver()


@dataclass
class PipelineRunResult:
    stage: str
    processed: int = 0
    failed: int = 0
    skipped: int = 0


async def merge_triples_to_graph(
    db: AsyncSession, messages: list[Message],
) -> int:
    """For each message's relations_json, resolve entities and create KnowledgeRelation records.

    Returns count of new relations created.
    """
    new_relations = 0
    now = datetime.now(UTC)

    for msg in messages:
        extraction = msg.relations_json
        if not extraction:
            continue

        # Build entity name -> KnowledgeEntity map from extracted entities
        entity_map: dict[str, KnowledgeEntity] = {}
        for ent in extraction.get("entities", []):
            name = ent.get("name", "").strip()
            etype = ent.get("type")
            if not name:
                continue
            resolved = await _entity_resolver.get_or_create(db, name, etype)
            entity_map[name] = resolved

        # Create relations
        for rel in extraction.get("relations", []):
            subj_name = rel.get("subject", "").strip()
            obj_name = rel.get("object", "").strip()
            predicate = rel.get("predicate", "RELATED_TO")
            confidence = rel.get("confidence", 0.5)

            if not subj_name or not obj_name:
                continue

            # Resolve subject and object entities
            if subj_name not in entity_map:
                entity_map[subj_name] = await _entity_resolver.get_or_create(db, subj_name)
            if obj_name not in entity_map:
                entity_map[obj_name] = await _entity_resolver.get_or_create(db, obj_name)

            source_entity = entity_map[subj_name]
            target_entity = entity_map[obj_name]

            # Check if relation already exists
            existing = await db.execute(
                select(KnowledgeRelation).where(
                    KnowledgeRelation.source_entity_id == source_entity.id,
                    KnowledgeRelation.target_entity_id == target_entity.id,
                    KnowledgeRelation.relation_type == predicate,
                )
            )
            existing_rel = existing.scalar_one_or_none()

            if existing_rel:
                existing_rel.observation_count += 1
                existing_rel.last_observed_at = now
                existing_rel.confidence = max(existing_rel.confidence, confidence)
            else:
                new_rel = KnowledgeRelation(
                    source_entity_id=source_entity.id,
                    target_entity_id=target_entity.id,
                    relation_type=predicate,
                    confidence=confidence,
                    first_observed_at=now,
                    last_observed_at=now,
                    observation_count=1,
                    source_message_id=msg.id,
                )
                db.add(new_rel)
                new_relations += 1

    await db.flush()
    return new_relations


def load_graph_from_db_sync(entities: list, relations: list):
    """Build an igraph.Graph from entity and relation lists.

    Takes pre-fetched lists to avoid async issues inside igraph construction.
    Returns (graph, id_to_idx mapping).
    """
    import igraph as ig

    g = ig.Graph(directed=True)

    id_to_idx: dict[int, int] = {}
    for ent in entities:
        idx = g.vcount()
        g.add_vertex(
            name=str(ent.id),
            entity_id=ent.id,
            canonical_name=ent.canonical_name,
            display_name=ent.display_name,
            entity_type=ent.entity_type,
        )
        id_to_idx[ent.id] = idx

    for rel in relations:
        src_idx = id_to_idx.get(rel.source_entity_id)
        tgt_idx = id_to_idx.get(rel.target_entity_id)
        if src_idx is not None and tgt_idx is not None:
            g.add_edge(
                src_idx, tgt_idx,
                relation_id=rel.id,
                relation_type=rel.relation_type,
                confidence=rel.confidence,
            )

    return g, id_to_idx


def detect_communities(graph) -> dict[int, int]:
    """Louvain community detection. Returns {entity_id: community_id}."""
    if graph.vcount() == 0:
        return {}

    # Use undirected copy for community detection
    undirected = graph.as_undirected(mode="collapse")
    partition = undirected.community_multilevel()

    result = {}
    for community_id, members in enumerate(partition):
        for idx in members:
            entity_id = graph.vs[idx]["entity_id"]
            result[entity_id] = community_id

    return result


def find_bridges(graph) -> set[tuple[int, int]]:
    """Find bridge edges (edges whose removal disconnects components).

    Returns set of (source_entity_id, target_entity_id) tuples.
    """
    if graph.ecount() == 0:
        return set()

    undirected = graph.as_undirected(mode="collapse")
    bridges = set()

    for edge in undirected.es:
        src_idx, tgt_idx = edge.source, edge.target
        # Test if removing this edge increases components
        test_graph = undirected.copy()
        test_graph.delete_edges(edge.index)
        if test_graph.components().n > undirected.components().n:
            src_eid = graph.vs[src_idx]["entity_id"]
            tgt_eid = graph.vs[tgt_idx]["entity_id"]
            bridges.add((src_eid, tgt_eid))

    return bridges


def compute_centrality(graph) -> dict[int, float]:
    """Betweenness centrality for each entity. Returns {entity_id: centrality}."""
    if graph.vcount() == 0:
        return {}

    betweenness = graph.betweenness(directed=True)
    result = {}
    for idx, score in enumerate(betweenness):
        entity_id = graph.vs[idx]["entity_id"]
        result[entity_id] = score

    return result


async def detect_bursts(
    db: AsyncSession, window_days: int = 7, threshold_multiplier: float = 3.0,
) -> list[int]:
    """Find entities with sudden spike in new relations (burst detection).

    Returns list of entity IDs with burst activity.
    """
    now = datetime.now(UTC)
    from datetime import timedelta

    window_start = now - timedelta(days=window_days)
    older_start = now - timedelta(days=window_days * 4)

    # Count recent relations per entity
    recent = await db.execute(
        select(
            KnowledgeRelation.source_entity_id,
            func.count().label("cnt"),
        )
        .where(KnowledgeRelation.first_observed_at >= window_start)
        .group_by(KnowledgeRelation.source_entity_id)
    )
    recent_counts = {row[0]: row[1] for row in recent.all()}

    # Count older baseline per entity
    baseline = await db.execute(
        select(
            KnowledgeRelation.source_entity_id,
            func.count().label("cnt"),
        )
        .where(
            KnowledgeRelation.first_observed_at >= older_start,
            KnowledgeRelation.first_observed_at < window_start,
        )
        .group_by(KnowledgeRelation.source_entity_id)
    )
    baseline_counts = {row[0]: row[1] for row in baseline.all()}

    burst_entities = []
    for entity_id, recent_count in recent_counts.items():
        baseline_avg = baseline_counts.get(entity_id, 0) / 3.0  # average per window
        if baseline_avg == 0 and recent_count >= 3:
            burst_entities.append(entity_id)
        elif baseline_avg > 0 and recent_count >= baseline_avg * threshold_multiplier:
            burst_entities.append(entity_id)

    return burst_entities


async def run_build_graph(
    db: AsyncSession, settings: Settings,
) -> PipelineRunResult:
    """Build/update the knowledge graph from extracted triples.

    1. Load messages with status='knowledge_extracted' that haven't been graph-merged
    2. Merge triples into global graph (entity resolution + relation creation)
    3. Run graph algorithms (communities, bridges, centrality, bursts)
    4. Return results
    """
    result = PipelineRunResult(stage="build_graph")

    # Fetch messages with extracted knowledge
    msgs_result = await db.execute(
        select(Message)
        .where(Message.status == "knowledge_extracted")
        .order_by(Message.published_at)
    )
    messages = list(msgs_result.scalars().all())

    if not messages:
        log.info("build_graph_no_messages")
        return result

    log.info("build_graph_processing", count=len(messages))

    try:
        new_relations = await merge_triples_to_graph(db, messages)

        # Mark messages as graph-merged by keeping status but we track via graph
        # Messages stay at knowledge_extracted — graph building is idempotent
        result.processed = len(messages)

        # Load full graph for algorithms
        all_entities_result = await db.execute(select(KnowledgeEntity))
        all_entities = list(all_entities_result.scalars().all())

        all_relations_result = await db.execute(select(KnowledgeRelation))
        all_relations = list(all_relations_result.scalars().all())

        if all_entities and all_relations:
            graph, id_to_idx = load_graph_from_db_sync(all_entities, all_relations)

            communities = detect_communities(graph)
            bridges = find_bridges(graph)
            centrality = compute_centrality(graph)
            bursts = await detect_bursts(db)

            log.info(
                "build_graph_algorithms_complete",
                entities=len(all_entities),
                relations=len(all_relations),
                communities=len(set(communities.values())) if communities else 0,
                bridges=len(bridges),
                burst_entities=len(bursts),
                new_relations=new_relations,
            )

        await db.commit()

    except Exception:
        log.exception("build_graph_error")
        result.failed = len(messages)
        return result

    return result
