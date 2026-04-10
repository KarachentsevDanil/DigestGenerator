from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import Message
from src.dedup.minhash import MinHashIndex
from src.dedup.vector_store import SimilarResult, VectorStore
from src.llm.client import OllamaClient
from src.llm.prompts import DEDUP_PROMPT
from src.pipelines.scrape import PipelineRunResult

log = structlog.get_logger()


def _compute_primary_score(msg: Message) -> float:
    """
    Proxy score for cluster primary election (before classification).
    Higher = better representative.
    """
    score = 0.0

    # Original > forward (+0.3)
    if not msg.forwarded_from_channel:
        score += 0.3

    # Longer content (+0.0 to +0.3, normalized)
    content_len = len(msg.content) if msg.content else 0
    score += min(0.3, content_len / 3000.0)

    # Higher views (+0.0 to +0.2)
    views = (msg.raw_metadata or {}).get("views", 0) if msg.raw_metadata else 0
    score += min(0.2, views / 50000.0)

    # Recency (+0.2 for newer messages, based on published_at)
    if msg.published_at:
        delta = datetime.now(UTC) - msg.published_at.replace(tzinfo=UTC)
        age_hours = delta.total_seconds() / 3600
        score += max(0.0, 0.2 - (age_hours / 720.0) * 0.2)  # decays over 30 days

    return score


def _elect_primary(cluster_messages: list[Message]) -> Message:
    """Elect the best representative of a cluster."""
    return max(cluster_messages, key=_compute_primary_score)


async def _slm_confirm_duplicate(
    client: OllamaClient, msg_a: Message, msg_b: Message
) -> bool:
    """Use SLM to confirm if two borderline messages are duplicates."""
    prompt = DEDUP_PROMPT.format(
        message_a=msg_a.content[:2000],
        message_b=msg_b.content[:2000],
    )
    try:
        result = await client.generate_json(prompt, temperature=0.1)
        return result.get("is_duplicate", False) is True
    except Exception:
        log.exception("slm_dedup_confirm_error")
        return False


async def run_deduplicate(db: AsyncSession, settings: Settings) -> PipelineRunResult:
    """
    Three-pass deduplication pipeline.
    Pass 1: MinHash LSH (near-exact text, Jaccard >= threshold)
    Pass 2: Embedding cosine similarity via ChromaDB
    Pass 3: SLM confirmation for borderline cases
    """
    result = PipelineRunResult(stage="deduplicate")
    dedup_cfg = settings.dedup
    window_start = datetime.now(UTC) - timedelta(hours=dedup_cfg.window_hours)

    # Fetch messages to process (status='embedded')
    new_msgs_result = await db.execute(
        select(Message)
        .where(Message.status == "embedded")
        .where(Message.scraped_at >= window_start)
        .order_by(Message.published_at)
    )
    new_messages = list(new_msgs_result.scalars().all())

    if not new_messages:
        log.info("dedup_no_messages_to_process")
        return result

    log.info("dedup_processing", count=len(new_messages))

    # Also fetch recently deduplicated messages for cross-batch matching
    existing_result = await db.execute(
        select(Message)
        .where(Message.status.in_(["deduplicated", "classified"]))
        .where(Message.scraped_at >= window_start)
        .where(Message.is_cluster_primary.is_(True))
    )
    existing_messages = list(existing_result.scalars().all())

    # Build MinHash LSH index from existing messages
    minhash_index = MinHashIndex(
        threshold=dedup_cfg.minhash_threshold,
        num_perm=dedup_cfg.minhash_num_perm,
    )
    minhash_index.build_from_messages(existing_messages)

    # Initialize ChromaDB vector store
    vector_store = VectorStore(persist_dir="data/chromadb")

    # Upsert existing message embeddings into ChromaDB
    for msg in existing_messages:
        embedding = VectorStore.embedding_from_message(msg)
        if embedding:
            vector_store.upsert(str(msg.id), embedding, {"status": msg.status})

    # Track clusters: message_id -> cluster_id
    clusters: dict[str, list[int]] = {}  # cluster_id -> [message_ids]
    msg_to_cluster: dict[int, str] = {}  # message_id -> cluster_id

    # Include existing cluster assignments
    for msg in existing_messages:
        if msg.dedup_cluster_id:
            cid = msg.dedup_cluster_id
            clusters.setdefault(cid, []).append(msg.id)
            msg_to_cluster[msg.id] = cid

    # Messages needing Pass 2 (not caught by MinHash)
    pass2_messages: list[Message] = []
    # Messages needing Pass 3 (borderline cosine)
    borderline_pairs: list[tuple[Message, Message]] = []

    # ── Pass 1: MinHash LSH ──
    for msg in new_messages:
        if not msg.minhash_signature:
            pass2_messages.append(msg)
            continue

        mh = MinHashIndex.deserialize(msg.minhash_signature)
        matches = minhash_index.query(mh)

        # Remove self-match
        matches = [m for m in matches if m != str(msg.id)]

        if matches:
            # Found near-duplicate — assign to existing cluster or create new one
            match_id = int(matches[0])
            if match_id in msg_to_cluster:
                cluster_id = msg_to_cluster[match_id]
            else:
                cluster_id = str(uuid.uuid4())
                clusters.setdefault(cluster_id, []).append(match_id)
                msg_to_cluster[match_id] = cluster_id

            clusters[cluster_id].append(msg.id)
            msg_to_cluster[msg.id] = cluster_id
            log.debug("dedup_minhash_match", msg_id=msg.id, cluster=cluster_id)
        else:
            pass2_messages.append(msg)

        # Insert into index for subsequent messages
        minhash_index.insert(str(msg.id), mh)

    log.info(
        "dedup_pass1_done",
        minhash_matches=len(new_messages) - len(pass2_messages),
        remaining=len(pass2_messages),
    )

    # ── Pass 2: Embedding cosine similarity ──
    pass3_candidates: list[Message] = []

    for msg in pass2_messages:
        embedding = VectorStore.embedding_from_message(msg)
        if not embedding:
            pass3_candidates.append(msg)
            continue

        # Query for similar messages (exclude self)
        similar = vector_store.query_similar(
            embedding,
            n_results=5,
            min_similarity=dedup_cfg.cosine_borderline_threshold,
            exclude_ids=[str(msg.id)],
        )

        definite_match: SimilarResult | None = None
        borderline_match: SimilarResult | None = None

        for s in similar:
            if s.similarity >= dedup_cfg.cosine_definite_threshold:
                definite_match = s
                break
            elif s.similarity >= dedup_cfg.cosine_borderline_threshold:
                borderline_match = s

        if definite_match:
            # Definite duplicate
            match_id = int(definite_match.id)
            if match_id in msg_to_cluster:
                cluster_id = msg_to_cluster[match_id]
            else:
                cluster_id = str(uuid.uuid4())
                clusters.setdefault(cluster_id, []).append(match_id)
                msg_to_cluster[match_id] = cluster_id

            clusters[cluster_id].append(msg.id)
            msg_to_cluster[msg.id] = cluster_id
            log.debug(
                "dedup_cosine_match",
                msg_id=msg.id,
                similarity=definite_match.similarity,
            )
        elif borderline_match:
            # Borderline — needs SLM confirmation
            match_msg_result = await db.execute(
                select(Message).where(Message.id == int(borderline_match.id))
            )
            match_msg = match_msg_result.scalar_one_or_none()
            if match_msg:
                borderline_pairs.append((msg, match_msg))
            else:
                pass3_candidates.append(msg)
        else:
            pass3_candidates.append(msg)

        # Upsert into ChromaDB for subsequent messages
        vector_store.upsert(str(msg.id), embedding, {"status": "embedded"})

    log.info(
        "dedup_pass2_done",
        cosine_matches=len(pass2_messages) - len(pass3_candidates) - len(borderline_pairs),
        borderline=len(borderline_pairs),
        unique=len(pass3_candidates),
    )

    # ── Pass 3: SLM confirmation for borderline pairs ──
    if borderline_pairs:
        ollama_client = OllamaClient(settings.ollama)
        for msg, match_msg in borderline_pairs:
            try:
                is_dup = await _slm_confirm_duplicate(ollama_client, msg, match_msg)
                if is_dup:
                    if match_msg.id in msg_to_cluster:
                        cluster_id = msg_to_cluster[match_msg.id]
                    else:
                        cluster_id = str(uuid.uuid4())
                        clusters.setdefault(cluster_id, []).append(match_msg.id)
                        msg_to_cluster[match_msg.id] = cluster_id

                    clusters[cluster_id].append(msg.id)
                    msg_to_cluster[msg.id] = cluster_id
                    log.debug("dedup_slm_confirmed", msg_id=msg.id)
                else:
                    pass3_candidates.append(msg)
                    log.debug("dedup_slm_rejected", msg_id=msg.id)
            except Exception:
                log.exception("dedup_slm_error", msg_id=msg.id)
                pass3_candidates.append(msg)

    log.info("dedup_pass3_done", unique_remaining=len(pass3_candidates))

    # ── Update DB: assign clusters, elect primaries ──
    # Process clustered messages
    for cluster_id, member_ids in clusters.items():
        # Only process clusters that contain new messages
        new_member_ids = [mid for mid in member_ids if any(m.id == mid for m in new_messages)]
        if not new_member_ids:
            continue

        # Load all cluster members
        all_member_result = await db.execute(
            select(Message).where(Message.id.in_(member_ids))
        )
        all_members = list(all_member_result.scalars().all())

        # Elect primary
        primary = _elect_primary(all_members)

        for member in all_members:
            member.dedup_cluster_id = cluster_id
            if member.id == primary.id:
                member.is_cluster_primary = True
                if member.status == "embedded":
                    member.status = "deduplicated"
                    member.deduplicated_at = datetime.now(UTC)
                    result.processed += 1
            else:
                member.is_cluster_primary = False
                if member.status == "embedded":
                    member.status = "skipped_duplicate"
                    member.deduplicated_at = datetime.now(UTC)
                    result.skipped += 1

    # Process unique messages (not in any cluster)
    for msg in pass3_candidates:
        if msg.id not in msg_to_cluster:
            msg.status = "deduplicated"
            msg.is_cluster_primary = True
            msg.deduplicated_at = datetime.now(UTC)
            result.processed += 1

    await db.commit()
    log.info(
        "dedup_complete",
        processed=result.processed,
        skipped=result.skipped,
        failed=result.failed,
    )

    return result
