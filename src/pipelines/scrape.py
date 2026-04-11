from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache

import numpy as np
import structlog
from datasketch import MinHash
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.connectors.base import NormalizedMessage
from src.connectors.telegram import TelegramConnector
from src.db.models import Message, Source

log = structlog.get_logger()


@dataclass
class PipelineRunResult:
    stage: str
    processed: int = 0
    failed: int = 0
    skipped: int = 0


@lru_cache(maxsize=1)
def _get_embedding_model(model_name: str):
    """Lazy-load sentence-transformers model (cached)."""
    from sentence_transformers import SentenceTransformer

    log.info("loading_embedding_model", model=model_name)
    return SentenceTransformer(model_name)


def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> bytes:
    """Generate a 384-dim float32 embedding and return as bytes."""
    model = _get_embedding_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _word_ngrams(text: str, n: int = 3) -> list[str]:
    """Generate word n-gram shingles from text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return [" ".join(words)] if words else [""]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def generate_minhash(text: str, num_perm: int = 128) -> bytes:
    """Generate a MinHash signature from text using word 3-gram shingles."""
    mh = MinHash(num_perm=num_perm)
    for shingle in _word_ngrams(text, n=3):
        mh.update(shingle.encode("utf-8"))
    return pickle.dumps(mh)


async def _check_forward_duplicate(
    db: AsyncSession,
    msg: NormalizedMessage,
) -> str | None:
    """
    If a message is forwarded from a known source, check if the original exists.
    Returns the original message's dedup_cluster_id if found, else None.
    """
    if not msg.forwarded_from_channel or not msg.forwarded_message_id:
        return None

    # Find the source matching the forwarded channel
    fwd_channel = msg.forwarded_from_channel.lstrip("@")
    result = await db.execute(
        select(Source).where(
            Source.source_identifier.in_([
                msg.forwarded_from_channel,
                f"@{fwd_channel}",
                fwd_channel,
            ])
        )
    )
    fwd_source = result.scalar_one_or_none()
    if not fwd_source:
        return None

    # Look up the original message
    result = await db.execute(
        select(Message).where(
            Message.source_id == fwd_source.id,
            Message.external_id == msg.forwarded_message_id,
        )
    )
    original = result.scalar_one_or_none()
    if not original:
        return None

    return original.dedup_cluster_id or f"fwd-{original.id}"


async def run_scrape(db: AsyncSession, settings: Settings) -> PipelineRunResult:
    """
    Scrape all active sources, generate embeddings and MinHash, store in DB.

    For each active source:
    1. Fetch new messages via connector
    2. Forward-based instant dedup (Pass 0)
    3. Generate embedding + MinHash for non-duplicate messages
    4. Insert into DB with status='embedded'
    5. Update source cursor
    """
    result = PipelineRunResult(stage="scrape")

    # Get all active sources
    sources_result = await db.execute(
        select(Source).where(Source.is_active.is_(True))
    )
    sources = list(sources_result.scalars().all())

    if not sources:
        log.info("scrape_no_sources")
        return result

    connector = TelegramConnector(settings)
    try:
        for source in sources:
            log.info("scraping_source", source=source.source_identifier)

            messages = await connector.fetch_new(source)
            if not messages:
                log.info("scrape_no_new_messages", source=source.source_identifier)
                continue

            highest_id = source.last_scraped_external_id
            for msg in messages:
                try:
                    # Track highest message ID for cursor update
                    if highest_id is None or int(msg.external_id) > int(highest_id):
                        highest_id = msg.external_id

                    # Forward-based instant dedup (Pass 0)
                    fwd_cluster_id = await _check_forward_duplicate(db, msg)
                    if fwd_cluster_id:
                        stmt = sqlite_insert(Message).values(
                            source_id=source.id,
                            external_id=msg.external_id,
                            content=msg.content,
                            content_url=msg.content_url,
                            media_type=msg.media_type,
                            published_at=msg.published_at,
                            scraped_at=datetime.now(UTC),
                            status="skipped_duplicate",
                            is_cluster_primary=False,
                            dedup_cluster_id=fwd_cluster_id,
                            forwarded_from_channel=msg.forwarded_from_channel,
                            forwarded_message_id=msg.forwarded_message_id,
                            raw_metadata=msg.raw_metadata,
                        ).on_conflict_do_nothing(
                            index_elements=["source_id", "external_id"]
                        )
                        await db.execute(stmt)
                        result.skipped += 1
                        continue

                    # Generate embedding and MinHash
                    embedding = generate_embedding(
                        msg.content, settings.embeddings.model
                    )
                    minhash = generate_minhash(
                        msg.content, settings.dedup.minhash_num_perm
                    )

                    stmt = sqlite_insert(Message).values(
                        source_id=source.id,
                        external_id=msg.external_id,
                        content=msg.content,
                        content_url=msg.content_url,
                        media_type=msg.media_type,
                        published_at=msg.published_at,
                        scraped_at=datetime.now(UTC),
                        status="embedded",
                        embedding_vector=embedding,
                        minhash_signature=minhash,
                        forwarded_from_channel=msg.forwarded_from_channel,
                        forwarded_message_id=msg.forwarded_message_id,
                        raw_metadata=msg.raw_metadata,
                    ).on_conflict_do_nothing(
                        index_elements=["source_id", "external_id"]
                    )
                    await db.execute(stmt)
                    result.processed += 1

                except Exception:
                    log.exception(
                        "scrape_message_error",
                        source=source.source_identifier,
                        external_id=msg.external_id,
                    )
                    result.failed += 1

            # Update source cursor
            if highest_id:
                source.last_scraped_external_id = highest_id
                source.last_scraped_at = datetime.now(UTC)

            await db.commit()
            log.info(
                "scrape_source_done",
                source=source.source_identifier,
                processed=result.processed,
                skipped=result.skipped,
            )

    finally:
        await connector.disconnect()

    return result
