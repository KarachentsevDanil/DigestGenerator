from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import Message
from src.llm.client import OllamaClient

log = structlog.get_logger()

VALID_ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "MODEL", "TECHNOLOGY", "PRODUCT",
    "EVENT", "REGULATION", "LOCATION", "CONCEPT", "DATASET",
}

VALID_RELATION_TYPES = {
    "CREATED_BY", "WORKS_AT", "SUBSIDIARY_OF", "HEADQUARTERED_IN",
    "MEMBER_OF", "FUNDED_BY", "COMPETES_WITH", "OUTPERFORMS",
    "BASED_ON", "SUCCESSOR_OF", "ALTERNATIVE_TO", "ANNOUNCED",
    "ACQUIRED", "LAUNCHED", "PARTNERED_WITH", "INVESTED_IN",
    "SUPPORTS", "OPPOSES", "REGULATES", "RELATED_TO",
}


@dataclass
class PipelineRunResult:
    stage: str
    processed: int = 0
    failed: int = 0
    skipped: int = 0


def _validate_extraction(result: dict) -> bool:
    """Validate the structure of SLM knowledge extraction output."""
    if not isinstance(result.get("entities"), list):
        return False
    if not isinstance(result.get("relations"), list):
        return False
    return True


def _normalize_extraction(result: dict) -> dict:
    """Normalize and filter extraction output to valid schema types."""
    # Filter entities to valid types
    entities = []
    for e in result.get("entities", []):
        if not isinstance(e, dict):
            continue
        if "name" not in e or "type" not in e:
            continue
        entity_type = str(e["type"]).upper()
        if entity_type in VALID_ENTITY_TYPES:
            entities.append({"name": str(e["name"]), "type": entity_type})

    # Filter relations to valid predicates and clamp confidence
    relations = []
    for r in result.get("relations", []):
        if not isinstance(r, dict):
            continue
        required = {"subject", "predicate", "object"}
        if not required.issubset(r.keys()):
            continue
        predicate = str(r["predicate"]).upper()
        if predicate not in VALID_RELATION_TYPES:
            continue
        confidence = float(r.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        relations.append({
            "subject": str(r["subject"]),
            "predicate": predicate,
            "object": str(r["object"]),
            "confidence": confidence,
        })

    return {"entities": entities, "relations": relations}


async def run_extract_knowledge(
    db: AsyncSession, settings: Settings,
) -> PipelineRunResult:
    """
    Process messages with status='classified'.
    For each message:
    1. Call SLM with EXTRACT_KNOWLEDGE_PROMPT
    2. Parse response -> store in message.relations_json
    3. Update status to 'knowledge_extracted'
    On failure after retry: status = 'knowledge_extract_failed'
    """
    result = PipelineRunResult(stage="extract_knowledge")

    # Fetch classified messages
    msgs_result = await db.execute(
        select(Message)
        .where(Message.status == "classified")
        .order_by(Message.published_at)
    )
    messages = list(msgs_result.scalars().all())

    if not messages:
        log.info("extract_knowledge_no_messages")
        return result

    log.info("extract_knowledge_processing", count=len(messages))

    ollama_client = OllamaClient(settings.ollama)

    for i, msg in enumerate(messages):
        try:
            extraction = await ollama_client.extract_knowledge(
                msg.content[:4000]
            )

            if not _validate_extraction(extraction):
                log.warning("extract_knowledge_invalid_output", msg_id=msg.id)
                msg.status = "knowledge_extract_failed"
                result.failed += 1
                continue

            extraction = _normalize_extraction(extraction)

            msg.relations_json = extraction
            msg.status = "knowledge_extracted"
            result.processed += 1

            if (i + 1) % 10 == 0:
                log.info(
                    "extract_knowledge_progress",
                    done=i + 1,
                    total=len(messages),
                )

        except Exception:
            log.exception("extract_knowledge_error", msg_id=msg.id)
            msg.status = "knowledge_extract_failed"
            result.failed += 1

    await db.commit()
    log.info(
        "extract_knowledge_complete",
        processed=result.processed,
        failed=result.failed,
    )

    return result
