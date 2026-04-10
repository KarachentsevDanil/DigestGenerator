from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import Category, Message
from src.llm.client import OllamaClient
from src.llm.prompts import CLASSIFY_PROMPT
from src.pipelines.scrape import PipelineRunResult

log = structlog.get_logger()


def _build_categories_block(categories: list[Category]) -> str:
    """Build the categories description block for the classification prompt."""
    lines = []
    for cat in categories:
        lines.append(f"- {cat.name}: {cat.description}")
    return "\n".join(lines)


def _validate_classification(result: dict) -> bool:
    """Validate the structure of SLM classification output."""
    if not isinstance(result.get("categories"), dict):
        return False
    if not isinstance(result.get("relevance"), (int, float)):
        return False
    if not isinstance(result.get("summary"), str):
        return False
    if not isinstance(result.get("entities"), list):
        return False
    return True


def _normalize_classification(result: dict) -> dict:
    """Normalize classification values to expected ranges."""
    # Clamp relevance to 0.0-1.0
    relevance = float(result.get("relevance", 0.0))
    result["relevance"] = max(0.0, min(1.0, relevance))

    # Clamp category scores to 0.0-1.0
    categories = result.get("categories", {})
    result["categories"] = {
        k: max(0.0, min(1.0, float(v)))
        for k, v in categories.items()
        if isinstance(v, (int, float))
    }

    # Ensure entities is a list of dicts
    entities = result.get("entities", [])
    result["entities"] = [
        e for e in entities
        if isinstance(e, dict) and "name" in e
    ]

    return result


async def run_classify(db: AsyncSession, settings: Settings) -> PipelineRunResult:
    """
    Classify all deduplicated cluster-primary messages.
    Only messages with status='deduplicated' AND is_cluster_primary=True are processed.
    """
    result = PipelineRunResult(stage="classify")

    # Fetch messages to classify
    msgs_result = await db.execute(
        select(Message)
        .where(Message.status == "deduplicated")
        .where(Message.is_cluster_primary.is_(True))
        .order_by(Message.published_at)
    )
    messages = list(msgs_result.scalars().all())

    if not messages:
        log.info("classify_no_messages")
        return result

    # Build categories block from global registry
    cats_result = await db.execute(select(Category).order_by(Category.name))
    categories = list(cats_result.scalars().all())

    if not categories:
        log.warning("classify_no_categories")
        # Mark all as failed since we can't classify without categories
        for msg in messages:
            msg.status = "classify_failed"
            result.failed += 1
        await db.commit()
        return result

    categories_block = _build_categories_block(categories)

    log.info("classify_processing", count=len(messages), categories=len(categories))

    ollama_client = OllamaClient(settings.ollama)

    for i, msg in enumerate(messages):
        try:
            prompt = CLASSIFY_PROMPT.format(
                categories_block=categories_block,
                message_content=msg.content[:4000],  # truncate very long messages
            )

            classification = await ollama_client.generate_json(prompt, temperature=0.1)

            if not _validate_classification(classification):
                log.warning("classify_invalid_output", msg_id=msg.id)
                msg.status = "classify_failed"
                result.failed += 1
                continue

            classification = _normalize_classification(classification)

            msg.category_scores_json = classification["categories"]
            msg.relevance_score = classification["relevance"]
            msg.summary = classification["summary"]
            msg.entities_json = classification["entities"]
            msg.classified_at = datetime.now(UTC)
            msg.status = "classified"
            result.processed += 1

            if (i + 1) % 10 == 0:
                log.info("classify_progress", done=i + 1, total=len(messages))

        except Exception:
            log.exception("classify_error", msg_id=msg.id)
            msg.status = "classify_failed"
            result.failed += 1

    await db.commit()
    log.info(
        "classify_complete",
        processed=result.processed,
        failed=result.failed,
    )

    return result
