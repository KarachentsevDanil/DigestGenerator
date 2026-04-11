from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings
from src.db.models import Message, PipelineRun
from dataclasses import dataclass


@dataclass
class PipelineRunResult:
    stage: str
    processed: int = 0
    failed: int = 0
    skipped: int = 0

log = structlog.get_logger()

# Map failed statuses back to their previous status for retry
_RETRY_STATUS_MAP = {
    "embed_failed": "unprocessed",
    "dedup_failed": "embedded",
    "classify_failed": "deduplicated",
    "knowledge_extract_failed": "classified",
}


@asynccontextmanager
async def track_pipeline_run(db: AsyncSession, stage: str):
    """
    Context manager that creates a PipelineRun record, tracks timing,
    and updates status on completion or failure.
    """
    run = PipelineRun(
        stage=stage,
        started_at=datetime.now(UTC),
        status="running",
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    result = PipelineRunResult(stage=stage)

    try:
        yield result
        run.status = "completed"
        run.processed_count = result.processed
        run.failed_count = result.failed
        run.skipped_count = result.skipped
    except Exception as e:
        run.status = "failed"
        run.error_detail = str(e)[:2000]
        run.processed_count = result.processed
        run.failed_count = result.failed
        run.skipped_count = result.skipped
        raise
    finally:
        run.finished_at = datetime.now(UTC)
        await db.commit()


async def run_all_stages(
    db: AsyncSession, settings: Settings
) -> list[PipelineRunResult]:
    """
    Run scrape -> deduplicate -> classify sequentially.
    Each stage tracked independently. Stops on stage failure.

    Cron usage:
      # Scrape + dedup + classify every 4 hours
      0 */4 * * *  curl -sf -X POST http://localhost:8000/pipeline/run-all
    """
    from src.knowledge.extractor import run_extract_knowledge
    from src.knowledge.graph_builder import run_build_graph
    from src.pipelines.classify import run_classify
    from src.pipelines.deduplicate import run_deduplicate
    from src.pipelines.scrape import run_scrape

    results: list[PipelineRunResult] = []

    stages = [
        ("scrape", run_scrape),
        ("deduplicate", run_deduplicate),
        ("classify", run_classify),
        ("extract_knowledge", run_extract_knowledge),
        ("build_graph", run_build_graph),
    ]

    for stage_name, stage_fn in stages:
        try:
            async with track_pipeline_run(db, stage_name) as tracked:
                result = await stage_fn(db, settings)
                tracked.processed = result.processed
                tracked.failed = result.failed
                tracked.skipped = result.skipped
                results.append(result)

            log.info(
                "pipeline_stage_done",
                stage=stage_name,
                processed=result.processed,
                failed=result.failed,
            )

        except Exception:
            log.exception("pipeline_stage_failed", stage=stage_name)
            results.append(PipelineRunResult(stage=stage_name))
            break  # Stop on failure

    return results


async def retry_failed(db: AsyncSession) -> dict[str, int]:
    """
    Reset all *_failed messages to their previous status for reprocessing.
    Returns count of messages reset per failed status.
    """
    counts: dict[str, int] = {}

    for failed_status, target_status in _RETRY_STATUS_MAP.items():
        stmt = (
            update(Message)
            .where(Message.status == failed_status)
            .values(status=target_status)
        )
        result = await db.execute(stmt)
        count = result.rowcount
        if count > 0:
            counts[failed_status] = count
            log.info(
                "retry_reset",
                from_status=failed_status,
                to_status=target_status,
                count=count,
            )

    await db.commit()
    return counts
