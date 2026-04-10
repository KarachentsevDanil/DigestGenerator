from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings, get_settings
from src.dependencies import get_app_settings, get_db


class PipelineResponse(BaseModel):
    run_id: int | None = None
    stage: str
    processed: int
    failed: int
    skipped: int
    duration_seconds: float


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    log = structlog.get_logger()
    settings = get_settings()
    log.info(
        "starting_up",
        database_url=settings.database.url,
        ollama_model=settings.ollama.model,
    )
    yield
    log.info("shutting_down")


app = FastAPI(
    title="Smart Digest Generator",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """System health check: verifies database connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    status = "ok" if db_status == "ok" else "degraded"
    return {"status": status, "database": db_status}


@app.get("/stats")
async def stats(settings=Depends(get_app_settings)):
    """Pipeline statistics. Placeholder — implemented in Phase 6."""
    return {"message": "not implemented"}


@app.post("/scrape", response_model=PipelineResponse)
async def scrape(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """Run the scrape pipeline: fetch messages, embed, store."""
    from src.pipelines.scrape import run_scrape

    start = time.time()
    result = await run_scrape(db, settings)
    duration = time.time() - start

    return PipelineResponse(
        stage=result.stage,
        processed=result.processed,
        failed=result.failed,
        skipped=result.skipped,
        duration_seconds=round(duration, 2),
    )


@app.post("/deduplicate", response_model=PipelineResponse)
async def deduplicate(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """Run the deduplication pipeline: MinHash LSH + cosine + SLM."""
    from src.pipelines.deduplicate import run_deduplicate

    start = time.time()
    result = await run_deduplicate(db, settings)
    duration = time.time() - start

    return PipelineResponse(
        stage=result.stage,
        processed=result.processed,
        failed=result.failed,
        skipped=result.skipped,
        duration_seconds=round(duration, 2),
    )


@app.post("/classify", response_model=PipelineResponse)
async def classify(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """Run the classification pipeline: SLM categorization + entity extraction."""
    from src.pipelines.classify import run_classify

    start = time.time()
    result = await run_classify(db, settings)
    duration = time.time() - start

    return PipelineResponse(
        stage=result.stage,
        processed=result.processed,
        failed=result.failed,
        skipped=result.skipped,
        duration_seconds=round(duration, 2),
    )
