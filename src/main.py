from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings, get_settings
from src.db.models import Message, PipelineRun, Source
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
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _ensure_data_dirs() -> None:
    """Ensure runtime data directories exist."""
    base = Path(__file__).resolve().parent.parent / "data"
    (base / "chromadb").mkdir(parents=True, exist_ok=True)
    (base / "sessions").mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_data_dirs()
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
async def health(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """System health check: verifies DB, Ollama, and ChromaDB connectivity."""
    # Check DB
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"

    # Check Ollama
    try:
        from src.llm.client import OllamaClient

        client = OllamaClient(settings.ollama)
        ollama_status = "ok" if await client.check_health() else "unavailable"
    except Exception:
        ollama_status = "error"

    # Check ChromaDB
    try:
        import chromadb

        chroma = chromadb.PersistentClient(path="data/chromadb")
        chroma.heartbeat()
        chromadb_status = "ok"
    except Exception:
        chromadb_status = "error"

    overall = "ok" if db_status == "ok" else "degraded"
    return {
        "status": overall,
        "database": db_status,
        "ollama": ollama_status,
        "chromadb": chromadb_status,
    }


@app.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)):
    """Pipeline statistics: messages by status, sources, clusters, last runs."""
    # Messages by status
    status_result = await db.execute(
        select(Message.status, func.count(Message.id)).group_by(Message.status)
    )
    by_status = dict(status_result.all())
    total_messages = sum(by_status.values())

    # Sources
    source_result = await db.execute(
        select(
            func.count(Source.id),
            func.count(Source.id).filter(Source.is_active.is_(True)),
        )
    )
    source_row = source_result.one()
    total_sources = source_row[0]
    active_sources = source_row[1]

    # Clusters
    cluster_result = await db.execute(
        select(func.count(func.distinct(Message.dedup_cluster_id))).where(
            Message.dedup_cluster_id.isnot(None)
        )
    )
    total_clusters = cluster_result.scalar_one()

    # Last pipeline runs
    last_runs = {}
    for stage in ["scrape", "deduplicate", "classify", "digest"]:
        run_result = await db.execute(
            select(PipelineRun)
            .where(PipelineRun.stage == stage, PipelineRun.status == "completed")
            .order_by(PipelineRun.finished_at.desc())
            .limit(1)
        )
        run = run_result.scalar_one_or_none()
        if run and run.finished_at:
            last_runs[f"last_{stage}"] = run.finished_at.isoformat()

    return {
        "messages": {
            "total": total_messages,
            "by_status": by_status,
        },
        "sources": {
            "total": total_sources,
            "active": active_sources,
        },
        "clusters": {"total": total_clusters},
        "pipeline_runs": last_runs,
    }


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


@app.post("/extract-knowledge", response_model=PipelineResponse)
async def extract_knowledge(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """Run knowledge extraction: SLM-based triple extraction from classified messages."""
    from src.knowledge.extractor import run_extract_knowledge

    start = time.time()
    result = await run_extract_knowledge(db, settings)
    duration = time.time() - start

    return PipelineResponse(
        stage=result.stage,
        processed=result.processed,
        failed=result.failed,
        skipped=result.skipped,
        duration_seconds=round(duration, 2),
    )


@app.post("/build-graph", response_model=PipelineResponse)
async def build_graph(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """Build/update knowledge graph from extracted triples. Pure computation, no SLM."""
    from src.knowledge.graph_builder import run_build_graph

    start = time.time()
    result = await run_build_graph(db, settings)
    duration = time.time() - start

    return PipelineResponse(
        stage=result.stage,
        processed=result.processed,
        failed=result.failed,
        skipped=result.skipped,
        duration_seconds=round(duration, 2),
    )


@app.post("/generate-digest", response_model=PipelineResponse)
async def generate_digest(
    digest_type: str = Query(default="daily", pattern="^(daily|weekly)$"),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """Generate and deliver digests for eligible users."""
    from src.pipelines.digest import run_generate_digest

    start = time.time()
    result = await run_generate_digest(db, settings, digest_type)
    duration = time.time() - start

    return PipelineResponse(
        stage=result.stage,
        processed=result.processed,
        failed=result.failed,
        skipped=result.skipped,
        duration_seconds=round(duration, 2),
    )


@app.post("/webhook")
async def webhook(request: Request):
    """Telegram Bot API webhook receiver."""
    from telegram import Update as TGUpdate
    from telegram.ext import Application

    from src.bot.handlers import register_handlers

    settings = get_settings()
    if not settings.telegram_bot_token:
        return Response(status_code=503)

    body = await request.json()
    bot_app = Application.builder().token(settings.telegram_bot_token).build()
    register_handlers(bot_app)

    async with bot_app:
        update = TGUpdate.de_json(body, bot_app.bot)
        await bot_app.process_update(update)

    return Response(status_code=200)


@app.post("/pipeline/run-all")
async def pipeline_run_all(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
):
    """Run scrape -> deduplicate -> classify sequentially."""
    from src.pipelines.runner import run_all_stages

    start = time.time()
    results = await run_all_stages(db, settings)
    duration = time.time() - start

    return {
        "stages": [
            {
                "stage": r.stage,
                "processed": r.processed,
                "failed": r.failed,
                "skipped": r.skipped,
            }
            for r in results
        ],
        "total_duration_seconds": round(duration, 2),
    }


@app.post("/pipeline/retry-failed")
async def pipeline_retry_failed(db: AsyncSession = Depends(get_db)):
    """Reset *_failed messages to their previous status for retry."""
    from src.pipelines.runner import retry_failed

    counts = await retry_failed(db)
    return {"reset_counts": counts, "total_reset": sum(counts.values())}


@app.get("/pipeline/runs")
async def pipeline_runs(
    stage: str | None = None,
    status: str | None = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Query pipeline run history."""
    query = select(PipelineRun).order_by(PipelineRun.started_at.desc())

    if stage:
        query = query.where(PipelineRun.stage == stage)
    if status:
        query = query.where(PipelineRun.status == status)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    runs = result.scalars().all()

    return [
        {
            "id": r.id,
            "stage": r.stage,
            "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "processed": r.processed_count,
            "failed": r.failed_count,
            "skipped": r.skipped_count,
            "error": r.error_detail,
        }
        for r in runs
    ]
