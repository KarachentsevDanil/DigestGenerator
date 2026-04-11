# Phase 6: Pipeline Orchestration

## Goal

Implement the `PipelineRun` tracking system and orchestration endpoints: `run-all` (sequential scrape + dedup + classify), `retry-failed` (reset failed messages for reprocessing), and observability endpoints. After this phase, the full pipeline can be triggered by a single cron job and all runs are tracked for debugging.

---

## Dependencies on Phases 2-5

- All four pipeline stages implemented: scrape, deduplicate, classify, digest
- PipelineRun model exists in DB
- All individual stage endpoints working

---

## Files to Create/Modify

```
src/
├── pipelines/
│   └── runner.py               # PipelineRun tracking + run-all orchestration
├── main.py                     # Add orchestration endpoints
```

---

## Detailed Spec

### pipelines/runner.py — Pipeline Run Tracking

```python
@asynccontextmanager
async def track_pipeline_run(db: AsyncSession, stage: str):
    """
    Context manager that creates a PipelineRun record, tracks timing,
    and updates status on completion or failure.

    Usage:
        async with track_pipeline_run(db, "scrape") as run:
            result = await run_scrape(db, settings)
            run.processed_count = result.processed
            run.failed_count = result.failed
            run.skipped_count = result.skipped
    """
```

- Creates `PipelineRun` with `status='running'`, `started_at=utcnow()`
- On success: `status='completed'`, `finished_at=utcnow()`
- On exception: `status='failed'`, `error_detail=str(exception)`, `finished_at=utcnow()`
- Always commits the run record (even on failure)

**Run-all orchestration:**

```python
async def run_all_stages(db: AsyncSession, settings: Settings) -> list[PipelineRunResult]:
    """
    Run scrape -> deduplicate -> classify sequentially.
    Each stage tracked independently. Stops on stage failure.
    """
```

1. Run scrape (tracked)
2. If scrape succeeded → run deduplicate (tracked)
3. If dedup succeeded → run classify (tracked)
4. Return list of all run results
5. If any stage fails, stop and return results so far with the failure

**Retry failed:**

```python
async def retry_failed(db: AsyncSession) -> dict[str, int]:
    """
    Reset all *_failed messages to their previous status for reprocessing.
    - embed_failed -> unprocessed
    - dedup_failed -> embedded
    - classify_failed -> deduplicated
    Returns count of messages reset per stage.
    """
```

### Wire orchestration endpoints

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `POST /pipeline/run-all` | `run_all_stages()` | Run scrape → dedup → classify sequentially |
| `POST /pipeline/retry-failed` | `retry_failed()` | Reset failed messages for retry |
| `GET /pipeline/runs` | query `pipeline_runs` table | List recent pipeline runs with filters |
| `GET /stats` | aggregate queries | Messages by status, source stats, cluster counts |
| `GET /health` | enhanced | Check DB + Ollama + ChromaDB connectivity |

**`GET /pipeline/runs` query params:**
- `stage` (optional): filter by stage name
- `status` (optional): filter by run status
- `limit` (default 20): max results
- `offset` (default 0): pagination

**`GET /stats` response:**

```json
{
  "messages": {
    "total": 1250,
    "by_status": {
      "unprocessed": 0,
      "embedded": 12,
      "deduplicated": 5,
      "classified": 890,
      "skipped_duplicate": 340,
      "classify_failed": 3
    }
  },
  "sources": {
    "total": 8,
    "active": 7
  },
  "clusters": {
    "total": 180,
    "avg_size": 2.3
  },
  "pipeline_runs": {
    "last_scrape": "2026-04-10T08:00:00Z",
    "last_classify": "2026-04-10T08:02:30Z"
  }
}
```

**`GET /health` enhanced response:**

```json
{
  "status": "ok",
  "database": "ok",
  "ollama": "ok",
  "chromadb": "ok"
}
```

- Check DB: simple query
- Check Ollama: `GET {base_url}/api/tags` (list models)
- Check ChromaDB: `client.heartbeat()`

### Cron job documentation

Document the expected cron setup (in comments/docstrings, not a separate file):

```bash
# Scrape + dedup + classify every 4 hours
0 */4 * * *  curl -sf -X POST http://localhost:8000/pipeline/run-all

# Check if any user needs a daily digest (hourly, handles timezones)
0 * * * *    curl -sf -X POST http://localhost:8000/generate-digest?type=daily

# Check if any user needs a weekly digest (hourly on Sundays)
0 * * * 0    curl -sf -X POST http://localhost:8000/generate-digest?type=weekly
```

---

## Definition of Success

### Checklist

- [ ] `PipelineRun` records created for every stage execution with correct timing
- [ ] `run-all` executes scrape → dedup → classify sequentially, stops on failure
- [ ] Each stage in `run-all` is tracked independently in `pipeline_runs` table
- [ ] `retry-failed` resets `*_failed` messages to correct previous status
- [ ] `GET /pipeline/runs` returns paginated run history with optional filters
- [ ] `GET /stats` returns accurate message counts by status, source counts, cluster stats
- [ ] `GET /health` checks DB, Ollama, and ChromaDB connectivity
- [ ] Failed pipeline run records include `error_detail` with exception info
- [ ] `POST /pipeline/run-all` returns list of all stage results
- [ ] Pipeline runs have correct `duration_seconds` calculation
- [ ] Cron job commands documented in code
