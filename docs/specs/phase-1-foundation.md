# Phase 1: Foundation

## Goal

Set up the project skeleton: dependency management with UV, configuration system, all database models, async DB session, Alembic migrations, and a minimal FastAPI app with `/health`. After this phase the app starts and responds to HTTP requests with a working database.

---

## Tech Stack (Pinned Versions)

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.12+ | Runtime |
| FastAPI | >=0.135.3 | API framework |
| uvicorn | >=0.44.0 | ASGI server |
| SQLAlchemy | >=2.0.49 | ORM (async) |
| Alembic | >=1.18.4 | Migrations |
| aiosqlite | >=0.22.1 | SQLite async driver |
| pydantic-settings | >=2.13.1 | Config from env + YAML |
| PyYAML | >=6.0.3 | YAML config parsing |
| structlog | >=25.5.0 | Structured logging |
| python-slugify | >=8.0.4 | Entity slug generation |
| httpx | >=0.28.1 | Async HTTP client |
| Jinja2 | >=3.1.6 | Template rendering |

---

## Files to Create

```
DigestGenerator/
├── pyproject.toml              # UV project, all dependencies, [project.scripts]
├── .python-version             # Pin Python 3.12
├── .env.example                # Template for secrets
├── config.yaml                 # Default application config
├── alembic.ini                 # Alembic configuration
├── alembic/
│   ├── env.py                  # Async Alembic env
│   └── versions/               # Migration scripts (auto-generated)
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, /health endpoint
│   ├── config.py               # Pydantic settings: .env + config.yaml
│   ├── dependencies.py         # FastAPI dependency providers (db session)
│   └── db/
│       ├── __init__.py
│       ├── models.py           # All 10 SQLAlchemy models
│       └── session.py          # Async engine + sessionmaker
└── data/                       # Runtime data directory
    ├── chromadb/               # (empty, for Phase 3)
    └── sessions/               # (empty, for Phase 2)
```

---

## Detailed Spec

### pyproject.toml

- Use `[project]` table with `name = "smart-digest"`, `version = "0.1.0"`, `requires-python = ">=3.12"`
- ALL dependencies listed under `[project.dependencies]` with minimum version pins (`>=`)
- Dev dependencies under `[dependency-groups]`: pytest, pytest-asyncio, pytest-cov, ruff
- Include ALL libraries from the full tech stack (even those used in later phases) so the environment is ready upfront
- Full dependency list: fastapi, uvicorn[standard], sqlalchemy[asyncio], alembic, aiosqlite, chromadb, telethon, python-telegram-bot, ollama, sentence-transformers, datasketch, jinja2, structlog, pydantic-settings, pyyaml, python-slugify, httpx

### config.py

- `Settings` class extending `pydantic_settings.BaseSettings`
- Load `.env` for secrets (`TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_BOT_TOKEN`, `WEBHOOK_URL`)
- Load `config.yaml` for app config using `@classmethod` settings customise sources
- Nested config sections as inner models:
  - `OllamaConfig`: base_url, model, timeout, temperature
  - `EmbeddingsConfig`: model name
  - `DedupConfig`: minhash_threshold, minhash_num_perm, cosine thresholds, window_hours
  - `ScrapeConfig`: messages_per_channel, delay_between_channels_seconds
  - `DigestConfig`: default_top_k, daily_relevance_floor, weekly_relevance_floor
  - `DatabaseConfig`: url (default `sqlite+aiosqlite:///data/digest.db`)
- Single `get_settings()` function with `@lru_cache`

### db/models.py

All 10 tables from the master spec, using SQLAlchemy 2.0 `DeclarativeBase` + `Mapped` type annotations:

1. **User** — id, name, telegram_chat_id (unique), timezone, daily/weekly prefs, is_active, created_at
2. **Category** — id, name (unique slug), display_name, description, created_by_user_id (FK), created_at
3. **UserCategory** — composite PK (user_id, category_id), top_k, min_confidence
4. **Source** — id, source_type, source_identifier, display_name, is_active, last_scraped_external_id, last_scraped_at. UNIQUE(source_type, source_identifier)
5. **UserSource** — composite PK (user_id, source_id)
6. **Message** — id, source_id (FK), external_id, content, content_url, media_type, published_at, scraped_at, status (default "unprocessed"), embedding_vector (bytes), minhash_signature (bytes), dedup fields, classification fields, telegram metadata, raw_metadata (JSON). UNIQUE(source_id, external_id)
7. **Digest** — id, user_id (FK), digest_type, window_start/end, generated_at, delivered_at, telegram_message_id, item_count
8. **DigestItem** — id, digest_id (FK), message_id (FK), category_name, rank_in_category, confidence_score
9. **UserKnowledge** — id, user_id (FK), entity_name, entity_type, canonical_name, first/last_seen_at, encounter_count, categories (JSON), embedding (bytes). UNIQUE(user_id, canonical_name)
10. **PipelineRun** — id, stage, started_at, finished_at, status, processed/failed/skipped counts, error_detail

Use `datetime` with `func.now()` defaults. JSON columns via `sqlalchemy.JSON`. Status as `String` (not enum — simpler migrations).

### db/session.py

- `create_async_engine()` from settings
- `async_sessionmaker` with `expire_on_commit=False`
- `async def get_session()` async generator for FastAPI `Depends()`
- `async def init_db()` — creates tables (for dev/test, production uses Alembic)

### Alembic Setup

- `alembic.ini` pointing to `alembic/` directory
- `sqlalchemy.url` set to `sqlite+aiosqlite:///data/digest.db`
- `alembic/env.py` configured for async with `run_async_migrations()`
- Import `Base` metadata from `src.db.models`
- Generate initial migration with all 10 tables

### main.py

- FastAPI app with `lifespan` async context manager
- On startup: init structlog, log config loaded
- Endpoints:
  - `GET /health` — returns `{"status": "ok", "database": "ok"}` after pinging DB
  - `GET /stats` — placeholder returning `{"message": "not implemented"}`
- Include placeholder routers for pipeline endpoints (empty for now)

### dependencies.py

- `get_db()` — yields async session from `get_session()`
- `get_settings()` — returns cached settings instance

---

## Message Status Lifecycle (Reference)

```
unprocessed -> embedded -> deduplicated -> classified
                              |
                              +-> skipped_duplicate (terminal)
Each stage has a *_failed variant for retry.
```

---

## Definition of Success

### Checklist

- [ ] `uv sync` installs all dependencies without errors
- [ ] `uv run alembic upgrade head` creates `data/digest.db` with all 10 tables
- [ ] `uv run uvicorn src.main:app --host 0.0.0.0 --port 8000` starts without errors
- [ ] `GET /health` returns `{"status": "ok", "database": "ok"}` (HTTP 200)
- [ ] `GET /stats` returns HTTP 200 with placeholder response
- [ ] All SQLAlchemy models have correct column types, FKs, and unique constraints
- [ ] `.env.example` documents all required environment variables
- [ ] `config.yaml` has sensible defaults for all config sections
- [ ] structlog is configured and produces JSON log output on startup
- [ ] `data/` directory structure exists (chromadb/, sessions/)
- [ ] Project can be linted with `uv run ruff check src/`
