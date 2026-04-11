# Architecture Document

## System Overview

Smart Digest Generator is a fully local, multi-user news digest system. It scrapes Telegram channels, deduplicates content, classifies and summarizes with a local SLM (Gemma 4 via Ollama), and delivers personalized daily/weekly digests via Telegram bot.

**Key Principles:**
- Fully local — no cloud dependencies, runs on a laptop or home server
- Batch pipeline — not real-time, triggered by cron
- SLM-powered — classification, summarization, and entity extraction via Ollama
- Dedup before classify — saves 30-40% SLM calls by eliminating duplicates first

---

## High-Level Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              System Cron                 │
                        │  0 */4 * * * POST /pipeline/run-all     │
                        │  0 * * * *   POST /generate-digest      │
                        └────────────────┬────────────────────────┘
                                         │ HTTP
                                         ▼
┌───────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                            │
│                                                                       │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │  Scrape   │→│  Deduplicate  │→│ Classify  │→│ Generate Digest  │  │
│  └─────┬─────┘  └──────┬───────┘  └────┬─────┘  └───────┬─────────┘  │
│        │               │               │                │            │
│        ▼               ▼               ▼                ▼            │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Telethon  │  │ MinHash LSH  │  │  Ollama   │  │ Telegram Bot │    │
│  │ (MTProto) │  │ + ChromaDB   │  │ (Gemma 4) │  │  (Bot API)   │    │
│  └──────────┘  └──────────────┘  └──────────┘  └──────────────┘    │
│        │               │               │                │            │
│        ▼               ▼               ▼                ▼            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    SQLite + ChromaDB                         │    │
│  │           data/digest.db        data/chromadb/              │    │
│  └─────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Architecture

### Four Sequential Stages

```
POST /scrape ──▶ POST /deduplicate ──▶ POST /classify ──▶ POST /generate-digest
```

Each stage is independently triggerable via HTTP. The `POST /pipeline/run-all` endpoint runs the first three sequentially. Digest generation runs on its own hourly schedule.

**Critical design decision: dedup BEFORE classify.**
- Embeddings cost ~1ms/msg (CPU)
- SLM calls cost ~2-3s/msg
- Dedup first → only classify unique messages → saves 30-40% SLM calls

### Message Status Lifecycle

```
unprocessed ──▶ embedded ──▶ deduplicated ──▶ classified
                                 │
                                 ├──▶ skipped_duplicate  (non-primary, never classified)
                                 │
           Each stage has a *_failed variant for retry:
           embed_failed, dedup_failed, classify_failed
```

**Key rule:** `status` tracks pipeline progress, NOT delivery. A message stays `classified` forever — delivery tracking is in `digest_items`.

---

## Two Telegram Clients

The system uses two separate Telegram integrations for distinct purposes:

### 1. Telethon (MTProto API) — Channel Reader

- Runs as YOUR user account (not a bot)
- Reads public channel history via `iter_messages()`
- Requires phone number + 2FA on first setup, then reuses `.session` file
- Cannot be replaced with Bot API (bots can't read channels unless added as admin)
- Used in: **Scrape stage only**

### 2. python-telegram-bot (Bot API) — User Interface

- A bot created via @BotFather
- Sends digests to users
- Handles `/commands` for configuration
- Cannot read channels
- Receives updates via webhook (`POST /webhook`), not polling
- Used in: **Digest delivery + user commands**

Both run inside the same FastAPI process.

---

## Data Flow

### Scrape Stage

```
Telegram Channels
       │
       ▼ (Telethon fetch)
   Raw Messages
       │
       ├──▶ Forward-based instant dedup (free, if forwarded from known source)
       │         │
       │         ▼ status = skipped_duplicate
       │
       ├──▶ Text normalization (strip formatting, extract URLs)
       │
       ├──▶ Embedding generation (sentence-transformers, 384-dim, ~1ms)
       │
       ├──▶ MinHash signature (datasketch, 128 perms, ~0.1ms)
       │
       ▼
   DB: Messages (status = embedded)
```

### Dedup Stage — Three-Pass

```
Messages (status = embedded)
       │
       ▼
   Pass 1: MinHash LSH (Jaccard >= 0.7)
       │         Near-exact text match (~0.1ms/msg)
       │         Catches: reposts, copy-paste with minor edits
       │
       ▼
   Pass 2: Embedding Cosine (ChromaDB)
       │         Semantic similarity (~1ms/msg)
       │         >= 0.88 → definite duplicate
       │         0.80-0.88 → borderline (→ Pass 3)
       │         < 0.80 → unique
       │
       ▼
   Pass 3: SLM Confirmation (borderline only, ~5-10% of messages)
       │         ~2s per pair via Ollama
       │         JSON response: {is_duplicate, reason}
       │
       ▼
   Cluster Management
       │
       ├──▶ Primary elected (proxy score: originality + length + views + recency)
       │         status = deduplicated, is_cluster_primary = True
       │
       └──▶ Non-primaries
                 status = skipped_duplicate, is_cluster_primary = False
```

### Classify Stage

```
Messages (status = deduplicated, is_cluster_primary = True)
       │
       ▼
   Ollama (Gemma 4, temperature=0.1, format=json)
       │
       │  Single SLM call produces ALL of:
       │  ├── category_scores_json  {"ai_ml": 0.92, "crypto": 0.1}
       │  ├── relevance_score       0.85
       │  ├── summary               "Google releases Gemma 4..."
       │  └── entities_json         [{"name": "Gemma 4", "type": "model"}]
       │
       ▼
   DB: Messages (status = classified)
```

### Digest Generation

```
   Hourly cron trigger
       │
       ▼
   For each active user:
       │
       ├── Check timezone: is it their preferred hour?
       ├── Check not already sent today/this week
       │
       ▼
   select_items()
       │
       ├── For each subscribed category:
       │   ├── Query classified primaries in time window
       │   ├── Filter: category_score >= threshold
       │   ├── Filter: relevance >= floor (0.2 daily, 0.4 weekly)
       │   ├── Exclude already-sent items
       │   ├── Compute novelty score (knowledge graph)
       │   ├── Rank by composite score
       │   └── Take top_k
       │
       ▼
   Render (Jinja2 template, no SLM call)
       │
       ▼
   Deliver (python-telegram-bot)
       │
       ├── Split at 4096 char boundaries if needed
       ├── Record in digests + digest_items tables
       └── Update user_knowledge with delivered entities
```

---

## Data Model

### Entity Relationship Diagram

```
┌──────────┐       ┌───────────────┐       ┌──────────┐
│   User   │──1:N──│ UserCategory  │──N:1──│ Category │
│          │       └───────────────┘       └──────────┘
│          │
│          │──1:N──┌───────────────┐       ┌──────────┐
│          │       │  UserSource   │──N:1──│  Source   │──1:N──┐
│          │       └───────────────┘       └──────────┘       │
│          │                                                   │
│          │──1:N──┌──────────┐       ┌─────────────┐          │
│          │       │  Digest  │──1:N──│ DigestItem  │──N:1─────┤
│          │       └──────────┘       └─────────────┘          │
│          │                                                   │
│          │──1:N──┌───────────────┐                 ┌─────────┴──┐
│          │       │UserKnowledge  │                 │  Message   │
└──────────┘       └───────────────┘                 └────────────┘
                                                           │
                                                     ┌─────┴──────┐
                                                     │PipelineRun │
                                                     └────────────┘
```

### 10 Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| `users` | Registered bot users | telegram_chat_id, timezone, daily/weekly prefs |
| `categories` | Global category registry | name (slug), display_name, description |
| `user_categories` | Per-user category subscriptions | top_k, min_confidence |
| `sources` | Content sources (Telegram channels) | source_type, source_identifier, cursor |
| `user_sources` | Which users subscribe to which sources | user_id, source_id |
| `messages` | Raw + processed messages | content, status, embeddings, classifications |
| `digests` | Generated digest records | user_id, type, window, delivered_at |
| `digest_items` | Message-to-digest mapping | digest_id, message_id, rank |
| `user_knowledge` | Entity knowledge graph per user | entity_name, canonical_name, encounter_count |
| `pipeline_runs` | Pipeline execution history | stage, status, counts, timing |

---

## Unit Architecture

### src/config.py — Configuration

Loads secrets from `.env` and tuning knobs from `config.yaml` via `pydantic-settings`.

```
.env (secrets)          config.yaml (tuning)
     │                        │
     └────────┬───────────────┘
              ▼
        Settings (BaseSettings)
        ├── telegram_api_id
        ├── telegram_api_hash
        ├── telegram_bot_token
        ├── OllamaConfig (model, temperature, timeout)
        ├── EmbeddingsConfig (model name)
        ├── DedupConfig (thresholds, window)
        ├── ScrapeConfig (limits, delays)
        └── DigestConfig (top_k, relevance floors)
```

Single `get_settings()` function with `@lru_cache`.

### src/db/ — Database Layer

- **models.py**: All 10 SQLAlchemy 2.0 models with `Mapped[]` type annotations
- **session.py**: Async engine (aiosqlite) + `async_sessionmaker` + `get_session()` generator

SQLite chosen for simplicity (single-file, no server). All access is async via aiosqlite.

### src/connectors/ — Source Connectors

```
BaseConnector (ABC)
├── source_type: str
├── fetch_new(source) -> list[NormalizedMessage]
└── validate_source(identifier) -> bool

TelegramConnector(BaseConnector)
├── Telethon client with session file
├── Handles FloodWaitError, ChannelPrivateError
└── Returns NormalizedMessage with forwarded-from metadata
```

Only `TelegramConnector` implemented now. Future connectors (X, Gmail, RSS) implement `BaseConnector`. The pipeline doesn't care about connector internals.

### src/dedup/ — Deduplication Engine

```
MinHashIndex
├── Wraps datasketch.MinHashLSH
├── build_from_messages() — bulk load from DB
├── query() — find near-duplicates above Jaccard threshold
└── Word 3-gram shingles, 128 permutations

VectorStore
├── Wraps ChromaDB PersistentClient
├── Cosine similarity space (hnsw:space = cosine)
├── upsert() / batch_upsert() — add embeddings
└── query_similar() — find semantic neighbors with similarity scores
```

### src/llm/ — SLM Client

```
OllamaClient
├── AsyncClient (official ollama-python library)
├── generate_json() — structured output with format="json"
├── Retry: on JSON parse failure, retry once with stricter prompt
└── Temperature 0.1 for consistent classification

Prompts (constants):
├── CLASSIFY_PROMPT — multi-label classification + entity extraction
└── DEDUP_PROMPT — same-event confirmation for borderline pairs
```

### src/pipelines/ — Pipeline Stages

```
scrape.py
├── run_scrape() — fetch, embed, hash, persist
├── Forward-based instant dedup (Pass 0)
├── sentence-transformers encoding (~1ms/msg)
└── datasketch MinHash generation (~0.1ms/msg)

deduplicate.py
├── run_deduplicate() — three-pass dedup
├── Pass 1: MinHash LSH (Jaccard >= 0.7)
├── Pass 2: Cosine similarity via ChromaDB
├── Pass 3: SLM confirmation (borderline only)
└── Cluster primary election

classify.py
├── run_classify() — SLM classification of primaries only
├── Multi-label category scores
├── Relevance scoring + summarization
├── Entity extraction (free, same SLM call)
└── Post-classification primary re-election

digest.py
├── run_generate_digest() — select, render, deliver
├── select_items() — per-user, per-category selection
├── Jinja2 template rendering (no SLM call)
├── Telegram delivery with message splitting
└── Knowledge graph update post-delivery

runner.py
├── track_pipeline_run() — context manager for PipelineRun records
├── run_all_stages() — scrape → dedup → classify sequential
└── retry_failed() — reset *_failed messages to previous status
```

### src/bot/ — Telegram Bot

```
handlers.py
├── /start — user registration
├── /addcat, /rmcat — category management
├── /addsource, /rmsource — source management
├── /topk, /threshold — preference tuning
├── /digest — manual trigger
├── /settings, /stats — info
└── /knowledge, /forget — knowledge graph

delivery.py
├── send_digest() — deliver with message splitting
├── Split at 4096 char category boundaries
├── MarkdownV2 escaping
└── 0.5s delay between split messages
```

### src/knowledge/ — Knowledge Graph

```
tracker.py
├── update_user_knowledge() — create/update entities after digest delivery
├── compute_novelty() — % of new entities per message (0.0-1.0)
├── Entity resolution: exact slug match + embedding cosine fallback
└── Stale knowledge (>30 days) = partial novelty (0.3)
```

### src/templates/ — Jinja2 Templates

- `daily_digest.md.j2` — daily format with relevance icons
- `weekly_digest.md.j2` — weekly "best of" format

Templates use pre-computed summaries from classification. No SLM call during rendering.

---

## External Orchestration

No in-process scheduler. System cron triggers pipeline stages via HTTP:

```bash
# Scrape + dedup + classify every 4 hours
0 */4 * * *  curl -sf -X POST http://localhost:8000/pipeline/run-all

# Check if any user needs a daily digest (hourly, handles timezones)
0 * * * *    curl -sf -X POST http://localhost:8000/generate-digest?type=daily

# Check if any user needs a weekly digest (hourly on Sundays)
0 * * * 0    curl -sf -X POST http://localhost:8000/generate-digest?type=weekly
```

This survives app restarts and is easier to debug than an in-process scheduler.

---

## Idempotency Guarantees

| Stage | Mechanism |
|-------|-----------|
| Scrape | `UNIQUE(source_id, external_id)` + INSERT OR IGNORE |
| Dedup | Only processes `status='embedded'` |
| Classify | Only processes `status='deduplicated' AND is_cluster_primary` |
| Digest | `digest_items` prevents re-sending same message to same user |
| Digest schedule | Checks `digests` table for existing digest today/this week |

Every stage is safe to re-run. Failed messages get `*_failed` status and can be retried via `POST /pipeline/retry-failed`.

---

## Knowledge Graph (Phase 1)

**The digest IS the knowledge.** Every entity extracted during classification that gets delivered to a user becomes part of their knowledge graph. Zero extra SLM cost.

```
Classification (Phase 4)        Digest Delivery (Phase 5)
       │                                │
       │ entities_json                  │ for each delivered entity:
       │ extracted for FREE             │
       ▼                                ▼
  Message.entities_json ─────▶ UserKnowledge table
                                        │
                                        ▼
                               compute_novelty()
                                        │
                            ┌───────────┴───────────┐
                            │ Composite Ranking      │
                            │ daily:  0.4C + 0.3R + 0.3N │
                            │ weekly: 0.35C + 0.35R + 0.3N │
                            └───────────────────────┘
```

Where C = category_confidence, R = relevance, N = novelty.

---

## Project Structure

```
DigestGenerator/
├── .env                        # Secrets (not in git)
├── .env.example                # Template for .env
├── .python-version             # Python 3.12
├── config.yaml                 # Application configuration
├── pyproject.toml              # UV project + all dependencies
├── uv.lock                     # Lockfile (committed)
├── alembic.ini                 # Alembic configuration
├── alembic/
│   ├── env.py                  # Async Alembic environment
│   └── versions/               # Migration scripts
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, lifespan, endpoints
│   ├── config.py               # Pydantic settings (.env + YAML)
│   ├── dependencies.py         # FastAPI dependency injection
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py           # All 10 SQLAlchemy models
│   │   └── session.py          # Async engine + session factory
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseConnector ABC + NormalizedMessage
│   │   └── telegram.py         # TelegramConnector (Telethon)
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── scrape.py           # Scrape + embed + hash
│   │   ├── deduplicate.py      # Three-pass dedup
│   │   ├── classify.py         # SLM classification
│   │   ├── digest.py           # Selection + render + deliver
│   │   └── runner.py           # PipelineRun tracking + orchestration
│   ├── dedup/
│   │   ├── __init__.py
│   │   ├── minhash.py          # MinHash LSH index (datasketch)
│   │   └── vector_store.py     # ChromaDB wrapper
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py           # Ollama async client + retry
│   │   └── prompts.py          # Prompt templates
│   ├── knowledge/
│   │   ├── __init__.py
│   │   └── tracker.py          # Entity tracking + novelty
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── handlers.py         # Telegram bot command handlers
│   │   └── delivery.py         # Digest sending + message splitting
│   └── templates/
│       ├── daily_digest.md.j2  # Daily digest template
│       └── weekly_digest.md.j2 # Weekly digest template
├── tests/
│   ├── test_dedup.py           # Dedup logic tests
│   ├── test_classify.py        # Classification parsing tests
│   ├── test_selection.py       # Digest selection tests
│   ├── test_knowledge.py       # Knowledge graph tests
│   └── fixtures/
│       └── sample_messages.json
├── scripts/
│   └── setup_session.py        # Interactive Telethon auth
├── data/
│   ├── digest.db               # SQLite database
│   ├── chromadb/               # ChromaDB persistent storage
│   └── sessions/               # Telethon session files
└── docs/
    ├── setup.md                # Setup guide
    ├── architecture.md         # This document
    └── specs/
        ├── phase-1-foundation.md
        ├── phase-2-telegram-connector.md
        ├── phase-3-deduplication.md
        ├── phase-4-classification.md
        ├── phase-5-bot-digest.md
        ├── phase-6-orchestration.md
        └── phase-7-knowledge-graph.md
```

---

## API Endpoints

### Pipeline

| Method | Path | Description |
|--------|------|-------------|
| POST | `/scrape` | Run scrape stage |
| POST | `/deduplicate` | Run dedup stage |
| POST | `/classify` | Run classify stage |
| POST | `/generate-digest?type=daily\|weekly` | Generate & deliver digests |
| POST | `/pipeline/run-all` | Run scrape -> dedup -> classify |
| POST | `/pipeline/retry-failed` | Reset failed messages for retry |
| GET | `/pipeline/runs` | Query pipeline run history |

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | System health (DB, Ollama, ChromaDB) |
| GET | `/stats` | Messages by status, source stats |

### Bot

| Method | Path | Description |
|--------|------|-------------|
| POST | `/webhook` | Telegram Bot API webhook receiver |

### Standard Response

```json
{
  "run_id": 42,
  "stage": "scrape",
  "processed": 150,
  "failed": 2,
  "skipped": 10,
  "duration_seconds": 12.5
}
```

---

## Future Extensions (Not In Scope)

The architecture supports these without major changes:

- **X/Twitter connector** — implement `BaseConnector`, add `source_type="x"`
- **Gmail connector** — implement `BaseConnector`, add `source_type="gmail"`
- **RSS connector** — implement `BaseConnector`, add `source_type="rss"`
- **Knowledge graph visualization** — React app reading from `/api/knowledge/*`
- **Adaptive summaries** — user-specific summaries based on knowledge gaps
- **Knowledge decay** — time-based confidence reduction for stale facts
- **Web UI** — dashboard for non-Telegram users
