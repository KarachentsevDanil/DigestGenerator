# Phase 2: Telegram Connector + Scrape Pipeline

## Goal

Build the Telegram channel scraper using Telethon (MTProto userbot). Implement the connector abstraction, text normalization, embedding generation (sentence-transformers), MinHash signature creation (datasketch), and the `POST /scrape` endpoint. After this phase, real Telegram messages are scraped, embedded, and stored in the database.

---

## Dependencies on Phase 1

- Database models (Message, Source) must exist
- Async session factory must work
- Config system must provide Telegram credentials and scrape settings

---

## Files to Create

```
src/
├── connectors/
│   ├── __init__.py
│   ├── base.py                 # BaseConnector ABC + NormalizedMessage dataclass
│   └── telegram.py             # TelegramConnector implementation
├── pipelines/
│   ├── __init__.py
│   └── scrape.py               # Scrape orchestration logic
scripts/
└── setup_session.py            # Interactive Telethon session setup
```

---

## Detailed Spec

### connectors/base.py

```python
@dataclass
class NormalizedMessage:
    external_id: str
    content: str
    content_url: str | None = None
    media_type: str = "text"
    published_at: datetime = field(default_factory=datetime.utcnow)
    raw_metadata: dict = field(default_factory=dict)
    forwarded_from_channel: str | None = None
    forwarded_message_id: str | None = None

class BaseConnector(ABC):
    source_type: str

    @abstractmethod
    async def fetch_new(self, source: Source) -> list[NormalizedMessage]:
        """Fetch new messages since source's cursor."""
        ...

    @abstractmethod
    async def validate_source(self, identifier: str) -> bool:
        """Check if source exists and is accessible."""
        ...
```

Minimal abstraction. No deep generics. Only implement when a second connector is needed.

### connectors/telegram.py — TelegramConnector

- Initialize `TelethonClient` using `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`
- Session file stored at `data/sessions/digest_session.session`
- `fetch_new(source)`:
  1. Resolve channel by `source.source_identifier` (handles both `@username` and numeric ID)
  2. Fetch messages using `client.iter_messages()` with `min_id` from `source.last_scraped_external_id`
  3. Limit to `config.scrape.messages_per_channel` per call
  4. For each message:
     - Skip if no text and no caption
     - Extract `message.text` or `message.caption`
     - Extract URLs from entities
     - Detect forwarded-from: `message.forward.chat.username` if available
     - Build `NormalizedMessage`
  5. Return list sorted by message ID ascending
- `validate_source(identifier)`: Try `client.get_entity(identifier)`, return True/False
- Handle Telethon exceptions: `FloodWaitError` (wait and retry), `ChannelPrivateError` (log and skip)

### scripts/setup_session.py

- Interactive script for first-time Telethon auth
- Prompts for phone number, receives code, handles 2FA
- Creates session file at `data/sessions/digest_session.session`
- Run with: `uv run python scripts/setup_session.py`

### pipelines/scrape.py

Main orchestration:

```python
async def run_scrape(db: AsyncSession, settings: Settings) -> PipelineRunResult:
    """
    For each active source:
    1. Fetch new messages via connector
    2. Forward-based instant dedup (if forwarded from known source, skip)
    3. Normalize text
    4. Generate embedding (sentence-transformers, all-MiniLM-L6-v2)
    5. Generate MinHash signature (datasketch)
    6. Insert into DB with status='embedded'
    7. Update source cursor
    """
```

**Embedding generation:**
- Load `SentenceTransformer('all-MiniLM-L6-v2')` once (module-level or cached)
- `model.encode(text)` returns 384-dim float32 array
- Store as `bytes` via `numpy.ndarray.tobytes()`
- ~1ms per message on CPU

**MinHash generation:**
- Create `datasketch.MinHash(num_perm=128)` per message
- Shingle at word 3-gram level: `["word1 word2 word3", "word2 word3 word4", ...]`
- `minhash.update(shingle.encode('utf-8'))` for each shingle
- Store as `pickle.dumps(minhash)`
- ~0.1ms per message

**Forward-based instant dedup (Pass 0):**
- If message has `forwarded_from_channel` AND that channel matches a known `source.source_identifier`:
  - Look up original by `(source_id, forwarded_message_id)`
  - If found: set `status='skipped_duplicate'`, `is_cluster_primary=False`, `dedup_cluster_id` = original's cluster
  - Skip embedding entirely

**Text normalization:**
- Strip Telegram formatting entities (bold, italic, etc.) — keep plain text
- Extract first URL as `content_url`
- Collapse multiple newlines
- Strip leading/trailing whitespace

**Idempotency:**
- `INSERT OR IGNORE` via SQLAlchemy's `insert().on_conflict_do_nothing()` on `UNIQUE(source_id, external_id)`
- Safe to re-run; already-scraped messages are skipped

### Wire POST /scrape endpoint

In `main.py` or a new router file:
- `POST /scrape` — calls `run_scrape()`, returns `PipelineResponse`
- Create `PipelineResponse` Pydantic model: `run_id, stage, processed, failed, skipped, duration_seconds`

---

## Definition of Success

### Checklist

- [ ] `BaseConnector` ABC and `NormalizedMessage` dataclass defined in `connectors/base.py`
- [ ] `TelegramConnector` implements `fetch_new()` and `validate_source()`
- [ ] `scripts/setup_session.py` exists and can create a Telethon session interactively
- [ ] `run_scrape()` fetches messages, generates embeddings (384-dim), generates MinHash signatures
- [ ] Forward-based instant dedup marks known forwards as `skipped_duplicate`
- [ ] Messages stored in DB with `status='embedded'`, non-null `embedding_vector` and `minhash_signature`
- [ ] Source cursor (`last_scraped_external_id`, `last_scraped_at`) is updated after scrape
- [ ] `POST /scrape` returns `PipelineResponse` with correct counts
- [ ] Re-running `/scrape` is idempotent — no duplicate inserts
- [ ] `sentence-transformers` model loads and encodes text without errors
- [ ] `datasketch` MinHash generates correct 128-permutation signatures
- [ ] Telethon `FloodWaitError` is handled gracefully (wait + retry)
