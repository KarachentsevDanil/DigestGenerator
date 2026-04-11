# Phase 3: Deduplication

## Goal

Implement the three-pass deduplication pipeline: MinHash LSH for near-exact text matching, ChromaDB embedding cosine similarity for semantic matching, and SLM confirmation for borderline cases. After this phase, duplicate messages are clustered and only cluster primaries proceed to classification.

---

## Dependencies on Phase 2

- Messages in DB with `status='embedded'`, non-null `embedding_vector` and `minhash_signature`
- Forward-based dedup (Pass 0) already handled during scrape

---

## Files to Create

```
src/
├── dedup/
│   ├── __init__.py
│   ├── minhash.py              # MinHash LSH index wrapper
│   └── vector_store.py         # ChromaDB wrapper
├── pipelines/
│   └── deduplicate.py          # Three-pass dedup orchestration
├── llm/
│   ├── __init__.py
│   ├── client.py               # Ollama client (basic, extended in Phase 4)
│   └── prompts.py              # DEDUP_PROMPT (extended in Phase 4)
tests/
└── test_dedup.py               # Unit tests for dedup logic
```

---

## Detailed Spec

### dedup/minhash.py — MinHash LSH Index

Thin wrapper around `datasketch.MinHashLSH`:

```python
class MinHashIndex:
    def __init__(self, threshold: float = 0.7, num_perm: int = 128):
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.num_perm = num_perm

    def insert(self, key: str, minhash: MinHash) -> None:
        """Insert a MinHash with a unique key (message ID)."""

    def query(self, minhash: MinHash) -> list[str]:
        """Return keys of similar items above threshold."""

    def build_from_messages(self, messages: list[Message]) -> None:
        """Bulk load from DB messages (deserialize minhash_signature)."""

    @staticmethod
    def create_minhash(text: str, num_perm: int = 128) -> MinHash:
        """Create MinHash from text using word 3-gram shingles."""
```

- Rebuild index from DB each dedup run (messages within the time window)
- Query before insert — if matches found, it's a duplicate
- Key = `str(message.id)`

### dedup/vector_store.py — ChromaDB Wrapper

```python
class VectorStore:
    def __init__(self, persist_dir: str = "data/chromadb"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="message_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, message_id: str, embedding: list[float], metadata: dict) -> None:
        """Upsert a single message embedding."""

    def query_similar(self, embedding: list[float], n_results: int = 10,
                      min_similarity: float = 0.80) -> list[SimilarResult]:
        """Find similar embeddings. Returns (id, similarity) pairs."""

    def batch_upsert(self, ids: list[str], embeddings: list[list[float]],
                     metadatas: list[dict]) -> None:
        """Bulk upsert for efficiency."""
```

- `SimilarResult` = simple dataclass with `id: str, similarity: float`
- ChromaDB returns distances; convert to similarity: `1 - distance` for cosine
- Persistent storage at `data/chromadb/`

### pipelines/deduplicate.py — Three-Pass Dedup

```python
async def run_deduplicate(db: AsyncSession, settings: Settings) -> PipelineRunResult:
    """
    Process messages with status='embedded' within dedup window.
    Three passes: MinHash LSH -> Embedding cosine -> SLM confirmation.
    """
```

**Processing flow:**

1. Query all messages with `status='embedded'` AND `scraped_at >= now - window_hours`
2. Build MinHash LSH index from these messages + recent `deduplicated` messages (for cross-batch matching)

**Pass 1 — MinHash LSH (threshold 0.7):**
- For each new message, query LSH index
- If matches found → assign to existing cluster or create new cluster with UUID
- Mark as duplicate (definite)

**Pass 2 — Embedding cosine similarity:**
- For messages NOT caught by Pass 1, upsert into ChromaDB
- Query for neighbors:
  - similarity >= 0.88 → definite duplicate, assign to cluster
  - 0.80 <= similarity < 0.88 → borderline, collect for Pass 3
  - similarity < 0.80 → unique

**Pass 3 — SLM confirmation (borderline only):**
- For borderline pairs (~5-10% of messages):
  - Call Ollama with `DEDUP_PROMPT`
  - Parse JSON response: `{"is_duplicate": true/false, "reason": "..."}`
  - If duplicate → assign to cluster
  - If not → mark as unique

**Cluster management:**
- Each cluster gets a `uuid4()` ID
- **Primary election** (proxy score before classification):
  - Original post > forward: +0.3
  - Longer content: +0.0 to +0.3 (normalized by max length in cluster)
  - Higher views (from raw_metadata): +0.0 to +0.2
  - More recent: +0.2 for newest in cluster
- Primary → `status='deduplicated'`, `is_cluster_primary=True`
- Non-primaries → `status='skipped_duplicate'`, `is_cluster_primary=False`
- Unique messages (no cluster) → `status='deduplicated'`, `is_cluster_primary=True`, `dedup_cluster_id=NULL`

### llm/client.py — Ollama Client (Basic)

```python
class OllamaClient:
    def __init__(self, settings: OllamaConfig):
        self.client = ollama.AsyncClient(host=settings.base_url)
        self.model = settings.model
        self.timeout = settings.timeout

    async def generate_json(self, prompt: str, temperature: float = 0.1) -> dict:
        """Generate structured JSON output. Retry once on parse failure."""
```

- Use `format="json"` parameter for guaranteed JSON output
- Set `temperature=0.1` for consistent results
- Retry logic: on JSON parse failure, retry once with stricter prompt suffix
- On second failure, raise exception (caller decides status)

### llm/prompts.py — Dedup Prompt

```python
DEDUP_PROMPT = """Are these two messages about the same news event?

Message A:
---
{message_a}
---

Message B:
---
{message_b}
---

Return ONLY valid JSON:
{{"is_duplicate": true/false, "reason": "<brief explanation>"}}
"""
```

### Wire POST /deduplicate endpoint

- `POST /deduplicate` — calls `run_deduplicate()`, returns `PipelineResponse`
- Response includes: processed (total), skipped (duplicates), failed counts

---

## Tests (test_dedup.py)

Focus on critical dedup logic:

1. **MinHash near-duplicate detection** — two messages with 80% word overlap should cluster
2. **MinHash non-duplicate** — two unrelated messages should NOT cluster
3. **Cosine similarity thresholds** — test definite (>=0.88), borderline (0.80-0.88), unique (<0.80)
4. **Cluster primary election** — verify scoring: original > forward, longer > shorter
5. **Forward-based dedup** — forwarded messages from known sources are caught at scrape time
6. **Idempotency** — re-running dedup on already-deduplicated messages is a no-op
7. **SLM confirmation parsing** — valid and malformed JSON responses handled correctly

Use fixtures with realistic Telegram messages (`tests/fixtures/sample_messages.json`).

---

## Definition of Success

### Checklist

- [ ] `MinHashIndex` builds from DB messages, queries return correct near-duplicates
- [ ] `VectorStore` upserts embeddings to ChromaDB, queries return similar items with correct cosine scores
- [ ] Three-pass dedup runs in correct order: MinHash -> Cosine -> SLM
- [ ] Definite duplicates (MinHash >= 0.7 OR cosine >= 0.88) are clustered without SLM call
- [ ] Borderline cases (cosine 0.80-0.88) are confirmed/rejected via SLM
- [ ] Unique messages (cosine < 0.80) pass through unclustered
- [ ] Cluster primary election uses proxy score (originality + length + views + recency)
- [ ] Non-primary messages get `status='skipped_duplicate'` (never classified)
- [ ] Primary messages get `status='deduplicated'`, `is_cluster_primary=True`
- [ ] `POST /deduplicate` returns correct `PipelineResponse` counts
- [ ] ChromaDB data persists at `data/chromadb/`
- [ ] `test_dedup.py` passes — all dedup scenarios covered
- [ ] Dedup is idempotent — re-running doesn't re-cluster already-processed messages
