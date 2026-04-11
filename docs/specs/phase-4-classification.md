# Phase 4: Classification

## Goal

Implement SLM-based classification using Ollama + Gemma 4. Only cluster-primary messages are classified, saving 30-40% SLM calls. Each message gets multi-label category scores, a relevance score, a summary, and extracted entities — all in a single SLM call. After this phase, messages are fully classified and ready for digest generation.

---

## Dependencies on Phase 3

- Messages with `status='deduplicated'` and `is_cluster_primary=True` exist in DB
- `OllamaClient` basic structure from Phase 3 (extend it here)
- Category table populated (at least test categories)

---

## Files to Create/Modify

```
src/
├── llm/
│   ├── client.py               # Extend with classify-specific methods
│   └── prompts.py              # Add CLASSIFY_PROMPT
├── pipelines/
│   └── classify.py             # Classification pipeline
tests/
├── test_classify.py            # Test prompt parsing, retry, malformed output
└── fixtures/
    └── sample_messages.json    # Realistic messages for testing
```

---

## Detailed Spec

### What Gets Classified

**ONLY** messages where:
- `status = 'deduplicated'`
- `is_cluster_primary = True`

This is the key optimization: duplicates (30-40% of messages) are never sent to the SLM.

### llm/prompts.py — Classification Prompt

```python
CLASSIFY_PROMPT = """Classify this message and extract key entities.

CATEGORIES:
{categories_block}

MESSAGE:
---
{message_content}
---

Return ONLY valid JSON:
{{
  "categories": {{"<category_name>": <float 0.0-1.0>, ...}},
  "relevance": <float 0.0-1.0>,
  "summary": "<1-2 sentence factual summary>",
  "entities": [
    {{"name": "<entity name>", "type": "<model|company|person|technology|event|regulation|product|concept>"}}
  ]
}}

Category scoring: 0.0=unrelated, 0.5=tangential, 0.8=strong match, 1.0=core topic.
A message CAN score high in multiple categories.
Relevance: 1.0=breaking news, 0.7=notable, 0.4=routine, 0.1=noise/spam.
Summary: key fact or claim, neutral tone, include names and numbers.
Entities: extract ALL named entities (people, companies, models, products, regulations).
"""
```

**Key design choices:**
- `format="json"` in ollama-python enforces valid JSON output
- `temperature=0.1` for consistent classification
- Categories are from the **global** registry (not per-user) — a message is classified ONCE
- Entities extracted for FREE — same SLM call, ~20 extra output tokens

### categories_block Format

Built from the `categories` table:

```
- ai_ml: Artificial intelligence, LLMs, ML research, neural networks
- crypto: Cryptocurrency, blockchain, DeFi, Web3
- tech_industry: Big tech news, startups, funding, acquisitions
```

Each line: `{name}: {description}` — gives the SLM context for scoring.

### llm/client.py — Extended

Add to existing `OllamaClient`:

```python
async def classify_message(self, content: str, categories_block: str) -> ClassificationResult:
    """Classify a message against categories. Returns parsed result."""
    prompt = CLASSIFY_PROMPT.format(
        categories_block=categories_block,
        message_content=content
    )
    return await self.generate_json(prompt, temperature=0.1)
```

**Retry logic:**
1. First attempt: standard prompt with `format="json"`
2. On JSON parse failure: retry with appended suffix: `"\n\nIMPORTANT: Return ONLY valid JSON, no other text."`
3. On second failure: mark message `status='classify_failed'`, log error, continue

### pipelines/classify.py

```python
async def run_classify(db: AsyncSession, settings: Settings) -> PipelineRunResult:
    """
    Classify all deduplicated cluster-primary messages.
    1. Fetch messages with status='deduplicated' AND is_cluster_primary=True
    2. Build categories_block from global categories table
    3. For each message, call SLM classification
    4. Parse and store results
    5. Update status to 'classified'
    """
```

**Per-message processing:**
1. Call `ollama_client.classify_message(message.content, categories_block)`
2. Parse response and store:
   - `category_scores_json` = `{"ai_ml": 0.92, "crypto": 0.1, ...}`
   - `relevance_score` = `0.0-1.0`
   - `summary` = SLM-generated 1-2 sentence summary
   - `entities_json` = `[{"name": "Gemma 4", "type": "model"}, ...]`
3. Set `classified_at = utcnow()`
4. Set `status = 'classified'`

**Multi-label classification:**
A message gets scores against EVERY category. The same message can appear in multiple users' digests under different categories, if their per-category confidence threshold is met.

Example: A message about "Google releases Gemma 4" scores:
```json
{"ai_ml": 0.92, "tech_industry": 0.55, "crypto": 0.03, "geopolitics": 0.01}
```

**Batch considerations:**
- Process one message at a time (SLM calls are ~2-3s each, CPU-bound)
- Log progress every 10 messages
- Track `PipelineRun` with processed/failed/skipped counts

### Wire POST /classify endpoint

- `POST /classify` — calls `run_classify()`, returns `PipelineResponse`
- Response includes: processed (classified), failed (parse errors), skipped (non-primary)

---

## Tests (test_classify.py)

1. **Valid JSON parsing** — mock SLM response, verify all fields extracted correctly
2. **Multi-label scores** — verify a message can score >0.5 in multiple categories
3. **Malformed JSON retry** — first call returns garbage, second returns valid JSON → succeeds
4. **Double failure** — both calls fail → message gets `classify_failed` status
5. **Empty categories** — handle case with no categories gracefully
6. **Entity extraction** — verify entities parsed from various response formats
7. **Only primaries classified** — non-primary messages are skipped
8. **Idempotency** — already-classified messages are not re-processed

---

## Definition of Success

### Checklist

- [ ] `CLASSIFY_PROMPT` produces valid JSON via Ollama with `format="json"`
- [ ] Only `status='deduplicated' AND is_cluster_primary=True` messages are classified
- [ ] Each classified message has: `category_scores_json`, `relevance_score`, `summary`, `entities_json`
- [ ] Multi-label: a message can score high in multiple categories simultaneously
- [ ] Retry logic: single JSON parse failure is retried, second failure marks `classify_failed`
- [ ] `classified_at` timestamp is set on successful classification
- [ ] `status` transitions to `classified` on success, `classify_failed` on failure
- [ ] `POST /classify` returns correct `PipelineResponse` counts
- [ ] Classification is idempotent — already-classified messages are skipped
- [ ] `test_classify.py` passes — parsing, retry, multi-label all covered
- [ ] SLM temperature is 0.1 for consistency
- [ ] Categories block is built from global `categories` table (classified once, not per-user)
