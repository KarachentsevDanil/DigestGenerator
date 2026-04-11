# Phase 7: Knowledge Graph

## Goal

Implement Phase 1 of the knowledge graph: track entities users encounter via digests, compute novelty scores for ranking, and expose knowledge via bot commands. The key insight is that the digest IS the knowledge — every entity delivered to a user becomes part of their graph at zero extra SLM cost (entities are already extracted during classification).

---

## Dependencies on Phase 5

- Digest delivery working (items delivered to users)
- Messages have `entities_json` populated during classification (Phase 4)
- `UserKnowledge` model exists in DB
- Bot command system set up (Phase 5)

---

## Files to Create/Modify

```
src/
├── knowledge/
│   ├── __init__.py
│   └── tracker.py              # Knowledge update + novelty computation
├── pipelines/
│   └── digest.py               # Integrate novelty into ranking (modify)
├── bot/
│   └── handlers.py             # Add /knowledge, /forget commands (modify)
tests/
└── test_knowledge.py           # Test novelty computation + entity tracking
```

---

## Detailed Spec

### knowledge/tracker.py — Entity Tracking + Novelty

**Update knowledge after digest delivery:**

```python
async def update_user_knowledge(db: AsyncSession, user_id: int,
                                 digest_items: list[DigestItem]) -> int:
    """
    After delivering a digest, update the user's knowledge graph.
    For each entity in delivered items:
    - New entity -> create UserKnowledge record
    - Known entity -> increment encounter_count, update last_seen_at
    Returns count of new entities added.
    """
```

For each digest item:
1. Load the message's `entities_json`
2. For each entity:
   - Compute `canonical_name = slugify(entity["name"])`
   - Query `UserKnowledge` by `(user_id, canonical_name)` — exact match first
   - If not found: try embedding similarity fallback (cosine >= 0.75 on entity name embeddings)
   - If still not found → **new entity**: create `UserKnowledge` record
   - If found → **known entity**: `encounter_count += 1`, `last_seen_at = utcnow()`
3. Update category associations on the knowledge record

**Entity resolution:**
- Primary: exact match on `canonical_name` (slugified via `python-slugify`)
- Fallback: embedding cosine similarity on entity name embeddings (threshold 0.75)
  - Generate entity name embedding with same sentence-transformers model
  - Compare against existing `UserKnowledge.embedding` vectors
  - This catches "GPT-5" matching "GPT-5 Turbo" etc.
- YAGNI: don't build complex entity resolution until it fails in practice

**Compute novelty for digest ranking:**

```python
def compute_novelty(db: AsyncSession, user_id: int, message: Message) -> float:
    """
    What % of this message's entities are new to the user?
    No SLM call — pure DB lookups.

    Returns 0.0 (all known) to 1.0 (all new).
    """
    if not message.entities_json:
        return 0.5  # can't assess, neutral score

    entities = message.entities_json
    new_count = 0
    for entity in entities:
        known = db.query(UserKnowledge).filter(
            UserKnowledge.user_id == user_id,
            UserKnowledge.canonical_name == slugify(entity["name"])
        ).first()
        if not known:
            new_count += 1
        elif (utcnow() - known.last_seen_at).days > 30:
            new_count += 0.3  # stale knowledge = partial novelty

    return min(1.0, new_count / max(len(entities), 1))
```

### Integrate novelty into digest selection (modify digest.py)

**Composite ranking with novelty:**

```python
# Daily score
daily_score = 0.4 * category_confidence + 0.3 * relevance + 0.3 * novelty

# Weekly score
weekly_score = 0.35 * category_confidence + 0.35 * relevance + 0.3 * novelty
```

Modify `select_items()` in `pipelines/digest.py`:
1. After initial query and filtering, compute `novelty` for each candidate
2. Calculate composite score based on digest type
3. Re-sort by composite score
4. Take top_k

This replaces the simpler Phase 5 ranking (category_confidence + relevance only).

### Bot commands for knowledge (modify handlers.py)

| Command | Behavior |
|---------|----------|
| `/knowledge` | Show stats: total entities known, breakdown by category, 5 most recent |
| `/knowledge <query>` | Search knowledge for entity (e.g., `/knowledge Gemma`). Show: entity name, type, encounter count, first/last seen, categories |
| `/forget <entity>` | Remove entity from user's knowledge graph. Forces re-explanation in future digests (higher novelty score next time) |

**`/knowledge` (no args) response format:**

```
Your Knowledge Graph

Total entities: 142
By category: AI/ML (68), Tech (45), Crypto (29)

Recent:
- Gemma 4 (model) — seen 5 times, last 2 days ago
- OpenAI (company) — seen 12 times, last 1 day ago
- EU AI Act (regulation) — seen 3 times, last 5 days ago
- Claude 4 (model) — seen 2 times, last 1 day ago
- Stripe (company) — seen 1 time, last 3 days ago
```

**`/knowledge Gemma` response format:**

```
Gemma 4 (model)
First seen: 2026-03-15
Last seen: 2026-04-09
Encountered: 5 times
Categories: AI/ML, Tech Industry
```

---

## Tests (test_knowledge.py)

1. **New entity creation** — deliver digest with unknown entity → UserKnowledge created
2. **Known entity update** — deliver digest with known entity → encounter_count increments
3. **Stale entity partial novelty** — entity not seen in >30 days → counts as 0.3 new
4. **Novelty score: all new** — message with 3 unknown entities → novelty = 1.0
5. **Novelty score: all known** — message with 3 recently-seen entities → novelty = 0.0
6. **Novelty score: mixed** — 1 new + 2 known → novelty = 0.33
7. **No entities** — message with empty entities_json → novelty = 0.5 (neutral)
8. **Entity resolution: exact match** — "Gemma 4" matches existing "Gemma 4" via canonical slug
9. **Forget command** — `/forget Gemma 4` removes record, next encounter creates fresh
10. **Composite ranking** — verify novelty affects final ranking order correctly

---

## Definition of Success

### Checklist

- [ ] `update_user_knowledge()` creates new `UserKnowledge` records for unknown entities
- [ ] Known entities get `encounter_count` incremented and `last_seen_at` updated
- [ ] `compute_novelty()` returns correct scores: 0.0 (all known) to 1.0 (all new)
- [ ] Stale entities (>30 days) count as 0.3 novelty (partial)
- [ ] Messages without entities get neutral novelty score (0.5)
- [ ] Composite ranking integrated into `select_items()` for both daily and weekly
- [ ] Daily: `0.4 * confidence + 0.3 * relevance + 0.3 * novelty`
- [ ] Weekly: `0.35 * confidence + 0.35 * relevance + 0.3 * novelty`
- [ ] `/knowledge` command shows entity stats and recent entities
- [ ] `/knowledge <query>` searches and displays specific entity info
- [ ] `/forget <entity>` removes entity from user's graph
- [ ] Entity resolution: exact slug match works, embedding fallback catches near-matches
- [ ] `test_knowledge.py` passes — novelty computation, entity tracking, staleness all covered
- [ ] Knowledge updates happen automatically after every digest delivery
- [ ] No extra SLM calls — all entity data comes from classification phase
