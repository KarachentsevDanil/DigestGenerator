# Phase 7 v2: Real Knowledge Graph

> **Replaces**: `phase-7-knowledge-graph.md` (simple entity frequency counter)
> **Why**: Tracking "Gemma 4 seen 5 times" is not knowledge. A real knowledge graph captures **relationships between entities**, detects **structural novelty**, and tracks **evolving narratives**.

---

## What Changes From v1

| Aspect | v1 (Old) | v2 (New) |
|--------|----------|----------|
| Data model | Flat table: entity + count | Graph: nodes + typed edges + user overlay |
| Extraction | Entities only (during classify) | Entities + relationship triples (dedicated pipeline stage) |
| Novelty | "Have you seen this keyword?" | Multi-dimensional: new entities, new relationships, bridge connections, story evolution |
| Storage | Single SQLAlchemy table | 6 tables (adjacency model) + igraph for algorithms |
| Pipeline | Baked into digest generation | Two new dedicated stages: extract-knowledge, build-graph |
| Cost | Zero (piggybacks on classify) | ~2-3s per message SLM call (separate stage, only on classified primaries) |

---

## Pipeline Changes

The pipeline grows from 4 to 6 stages. Knowledge extraction is its own granular step because SLM calls are expensive and should be independently retriable.

```
Scrape → Deduplicate → Classify → Extract Knowledge → Build Graph → Generate Digest
                                   ~~~~~~~~~~~~~~~     ~~~~~~~~~~~
                                    NEW (SLM call)     NEW (pure computation)
```

New message status: `classified → knowledge_extracted`

**POST /extract-knowledge** — SLM-based triple extraction. Processes messages with `status=classified`. Calls Ollama with a dedicated relationship extraction prompt. Stores triples in `relations_json` on the message. Updates status to `knowledge_extracted`. ~2-3s per message.

**POST /build-graph** — Pure computation, no SLM. Merges extracted triples into the global knowledge graph. Resolves entities (dedup "GPT-5" vs "GPT 5"). Runs graph algorithms (community detection, bridge identification). Computes novelty scores. Fast — seconds for the full graph.

---

## Graph Storage Architecture

**Decision: igraph + existing SQLite database.**

Why not a dedicated graph DB:
- Neo4j requires a JVM server — too heavy for a laptop app
- KuzuDB was archived October 2025 — dead project
- Memgraph requires 1GB+ RAM and runs as a separate server
- DuckDB+DuckPGQ is promising but Onager (algorithm extension) is too immature
- GraphQLite (SQLite+Cypher) is v0.4.3 — too young to bet on

Why igraph + SQLite:
- igraph has a C core — 10-50x faster than NetworkX for graph algorithms
- pip-installable, no system dependencies
- 500K nodes / 2M edges fits in ~200-500MB RAM
- Full algorithm suite: centrality, community detection (Louvain/Leiden), bridges, shortest paths
- SQLite is already in the stack — no new database engine, adjacency tables via SQLAlchemy
- Load from SQLite into igraph when algorithms are needed, store results back

**Optional future upgrade**: If graph pattern matching queries become complex, migrate the graph tables to DuckDB+DuckPGQ for SQL/PGQ MATCH syntax. The adjacency table schema is compatible.

---

## Data Model

Replace the single flat `user_knowledge` table with 6 tables:

### Global Knowledge Graph (extracted once, shared across all users)

**knowledge_entities** — Graph nodes. Each entity has a canonical name (slugified for dedup), a type from a fixed schema, optional properties as JSON, and an embedding vector for fuzzy entity matching. Tracks first/last seen timestamps and total mention count across all messages.

**knowledge_relations** — Graph edges. Each relation connects two entities with a typed predicate from a fixed schema. Has a confidence score (0.0-1.0), optional properties as JSON, timestamps, and a reference to the source message for provenance. Observation count tracks how many times this relationship has been independently observed.

**entity_aliases** — Maps variant names to canonical entities. "GPT-5", "GPT5", "GPT-5 Turbo" all map to the same entity. Populated during entity resolution.

### User Exposure Overlay (per-user view of the global graph)

**user_entity_exposure** — Which entities each user has been exposed to via digests. Tracks first/last exposure timestamps and exposure count. This is what was previously the entire `user_knowledge` table.

**user_relation_exposure** — Which relationships each user has been exposed to. This is the critical addition — knowing the user has seen "Gemma 4" is less valuable than knowing they've seen "Gemma 4 CREATED_BY Google" and "Gemma 4 COMPETES_WITH Llama 4".

### Narrative Tracking

**knowledge_narratives** — Groups of messages about the same evolving story. Core entity IDs, first/last message timestamps, message count, auto-generated title, status (active/stale/concluded). Detected by finding messages that share 3+ entities within a rolling 14-day window.

### Message Model Extension

Add `relations_json` field to the existing `messages` table. Stores extracted triples as a list of {subject, predicate, object, confidence} dicts. Populated by the extract-knowledge stage.

---

## Schema-Constrained Extraction

SLMs produce garbage with open-ended extraction. The extraction prompt uses a fixed menu:

### Entity Types (10)

PERSON, ORGANIZATION, MODEL, TECHNOLOGY, PRODUCT, EVENT, REGULATION, LOCATION, CONCEPT, DATASET

### Relation Types (20)

**Structural**: CREATED_BY, WORKS_AT, SUBSIDIARY_OF, HEADQUARTERED_IN, MEMBER_OF, FUNDED_BY

**Competitive/Comparative**: COMPETES_WITH, OUTPERFORMS, BASED_ON, SUCCESSOR_OF, ALTERNATIVE_TO

**Event-driven**: ANNOUNCED, ACQUIRED, LAUNCHED, PARTNERED_WITH, INVESTED_IN

**Stance**: SUPPORTS, OPPOSES, REGULATES

**Catch-all**: RELATED_TO (refined in post-processing)

### Extraction Prompt Design

The prompt for `POST /extract-knowledge` follows these principles:

**Schema-guided**: The prompt lists all valid entity types and relation types. The SLM picks from the menu, not inventing its own ontology.

**Few-shot**: 2-3 worked examples of news text → triples are included in the prompt. This anchors the output format and granularity for SLMs that struggle with zero-shot structured extraction.

**Two-pass with self-verification (PiVe pattern)**: First pass extracts triples. Second pass (same call, chain-of-thought) verifies each triple is explicitly stated in the text, not hallucinated. This catches 20-30% of hallucinated relations in empirical tests.

**JSON mode enforced**: Ollama `format="json"`, `temperature=0.1`.

**Retry logic**: Same as classification — one retry with stricter prompt on parse failure, then `status=knowledge_extract_failed`.

---

## Entity Resolution

Three-tier, cheapest first:

**Tier 1 — Exact canonical match.** Slugify entity name via python-slugify. Look up in `knowledge_entities.canonical_name`. Handles "Gemma 4" = "gemma-4". Free.

**Tier 2 — Alias table lookup.** Check `entity_aliases` for known variants. "GPT5" → "gpt-5", "WHO" → "world-health-organization". Populated over time. Free.

**Tier 3 — Embedding similarity.** Encode entity name with sentence-transformers (already loaded for message embeddings). Compare against existing entity embeddings. Cosine similarity >= 0.85 = likely same entity. Catches "World Health Organization" vs "WHO" that alias table misses. ~1ms.

**Periodic SLM review** (during build-graph): Batch near-miss pairs (cosine 0.75-0.85) and ask the SLM "Are these the same entity?" to expand the alias table. Run infrequently.

---

## Graph Algorithms for Novelty

All run during `POST /build-graph`. Pure computation via igraph, zero SLM cost.

### Community Detection (Louvain/Leiden)

Partition the global graph into topic communities. Example communities: {Google, Gemma 4, TPU, DeepMind} = "Google AI ecosystem", {EU AI Act, GDPR, Digital Markets Act} = "EU Tech Regulation".

**Novelty signal**: A new edge that connects entities in DIFFERENT communities is structurally novel — it bridges knowledge domains.

### Bridge Detection

A bridge edge is one whose removal would disconnect parts of the graph. New triples that are bridges represent genuinely new connections between previously unrelated topics.

**Novelty signal**: Messages that introduce bridge edges get a significant novelty boost.

### Betweenness Centrality Delta

Compute betweenness centrality before and after adding new triples from recent messages. Entities whose centrality jumps sharply have become new connectors — they're structurally important.

**Novelty signal**: High centrality delta = the entity just became a hub connecting multiple topics.

### Temporal Burst Detection

Track the rate of new triples involving an entity over a sliding window. A sudden spike (e.g., 2 mentions/week → 15 mentions/week) signals breaking activity.

**Novelty signal**: Burst entities are trending and warrant attention.

---

## Multi-Dimensional Novelty Score

Replaces the v1 model of "% of entities that are new." The new composite score:

### Five Novelty Dimensions

**Entity novelty (weight: 0.15)** — What fraction of this message's entities are unknown to the user? Same as v1 but now just one dimension of five.

**Relation novelty (weight: 0.30)** — What fraction of this message's relationship triples are new to the user? This is the most valuable signal — knowing "OpenAI" is not novel, but knowing "OpenAI ACQUIRED Windsurf" is.

**Bridge novelty (weight: 0.25)** — Does this message connect two previously disconnected communities in the user's knowledge graph? Binary signal (is bridge / is not), scaled by the size of the communities being connected.

**Evolution novelty (weight: 0.15)** — Does this message update or contradict existing knowledge? E.g., "OpenAI valuation: $300B" when the user's graph has "OpenAI valuation: $157B". Detected by finding relations with the same subject+predicate but different object.

**Narrative novelty (weight: 0.15)** — Is this message part of a story the user is following but hasn't seen an update on recently? Inverse of time-since-last-update for the narrative.

### Staleness Decay

Known entities and relations decay over time. An entity not seen in 30+ days counts as 30% novel (partial decay). A relation not seen in 60+ days counts as 50% novel. This ensures the digest resurfaces important evolving topics even if the user has seen the entities before.

### Final Digest Ranking (replaces v1 formulas)

```
daily_score  = 0.30 * category_confidence + 0.25 * relevance + 0.45 * composite_novelty
weekly_score = 0.25 * category_confidence + 0.30 * relevance + 0.45 * composite_novelty
```

Knowledge novelty gets the largest weight (0.45) because it's the primary differentiator from a dumb RSS reader.

---

## Narrative / Story Tracking

### Detection

During `POST /build-graph`, identify message clusters that form evolving stories:

1. Group messages that share 3+ entities AND fall within a rolling 14-day window
2. If a group has 3+ messages, it's a candidate narrative
3. SLM generates a short title for the narrative (one call per new narrative, amortized cost)
4. New messages matching the entity set extend the narrative

### Status Lifecycle

**Active** — received a new message within the last 7 days.
**Stale** — no new message in 7-30 days. The user can be notified "this story you were following has gone quiet."
**Concluded** — no new message in 30+ days, or SLM determines the story has a resolution.

### User Interaction

Users see which narratives they're implicitly following (based on digest exposure). They can explicitly follow/unfollow via bot commands.

---

## Bot Commands (Enhanced)

**/knowledge** — Show graph stats: total entities, total relations, community count, top 5 most-connected entities, recent new connections.

**/knowledge [query]** — Search the graph. Show the entity, its type, direct relations (who/what it's connected to), communities it belongs to, first/last seen, exposure count.

**/story** — List active narratives the user is following with last-update timestamps.

**/story [query]** — Show narrative details: title, involved entities, message count, timeline.

**/forget [entity]** — Remove entity AND its relations from user's exposure overlay. Makes it novel again for future digests.

**/graph** — Export user's knowledge subgraph as a simple text-based adjacency list (for future visualization).

---

## Files to Create/Modify

| File | Purpose |
|------|---------|
| `src/knowledge/__init__.py` | Package init |
| `src/knowledge/extractor.py` | SLM-based triple extraction (extract-knowledge stage) |
| `src/knowledge/graph_builder.py` | Entity resolution, graph merge, algorithm runner (build-graph stage) |
| `src/knowledge/entity_resolver.py` | Three-tier entity resolution |
| `src/knowledge/novelty.py` | Multi-dimensional novelty scoring |
| `src/knowledge/narratives.py` | Story detection and lifecycle |
| `src/llm/prompts.py` (modify) | Add EXTRACT_KNOWLEDGE_PROMPT |
| `src/db/models.py` (modify) | Add 6 new tables, add relations_json to Message |
| `src/pipelines/digest.py` (modify) | Integrate new novelty scores into ranking |
| `src/bot/handlers.py` (modify) | Add /knowledge, /story, /forget, /graph commands |
| `src/main.py` (modify) | Register /extract-knowledge and /build-graph endpoints |
| `tests/test_knowledge_graph.py` | Graph construction, entity resolution, novelty |
| `tests/test_narratives.py` | Story detection and lifecycle |

---

## Dependencies Added

| Library | Purpose | Version |
|---------|---------|---------|
| igraph | Graph algorithms (C core, fast) | latest (pip install python-igraph) |

No new database engine. Graph data stored in existing SQLite via SQLAlchemy adjacency tables.

---

## Definition of Success

### Extract Knowledge Stage
- [ ] Dedicated SLM prompt extracts typed triples with schema-constrained entity/relation types
- [ ] Few-shot examples in prompt produce consistent structured output from Gemma 4
- [ ] Self-verification step in prompt reduces hallucinated triples
- [ ] `POST /extract-knowledge` processes `status=classified` messages independently
- [ ] `relations_json` populated on messages; status transitions to `knowledge_extracted`
- [ ] Retry logic: one retry on parse failure, then `knowledge_extract_failed` status
- [ ] Stage is idempotent — re-running skips already-extracted messages

### Build Graph Stage
- [ ] Entity resolution correctly merges "GPT-5" / "GPT5" / "GPT-5 Turbo" into one node
- [ ] Alias table grows over time from resolution results
- [ ] Triples merged into `knowledge_entities` + `knowledge_relations` tables
- [ ] igraph loads the full graph from SQLite in under 5 seconds for graphs up to 100K nodes
- [ ] Community detection (Louvain) partitions graph into meaningful topic clusters
- [ ] Bridge detection identifies edges connecting different communities
- [ ] `POST /build-graph` completes in seconds (pure computation, no SLM)

### Novelty Scoring
- [ ] Five novelty dimensions computed: entity, relation, bridge, evolution, narrative
- [ ] Relation novelty weighted highest (0.30) — "new relationship" > "new keyword"
- [ ] Bridge novelty boosts messages that connect previously disconnected knowledge areas
- [ ] Evolution novelty detects updated/contradicted facts
- [ ] Staleness decay: entities >30 days = 30% novel, relations >60 days = 50% novel
- [ ] Composite novelty integrated into digest ranking with 0.45 weight

### Narrative Tracking
- [ ] Messages sharing 3+ entities within 14 days grouped into narratives
- [ ] Narratives have auto-generated titles (one SLM call per narrative)
- [ ] Narrative lifecycle: active → stale (7d no update) → concluded (30d)
- [ ] User exposure to narratives tracked

### Bot Commands
- [ ] `/knowledge` shows graph stats: entities, relations, communities, top connectors
- [ ] `/knowledge [query]` shows entity details with relations and community membership
- [ ] `/story` lists active narratives user is following
- [ ] `/forget [entity]` removes entity + relations from user exposure overlay

### Tests
- [ ] Entity resolution: exact slug, alias lookup, embedding similarity all work
- [ ] Novelty scoring: all-new entities → high score, all-known → low score
- [ ] Bridge detection: message connecting two communities scores high bridge novelty
- [ ] Evolution detection: updated fact (same subject+predicate, different object) detected
- [ ] Narrative grouping: 3+ messages sharing 3+ entities form a narrative
- [ ] Graph algorithms run on igraph without errors for realistic graph sizes
