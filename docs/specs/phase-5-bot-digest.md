# Phase 5: Bot + Digest Delivery

## Goal

Implement the Telegram bot (python-telegram-bot) for user interaction and the digest generation pipeline. The bot handles user registration, category/source management via commands. The digest pipeline selects top items per user per category, renders them with Jinja2 templates, and delivers via bot. After this phase, users can configure their preferences and receive personalized digests.

---

## Dependencies on Phase 4

- Classified messages in DB (`status='classified'`) with category scores, summaries, entities
- User, Category, UserCategory, Source, UserSource, Digest, DigestItem models exist
- Config system provides `TELEGRAM_BOT_TOKEN` and `WEBHOOK_URL`

---

## Files to Create

```
src/
├── bot/
│   ├── __init__.py
│   ├── handlers.py             # All bot command handlers
│   └── delivery.py             # Digest sending + message splitting
├── pipelines/
│   └── digest.py               # Selection algorithm + rendering
├── templates/
│   ├── daily_digest.md.j2      # Daily digest Jinja2 template
│   └── weekly_digest.md.j2     # Weekly digest Jinja2 template
tests/
└── test_selection.py           # Test digest selection algorithm
```

---

## Detailed Spec

### bot/handlers.py — Command Handlers

Using `python-telegram-bot` v22+ with `Application` builder:

| Command | Handler | Behavior |
|---------|---------|----------|
| `/start` | `start_handler` | Register user: create `User` with `telegram_chat_id`, send welcome message |
| `/help` | `help_handler` | List all available commands with descriptions |
| `/categories` | `categories_handler` | List user's subscribed categories with top_k and min_confidence |
| `/addcat <name> \| <description>` | `addcat_handler` | Create global category (if new) + subscribe user. Parse `name \| description` format |
| `/rmcat <name>` | `rmcat_handler` | Unsubscribe user from category (don't delete global category) |
| `/topk <category> <n>` | `topk_handler` | Update `UserCategory.top_k` for the specified category |
| `/threshold <category> <score>` | `threshold_handler` | Update `UserCategory.min_confidence` |
| `/sources` | `sources_handler` | List user's subscribed sources with last-scraped info |
| `/addsource <@channel>` | `addsource_handler` | Create source (if new) + subscribe user |
| `/rmsource <@channel>` | `rmsource_handler` | Unsubscribe user from source |
| `/digest` | `digest_handler` | Trigger immediate manual digest generation for this user |
| `/settings` | `settings_handler` | Show current settings (timezone, daily/weekly prefs) |
| `/stats` | `stats_handler` | Show pipeline stats: total messages, clusters, classified count |

**Registration flow (`/start`):**
1. Check if user with this `chat_id` already exists → welcome back message
2. If new: create `User` with `telegram_chat_id=update.effective_chat.id`, `name=update.effective_user.first_name`
3. Send welcome message with quick-start instructions

**Bot setup in main.py:**
- Create `Application` with `bot_token`
- Register all handlers
- Set webhook via `application.bot.set_webhook(url=WEBHOOK_URL + "/webhook")`
- Mount webhook endpoint: `POST /webhook` in FastAPI that feeds updates to the bot application
- On startup: initialize bot; on shutdown: close bot

### bot/delivery.py — Digest Sending

```python
async def send_digest(bot: Bot, chat_id: int, rendered_text: str) -> list[int]:
    """
    Send rendered digest to user. Handle Telegram's 4096 char limit.
    Returns list of sent message IDs.
    """
```

**Message splitting logic:**
1. If `len(rendered_text) <= 4096` → send as single message
2. If longer → split at category boundaries (find `\n\n` between sections)
3. If a single category section exceeds 4096 → split at item boundaries within that category
4. Send each chunk with `parse_mode=ParseMode.MARKDOWN_V2` and 0.5s delay between sends
5. Return all `message_id`s for tracking

**Markdown V2 escaping:**
- Telegram MarkdownV2 requires escaping: `_`, `*`, `[`, `]`, `(`, `)`, `~`, `` ` ``, `>`, `#`, `+`, `-`, `=`, `|`, `{`, `}`, `.`, `!`
- Escape special chars in user-generated content (summaries, entity names) before rendering
- Keep intentional formatting (bold, italic, links) unescaped

### pipelines/digest.py — Selection + Rendering

**Two digest types:**

| | Daily | Weekly |
|--|-------|--------|
| Window | Last 24 hours | Last 7 days |
| Confidence threshold | User's `daily_min_confidence` (default 0.5) | `max(per-category, weekly_min_confidence=0.7)` |
| Relevance floor | 0.2 | 0.4 |
| Ranking | category_confidence DESC, then relevance DESC | `relevance * category_confidence` composite DESC |
| Already-sent exclusion | Exclude previous daily items | CAN re-include daily items (best-of-week), exclude previous weekly |

**Selection algorithm:**

```python
async def select_items(user: User, digest_type: str,
                       subscriptions: list[UserCategory],
                       db: AsyncSession) -> dict[str, list[DigestCandidate]] | None:
    """
    Select top items per category for a user's digest.
    Returns {category_display_name: [items]} or None if nothing to send.
    """
```

1. Calculate time window based on digest type
2. Get already-sent message IDs from `digest_items` (scope depends on digest type)
3. For each user category subscription:
   a. Query classified, primary messages within window
   b. Filter by category score >= threshold AND relevance >= floor
   c. Exclude already-sent IDs
   d. Sort by ranking (type-dependent)
   e. Take top `sub.top_k` items
4. Return grouped results, or `None` if no items pass filters

**Rendering with Jinja2:**

```python
async def render_digest(user: User, digest_type: str,
                        items_by_category: dict, stats: dict) -> str:
    """Render digest using Jinja2 template. No SLM call."""
```

- Load template from `src/templates/`
- Pass: digest_type, date, categories with items, emoji map, stats
- Return rendered Markdown string

**Digest generation endpoint:**

```python
async def run_generate_digest(db: AsyncSession, settings: Settings,
                              digest_type: str) -> PipelineRunResult:
    """
    For each eligible user:
    1. Check if current hour matches their preferred hour (in their timezone)
    2. Check no digest already sent today/this week
    3. Select items
    4. Render digest
    5. Deliver via bot
    6. Record in digests + digest_items tables
    """
```

**Timezone handling:**
- `POST /generate-digest?type=daily` runs hourly via cron
- Iterate all active users
- Convert current UTC time to user's timezone
- Check if hour matches `user.daily_hour` (or `weekly_hour` on correct weekday)
- Check `digests` table: no existing digest of this type for today/this week
- If eligible → generate and send

### templates/daily_digest.md.j2

```jinja2
*Your Daily Digest — {{ date }}*

{% for cat_name, items in categories.items() %}
*{{ emojis.get(cat_name, '📌') }} {{ cat_name }}* ({{ items|length }})
{% for item in items %}
{{ '🔴' if item.relevance_score >= 0.7 else '🟡' if item.relevance_score >= 0.4 else '⚪' }} {{ item.summary }}
{% if item.content_url %}[source]({{ item.content_url }}){% endif %} _via {{ item.source_name }}_{% if item.cluster_size > 1 %} _(+{{ item.cluster_size - 1 }} similar)_{% endif %}

{% endfor %}
{% endfor %}
_{{ total_items }} items from {{ source_count }} sources · {{ dupes_removed }} duplicates filtered_
```

### templates/weekly_digest.md.j2

Same structure but with "Weekly Digest" header and potentially different emoji/formatting.

### Wire endpoints

- `POST /generate-digest?type=daily` — hourly cron trigger for daily digests
- `POST /generate-digest?type=weekly` — hourly cron trigger on Sundays for weekly digests
- `POST /webhook` — Telegram bot webhook receiver

---

## Tests (test_selection.py)

1. **Basic selection** — user with 1 category, 3 classified messages → top_k=2 selected
2. **Threshold filtering** — messages below confidence threshold excluded
3. **Relevance floor** — messages below relevance floor excluded
4. **Already-sent exclusion (daily)** — previously sent daily items not re-sent
5. **Weekly re-inclusion** — daily items CAN appear in weekly digest
6. **Weekly exclusion** — previous weekly items NOT in next weekly
7. **Multi-category** — same message appears in 2 categories for different users
8. **Empty digest** — no qualifying items → returns None, no message sent
9. **Message splitting** — digest >4096 chars splits correctly at category boundaries
10. **Timezone matching** — user at UTC+5 with daily_hour=8 gets digest at 08:00 their time

---

## Definition of Success

### Checklist

- [ ] All bot commands registered and responding: /start, /help, /categories, /addcat, /rmcat, /topk, /threshold, /sources, /addsource, /rmsource, /digest, /settings, /stats
- [ ] `/start` creates a new user with correct `telegram_chat_id`
- [ ] `/addcat` creates global category + user subscription
- [ ] Webhook endpoint `POST /webhook` receives and processes bot updates
- [ ] `select_items()` correctly filters by category confidence, relevance floor, and already-sent
- [ ] Daily digest excludes previous daily items; weekly CAN re-include daily items
- [ ] Jinja2 templates render valid Telegram MarkdownV2
- [ ] Digests exceeding 4096 chars are split at category boundaries
- [ ] `POST /generate-digest?type=daily` checks user timezone and sends at correct hour
- [ ] No duplicate digests: existing digest for today/this week prevents re-generation
- [ ] Digest and DigestItem records created in DB after delivery
- [ ] `test_selection.py` passes — all selection scenarios covered
- [ ] MarkdownV2 special characters properly escaped in user content
