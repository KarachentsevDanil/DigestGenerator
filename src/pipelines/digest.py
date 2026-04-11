from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog
from jinja2 import Environment, FileSystemLoader
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.bot.delivery import escape_markdown_v2
from src.config import Settings
from src.db.models import (
    Category,
    Digest,
    DigestItem,
    Message,
    Source,
    User,
    UserCategory,
)

log = structlog.get_logger()

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

# Default category emojis
CATEGORY_EMOJIS: dict[str, str] = {
    "ai_ml": "\U0001f916",
    "crypto": "\U0001f4b0",
    "tech_industry": "\U0001f4bb",
    "geopolitics": "\U0001f30d",
    "science": "\U0001f52c",
    "finance": "\U0001f4c8",
    "regulation": "\u2696\ufe0f",
    "startups": "\U0001f680",
}


@dataclass
class PipelineRunResult:
    stage: str
    processed: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class DigestCandidate:
    message: Message
    category_name: str
    category_display_name: str
    confidence: float
    source_name: str
    cluster_size: int

    @property
    def summary(self) -> str:
        return self.message.summary or self.message.content[:200]

    @property
    def relevance_score(self) -> float:
        return self.message.relevance_score or 0.0

    @property
    def content_url(self) -> str | None:
        return self.message.content_url


def _now_in_user_tz(user: User) -> datetime:
    """Get current time in user's timezone."""
    try:
        tz = ZoneInfo(user.timezone)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz)


async def _get_already_sent_ids(
    db: AsyncSession, user_id: int, digest_type: str
) -> set[int]:
    """Get message IDs already sent to this user in previous digests."""
    if digest_type == "daily":
        # Exclude items from ALL previous daily digests
        result = await db.execute(
            select(DigestItem.message_id)
            .join(Digest, DigestItem.digest_id == Digest.id)
            .where(Digest.user_id == user_id, Digest.digest_type == "daily")
        )
    else:
        # Weekly: exclude only previous weekly digest items (daily items CAN re-appear)
        result = await db.execute(
            select(DigestItem.message_id)
            .join(Digest, DigestItem.digest_id == Digest.id)
            .where(Digest.user_id == user_id, Digest.digest_type == "weekly")
        )
    return set(result.scalars().all())


async def _get_cluster_size(db: AsyncSession, message: Message) -> int:
    """Get the number of messages in this message's cluster."""
    if not message.dedup_cluster_id:
        return 1
    result = await db.execute(
        select(func.count(Message.id)).where(
            Message.dedup_cluster_id == message.dedup_cluster_id
        )
    )
    return result.scalar_one()


async def select_items(
    db: AsyncSession,
    user: User,
    digest_type: str,
    settings: Settings,
) -> dict[str, list[DigestCandidate]] | None:
    """
    Select top items per category for a user's digest.
    Returns {category_display_name: [candidates]} or None if nothing qualifies.
    """
    window = timedelta(hours=24) if digest_type == "daily" else timedelta(days=7)
    window_start = datetime.now(UTC) - window

    relevance_floor = (
        settings.digest.daily_relevance_floor
        if digest_type == "daily"
        else settings.digest.weekly_relevance_floor
    )

    # Get user's category subscriptions
    subs_result = await db.execute(
        select(UserCategory, Category)
        .join(Category, UserCategory.category_id == Category.id)
        .where(UserCategory.user_id == user.id)
    )
    subscriptions = list(subs_result.all())

    if not subscriptions:
        return None

    already_sent = await _get_already_sent_ids(db, user.id, digest_type)

    result: dict[str, list[DigestCandidate]] = {}

    for uc, category in subscriptions:
        threshold = uc.min_confidence
        if digest_type == "weekly":
            threshold = max(threshold, user.weekly_min_confidence)

        # Query classified primary messages in the time window
        query = select(Message).where(
            Message.is_cluster_primary.is_(True),
            Message.status == "classified",
            Message.published_at >= window_start,
        )
        if already_sent:
            query = query.where(Message.id.notin_(already_sent))

        msgs_result = await db.execute(query)
        messages = list(msgs_result.scalars().all())

        candidates: list[DigestCandidate] = []
        for msg in messages:
            scores = msg.category_scores_json or {}
            cat_score = scores.get(category.name, 0.0)

            if cat_score < threshold:
                continue
            if (msg.relevance_score or 0.0) < relevance_floor:
                continue

            # Get source name
            source_result = await db.execute(
                select(Source).where(Source.id == msg.source_id)
            )
            source = source_result.scalar_one_or_none()
            source_name = (
                source.display_name or source.source_identifier
                if source
                else "unknown"
            )

            cluster_size = await _get_cluster_size(db, msg)

            candidates.append(
                DigestCandidate(
                    message=msg,
                    category_name=category.name,
                    category_display_name=category.display_name,
                    confidence=cat_score,
                    source_name=source_name,
                    cluster_size=cluster_size,
                )
            )

        # Ranking per spec:
        # daily = category_confidence DESC, then relevance DESC
        # weekly = relevance * category_confidence composite DESC
        if digest_type == "daily":
            candidates.sort(
                key=lambda c: (c.confidence, c.relevance_score),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda c: c.relevance_score * c.confidence,
                reverse=True,
            )

        # Take top_k
        top = candidates[: uc.top_k]
        if top:
            result[category.display_name] = top

    return result if result else None


def render_digest(
    digest_type: str,
    items_by_category: dict[str, list[DigestCandidate]],
    dupes_removed: int = 0,
) -> str:
    """Render digest using Jinja2 template."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
    )
    template_name = f"{digest_type}_digest.md.j2"
    template = env.get_template(template_name)

    # Count totals
    total_items = sum(len(items) for items in items_by_category.values())
    source_names = set()
    for items in items_by_category.values():
        for item in items:
            source_names.add(item.source_name)

    # Escape summaries and source names for MarkdownV2
    categories = {}
    for cat_name, items in items_by_category.items():
        escaped_items = []
        for item in items:
            escaped_items.append(
                type("EscapedItem", (), {
                    "summary": escape_markdown_v2(item.summary),
                    "relevance_score": item.relevance_score,
                    "content_url": item.content_url,
                    "source_name": escape_markdown_v2(item.source_name),
                    "cluster_size": item.cluster_size,
                })()
            )
        categories[escape_markdown_v2(cat_name)] = escaped_items

    return template.render(
        date=escape_markdown_v2(datetime.now(UTC).strftime("%Y-%m-%d")),
        categories=categories,
        emojis=CATEGORY_EMOJIS,
        total_items=total_items,
        source_count=len(source_names),
        dupes_removed=dupes_removed,
    )


async def generate_user_digest(
    db: AsyncSession,
    user: User,
    digest_type: str,
    settings: Settings,
) -> bool:
    """
    Generate and deliver a digest for a single user.
    Returns True if a digest was sent, False if nothing to send.
    """
    items = await select_items(db, user, digest_type, settings)
    if not items:
        return False

    # Render
    rendered = render_digest(digest_type, items)

    # Deliver via bot
    from telegram import Bot

    bot = Bot(token=settings.telegram_bot_token)

    from src.bot.delivery import send_digest

    message_ids = await send_digest(bot, user.telegram_chat_id, rendered)

    # Record digest
    window = timedelta(hours=24) if digest_type == "daily" else timedelta(days=7)
    now = datetime.now(UTC)
    total_items = sum(len(v) for v in items.values())

    digest = Digest(
        user_id=user.id,
        digest_type=digest_type,
        window_start=now - window,
        window_end=now,
        generated_at=now,
        delivered_at=now if message_ids else None,
        telegram_message_id=message_ids[0] if message_ids else None,
        item_count=total_items,
    )
    db.add(digest)
    await db.commit()
    await db.refresh(digest)

    # Record digest items
    for cat_name, cat_items in items.items():
        for rank, item in enumerate(cat_items):
            di = DigestItem(
                digest_id=digest.id,
                message_id=item.message.id,
                category_name=item.category_name,
                rank_in_category=rank + 1,
                confidence_score=item.confidence,
            )
            db.add(di)

    await db.commit()

    log.info(
        "digest_generated",
        user_id=user.id,
        type=digest_type,
        items=total_items,
        messages_sent=len(message_ids),
    )

    return True


async def run_generate_digest(
    db: AsyncSession, settings: Settings, digest_type: str
) -> PipelineRunResult:
    """
    Check all users and generate digests for those whose schedule matches.
    Called hourly by cron.
    """
    result = PipelineRunResult(stage="digest")

    users_result = await db.execute(
        select(User).where(User.is_active.is_(True))
    )
    users = list(users_result.scalars().all())

    for user in users:
        try:
            user_now = _now_in_user_tz(user)
            current_hour = user_now.hour
            current_weekday = user_now.weekday()

            should_send = False
            if digest_type == "daily" and user.daily_enabled:
                should_send = current_hour == user.daily_hour
            elif digest_type == "weekly" and user.weekly_enabled:
                should_send = (
                    current_weekday == user.weekly_day
                    and current_hour == user.weekly_hour
                )

            if not should_send:
                result.skipped += 1
                continue

            # Check if digest already sent today/this week
            now = datetime.now(UTC)
            if digest_type == "daily":
                check_since = now - timedelta(hours=20)  # allow some buffer
            else:
                check_since = now - timedelta(days=6)

            existing = await db.execute(
                select(Digest).where(
                    Digest.user_id == user.id,
                    Digest.digest_type == digest_type,
                    Digest.generated_at >= check_since,
                )
            )
            if existing.scalar_one_or_none():
                result.skipped += 1
                continue

            sent = await generate_user_digest(db, user, digest_type, settings)
            if sent:
                result.processed += 1
            else:
                result.skipped += 1

        except Exception:
            log.exception("digest_user_error", user_id=user.id)
            result.failed += 1

    return result
