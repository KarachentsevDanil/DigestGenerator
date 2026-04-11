from __future__ import annotations

import structlog
from slugify import slugify
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from src.db.models import Category, Source, User, UserCategory, UserKnowledge, UserSource
from src.db.session import get_session

log = structlog.get_logger()


async def _get_db() -> AsyncSession:
    """Get a DB session for bot handlers."""
    async for session in get_session():
        return session
    raise RuntimeError("Failed to get DB session")


async def _get_or_create_user(db: AsyncSession, update: Update) -> User:
    """Get existing user or create a new one from the Telegram update."""
    chat_id = update.effective_chat.id
    result = await db.execute(
        select(User).where(User.telegram_chat_id == chat_id)
    )
    user = result.scalar_one_or_none()
    if user:
        return user

    user = User(
        name=update.effective_user.first_name or "Unknown",
        telegram_chat_id=chat_id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — register user."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        await update.message.reply_text(
            f"Welcome, {user.name}! You're registered.\n\n"
            "Use /help to see available commands.\n"
            "Start by adding categories with /addcat and sources with /addsource."
        )
    finally:
        await db.close()


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — list commands."""
    await update.message.reply_text(
        "Available commands:\n\n"
        "/start — Register\n"
        "/categories — List your categories\n"
        "/addcat Name | Description — Add a category\n"
        "/rmcat Name — Remove a category\n"
        "/topk Category N — Set max items per category\n"
        "/threshold Category 0.7 — Set confidence threshold\n"
        "/sources — List your sources\n"
        "/addsource @channel — Subscribe to a channel\n"
        "/rmsource @channel — Unsubscribe from a channel\n"
        "/digest — Get a digest now\n"
        "/settings — View your settings\n"
        "/stats — Pipeline statistics\n"
        "/knowledge — Your knowledge graph\n"
        "/help — This message"
    )


async def categories_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /categories — list user's categories."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        result = await db.execute(
            select(UserCategory, Category)
            .join(Category, UserCategory.category_id == Category.id)
            .where(UserCategory.user_id == user.id)
        )
        rows = result.all()

        if not rows:
            await update.message.reply_text(
                "No categories yet. Add one with /addcat Name | Description"
            )
            return

        lines = ["Your categories:\n"]
        for uc, cat in rows:
            lines.append(
                f"  {cat.display_name} (top_k={uc.top_k}, "
                f"threshold={uc.min_confidence})"
            )
        await update.message.reply_text("\n".join(lines))
    finally:
        await db.close()


async def addcat_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /addcat Name | Description — add a category."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        text = " ".join(context.args) if context.args else ""

        if "|" in text:
            name_part, desc_part = text.split("|", 1)
        else:
            name_part = text
            desc_part = text

        display_name = name_part.strip()
        description = desc_part.strip()

        if not display_name:
            await update.message.reply_text("Usage: /addcat Name | Description")
            return

        cat_slug = slugify(display_name, separator="_")

        # Get or create the global category
        result = await db.execute(
            select(Category).where(Category.name == cat_slug)
        )
        category = result.scalar_one_or_none()

        if not category:
            category = Category(
                name=cat_slug,
                display_name=display_name,
                description=description,
                created_by_user_id=user.id,
            )
            db.add(category)
            await db.commit()
            await db.refresh(category)

        # Check if already subscribed
        result = await db.execute(
            select(UserCategory).where(
                UserCategory.user_id == user.id,
                UserCategory.category_id == category.id,
            )
        )
        if result.scalar_one_or_none():
            await update.message.reply_text(
                f"Already subscribed to {category.display_name}"
            )
            return

        uc = UserCategory(user_id=user.id, category_id=category.id)
        db.add(uc)
        await db.commit()

        await update.message.reply_text(
            f"Subscribed to {category.display_name} "
            f"(top_k=5, threshold=0.5)"
        )
    finally:
        await db.close()


async def rmcat_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /rmcat Name — unsubscribe from category."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        name = " ".join(context.args) if context.args else ""
        cat_slug = slugify(name, separator="_")

        result = await db.execute(
            select(UserCategory, Category)
            .join(Category, UserCategory.category_id == Category.id)
            .where(UserCategory.user_id == user.id, Category.name == cat_slug)
        )
        row = result.first()

        if not row:
            await update.message.reply_text(f"Not subscribed to '{name}'")
            return

        uc, _cat = row
        await db.delete(uc)
        await db.commit()
        await update.message.reply_text(f"Unsubscribed from {name}")
    finally:
        await db.close()


async def topk_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /topk Category N — set max items."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        args = context.args or []
        if len(args) < 2:
            await update.message.reply_text("Usage: /topk Category N")
            return

        cat_name = slugify(args[0], separator="_")
        try:
            top_k = int(args[1])
        except ValueError:
            await update.message.reply_text("N must be a number")
            return

        result = await db.execute(
            select(UserCategory, Category)
            .join(Category, UserCategory.category_id == Category.id)
            .where(UserCategory.user_id == user.id, Category.name == cat_name)
        )
        row = result.first()

        if not row:
            await update.message.reply_text(f"Not subscribed to '{args[0]}'")
            return

        uc, cat = row
        uc.top_k = top_k
        await db.commit()
        await update.message.reply_text(
            f"Set top_k={top_k} for {cat.display_name}"
        )
    finally:
        await db.close()


async def threshold_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /threshold Category 0.7 — set confidence threshold."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        args = context.args or []
        if len(args) < 2:
            await update.message.reply_text("Usage: /threshold Category 0.7")
            return

        cat_name = slugify(args[0], separator="_")
        try:
            threshold = float(args[1])
        except ValueError:
            await update.message.reply_text("Threshold must be a number (0.0-1.0)")
            return

        result = await db.execute(
            select(UserCategory, Category)
            .join(Category, UserCategory.category_id == Category.id)
            .where(UserCategory.user_id == user.id, Category.name == cat_name)
        )
        row = result.first()

        if not row:
            await update.message.reply_text(f"Not subscribed to '{args[0]}'")
            return

        uc, cat = row
        uc.min_confidence = max(0.0, min(1.0, threshold))
        await db.commit()
        await update.message.reply_text(
            f"Set threshold={uc.min_confidence} for {cat.display_name}"
        )
    finally:
        await db.close()


async def sources_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /sources — list user's sources."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        result = await db.execute(
            select(UserSource, Source)
            .join(Source, UserSource.source_id == Source.id)
            .where(UserSource.user_id == user.id)
        )
        rows = result.all()

        if not rows:
            await update.message.reply_text(
                "No sources yet. Add one with /addsource @channel"
            )
            return

        lines = ["Your sources:\n"]
        for _us, source in rows:
            last = (
                source.last_scraped_at.strftime("%Y-%m-%d %H:%M")
                if source.last_scraped_at else "never"
            )
            lines.append(f"  {source.source_identifier} (last scraped: {last})")
        await update.message.reply_text("\n".join(lines))
    finally:
        await db.close()


async def addsource_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /addsource @channel — subscribe to channel."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        identifier = " ".join(context.args) if context.args else ""

        if not identifier:
            await update.message.reply_text("Usage: /addsource @channel_name")
            return

        if not identifier.startswith("@"):
            identifier = f"@{identifier}"

        # Get or create source
        result = await db.execute(
            select(Source).where(
                Source.source_type == "telegram",
                Source.source_identifier == identifier,
            )
        )
        source = result.scalar_one_or_none()

        if not source:
            source = Source(
                source_type="telegram",
                source_identifier=identifier,
                display_name=identifier,
            )
            db.add(source)
            await db.commit()
            await db.refresh(source)

        # Check if already subscribed
        result = await db.execute(
            select(UserSource).where(
                UserSource.user_id == user.id,
                UserSource.source_id == source.id,
            )
        )
        if result.scalar_one_or_none():
            await update.message.reply_text(f"Already subscribed to {identifier}")
            return

        us = UserSource(user_id=user.id, source_id=source.id)
        db.add(us)
        await db.commit()

        await update.message.reply_text(f"Subscribed to {identifier}")
    finally:
        await db.close()


async def rmsource_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /rmsource @channel — unsubscribe from channel."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        identifier = " ".join(context.args) if context.args else ""

        if not identifier.startswith("@"):
            identifier = f"@{identifier}"

        result = await db.execute(
            select(UserSource, Source)
            .join(Source, UserSource.source_id == Source.id)
            .where(
                UserSource.user_id == user.id,
                Source.source_identifier == identifier,
            )
        )
        row = result.first()

        if not row:
            await update.message.reply_text(f"Not subscribed to {identifier}")
            return

        us, _source = row
        await db.delete(us)
        await db.commit()
        await update.message.reply_text(f"Unsubscribed from {identifier}")
    finally:
        await db.close()


async def settings_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /settings — show user settings."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        await update.message.reply_text(
            f"Your settings:\n\n"
            f"Timezone: {user.timezone}\n"
            f"Daily digest: {'ON' if user.daily_enabled else 'OFF'} "
            f"at {user.daily_hour:02d}:00\n"
            f"Weekly digest: {'ON' if user.weekly_enabled else 'OFF'} "
            f"on {days[user.weekly_day]} at {user.weekly_hour:02d}:00\n"
            f"Daily min confidence: {user.daily_min_confidence}\n"
            f"Weekly min confidence: {user.weekly_min_confidence}"
        )
    finally:
        await db.close()


async def stats_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /stats — show pipeline stats."""
    from sqlalchemy import func

    from src.db.models import Message

    db = await _get_db()
    try:
        result = await db.execute(
            select(Message.status, func.count(Message.id)).group_by(Message.status)
        )
        counts = dict(result.all())
        total = sum(counts.values())

        lines = [f"Pipeline stats:\n\nTotal messages: {total}"]
        for status, count in sorted(counts.items()):
            lines.append(f"  {status}: {count}")

        await update.message.reply_text("\n".join(lines))
    finally:
        await db.close()


async def digest_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /digest — trigger manual digest."""
    await update.message.reply_text(
        "Generating your digest... This may take a moment."
    )
    # The actual generation is triggered via POST /generate-digest
    # For manual trigger, we'll import and call directly
    db = await _get_db()
    try:
        from src.config import get_settings
        from src.pipelines.digest import generate_user_digest

        user = await _get_or_create_user(db, update)
        settings = get_settings()
        result = await generate_user_digest(db, user, "daily", settings)
        if result:
            await update.message.reply_text("Digest sent!")
        else:
            await update.message.reply_text("No new items for your digest.")
    except Exception:
        log.exception("manual_digest_error")
        await update.message.reply_text("Error generating digest. Try again later.")
    finally:
        await db.close()


async def knowledge_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /knowledge — show knowledge graph stats or search."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        query_text = " ".join(context.args) if context.args else ""

        if query_text:
            # Search for specific entity
            search = f"%{query_text}%"
            result = await db.execute(
                select(UserKnowledge)
                .where(
                    UserKnowledge.user_id == user.id,
                    UserKnowledge.entity_name.ilike(search),
                )
                .limit(5)
            )
            entities = list(result.scalars().all())

            if not entities:
                await update.message.reply_text(
                    f"No knowledge about '{query_text}'"
                )
                return

            lines = []
            for e in entities:
                lines.append(
                    f"{e.entity_name} ({e.entity_type or 'unknown'})\n"
                    f"  First seen: {e.first_seen_at.strftime('%Y-%m-%d')}\n"
                    f"  Last seen: {e.last_seen_at.strftime('%Y-%m-%d')}\n"
                    f"  Encountered: {e.encounter_count} times\n"
                    f"  Categories: {', '.join(e.categories or [])}"
                )
            await update.message.reply_text("\n\n".join(lines))
        else:
            # Show stats
            from sqlalchemy import func as sa_func

            total_result = await db.execute(
                select(sa_func.count(UserKnowledge.id)).where(
                    UserKnowledge.user_id == user.id
                )
            )
            total = total_result.scalar_one()

            if total == 0:
                await update.message.reply_text(
                    "Your knowledge graph is empty. "
                    "It will grow as you receive digests."
                )
                return

            # Recent entities
            recent_result = await db.execute(
                select(UserKnowledge)
                .where(UserKnowledge.user_id == user.id)
                .order_by(UserKnowledge.last_seen_at.desc())
                .limit(5)
            )
            recent = list(recent_result.scalars().all())

            lines = [f"Your Knowledge Graph\n\nTotal entities: {total}\n\nRecent:"]
            for e in recent:
                lines.append(
                    f"- {e.entity_name} ({e.entity_type or '?'}) "
                    f"— seen {e.encounter_count} times"
                )
            await update.message.reply_text("\n".join(lines))
    finally:
        await db.close()


async def forget_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle /forget <entity> — remove entity from knowledge graph."""
    db = await _get_db()
    try:
        user = await _get_or_create_user(db, update)
        name = " ".join(context.args) if context.args else ""

        if not name:
            await update.message.reply_text("Usage: /forget Entity Name")
            return

        canonical = slugify(name, separator="_")
        result = await db.execute(
            select(UserKnowledge).where(
                UserKnowledge.user_id == user.id,
                UserKnowledge.canonical_name == canonical,
            )
        )
        entity = result.scalar_one_or_none()

        if not entity:
            await update.message.reply_text(f"No knowledge of '{name}'")
            return

        await db.delete(entity)
        await db.commit()
        await update.message.reply_text(
            f"Forgot '{entity.entity_name}'. "
            f"It will appear as new in future digests."
        )
    finally:
        await db.close()


def register_handlers(application: Application) -> None:
    """Register all bot command handlers."""
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("categories", categories_handler))
    application.add_handler(CommandHandler("addcat", addcat_handler))
    application.add_handler(CommandHandler("rmcat", rmcat_handler))
    application.add_handler(CommandHandler("topk", topk_handler))
    application.add_handler(CommandHandler("threshold", threshold_handler))
    application.add_handler(CommandHandler("sources", sources_handler))
    application.add_handler(CommandHandler("addsource", addsource_handler))
    application.add_handler(CommandHandler("rmsource", rmsource_handler))
    application.add_handler(CommandHandler("settings", settings_handler))
    application.add_handler(CommandHandler("stats", stats_handler))
    application.add_handler(CommandHandler("digest", digest_handler))
    application.add_handler(CommandHandler("knowledge", knowledge_handler))
    application.add_handler(CommandHandler("forget", forget_handler))
