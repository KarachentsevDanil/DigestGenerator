from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

import structlog
from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, FloodWaitError
from telethon.tl.types import MessageEntityTextUrl, MessageEntityUrl

from src.config import Settings
from src.connectors.base import BaseConnector, NormalizedMessage
from src.db.models import Source

log = structlog.get_logger()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _normalize_text(text: str) -> str:
    """Strip excessive whitespace, collapse multiple newlines."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_urls_from_entities(message) -> str | None:
    """Extract the first URL from message entities."""
    if not message.entities:
        return None
    for entity in message.entities:
        if isinstance(entity, MessageEntityTextUrl):
            return entity.url
        if isinstance(entity, MessageEntityUrl):
            start = entity.offset
            end = start + entity.length
            return (message.message or "")[start:end]
    return None


class TelegramConnector(BaseConnector):
    source_type = "telegram"

    def __init__(self, settings: Settings):
        self._settings = settings
        session_path = _PROJECT_ROOT / "data" / "sessions" / "digest_session"
        self._client = TelegramClient(
            str(session_path),
            settings.telegram_api_id,
            settings.telegram_api_hash,
        )
        self._connected = False

    async def connect(self) -> None:
        if not self._connected:
            await self._client.start()
            self._connected = True
            log.info("telegram_connector_connected")

    async def disconnect(self) -> None:
        if self._connected:
            await self._client.disconnect()
            self._connected = False

    async def fetch_new(self, source: Source) -> list[NormalizedMessage]:
        """Fetch new messages from a Telegram channel since the last cursor."""
        await self.connect()

        identifier = source.source_identifier
        min_id = int(source.last_scraped_external_id) if source.last_scraped_external_id else 0

        messages: list[NormalizedMessage] = []
        try:
            entity = await self._client.get_entity(identifier)
            async for msg in self._client.iter_messages(
                entity,
                limit=self._settings.scrape.messages_per_channel,
                min_id=min_id,
            ):
                text = msg.message or (msg.caption if hasattr(msg, "caption") else None) or ""
                if not text.strip():
                    continue

                # Detect media type
                media_type = "text"
                if msg.photo:
                    media_type = "photo"
                elif msg.video:
                    media_type = "video"
                elif msg.document:
                    media_type = "document"

                # Extract forwarded-from info
                forwarded_from_channel = None
                forwarded_message_id = None
                if msg.forward:
                    fwd = msg.forward
                    if hasattr(fwd, "chat") and fwd.chat:
                        has_username = (
                            hasattr(fwd.chat, "username") and fwd.chat.username
                        )
                        forwarded_from_channel = (
                            f"@{fwd.chat.username}" if has_username
                            else str(fwd.chat.id)
                        )
                    elif hasattr(fwd, "from_id") and fwd.from_id:
                        forwarded_from_channel = str(fwd.from_id)
                    if hasattr(fwd, "channel_post") and fwd.channel_post:
                        forwarded_message_id = str(fwd.channel_post)

                content = _normalize_text(text)
                content_url = _extract_urls_from_entities(msg)

                # Build raw metadata
                raw_metadata = {}
                if hasattr(msg, "views") and msg.views is not None:
                    raw_metadata["views"] = msg.views
                if hasattr(msg, "forwards") and msg.forwards is not None:
                    raw_metadata["forwards"] = msg.forwards

                published = msg.date
                if published and published.tzinfo is None:
                    published = published.replace(tzinfo=UTC)

                messages.append(
                    NormalizedMessage(
                        external_id=str(msg.id),
                        content=content,
                        content_url=content_url,
                        media_type=media_type,
                        published_at=published or datetime.now(UTC),
                        raw_metadata=raw_metadata,
                        forwarded_from_channel=forwarded_from_channel,
                        forwarded_message_id=forwarded_message_id,
                    )
                )

        except FloodWaitError as e:
            log.warning("telegram_flood_wait", seconds=e.seconds, source=identifier)
            await asyncio.sleep(min(e.seconds, 60))
            return []
        except ChannelPrivateError:
            log.warning("telegram_channel_private", source=identifier)
            return []
        except Exception:
            log.exception("telegram_fetch_error", source=identifier)
            return []

        # Return sorted by message ID ascending (oldest first)
        messages.sort(key=lambda m: int(m.external_id))
        return messages

    async def validate_source(self, identifier: str) -> bool:
        """Check if a Telegram channel exists and is accessible."""
        await self.connect()
        try:
            await self._client.get_entity(identifier)
            return True
        except Exception:
            return False
