from __future__ import annotations

import asyncio
import re

import structlog
from telegram import Bot
from telegram.constants import ParseMode

log = structlog.get_logger()

# Characters that must be escaped in Telegram MarkdownV2
_MARKDOWNV2_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _MARKDOWNV2_SPECIAL.sub(r"\\\1", text)


def _split_at_boundaries(text: str, max_len: int = 4096) -> list[str]:
    """
    Split text at section boundaries (double newlines).
    Falls back to splitting at single newlines if a section is too long.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    sections = text.split("\n\n")
    current = ""

    for section in sections:
        candidate = f"{current}\n\n{section}" if current else section

        if len(candidate) <= max_len:
            current = candidate
        else:
            # Current chunk is full, save it
            if current:
                chunks.append(current.strip())

            # Check if this single section is too long
            if len(section) > max_len:
                # Split within section at newline boundaries
                lines = section.split("\n")
                current = ""
                for line in lines:
                    line_candidate = f"{current}\n{line}" if current else line
                    if len(line_candidate) <= max_len:
                        current = line_candidate
                    else:
                        if current:
                            chunks.append(current.strip())
                        # If a single line is too long, force-split it
                        if len(line) > max_len:
                            for i in range(0, len(line), max_len):
                                chunks.append(line[i : i + max_len])
                            current = ""
                        else:
                            current = line
            else:
                current = section

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


async def send_digest(
    bot: Bot,
    chat_id: int,
    rendered_text: str,
) -> list[int]:
    """
    Send rendered digest to user. Handles Telegram's 4096 char limit
    by splitting at category boundaries.
    Returns list of sent message IDs.
    """
    chunks = _split_at_boundaries(rendered_text, max_len=4096)
    message_ids: list[int] = []

    for i, chunk in enumerate(chunks):
        try:
            msg = await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                parse_mode=ParseMode.MARKDOWN_V2,
            )
            message_ids.append(msg.message_id)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
        except Exception:
            log.exception("delivery_send_error", chat_id=chat_id, chunk=i)
            # Try without markdown as fallback
            try:
                msg = await bot.send_message(chat_id=chat_id, text=chunk)
                message_ids.append(msg.message_id)
            except Exception:
                log.exception("delivery_fallback_error", chat_id=chat_id, chunk=i)

    return message_ids
