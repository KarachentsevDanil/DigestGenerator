"""Interactive Telethon session setup.

Run once to authenticate with your Telegram account:
    uv run python scripts/setup_session.py

Creates a session file at data/sessions/digest_session.session
that persists auth for all subsequent headless runs.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telethon import TelegramClient

from src.config import get_settings


async def main():
    settings = get_settings()

    if not settings.telegram_api_id or not settings.telegram_api_hash:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        print("Get them from https://my.telegram.org")
        sys.exit(1)

    session_dir = Path(__file__).resolve().parent.parent / "data" / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / "digest_session"

    print("Setting up Telethon session...")
    print(f"Session file: {session_path}.session")
    print()

    client = TelegramClient(
        str(session_path),
        settings.telegram_api_id,
        settings.telegram_api_hash,
    )

    await client.start()
    me = await client.get_me()
    print(f"\nAuthenticated as: {me.first_name} ({me.phone})")
    print("Session file created successfully. You can now run the scraper headlessly.")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
