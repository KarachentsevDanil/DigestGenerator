from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings, get_settings
from src.db.session import get_session


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an async DB session."""
    async for session in get_session():
        yield session


def get_app_settings() -> Settings:
    """FastAPI dependency: returns cached settings."""
    return get_settings()
