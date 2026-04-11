from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_settings

_engine = None
_session_factory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_url = settings.database.url

        # Ensure the data directory exists for SQLite
        if "sqlite" in db_url:
            db_path = db_url.split("///")[-1] if "///" in db_url else None
            if db_path:
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _engine = create_async_engine(db_url, echo=False)
    return _engine


def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            _get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async generator yielding a DB session. Use with FastAPI Depends()."""
    factory = _get_session_factory()
    async with factory() as session:
        yield session


async def init_db() -> None:
    """Create all tables. For dev/test use; production uses Alembic."""
    from src.db.models import Base

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
