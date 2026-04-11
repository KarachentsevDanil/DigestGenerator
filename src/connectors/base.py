from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.db.models import Source


@dataclass
class NormalizedMessage:
    external_id: str
    content: str
    content_url: str | None = None
    media_type: str = "text"
    published_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_metadata: dict = field(default_factory=dict)
    forwarded_from_channel: str | None = None
    forwarded_message_id: str | None = None


class BaseConnector(ABC):
    source_type: str

    @abstractmethod
    async def fetch_new(self, source: Source) -> list[NormalizedMessage]:
        """Fetch new messages since source's cursor. Update cursor internally."""
        ...

    @abstractmethod
    async def validate_source(self, identifier: str) -> bool:
        """Check if source exists and is accessible."""
        ...
