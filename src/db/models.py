from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    telegram_chat_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    daily_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    daily_hour: Mapped[int] = mapped_column(Integer, default=8)
    weekly_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    weekly_day: Mapped[int] = mapped_column(Integer, default=6)  # 0=Mon, 6=Sun
    weekly_hour: Mapped[int] = mapped_column(Integer, default=9)
    daily_min_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    weekly_min_confidence: Mapped[float] = mapped_column(Float, default=0.7)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    categories: Mapped[list[UserCategory]] = relationship(back_populates="user")
    sources: Mapped[list[UserSource]] = relationship(back_populates="user")
    digests: Mapped[list[Digest]] = relationship(back_populates="user")
    knowledge: Mapped[list[UserKnowledge]] = relationship(back_populates="user")


class Category(Base):
    __tablename__ = "categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user_categories: Mapped[list[UserCategory]] = relationship(back_populates="category")
    created_by: Mapped[User] = relationship()


class UserCategory(Base):
    __tablename__ = "user_categories"

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), primary_key=True
    )
    category_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("categories.id"), primary_key=True
    )
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    min_confidence: Mapped[float] = mapped_column(Float, default=0.5)

    user: Mapped[User] = relationship(back_populates="categories")
    category: Mapped[Category] = relationship(back_populates="user_categories")


class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (
        UniqueConstraint("source_type", "source_identifier", name="uq_source_type_identifier"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(50), default="telegram")
    source_identifier: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_scraped_external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_scraped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user_sources: Mapped[list[UserSource]] = relationship(back_populates="source")
    messages: Mapped[list[Message]] = relationship(back_populates="source")


class UserSource(Base):
    __tablename__ = "user_sources"

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), primary_key=True
    )
    source_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sources.id"), primary_key=True
    )

    user: Mapped[User] = relationship(back_populates="sources")
    source: Mapped[Source] = relationship(back_populates="user_sources")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_source_external_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[int] = mapped_column(Integer, ForeignKey("sources.id"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_type: Mapped[str] = mapped_column(String(50), default="text")
    published_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Pipeline status
    status: Mapped[str] = mapped_column(String(50), default="unprocessed", index=True)

    # Embeddings
    embedding_vector: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    minhash_signature: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # Dedup
    dedup_cluster_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    is_cluster_primary: Mapped[bool] = mapped_column(Boolean, default=True)
    deduplicated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Classification
    category_scores_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    entities_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    classified_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Knowledge extraction (phase 7 v2)
    relations_json: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Telegram metadata
    forwarded_from_channel: Mapped[str | None] = mapped_column(String(255), nullable=True)
    forwarded_message_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    raw_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    source: Mapped[Source] = relationship(back_populates="messages")
    digest_items: Mapped[list[DigestItem]] = relationship(back_populates="message")


class Digest(Base):
    __tablename__ = "digests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    digest_type: Mapped[str] = mapped_column(String(20), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    telegram_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    item_count: Mapped[int] = mapped_column(Integer, default=0)

    user: Mapped[User] = relationship(back_populates="digests")
    items: Mapped[list[DigestItem]] = relationship(back_populates="digest")


class DigestItem(Base):
    __tablename__ = "digest_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    digest_id: Mapped[int] = mapped_column(Integer, ForeignKey("digests.id"), nullable=False)
    message_id: Mapped[int] = mapped_column(Integer, ForeignKey("messages.id"), nullable=False)
    category_name: Mapped[str] = mapped_column(String(100), nullable=False)
    rank_in_category: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    digest: Mapped[Digest] = relationship(back_populates="items")
    message: Mapped[Message] = relationship(back_populates="digest_items")


class UserKnowledge(Base):
    __tablename__ = "user_knowledge"
    __table_args__ = (
        UniqueConstraint("user_id", "canonical_name", name="uq_user_canonical_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    entity_name: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    encounter_count: Mapped[int] = mapped_column(Integer, default=1)
    categories: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    user: Mapped[User] = relationship(back_populates="knowledge")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="running")
    processed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Knowledge Graph v2 models (Phase 7)
# ---------------------------------------------------------------------------


class KnowledgeEntity(Base):
    __tablename__ = "knowledge_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    properties_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    aliases: Mapped[list[EntityAlias]] = relationship(back_populates="entity")
    outgoing_relations: Mapped[list[KnowledgeRelation]] = relationship(
        back_populates="source_entity",
        foreign_keys="KnowledgeRelation.source_entity_id",
    )
    incoming_relations: Mapped[list[KnowledgeRelation]] = relationship(
        back_populates="target_entity",
        foreign_keys="KnowledgeRelation.target_entity_id",
    )
    user_exposures: Mapped[list[UserEntityExposure]] = relationship(
        back_populates="entity",
    )


class KnowledgeRelation(Base):
    __tablename__ = "knowledge_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_entities.id"), nullable=False
    )
    target_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_entities.id"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    properties_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    first_observed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_observed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    observation_count: Mapped[int] = mapped_column(Integer, default=1)
    source_message_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("messages.id"), nullable=True
    )

    source_entity: Mapped[KnowledgeEntity] = relationship(
        back_populates="outgoing_relations",
        foreign_keys=[source_entity_id],
    )
    target_entity: Mapped[KnowledgeEntity] = relationship(
        back_populates="incoming_relations",
        foreign_keys=[target_entity_id],
    )
    source_message: Mapped[Message | None] = relationship()
    user_exposures: Mapped[list[UserRelationExposure]] = relationship(
        back_populates="relation",
    )


class EntityAlias(Base):
    __tablename__ = "entity_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alias: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_entities.id"), nullable=False
    )

    entity: Mapped[KnowledgeEntity] = relationship(back_populates="aliases")


class UserEntityExposure(Base):
    __tablename__ = "user_entity_exposure"
    __table_args__ = (
        UniqueConstraint("user_id", "entity_id", name="uq_user_entity_exposure"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_entities.id"), nullable=False
    )
    first_exposed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_exposed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    exposure_count: Mapped[int] = mapped_column(Integer, default=1)

    user: Mapped[User] = relationship()
    entity: Mapped[KnowledgeEntity] = relationship(back_populates="user_exposures")


class UserRelationExposure(Base):
    __tablename__ = "user_relation_exposure"
    __table_args__ = (
        UniqueConstraint("user_id", "relation_id", name="uq_user_relation_exposure"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    relation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_relations.id"), nullable=False
    )
    first_exposed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_exposed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped[User] = relationship()
    relation: Mapped[KnowledgeRelation] = relationship(back_populates="user_exposures")


class KnowledgeNarrative(Base):
    __tablename__ = "knowledge_narratives"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    entity_ids_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    first_message_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_message_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    message_count: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(20), default="active")
