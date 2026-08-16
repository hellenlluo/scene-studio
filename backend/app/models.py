from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    state: Mapped[JobState] = mapped_column(String(16), default=JobState.PENDING)
    image_path: Mapped[str] = mapped_column(String(512))

    current_stage: Mapped[str | None] = mapped_column(String(32), default=None)
    # Stage name -> {"state": ..., "seconds": ...}. Lets the client render staged
    # progress and partial results while later stages are still running.
    stage_status: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    scene: Mapped["Scene | None"] = relationship(back_populates="job", uselist=False)


class Scene(Base):
    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    name: Mapped[str] = mapped_column(String(128), default="Untitled scene")
    image_path: Mapped[str] = mapped_column(String(512))

    # Serialized SceneSpec. Rewritten in place when the user commits an edit.
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    job: Mapped[Job] = relationship(back_populates="scene")
