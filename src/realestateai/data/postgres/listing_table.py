from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from realestateai.data.postgres.db import Base, session_scope


class HtmlFileCreate(BaseModel):
    offer_id: str = Field(min_length=1, max_length=256)
    url: str = Field(min_length=1)
    minio_key: str = Field(min_length=1, max_length=512)
    scraped_at: datetime | None = None


class HtmlFile(Base):
    __tablename__ = "html_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    offer_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    minio_key: Mapped[str] = mapped_column(String(512), nullable=False)

    # Лучше хранить в UTC. server_default=func.now() — дефолт на стороне БД.
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


def save_scraping_metadata(payload: HtmlFileCreate) -> int:
    # Если scraped_at не передали — ставим UTC “сейчас”.
    scraped_at = payload.scraped_at or datetime.utcnow()

    with session_scope() as session:
        row = HtmlFile(
            offer_id=payload.offer_id,
            url=payload.url,
            minio_key=payload.minio_key,
            scraped_at=scraped_at,
        )
        session.add(row)
        session.flush()  # получаем row.id до commit
        return row.id
