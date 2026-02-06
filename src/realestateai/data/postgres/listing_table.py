from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from realestateai.data.postgres.db import Base, session_scope


class HtmlFileCreate(BaseModel):
    offer_id: str = Field(min_length=1, max_length=512)
    url: str = Field(min_length=1)
    minio_key: str = Field(min_length=1, max_length=512)
    scraped_at: datetime | None = None
    address: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=256)
    ad_id: int | None = None


class HtmlFile(Base):
    __tablename__ = "html_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    offer_id: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    minio_key: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=True)
    address: Mapped[str] = mapped_column(String(256), nullable=True)
    ad_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

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
            address=payload.address,
            title=payload.title,
            ad_id=payload.ad_id,
        )
        session.add(row)
        session.flush()  # получаем row.id до commit
        return row.id


def load_html_file(row_id: int | None = None, **kwargs: Any) -> HtmlFile | None:
    """
    Load a row from html_files table by row_id or other search parameters.

    Args:
        row_id: Primary key (id) of the row
        **kwargs: Additional search parameters (e.g., offer_id, minio_key, url, etc.)
                  If multiple kwargs provided, they are combined with AND logic.

    Returns:
        HtmlFile instance if found, None otherwise

    Examples:
        # Load by primary key
        row = load_html_file(row_id=1)

        # Load by offer_id
        row = load_html_file(offer_id="12345")

        # Load by multiple criteria
        row = load_html_file(offer_id="12345", minio_key="realestate/listings/12345.html")
    """
    with session_scope() as session:
        query = session.query(HtmlFile)

        # Filter by row_id if provided
        if row_id is not None:
            query = query.filter(HtmlFile.id == row_id)

        # Apply additional filters from kwargs
        for key, value in kwargs.items():
            if hasattr(HtmlFile, key) and value is not None:
                query = query.filter(getattr(HtmlFile, key) == value)

        return query.first()
