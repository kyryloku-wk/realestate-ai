from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator
from sqlalchemy import BigInteger, DateTime, Integer, String, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from realestateai.data.postgres.db import Base, init_db, session_scope


class ListingPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str = Field(..., min_length=1, max_length=32)
    ad_id: int

    url: HttpUrl | None = None
    status: str | None = None

    created_at: datetime | None = None
    modified_at: datetime | None = None
    pushed_up_at: datetime | None = None

    @field_validator("created_at", "modified_at", "pushed_up_at", mode="before")
    @classmethod
    def parse_dt(cls, v: Any) -> Any:
        if v is None or isinstance(v, datetime):
            return v
        if isinstance(v, str) and v.strip():
            s = v.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s)
        return v

    @field_validator("source")
    @classmethod
    def normalize_source(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("status")
    @classmethod
    def normalize_status(cls, v: str | None) -> str | None:
        return v.strip().lower() if isinstance(v, str) else v


class ListingBronze(Base):
    __tablename__ = "listings_bronze"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ad_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    url: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pushed_up_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (UniqueConstraint("source", "ad_id", name="uq_listings_bronze_source_ad_id"),)


UPSERT_SQL = text("""
INSERT INTO listings_bronze (
  source, ad_id, url, status, created_at, modified_at, pushed_up_at, payload
)
VALUES (
  :source, :ad_id, :url, :status, :created_at, :modified_at, :pushed_up_at, CAST(:payload AS jsonb)
)
ON CONFLICT (source, ad_id)
DO UPDATE SET
  url          = EXCLUDED.url,
  status       = EXCLUDED.status,
  created_at   = COALESCE(EXCLUDED.created_at, listings_bronze.created_at),

  modified_at  = COALESCE(
                    GREATEST(listings_bronze.modified_at, EXCLUDED.modified_at),
                    listings_bronze.modified_at,
                    EXCLUDED.modified_at
                 ),

  pushed_up_at = COALESCE(
                    GREATEST(listings_bronze.pushed_up_at, EXCLUDED.pushed_up_at),
                    listings_bronze.pushed_up_at,
                    EXCLUDED.pushed_up_at
                 ),

  payload      = EXCLUDED.payload,
  ingested_at  = now()
RETURNING id;
""")


def upsert_broze_parsing(payload_dict: dict[str, Any]) -> int:
    parsed = ListingPayload.model_validate(payload_dict)

    params = {
        "source": parsed.source,
        "ad_id": int(parsed.ad_id),
        "url": str(parsed.url) if parsed.url else None,
        "status": parsed.status,
        "created_at": parsed.created_at,
        "modified_at": parsed.modified_at,
        "pushed_up_at": parsed.pushed_up_at,
        # сохраняем исходный payload "как есть"
        "payload": json.dumps(payload_dict, ensure_ascii=False),
    }

    with session_scope() as session:
        row_id = session.execute(UPSERT_SQL, params).scalar_one()
        return int(row_id)


def load_listing_raw(row_id: int | None = None, **kwargs: Any) -> ListingBronze | None:
    with session_scope() as session:
        query = session.query(ListingBronze)

        # Filter by row_id if provided
        if row_id is not None:
            query = query.filter(ListingBronze.id == row_id)

        # Apply additional filters from kwargs
        for key, value in kwargs.items():
            if hasattr(ListingBronze, key) and value is not None:
                query = query.filter(getattr(ListingBronze, key) == value)

        return query.first()


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    # init_db will raise if DB not configured
    init_db()

    # Example: load from file or pass dict directly
    import json

    path = "D:\\Projects\\RealEstateAgent\\garbage\\parsed_output.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    row_id = upsert_broze_parsing(data)
    print(f"Upserted listings_bronze.id={row_id}")
