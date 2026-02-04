from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(default="", alias="DATABASE_URL")


@lru_cache
def get_settings() -> Settings:
    return Settings()


class Base(DeclarativeBase):
    pass


def make_engine() -> Engine | None:
    settings = get_settings()
    url = (settings.database_url or "").strip()
    if not url:
        return None
    return create_engine(url, pool_pre_ping=True, future=True)


engine = make_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


@contextmanager
def session_scope() -> Iterator[Session]:
    if engine is None:
        raise RuntimeError("DATABASE_URL is empty. Set it in .env (DATABASE_URL=...).")

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    if engine is None:
        raise RuntimeError("DATABASE_URL is empty. Set it in .env (DATABASE_URL=...).")

    Base.metadata.create_all(bind=engine)
