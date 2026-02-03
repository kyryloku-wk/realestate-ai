import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# --- Database helpers (SQLAlchemy Core) ---

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()

    html_files = Table(
        "html_files",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("offer_id", String(256), nullable=False),
        Column("url", Text, nullable=False),
        Column("minio_key", String(512), nullable=False),
        Column("scraped_at", DateTime, nullable=False),
    )

    try:
        metadata.create_all(engine)
    except SQLAlchemyError:
        # If DB not available now, table creation will be retried on insert
        engine = None
else:
    engine = None
    html_files = None


def save_scraping_metadata(offer_id: str, url: str, minio_key: str, scraped_at: datetime = None):
    """Insert a metadata record into Postgres and return inserted id. If DB is not configured, raise."""
    if scraped_at is None:
        scraped_at = datetime.utcnow()
    if engine is None:
        raise RuntimeError("Database is not configured. Set DATABASE_URL in environment.")

    with engine.begin() as conn:
        result = conn.execute(
            html_files.insert().values(
                offer_id=offer_id, url=url, minio_key=minio_key, scraped_at=scraped_at
            )
        )
        return result.inserted_primary_key[0]


# Utility: run arbitrary SQL and return a pandas DataFrame
def query_to_dataframe(sql: str, params: dict | None = None):
    """Execute a SQL query and return a pandas DataFrame.

    Args:
        sql: SQL statement or query string. Use parameter placeholders as :name for SQLAlchemy text.
        params: Optional dict of parameters for parameterized queries.

    Returns:
        pandas.DataFrame with the query results.

    Raises:
        RuntimeError: if DATABASE_URL/engine is not configured.
    """
    try:
        import pandas as pd
        from sqlalchemy import text
    except Exception:
        raise

    if engine is None:
        raise RuntimeError("Database is not configured. Set DATABASE_URL in environment.")

    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)
    return df
