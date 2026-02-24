# pg_pandas_io.py
from __future__ import annotations

import traceback
from collections.abc import Iterable

import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB

from realestateai.data.postgres.db import engine


def _jsonb_columns(df: pd.DataFrame, candidates: Iterable[str] | None = None) -> list[str]:
    """
    Detect columns that likely should be JSONB (lists/dicts).
    You can pass candidates to force-list specific columns as JSONB.
    """
    cols: list[str] = []
    if candidates:
        cols.extend([c for c in candidates if c in df.columns])

    for c in df.columns:
        if c in cols:
            continue
        s = df[c]
        # if any value is list/dict, treat as JSONB
        if s.map(lambda x: isinstance(x, (list, dict))).to_numpy().any():
            cols.append(c)

    return cols


def save_df_to_postgres(
    df: pd.DataFrame,
    table: str,
    schema: str | None = None,
    if_exists: str = "overwrite",
    chunksize: int = 10_000,
) -> None:
    """
    Saves df to Postgres using pandas.to_sql.
    - Keeps JSON/list columns as JSONB where possible (dtype mapping).
    - Writes into schema (cfg.schema by default).
    """

    default_schema: str = "public"
    schema = schema or default_schema

    # Make sure NULLs become None (helps psycopg)
    df_to_write = df.where(pd.notna(df), None)

    # Rename any blank column names
    df_to_write.columns = [
        f"col_{i}" if not str(c).strip() else c for i, c in enumerate(df_to_write.columns)
    ]

    # If you have list/dict columns, strongly recommend mapping them to JSONB
    jsonb_cols = _jsonb_columns(df_to_write)
    dtype_map = {c: JSONB for c in jsonb_cols}

    with engine.connect() as conn:
        try:
            df_to_write.to_sql(
                name=table,
                con=conn,
                schema=schema,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=chunksize,
                dtype=dtype_map or None,
            )
        except Exception as e:
            print("EXC TYPE:", type(e))
            print("EXC REPR:", repr(e))
            # Если это SQLAlchemyError — у него часто есть .orig/.statement/.params
            for attr in ["orig", "statement", "params"]:
                if hasattr(e, attr):
                    val = getattr(e, attr)
                    print(f"{attr.upper()}:\n{val}\n")
            traceback.print_exc()
            raise
