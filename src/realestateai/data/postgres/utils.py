from .db import engine


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
