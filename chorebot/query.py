"""Read-only SQL over the sheet tabs using DuckDB."""
from __future__ import annotations

import re

import duckdb
import pandas as pd

from .storage import SCHEMA

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|copy|install|load|pragma|export)\b|read_\w+\(",
                       re.IGNORECASE)


def run_query(store, sql: str, limit: int = 50) -> list[dict]:
    sql = sql.strip().rstrip(";")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE) or ";" in sql or FORBIDDEN.search(sql):
        raise ValueError("Only a single SELECT query is allowed.")
    con = duckdb.connect(":memory:")
    try:
        for tab in ("Members", "Chores", "Tasks", "Busy", "Log"):
            df = pd.DataFrame(store.rows(tab), columns=SCHEMA[tab]).astype(str)
            con.register(tab.lower(), df)
        cur = con.execute(f"SELECT * FROM ({sql}) LIMIT {limit}")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, map(str, row))) for row in cur.fetchall()]
    finally:
        con.close()
