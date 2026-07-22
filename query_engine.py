"""Runs SQL queries the model writes, but only if they are safely read-only.

The model is a text generator - nothing stops it from writing a query that
deletes data, even by mistake. This module is the guardrail: it inspects
the query text before DuckDB ever sees it, and only lets SELECT statements
through.
"""

import duckdb
import pandas as pd

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH",
    "DETACH", "COPY", "PRAGMA", "EXPORT", "IMPORT", "INSTALL", "LOAD",
    "CALL", "VACUUM", "REPLACE", "TRUNCATE",
]
MAX_ROWS = 1000


class UnsafeQueryError(Exception):
    pass


def _validate(sql: str) -> None:
    stripped = sql.strip().rstrip(";")

    if ";" in stripped:
        raise UnsafeQueryError("Only a single statement is allowed (no ';' inside the query).")

    if "--" in stripped or "/*" in stripped:
        raise UnsafeQueryError("SQL comments are not allowed in a query.")

    if not stripped.upper().startswith("SELECT"):
        raise UnsafeQueryError("Only SELECT statements are allowed.")

    upper = stripped.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        # word-boundary check so e.g. a column named "updated_at" doesn't false-trigger
        if f" {keyword} " in f" {upper} " or upper.startswith(f"{keyword} "):
            raise UnsafeQueryError(f"Keyword '{keyword}' is not allowed in a read-only query.")


def run_safe_query(df: pd.DataFrame, sql: str, table_name: str = "sales") -> pd.DataFrame:
    _validate(sql)

    con = duckdb.connect(":memory:")
    con.register(table_name, df)
    result = con.execute(sql).fetchdf()
    con.close()

    if len(result) > MAX_ROWS:
        result = result.head(MAX_ROWS)

    return result


def format_result(df: pd.DataFrame) -> str:
    """Renders a query result as plain decimal numbers for the model.

    pandas' default to_string() switches large floats to scientific notation
    (e.g. 9.747748e+06), which a small LLM tends to misread when restating
    the number in prose. Forcing fixed-point formatting avoids that.
    """
    with pd.option_context("display.float_format", "{:,.2f}".format):
        return df.to_string(index=False)


if __name__ == "__main__":
    from data_loader import load_data

    data = load_data()
    print(run_safe_query(data, "SELECT country, COUNT(*) AS n FROM sales GROUP BY country ORDER BY n DESC LIMIT 5"))

    try:
        run_safe_query(data, "DROP TABLE sales")
    except UnsafeQueryError as e:
        print(f"\nBlocked as expected: {e}")
