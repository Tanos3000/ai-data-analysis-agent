"""Builds a text description of a DataFrame's schema to give the LLM as context.

The model never sees the raw data - only this description. Without it, the
model has to guess column names and types, and guesses wrong often enough
that any query it writes fails or silently answers the wrong question.
"""

import pandas as pd


def _describe_column(df: pd.DataFrame, column: str) -> str:
    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        return f"- {column} ({series.dtype}): range {series.min()} to {series.max()}"

    if pd.api.types.is_datetime64_any_dtype(series):
        return f"- {column} ({series.dtype}): range {series.min()} to {series.max()}"

    examples = series.dropna().unique()[:5]
    return f"- {column} ({series.dtype}): {series.nunique()} unique values, e.g. {list(examples)}"


def build_schema_context(df: pd.DataFrame, table_name: str = "sales") -> str:
    lines = [
        f"Table name: {table_name}",
        f"Row count: {len(df)}",
        "Columns:",
    ]
    lines += [_describe_column(df, col) for col in df.columns]
    return "\n".join(lines)


if __name__ == "__main__":
    from data_loader import load_data

    print(build_schema_context(load_data()))
