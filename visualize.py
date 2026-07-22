"""Turns a query result into a chart when the shape of the data makes one useful.

A single number doesn't deserve a chart; a "value per category" or
"value over time" table does. The chart type is picked from the shape of
the result itself, instead of asking the LLM to also write plotting code -
simpler, and it can't get the plotting code wrong.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CHART_DIR = Path(__file__).parent / "charts"
MAX_CATEGORIES = 15


def maybe_create_chart(df: pd.DataFrame, filename: str = "last_chart.png") -> str | None:
    if df is None or len(df) < 2 or df.shape[1] < 2:
        return None

    x_col = df.columns[0]
    numeric_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        return None
    y_col = numeric_cols[0]

    CHART_DIR.mkdir(exist_ok=True)
    path = CHART_DIR / filename

    fig, ax = plt.subplots(figsize=(8, 5))

    if pd.api.types.is_datetime64_any_dtype(df[x_col]):
        ax.plot(df[x_col], df[y_col], marker="o")
    else:
        plot_df = df.head(MAX_CATEGORIES)
        ax.bar(plot_df[x_col].astype(str), plot_df[y_col])
        plt.xticks(rotation=45, ha="right")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)

    return str(path)
