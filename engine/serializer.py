"""
serializer.py
-------------
Text representation of a table for the sentence transformer.

Output is the advanced "Executive Summary" + schema + 3 sample rows:

    This dataset has 5 columns: 3 numeric columns (...); 1 text column
    (CountryRegionCode (2-char values): AD, AE, AF, ...). Columns have
    moderate entropy. Data is complete.

    Table Name: my_table.csv
    Columns: ...
    Sample Data:
     ...

Per text column we include 10 most-frequent values and a length descriptor
("2-char values" / "4-32 char values") — the model learns column format
from this and can match columns of compatible formats across tables.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _entropy(series: pd.Series) -> float:
    p = series.dropna().value_counts(normalize=True).values
    return float(-np.sum(p * np.log2(p + 1e-10)))


def serialize(df: pd.DataFrame, table_name: str, profile: Dict) -> str:
    """Build the executive summary + schema + 3 sample rows for embedding."""
    base = _serialize_schema_and_sample(df, table_name)
    summary = _executive_summary(df, profile)
    return f"{summary}\n\n{base}"


def _serialize_schema_and_sample(df: pd.DataFrame, table_name: str) -> str:
    columns = ", ".join(df.columns.astype(str))
    n = min(3, len(df))
    sample = df.sample(n=n, random_state=42).to_string(index=False) if n > 0 else ""
    return f"Table Name: {table_name}\nColumns: {columns}\nSample Data:\n{sample}"


def _executive_summary(df: pd.DataFrame, profile: Dict) -> str:
    null_ratio = profile.get("null_ratio", 0.0)
    col_length_stats = profile.get("col_length_stats", {})

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    string_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

    parts = []
    if numeric_cols:
        n, s = len(numeric_cols), "s" if len(numeric_cols) > 1 else ""
        names = ", ".join(numeric_cols[:3]) + (f" and {n-3} more" if n > 3 else "")
        parts.append(f"{n} numeric column{s} ({names})")

    if string_cols:
        n, s = len(string_cols), "s" if len(string_cols) > 1 else ""
        bits = []
        for col in string_cols[:3]:
            top = df[col].dropna().astype(str).value_counts().head(10).index.tolist()
            lens = col_length_stats.get(col)
            if lens:
                size = (
                    f"{lens['min']}-char values"
                    if lens["min"] == lens["max"]
                    else f"{lens['min']}-{lens['max']} char values"
                )
                prefix = f"{col} ({size})"
            else:
                prefix = col
            val_str = f": {', '.join(top)}" if top else ""
            bits.append(f"{prefix}{val_str}")
        more = f" and {n-3} more" if n > 3 else ""
        parts.append(f"{n} text column{s} ({'; '.join(bits)}{more})")

    if date_cols:
        n, s = len(date_cols), "s" if len(date_cols) > 1 else ""
        parts.append(f"{n} date column{s} ({', '.join(date_cols[:3])})")

    col_desc = "; ".join(parts) if parts else "unknown column types"

    mean_entropy = (
        float(np.mean([_entropy(df[c]) for c in df.columns]))
        if len(df.columns) > 0
        else 0.0
    )
    if mean_entropy > 4.0:
        entropy_desc = "Columns have high entropy (many distinct values)."
    elif mean_entropy > 2.0:
        entropy_desc = "Columns have moderate entropy."
    else:
        entropy_desc = "Columns have low entropy (few dominant values)."

    if null_ratio < 0.02:
        complete_desc = "Data is complete."
    elif null_ratio < 0.2:
        complete_desc = "Data has some missing values."
    else:
        complete_desc = "Data is sparse with many missing values."

    return (
        f"This dataset has {len(df.columns)} columns: {col_desc}. "
        f"{entropy_desc} {complete_desc}"
    )
