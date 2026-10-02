"""
profiler.py
===========
Profile one CSV file into one structural feature dictionary.

Source
------
A CSV file already loaded as a pandas DataFrame, plus its file path
and a display name. We need both because:

  - Freyja's DataProfilerWorker reads the CSV with DuckDB SQL to
    compute ~25 per-column statistics in a single streaming pass
    (cardinality, uniqueness, value distribution, top-10 frequent
    values, soundex sketches, string-length stats per word, ...).

  - The pandas DataFrame is needed once more, locally, to derive
    per-VALUE character-length statistics used by the Executive Summary
    text on the semantic side ("2-char values", "4-32 char values").
    Freyja's lengths are per-word, which is not the same thing.

Freyja's profiler is vendored in engine/freyja (copied from Fya/app/core/profiling).

Output
------
A dictionary returned by profile(), or None if Freyja fails.

Conceptually the fields fall into six groups, each describing the
table from a different angle. Each row below names the field, what
it MEANS, and what it's USED FOR.

GROUP 1 — Basic shape ("how big is the table?")
-----------------------------------------------------------------
Weak signals on their own; mostly tiebreakers.

  table_name        str        File basename. Display + self-match exclusion.
  row_count         int        Number of rows. Recovered as
                                cardinality / uniqueness — no row scan.
                                Used by similarity (small weight).
  col_count         int        Number of columns. Used by similarity
                                (small weight).
  col_names         list[str]  Original headers, ordered. NOT scored
                                directly — consumed by the semantic
                                text serializer ("Columns: ...").

GROUP 2 — Schema ("what does the table talk about?")
-----------------------------------------------------------------
Strong signal when columns have descriptive English names.

  col_name_tokens   set[str]   Bag of lowercase word-tokens from every
                                column name, regex-extracted.
                                Example: "customer_id, total_amount" →
                                {customer, id, total, amount}.
                                Used by similarity (Jaccard, weight 0.15)
                                AND by joinability per-column-pair (0.25).

GROUP 3 — Role-derived aggregates  ★ the v2 lever ★
-----------------------------------------------------------------
Together these answer: "what KIND of columns does this table have?"
Strongest similarity signal in the pipeline.

  role_counts       dict       {role_name: int} — how many columns of
                                each of the 8 roles. Distinguishes
                                reference / fact / log / corpus tables.
                                Used by similarity (cosine, weight 0.20).
  cardinality_hist  4-tuple    Histogram of column cardinality_ratio
                                over buckets [0,.1] (.1,.5] (.5,.9] (.9,1].
                                Preserves more info than the raw mean.
                                Used by similarity (cosine, weight 0.10).
  length_hist       3-tuple    Histogram of text-column average length
                                (≤5 / 6–30 / >30 chars). Captures
                                "short codes" vs "long descriptions".
                                Used by similarity (cosine, weight 0.05).

GROUP 4 — Macro means and fractions (the classical v1 stats)
-----------------------------------------------------------------
Per-column statistics averaged across columns. v2 keeps them at lower
weight than Group 3 because averages throw information away.

  null_ratio        float      Mean Freyja `incompleteness`. Completeness
                                signal. similarity (0.05).
  cardinality       float      Mean Freyja `cardinality`. Average distinct
                                values per column. similarity (0.10).
  val_pct_max       float      Mean top-value dominance. High = many
                                flag/category columns; low = many unique
                                columns. similarity (0.10).
  avg_string_length float      Mean Freyja `len_avg_word` across text
                                columns. similarity (0.05).
  type_numeric_frac float      Numeric cols / total cols. } combined
  type_string_frac  float      String cols / total cols.  } into one
  type_date_frac    float      Date cols / total cols.    } "type"
                                                            component
                                                            for
                                                            similarity
                                                            (0.10).

GROUP 5 — Per-column value sketches  ★ the joinability evidence ★
-----------------------------------------------------------------
The only features in the profile that look at ACTUAL VALUES.
Without them, joinability collapses to "do the column names match?"

  col_value_sets    dict       {col: set[str]} top-10 lowercase values
                                per column (from Freyja
                                freq_word_containment).
                                Used by joinability (directional
                                containment, weight 0.45 per pair).
  col_length_stats  dict       {col: {min, max, avg}} per-VALUE
                                character length (full string, not per
                                word), derived from pandas. Powers the
                                semantic serializer's "(2-char values)"
                                annotations.

  (Per-column soundex sets, lex bounds and other value-level signals
   live in Group 6 under columns[col_name].)

GROUP 6 — Per-column structural detail (inside profile["columns"])
-----------------------------------------------------------------
A dict per column. Most fields exist to FEED the role classifier
(only `role` and a handful of values are directly scored).

  columns       dict       {col_name: {                                  } per
                              role:              str  (label),           } column
                              data_type:         str  (DuckDB SQL type),
                              cardinality:       float,
                              cardinality_ratio: float (cardinality/rows),
                              uniqueness:        float (≈ ratio),
                              null_ratio:        float,
                              val_pct_max:       float,
                              entropy:           float (reserved),
                              len_min:           int   (per-word),
                              len_max:           int   (per-word),
                              len_avg:           float (per-word),
                              fixed_length:      bool  (len_min == len_max
                                                       — the JOIN_CODE signal),
                              top_values:        set   (Freyja top-10),
                              soundex_values:    set   (phonetic top-10,
                                                       joinability weight 0.15),
                              lex_min:           str   (smallest value
                                                       alphabetically, reserved),
                              lex_max:           str   (largest, reserved),
                              is_binary:         bool  (2 distinct values
                                                       → FLAG role),
                            }}

How `columns` is consumed:
  - role + data_type   →  filter to JOIN_CODE / FK / CATEGORY for joinability
  - top_values         →  containment score (weight 0.45 per pair)
  - soundex_values     →  fuzzy fallback (weight 0.15)
  - len_min / len_max  →  length-range overlap (weight 0.15)
  - fixed_length, is_binary, etc.  →  inputs to the role classifier

Most useful (ranked)
--------------------
For SIMILARITY ("is this the same kind of data?"):
  1. role_counts        (weight 0.20)  — strongest single signal
  2. col_name_tokens    (weight 0.15)  — strong when names are descriptive
  3. cardinality_hist   (weight 0.10)  — better than the raw mean
  4. type composition   (weight 0.10)  — quick numeric-vs-string split

For JOINABILITY ("could I join these tables?"):
  1. top_values         (weight 0.45)  — the only feature probing value overlap
  2. role tag           (filter)       — gates which columns are considered
  3. col-name match     (weight 0.25)  — column-name Jaccard per pair
  4. soundex_values  +  (weight 0.15 each)  — fuzzy spelling + length-format
     len_min/len_max                          checks per pair

Removing top_values would make joinability collapse to schema matching;
removing role_counts would make similarity degrade to v1's mean-based
approach. col_name_tokens helps both modes. Everything else refines edges.

Currently unused (but kept in the profile — Freyja computes them for free
and they may be wired up later without a cache rebuild):
  - entropy (per column)
  - lex_min / lex_max
  - per-column null_ratio (only the table-level mean is scored)
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .minhash import K_MINHASH, minhash_signature
from .roles import (
    ALL_ROLES,
    cardinality_bucket,
    classify_column,
    length_bucket,
    type_class,
)


# ── Public class ────────────────────────────────────────────────────────────


class Profiler:
    """Wraps Freyja's DataProfilerWorker. One CSV → one profile dict."""

    def __init__(self) -> None:
        # Freyja's profiler is vendored in engine/freyja; imported lazily so that
        # browsing a lake never needs DuckDB.
        from .freyja.profiler import DataProfilerWorker, ProfilerConfig

        self._DataProfilerWorker = DataProfilerWorker
        self._ProfilerConfig = ProfilerConfig

    def profile(
        self, csv_path: str, df: pd.DataFrame, table_name: str
    ) -> Optional[Dict]:
        """Return one profile dict, or None if Freyja fails."""
        profile_df = self._run_freyja(csv_path)
        if profile_df is None:
            return None
        return _build_profile(profile_df, df, table_name)

    def _run_freyja(self, csv_path: str) -> Optional[pd.DataFrame]:
        config = self._ProfilerConfig(
            datalake_path=Path(csv_path).parent,
            output_profiles_path=Path(tempfile.gettempdir()) / "_lightweight_dummy.csv",
            varchar_only=False,
            max_workers=1,
        )
        worker = self._DataProfilerWorker(config)
        _, profile_df, err = worker.process_csv(Path(csv_path))
        if err or profile_df is None or profile_df.empty:
            print(f"[Profiler] Freyja error for {csv_path}: {err}")
            return None
        return profile_df


# ── Orchestrator ────────────────────────────────────────────────────────────


def _build_profile(profile_df: pd.DataFrame, df: pd.DataFrame, table_name: str) -> Dict:
    """Assemble the profile dict from Freyja's per-column DataFrame."""
    col_names = profile_df["attribute_name"].tolist()

    sql_types = profile_df["data_type"].apply(type_class)
    string_col_names = profile_df.loc[sql_types == "string", "attribute_name"].tolist()
    type_fractions = _type_fractions(sql_types, len(col_names))

    row_count = _estimate_row_count(profile_df)
    col_value_sets = _value_sets(profile_df, "freq_word_containment")
    col_soundex_sets = _value_sets(profile_df, "freq_word_soundex_containment")
    col_length_stats = _per_value_length_stats(df, string_col_names)
    means = _macro_means(profile_df)

    columns, role_counts, cardinality_hist, length_hist = _per_column_detail(
        profile_df, row_count, string_col_names, col_value_sets, col_soundex_sets,
    )

    # MinHash sketch over ALL distinct values (from the DataFrame, not Freyja's
    # top-10) — the value-containment evidence used by similarity reranking.
    col_minhash = _minhash_sets(df)
    for cname, cdict in columns.items():
        mh = col_minhash.get(cname)
        cdict["minhash"] = mh["sig"] if mh else ()
        cdict["minhash_n"] = mh["nd"] if mh else 0

    return {
        "table_name":        table_name,
        "row_count":         row_count,
        "col_count":         len(col_names),
        "col_names":         col_names,
        "col_name_tokens":   _col_name_tokens(col_names),
        "avg_string_length": means["avg_string_length"],
        "col_length_stats":  col_length_stats,
        "col_value_sets":    col_value_sets,
        "null_ratio":        means["null_ratio"],
        "cardinality":       means["cardinality"],
        "val_pct_max":       means["val_pct_max"],
        "type_numeric_frac": type_fractions["numeric"],
        "type_string_frac":  type_fractions["string"],
        "type_date_frac":    type_fractions["date"],
        "columns":           columns,
        "role_counts":       role_counts,
        "cardinality_hist":  cardinality_hist,
        "length_hist":       length_hist,
    }


# ── Coercion helpers ────────────────────────────────────────────────────────


def _num(value, default: float = 0.0) -> float:
    """Float conversion that tolerates None / NaN / missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(v):
        return float(default)
    return v


def _int(value, default: int = 0) -> int:
    """Int conversion that tolerates None / NaN / missing."""
    return int(_num(value, default))


# ── Aggregation helpers ─────────────────────────────────────────────────────


def _macro_means(profile_df: pd.DataFrame) -> Dict[str, float]:
    """Mean of four Freyja columns across the rows of profile_df."""
    return {
        "null_ratio":        _num(profile_df["incompleteness"].mean()),
        "cardinality":       _num(profile_df["cardinality"].mean()),
        "val_pct_max":       _num(profile_df["val_pct_max"].dropna().mean()),
        "avg_string_length": _num(profile_df["len_avg_word"].dropna().mean()),
    }


def _type_fractions(sql_types: pd.Series, n_cols: int) -> Dict[str, float]:
    """Fraction of columns of each SQL type-class."""
    n = max(n_cols, 1)
    return {
        "numeric": float((sql_types == "numeric").sum() / n),
        "string":  float((sql_types == "string").sum()  / n),
        "date":    float((sql_types == "date").sum()    / n),
    }


def _estimate_row_count(profile_df: pd.DataFrame) -> int:
    """Recover the table's row count from Freyja's per-column metrics.

    For each column,  cardinality / uniqueness ≈ row_count. Take the max
    across columns (robust to all-null columns where uniqueness → 0).
    """
    valid = profile_df["uniqueness"] > 0
    if not valid.any():
        return 0
    return int(
        (profile_df.loc[valid, "cardinality"]
         / profile_df.loc[valid, "uniqueness"]).round().max()
    )


def _value_sets(profile_df: pd.DataFrame, field: str) -> Dict[str, set]:
    """Extract a {column_name: lowercase-value-set} dict from a Freyja field."""
    out: Dict[str, set] = {}
    for _, prow in profile_df.iterrows():
        vals = prow[field]
        if isinstance(vals, (list, set)) and len(vals) > 0:
            out[prow["attribute_name"]] = {
                str(v).lower() for v in vals if v is not None
            }
    return out


def _per_value_length_stats(
    df: pd.DataFrame, string_col_names
) -> Dict[str, Dict[str, float]]:
    """Per-string-column min/max/avg of the full value's character length.

    Read from the pandas DataFrame (not Freyja) because Freyja's
    `len_min_word` / `len_max_word` measure per-word length, while the
    semantic serializer wants per-value length.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in string_col_names:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        lens = s.str.len()
        stats[col] = {
            "min": int(lens.min()),
            "max": int(lens.max()),
            "avg": float(lens.mean()),
        }
    return stats


def _minhash_sets(df: pd.DataFrame) -> Dict[str, Dict]:
    """Per-column bottom-k MinHash signature over the full distinct value set.

    Values are normalised (str, lower, strip) the same way the ground-truth
    profiler normalises them, then sketched. Returns {col: {sig, nd}} where `sig`
    is the bottom-k hash signature and `nd` the true distinct-value count.
    """
    out: Dict[str, Dict] = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        vals = s.astype(str).str.lower().str.strip().unique()
        out[str(col)] = {"sig": minhash_signature(vals, K_MINHASH), "nd": int(len(vals))}
    return out


def _col_name_tokens(col_names) -> set:
    """Union of lowercase regex tokens across all column names."""
    tokens: set = set()
    for name in col_names:
        tokens.update(re.findall(r"[a-z]+", str(name).lower()))
    return tokens


def _per_column_detail(
    profile_df: pd.DataFrame,
    row_count: int,
    string_col_names,
    value_sets: Dict[str, set],
    soundex_sets: Dict[str, set],
):
    """Build the per-column dict plus role / cardinality / length histograms.

    Returns:
      (columns_dict, role_counts_dict, cardinality_hist_tuple, length_hist_tuple)
    """
    columns: Dict[str, Dict] = {}
    role_counts: Dict[str, int] = {r: 0 for r in ALL_ROLES}
    card_hist = [0, 0, 0, 0]
    len_hist  = [0, 0, 0]

    for _, prow in profile_df.iterrows():
        cname = prow["attribute_name"]
        role = classify_column(prow, row_count)
        role_counts[role] += 1

        cardinality_col = _num(prow["cardinality"])
        card_ratio = (cardinality_col / row_count) if row_count > 0 else 0.0
        card_hist[cardinality_bucket(card_ratio)] += 1

        len_avg = _num(prow["len_avg_word"])
        if cname in string_col_names:
            len_hist[length_bucket(len_avg)] += 1

        len_min = _int(prow["len_min_word"])
        len_max = _int(prow["len_max_word"])
        fixed_length = (
            cname in string_col_names
            and len_min == len_max
            and len_min > 0
        )

        columns[cname] = {
            "role":              role,
            "data_type":         str(prow["data_type"] or ""),
            "cardinality":       cardinality_col,
            "cardinality_ratio": card_ratio,
            "uniqueness":        _num(prow["uniqueness"]),
            "null_ratio":        _num(prow["incompleteness"]),
            "val_pct_max":       _num(prow["val_pct_max"]),
            "entropy":           _num(prow.get("entropy")),
            "len_min":           len_min,
            "len_max":           len_max,
            "len_avg":           len_avg,
            "fixed_length":      fixed_length,
            "top_values":        value_sets.get(cname, set()),
            "soundex_values":    soundex_sets.get(cname, set()),
            "lex_min":           str(prow.get("first_word", "") or ""),
            "lex_max":           str(prow.get("last_word", "") or ""),
            "is_binary":         bool(int(prow.get("is_binary", 0) or 0) == 1),
        }

    return columns, role_counts, tuple(card_hist), tuple(len_hist)
