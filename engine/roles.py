"""
roles.py
--------
Column role classification.

Every column profiled by Freyja is tagged with one of eight roles. The
tag is the lever that lets downstream code "extract only the relevant
columns" per use case:

  - similarity scoring uses the histogram of roles plus all per-column
    shape signals (every column contributes).
  - joinability scoring restricts itself to columns whose role is in
    {JOIN_CODE, FOREIGN_KEY_CANDIDATE, CATEGORY} — i.e. columns whose
    values could plausibly match another table's column.

Classification reads only fields already present in Freyja's per-column
output. No extra passes over the data.
"""

from __future__ import annotations

from typing import Mapping


IDENTIFIER            = "IDENTIFIER"
JOIN_CODE             = "JOIN_CODE"
FOREIGN_KEY_CANDIDATE = "FOREIGN_KEY_CANDIDATE"
CATEGORY              = "CATEGORY"
MEASURE               = "MEASURE"
FREE_TEXT             = "FREE_TEXT"
TEMPORAL              = "TEMPORAL"
FLAG                  = "FLAG"

ALL_ROLES = (
    IDENTIFIER,
    JOIN_CODE,
    FOREIGN_KEY_CANDIDATE,
    CATEGORY,
    MEASURE,
    FREE_TEXT,
    TEMPORAL,
    FLAG,
)

# Roles whose values can plausibly match another table's values.
JOIN_KEY_ROLES = frozenset({JOIN_CODE, FOREIGN_KEY_CANDIDATE, CATEGORY})


def type_class(data_type: str) -> str:
    t = str(data_type or "").upper()
    if any(x in t for x in ("DATE", "TIMESTAMP", "TIME")):
        return "date"
    if any(
        x in t
        for x in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL", "BIGINT", "HUGEINT")
    ):
        return "numeric"
    return "string"


def classify_column(freyja_row: Mapping, row_count: int) -> str:
    """Assign a role label to a single column.

    `freyja_row` is one row of Freyja's profile DataFrame (any indexable
    mapping with the standard keys: `data_type`, `uniqueness`,
    `cardinality`, `val_pct_max`, `len_min_word`, `len_max_word`,
    `len_avg_word`, `words_cnt_avg`, `is_binary`).
    """
    tclass = type_class(freyja_row.get("data_type"))

    if tclass == "date":
        return TEMPORAL

    if int(freyja_row.get("is_binary", 0) or 0) == 1:
        return FLAG

    uniqueness = float(freyja_row.get("uniqueness", 0.0) or 0.0)
    cardinality = float(freyja_row.get("cardinality", 0.0) or 0.0)
    cardinality_ratio = (cardinality / row_count) if row_count > 0 else 0.0
    val_pct_max = float(freyja_row.get("val_pct_max", 0.0) or 0.0)

    len_min = freyja_row.get("len_min_word")
    len_max = freyja_row.get("len_max_word")
    len_avg = float(freyja_row.get("len_avg_word", 0.0) or 0.0)
    words_cnt_avg = float(freyja_row.get("words_cnt_avg", 0.0) or 0.0)

    fixed_length = (
        tclass == "string"
        and len_min is not None
        and len_max is not None
        and int(len_min) == int(len_max)
        and int(len_min) > 0
    )

    if uniqueness > 0.95:
        # Highly unique. A unique NUMERIC column is almost always a surrogate
        # / autoincrement key — independent sequences that do not join across
        # tables (and whose small integers cause false {1,2,3,...} overlaps),
        # so it is excluded as IDENTIFIER. A unique STRING column is a natural
        # key (email, username, SKU, ISO/UUID code) that does join, so it stays
        # join-eligible as JOIN_CODE — unless it is long free text.
        if tclass == "numeric":
            return IDENTIFIER
        if len_avg > 30 or words_cnt_avg > 3:
            return FREE_TEXT
        return JOIN_CODE

    if tclass == "numeric":
        if val_pct_max > 0.3 or cardinality < 50:
            return CATEGORY
        return MEASURE

    # tclass == "string"
    if fixed_length:
        return JOIN_CODE
    if val_pct_max > 0.3 or cardinality_ratio < 0.10:
        return CATEGORY
    if len_avg > 30 or words_cnt_avg > 3:
        return FREE_TEXT
    return FOREIGN_KEY_CANDIDATE


# ── Histogram helpers ───────────────────────────────────────────────────────


def cardinality_bucket(cardinality_ratio: float) -> int:
    """Bucket index in [0, 1, 2, 3] for [0,.1] (.1,.5] (.5,.9] (.9,1]."""
    r = cardinality_ratio
    if r <= 0.1:
        return 0
    if r <= 0.5:
        return 1
    if r <= 0.9:
        return 2
    return 3


def length_bucket(len_avg: float) -> int:
    """Bucket index in [0, 1, 2] for short / medium / long."""
    if len_avg <= 5:
        return 0
    if len_avg <= 30:
        return 1
    return 2
