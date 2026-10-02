"""
scoring.py
----------
The Approach-3 metaprofile and its scoring. Each table is reduced to a single
fixed-size **fingerprint** with two blocks, so a table-vs-table comparison is a
handful of dot products and one sketch overlap — no column-pair loop, cost
independent of cardinality and row count.

Fingerprint
~~~~~~~~~~~
  similarity block
    - the whole-table MiniLM embedding  (handled by SemanticIndex; ≈ table_semantic)
    - a 15-d structural vector: role histogram (8) ⊕ cardinality histogram (4) ⊕
      length histogram (3), each L1-normalised then the whole L2-normalised, so a
      dot product is a cosine in [0, 1]  (≈ "same KIND of table")
  joinability block  (per-join-column metaprofiles — NOT one pooled sketch)
    - for every join-eligible (JOIN_CODE / FK / CATEGORY) column: its bottom-k
      MinHash signature + distinct count (both already in the lightweight profile),
      plus a few cheap Freyja stats (type, value-length range) for a statistical
      compatibility prior.
    - online joinability of a table pair = method_heavy's two-stage aggregation over
      the (small) grid of query-join-col × candidate-join-col scores: per query
      column keep its best match (dedupe), then noisy-OR the top-3 keys. Each pair
      score is `containment × joinable_prior` — the prior demotes coincidental
      overlaps (incompatible type/length) the way heavy's column_semantic does.

Why per-column and not the union: pooling every join column into one sketch dilutes
the real key (a country code drowned by an unrelated category), which lost ~31% of
true joins to containment 0. Per-column sketches + noisy-OR restore heavy's
structure; the loop is bounded by the handful of join-eligible columns, so the cost
is still independent of cardinality and row count.

Fusions (mirroring method_heavy, table_semantic generalised to the richer
similarity block):
    similarity_block_score = W_EMB · table_semantic + W_STRUCT · struct_sim
    joinability_score      = noisy-OR over the top-3 per-query-column best pairs
    similarity_score       = √(joinability_score × similarity_block_score)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .minhash import estimate_containment
from .roles import ALL_ROLES, JOIN_KEY_ROLES, type_class

# Weighting of the two halves of the similarity block. The embedding carries the
# bulk of "are these about the same thing?"; the structural histograms are a
# cheap corrective for tables whose names/values mislead the encoder.
W_EMB = 0.7
W_STRUCT = 0.3

# Joinability aggregation: noisy-OR over the strongest few distinct join keys, as
# in method_heavy (HeavyScorer.TOP_K_KEYS). Capping avoids wide tables saturating
# to 1.0 while letting independent keys reinforce confidence.
TOP_K_KEYS = 3

# A query column counts as a join key when its best match reaches this score
# (method_heavy's STRONG_KEY_THRESHOLD). Reported only; it does not affect scores.
STRONG_KEY_THRESHOLD = 0.5

# Statistical compatibility prior. Each factor is a soft multiplier in
# [PRIOR_FLOOR, 1] (1 = fully compatible) so the prior REFINES containment — it
# demotes coincidental overlaps but never vetoes a real one to zero.
PRIOR_FLOOR = 0.5
TYPE_MISMATCH = 0.7  # same type_class → 1.0, else this

# 8 roles + 4 cardinality bins + 3 length bins.
STRUCT_DIM = len(ALL_ROLES) + 4 + 3


def _l1(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    return v / s if s > 0 else v


def structural_vector(profile: Dict) -> np.ndarray:
    """15-d L2-normalised structural fingerprint from a lightweight profile."""
    role_counts = profile.get("role_counts", {}) or {}
    roles = np.array([role_counts.get(r, 0) for r in ALL_ROLES], dtype=np.float32)
    card = np.array(profile.get("cardinality_hist", (0, 0, 0, 0)), dtype=np.float32)
    length = np.array(profile.get("length_hist", (0, 0, 0)), dtype=np.float32)
    vec = np.concatenate([_l1(roles), _l1(card), _l1(length)])
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def join_columns(profile: Dict) -> List[Dict]:
    """Per-join-column metaprofiles for one table (the joinability block).

    Reads only the cached lightweight profile — the per-column MinHash sketch and a
    few Freyja stats are already there, so no DataFrame pass is needed. Numeric
    CATEGORY columns are dropped (small-integer ordinals like {1..5} cause false
    overlaps — method_heavy skips them too); STRING / numeric codes are kept.
    """
    out: List[Dict] = []
    for name, d in (profile.get("columns") or {}).items():
        if d.get("role") not in JOIN_KEY_ROLES:
            continue
        tclass = type_class(d.get("data_type"))
        if d.get("role") == "CATEGORY" and tclass == "numeric":
            continue
        sig = d.get("minhash") or ()
        n = int(d.get("minhash_n") or 0)
        if not sig or n <= 0:
            continue
        out.append({
            "name": name,
            "sig": sig,
            "n": n,
            "type": tclass,
            "len_min": d.get("len_min"),
            "len_max": d.get("len_max"),
        })
    return out


# ── Online scoring (dot products + one sketch overlap) ───────────────────────


def table_semantic(q_vec: np.ndarray, c_vec: np.ndarray) -> float:
    """Cosine of two L2-normalised table embeddings, clamped to [0, 1]."""
    return float(max(0.0, np.dot(q_vec, c_vec)))


def structural_similarity(q_struct: np.ndarray, c_struct: np.ndarray) -> float:
    """Cosine of two structural fingerprints, in [0, 1]."""
    if q_struct is None or c_struct is None or len(q_struct) == 0 or len(c_struct) == 0:
        return 0.0
    return float(max(0.0, np.dot(q_struct, c_struct)))


def _len_compat(a: Dict, b: Dict) -> float:
    """Soft overlap of two columns' value-length ranges, in [PRIOR_FLOOR, 1]."""
    amin, amax, bmin, bmax = a.get("len_min"), a.get("len_max"), b.get("len_min"), b.get("len_max")
    if None in (amin, amax, bmin, bmax) or amax < amin or bmax < bmin:
        return 1.0  # unknown lengths → neutral
    inter = max(0, min(amax, bmax) - max(amin, bmin))
    union = max(amax, bmax) - min(amin, bmin)
    ov = 1.0 if union <= 0 else inter / union
    return PRIOR_FLOOR + (1.0 - PRIOR_FLOOR) * ov


def joinable_prior(a: Dict, b: Dict) -> float:
    """Statistical compatibility of two columns from Freyja stats (no values).

    A necessary-condition prior: type-class match and value-length overlap. Soft
    (never zero) so it refines containment rather than vetoing it.
    """
    type_factor = 1.0 if a["type"] == b["type"] else TYPE_MISMATCH
    return type_factor * _len_compat(a, b)


def pair_joinability(a: Dict, b: Dict) -> float:
    """One column pair: estimated containment weighted by the compatibility prior."""
    c = estimate_containment(a["sig"], a["n"], b["sig"], b["n"])
    if c <= 0.0:
        return 0.0
    return c * joinable_prior(a, b)


def aggregate_joinability(q_cols: List[Dict], c_cols: List[Dict]) -> float:
    """Table-level joinability via method_heavy's two-stage aggregation.

    (1) dedupe per query column — keep each query column's single best match;
    (2) noisy-OR over the top-TOP_K_KEYS strongest keys: 1 − ∏(1 − jᵢ).
    """
    return aggregate_joinability_detail(q_cols, c_cols)[0]


def aggregate_joinability_detail(
    q_cols: List[Dict], c_cols: List[Dict]
) -> Tuple[float, List[Tuple[str, str, float]]]:
    """Same score as `aggregate_joinability`, plus the pairs that produced it.

    Returns (joinability, pairs), where `pairs` holds each query column's best match
    as (query column, candidate column, pair joinability), strongest first, for the
    query columns that overlap at all. A pair scoring at least STRONG_KEY_THRESHOLD
    is a join key (method_heavy's `n_keys_matched`).
    """
    if not q_cols or not c_cols:
        return 0.0, []
    best = []  # (score, query column, candidate column)
    for q in q_cols:
        bj, bc = 0.0, None
        for c in c_cols:
            j = pair_joinability(q, c)
            if j > bj:
                bj, bc = j, c["name"]
        best.append((bj, q["name"], bc))
    best.sort(key=lambda t: t[0], reverse=True)
    prod = 1.0
    for j, _, _ in best[:TOP_K_KEYS]:
        prod *= (1.0 - j)
    pairs = [(qn, cn, j) for j, qn, cn in best if j > 0.0]
    return 1.0 - prod, pairs


def similarity_block_score(ts: float, ss: float) -> float:
    """Blend the embedding and structural halves of the similarity block."""
    return W_EMB * ts + W_STRUCT * ss


def similarity_score(join: float, sim_block: float) -> float:
    """√(joinability × similarity block) — method_heavy's similarity fusion."""
    return math.sqrt(max(0.0, join) * max(0.0, sim_block))
