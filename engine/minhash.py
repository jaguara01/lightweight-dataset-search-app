"""
minhash.py
----------
Bottom-k MinHash sketch for estimating directional value **containment** between
two columns from a small, fixed-size signature — replacing the old top-10-frequent
value sample.

Why this exists
~~~~~~~~~~~~~~~
The previous value sketch kept the 10 most-frequent values per column. Join keys
are high-cardinality, so the 10 most-frequent values are an unrepresentative slice
that misses the shared keys: ~53% of truly-joinable pairs scored containment 0,
even though their FULL value sets overlap strongly (median 0.82). A MinHash sketch
samples values *uniformly at random by hash* (not by frequency) and lets us
estimate set overlap from a bounded signature — recovering the full-set containment
(mean abs error ~0.05, Pearson ~0.95) at ~0.7 MB.

How it works
~~~~~~~~~~~~
Hash every distinct value with the SAME deterministic 64-bit hash and keep the
`k` smallest hashes (the "bottom-k" signature). Because both columns use the same
hash, their signatures keep the same elements where the sets agree, so the
signatures overlap iff the real sets overlap.

  Jaccard      J ≈ |sig_a ∩ sig_b restricted to the k smallest of the union| / k
  intersection |A∩B| ≈ J · (|A| + |B|) / (1 + J)
  containment  ≈ max(|A∩B|/|A|, |A∩B|/|B|)        # same directional formula as before

A deterministic hash (blake2b, not Python's salted `hash()`) is essential: the
signature is built offline and compared online in a different process.
"""

from __future__ import annotations

import hashlib
import heapq
from typing import Iterable, Sequence, Tuple

K_MINHASH = 128


def _hash64(value: str) -> int:
    """Deterministic 64-bit hash, stable across processes."""
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8", "ignore"), digest_size=8).digest(),
        "big",
    )


def minhash_signature(values: Iterable, k: int = K_MINHASH) -> Tuple[int, ...]:
    """Bottom-k signature: the k smallest distinct 64-bit hashes of `values`.

    If the column has <= k distinct values, the signature is the full hashed set
    (so containment is then estimated exactly).
    """
    hashes = {_hash64(str(v)) for v in values}
    if len(hashes) <= k:
        return tuple(sorted(hashes))
    return tuple(heapq.nsmallest(k, hashes))


def estimate_containment(
    sig_a: Sequence[int], n_a: int,
    sig_b: Sequence[int], n_b: int,
    k: int = K_MINHASH,
) -> float:
    """Directional containment in [0, 1] estimated from two bottom-k signatures.

    `n_a`, `n_b` are the true distinct-value counts of the two columns.
    """
    if not sig_a or not sig_b or n_a <= 0 or n_b <= 0:
        return 0.0
    sa, sb = set(sig_a), set(sig_b)
    union = heapq.nsmallest(k, sa | sb)   # k smallest hashes of the union
    m = len(union)
    if m == 0:
        return 0.0
    inter = sum(1 for x in union if x in sa and x in sb)
    jaccard = inter / m
    if jaccard <= 0.0:
        return 0.0
    inter_size = jaccard * (n_a + n_b) / (1.0 + jaccard)
    return min(1.0, max(inter_size / n_a, inter_size / n_b))
