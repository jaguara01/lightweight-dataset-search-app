"""
fingerprint.py
--------------
Build the non-embedding half of one table's metaprofile: the 15-d structural
vector and the union join sketch. The embedding half is produced by
method_lightweight's SemanticIndex from the serialized table text.

A fingerprint is a plain dict so it pickles trivially:

    {
      "struct_vec": np.ndarray(15,)   # role ⊕ cardinality ⊕ length histograms
      "join_cols":  list[dict]        # per-join-column metaprofile: MinHash sig +
                                      # distinct count + type/length stats
    }
"""

from __future__ import annotations

from typing import Dict

from .scoring import join_columns, structural_vector


def build_fingerprint(profile: Dict) -> Dict:
    """Structural vector + per-join-column metaprofiles for one table."""
    return {
        "struct_vec": structural_vector(profile),
        "join_cols": join_columns(profile),
    }
