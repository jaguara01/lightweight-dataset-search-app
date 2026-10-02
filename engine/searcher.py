"""
searcher.py
-----------
Online phase: load the metaprofile cache once, then answer queries. Every score
is a dot product or a sketch overlap over fixed-size fingerprints — no CSV reads
and no column-pair loop at query time.

    search_indexed(name)   — query already in the cache (a table of the shipped lake)
    search(csv)            — external query CSV: profile + encode + fingerprint online

Output columns (per candidate):
    candidate_table, table_semantic, struct_sim, joinability_score, similarity_score

`joinability_score` / `similarity_score` use method_heavy's names so the output is
directly comparable to heavy's aggregated ranking.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import LightweightConfig
from .profiler import Profiler
from .semantic_index import SemanticIndex
from .serializer import serialize

from .config import fingerprints_path
from .fingerprint import build_fingerprint
from . import scoring

PhaseTiming = Tuple[str, float, str]

_OUTPUT_COLUMNS = [
    "rank",
    "candidate_table",
    "similarity_score",
    "joinability_score",
    "table_semantic",
    "struct_sim",
    "join_columns",
    "n_join_keys",
    "best_query_column",
    "best_candidate_column",
]


def _format_pairs(pairs: List[Tuple[str, str, float]]) -> str:
    """Join keys as 'query ↔ candidate (score)'; the best weak pair if there is no key."""
    keys = [p for p in pairs if p[2] >= scoring.STRONG_KEY_THRESHOLD] or pairs[:1]
    return "; ".join(f"{q} ↔ {c} ({j:.2f})" for q, c, j in keys)


class BisSearcher:
    """Loaded once, queried many times."""

    def __init__(self, config: LightweightConfig) -> None:
        self.config = config
        self.semantic = SemanticIndex(config)
        self.fingerprints: Dict[str, Dict] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self._profiler: Optional[Profiler] = None
        self.last_search_timings: List[PhaseTiming] = []
        self.load_timings: List[PhaseTiming] = []

    # ── Load cache ──────────────────────────────────────────────────────────

    def load(self) -> "BisSearcher":
        timings: List[PhaseTiming] = []

        t = time.perf_counter()
        self.semantic.load()
        timings.append(("Load FAISS index", time.perf_counter() - t, f"{self.semantic.n_tables} tables"))

        t = time.perf_counter()
        with open(fingerprints_path(self.config), "rb") as f:
            self.fingerprints = pickle.load(f)
        # Materialise the stored embeddings once so scoring is pure dot products.
        self.embeddings = {
            name: self.semantic.reconstruct(name)[0]
            for name in self.semantic.name_to_id
        }
        timings.append(
            ("Load fingerprints", time.perf_counter() - t, f"{len(self.fingerprints)} fingerprints")
        )

        self.load_timings = timings
        return self

    # ── Public search APIs ──────────────────────────────────────────────────

    def search_indexed(
        self,
        query_name: str,
        top_n: Optional[int] = 10,
        rerank_pool: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query is already in the cache → reconstruct its fingerprint, no I/O."""
        timings: List[PhaseTiming] = []
        if query_name not in self.embeddings:
            raise KeyError(f"{query_name} is not in the index")
        if query_name not in self.fingerprints:
            raise KeyError(f"{query_name} has no cached fingerprint")

        t = time.perf_counter()
        q_vec = self.embeddings[query_name]
        q_fp = self.fingerprints[query_name]
        timings.append(("Reconstruct query", time.perf_counter() - t, query_name))

        result = self._rank(q_vec, q_fp, query_name, top_n, rerank_pool, timings)
        self.last_search_timings = timings
        return result

    def search(
        self,
        query: Union[str, pd.DataFrame],
        top_n: Optional[int] = 10,
        query_name: Optional[str] = None,
        rerank_pool: Optional[int] = None,
    ) -> pd.DataFrame:
        """External query CSV the cache has not seen: profile + encode + fingerprint."""
        timings: List[PhaseTiming] = []

        t = time.perf_counter()
        if isinstance(query, str):
            df = pd.read_csv(query, low_memory=False)
            csv_path = query
            if query_name is None:
                query_name = os.path.basename(query)
        else:
            df = query
            csv_path = None
            if query_name is None:
                query_name = "query"
        timings.append(("Query CSV read", time.perf_counter() - t, query_name))

        t = time.perf_counter()
        q_profile = self._profile(csv_path, query_name, df)
        if q_profile is None:
            raise RuntimeError(f"Freyja failed to profile {query_name}")
        q_fp = build_fingerprint(q_profile)
        timings.append(("Query profile + fingerprint", time.perf_counter() - t, f"{len(df.columns)} cols"))

        t = time.perf_counter()
        q_text = serialize(df, query_name, q_profile)
        q_vec = self.semantic.encode([q_text])[0]
        timings.append(("Serialise + encode", time.perf_counter() - t, "1 text"))

        result = self._rank(q_vec, q_fp, query_name, top_n, rerank_pool, timings)
        self.last_search_timings = timings
        return result

    # ── Internals ───────────────────────────────────────────────────────────

    @property
    def profiler(self) -> Profiler:
        if self._profiler is None:
            self._profiler = Profiler()
        return self._profiler

    def _profile(self, csv_path: Optional[str], name: str, df: pd.DataFrame) -> Optional[dict]:
        if csv_path is not None:
            return self.profiler.profile(csv_path, df, name)
        # Profiler needs a path for Freyja: spill the df to a temp file.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp = f.name
        try:
            df.to_csv(tmp, index=False)
            return self.profiler.profile(tmp, df, name)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _candidate_names(
        self, q_vec: np.ndarray, query_name: str, rerank_pool: Optional[int]
    ) -> List[str]:
        """Either every indexed table (full ranking) or a FAISS top-pool prefilter."""
        if rerank_pool is None:
            return [n for n in self.embeddings if n != query_name]
        raw = self.semantic.search_vec(q_vec, top_k=rerank_pool)
        return [n for n, _ in raw if n != query_name][: rerank_pool]

    def _rank(
        self,
        q_vec: np.ndarray,
        q_fp: Dict,
        query_name: str,
        top_n: Optional[int],
        rerank_pool: Optional[int],
        timings: List[PhaseTiming],
    ) -> pd.DataFrame:
        t = time.perf_counter()
        cand_names = self._candidate_names(q_vec, query_name, rerank_pool)
        q_struct = q_fp["struct_vec"]
        q_cols = q_fp["join_cols"]

        rows = []
        for name in cand_names:
            c_fp = self.fingerprints.get(name)
            if c_fp is None:
                continue
            ts = scoring.table_semantic(q_vec, self.embeddings[name])
            ss = scoring.structural_similarity(q_struct, c_fp["struct_vec"])
            jo, pairs = scoring.aggregate_joinability_detail(q_cols, c_fp["join_cols"])
            n_keys = sum(1 for p in pairs if p[2] >= scoring.STRONG_KEY_THRESHOLD)
            best_q, best_c = (pairs[0][0], pairs[0][1]) if pairs else (None, None)
            sim = scoring.similarity_score(jo, scoring.similarity_block_score(ts, ss))
            rows.append(
                {
                    "candidate_table": name,
                    "table_semantic": round(ts, 4),
                    "struct_sim": round(ss, 4),
                    "joinability_score": round(jo, 4),
                    "similarity_score": round(sim, 4),
                    "join_columns": _format_pairs(pairs),
                    "best_query_column": best_q,
                    "best_candidate_column": best_c,
                    "n_join_keys": n_keys,
                }
            )
        timings.append(("Score fingerprints", time.perf_counter() - t, f"{len(rows)} candidates"))

        if not rows:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)

        df = pd.DataFrame(rows).sort_values(
            "similarity_score", ascending=False, kind="mergesort"
        )
        if top_n is not None:
            df = df.head(top_n)
        df = df.reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df[_OUTPUT_COLUMNS]

    # ── Timing helper ─────────────────────────────────────────────────────────

    @staticmethod
    def print_timings(timings: List[PhaseTiming], title: str = "Timing") -> None:
        W = 32
        sep = "─" * (W + 28)
        total = sum(t for _, t, _ in timings)
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
        print(f"  {'Phase':<{W}} {'Time':>8}   {'(%)':>4}   Note")
        print(sep)
        for name, t, note in timings:
            pct = 100 * t / total if total > 0 else 0
            print(f"  {name:<{W}} {t:>7.3f}s  {pct:>4.0f}%  {note}")
        print(sep)
        print(f"  {'TOTAL':<{W}} {total:>7.3f}s")
        print(f"{sep}\n")
