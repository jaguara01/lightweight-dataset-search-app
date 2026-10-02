"""
semantic_index.py
-----------------
FAISS cosine HNSW index over sentence-transformer embeddings.

Only the cosine metric is supported (L2 was removed). Embeddings are
L2-normalised so inner product == cosine similarity.

Lifecycle
~~~~~~~~~
Offline:
    index = SemanticIndex(config).build(texts, names)
    index.save()

Online:
    index = SemanticIndex(config).load()
    score, name_list = index.search_vec(query_vec, top_k=20)
    stored_vec = index.reconstruct("AdventureWorks2014_CountryRegion.csv")
"""

from __future__ import annotations

import os
import pickle
from typing import Iterable, List, Tuple

import faiss
import numpy as np

from .config import LightweightConfig


class SemanticIndex:
    def __init__(self, config: LightweightConfig) -> None:
        self.config = config
        self._model = None
        self._index: faiss.Index | None = None
        self.table_mapping: dict[int, str] = {}
        self.name_to_id: dict[str, int] = {}

    # ── Model lazy-load ─────────────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    # ── Encoding ────────────────────────────────────────────────────────────

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Encode a batch of texts to L2-normalised float32 vectors."""
        if isinstance(texts, str):
            texts = [texts]
        embs = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        embs = np.ascontiguousarray(embs.astype(np.float32))
        faiss.normalize_L2(embs)
        return embs

    # ── Build ───────────────────────────────────────────────────────────────

    def build(self, texts: List[str], names: List[str]) -> "SemanticIndex":
        if len(texts) != len(names):
            raise ValueError("texts and names must have the same length")

        embeddings = self.encode(texts)
        dim = embeddings.shape[1]

        index = faiss.IndexHNSWFlat(dim, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.config.hnsw_ef_construction
        index.add(embeddings)

        self._index = index
        self.table_mapping = {i: n for i, n in enumerate(names)}
        self.name_to_id = {n: i for i, n in self.table_mapping.items()}
        return self

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        if self._index is None:
            raise RuntimeError("Nothing to save — call build() first.")
        os.makedirs(self.config.cache_dir, exist_ok=True)
        faiss.write_index(self._index, self.config.semantic_index_path)
        with open(self.config.semantic_mapping_path, "wb") as f:
            pickle.dump(self.table_mapping, f)

    def load(self) -> "SemanticIndex":
        self._index = faiss.read_index(self.config.semantic_index_path)
        with open(self.config.semantic_mapping_path, "rb") as f:
            self.table_mapping = pickle.load(f)
        self.name_to_id = {n: i for i, n in self.table_mapping.items()}
        return self

    # ── Search ──────────────────────────────────────────────────────────────

    def search_vec(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Search with a precomputed (already L2-normalised) query vector."""
        if self._index is None:
            raise RuntimeError("Index not built or loaded.")
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        k = min(top_k, self._index.ntotal)
        # HNSW is approximate: with the default efSearch (16) a search for k
        # neighbours returns far fewer than k (only the slice of the graph it
        # walked). Raise efSearch to at least k so the requested number of
        # candidates is actually returned — critical for the full-matrix run,
        # where we ask for all tables and must not silently drop ~20% before
        # the syntactic reranker ever sees them.
        hnsw = getattr(self._index, "hnsw", None)
        if hnsw is not None and hnsw.efSearch < k:
            hnsw.efSearch = k
        sims, indices = self._index.search(query_vec.astype(np.float32), k)
        out = []
        for s, idx in zip(sims[0], indices[0]):
            if idx != -1:
                out.append((self.table_mapping[idx], max(0.0, float(s))))
        return out

    def reconstruct(self, name: str) -> np.ndarray:
        """Return the stored embedding for an indexed table by name."""
        if self._index is None:
            raise RuntimeError("Index not loaded.")
        if name not in self.name_to_id:
            raise KeyError(f"{name} is not in the index")
        dim = self._index.d
        vec = np.zeros((1, dim), dtype=np.float32)
        self._index.reconstruct(self.name_to_id[name], vec[0])
        return vec

    @property
    def n_tables(self) -> int:
        return 0 if self._index is None else self._index.ntotal
