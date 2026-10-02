"""
config.py
---------
App-local configuration for the lightweight method. Merges
`method_lightweight/src/config.py` and `method_lightweight_bis/src/config.py`, with every
path resolved inside 02_app/ so the app needs nothing from the repository.

    lakes/<lake>/semantic.index        FAISS index of the table embeddings
    lakes/<lake>/semantic_mapping.pkl  index position -> table name
    lakes/<lake>/fingerprints.pkl      per-table metaprofile (no raw values)
    model/                             bundled all-MiniLM-L6-v2
"""

from __future__ import annotations

import os
from dataclasses import dataclass

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LAKES_DIR = os.path.join(APP_ROOT, "lakes")
MODEL_DIR = os.path.join(APP_ROOT, "model")
HUB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class LightweightConfig:
    cache_dir: str = os.path.join(LAKES_DIR, "freyja")
    # Local model if download_model.py has saved it, else the Hugging Face name.
    model_name: str = MODEL_DIR if os.path.isdir(MODEL_DIR) else HUB_MODEL_NAME
    hnsw_m: int = 32
    hnsw_ef_construction: int = 40

    @property
    def semantic_index_path(self) -> str:
        return os.path.join(self.cache_dir, "semantic.index")

    @property
    def semantic_mapping_path(self) -> str:
        return os.path.join(self.cache_dir, "semantic_mapping.pkl")


def make_config(lake: str) -> LightweightConfig:
    """Configuration for one shipped lake, by folder name under lakes/."""
    return LightweightConfig(cache_dir=os.path.join(LAKES_DIR, lake))


def fingerprints_path(cfg: LightweightConfig) -> str:
    """Where the per-table structural vectors + join sketches are pickled."""
    return os.path.join(cfg.cache_dir, "fingerprints.pkl")
