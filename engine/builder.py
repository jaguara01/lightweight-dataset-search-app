"""
builder.py
----------
Offline phase: build a lake cache from a folder of CSV files. Same steps as
`method_lightweight_bis/src/builder.py`: read -> profile (Freyja) -> fingerprint (MinHash
sketches + structural vector) -> serialise -> encode (MiniLM) -> FAISS index -> save.

Unlike the original, each table is profiled and serialised as soon as it is read, so only
one table is in memory at a time, and a callback reports progress to the caller.

    build_lake("path/to/csv_folder", "lakes/my_lake", name="My lake", description="...")
    # writes lakes/my_lake/{semantic.index, semantic_mapping.pkl, fingerprints.pkl, lake.json}
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import time
from typing import Callable, Dict, List, Optional

import pandas as pd

from .config import LightweightConfig, fingerprints_path
from .fingerprint import build_fingerprint
from .profiler import Profiler
from .semantic_index import SemanticIndex
from .serializer import serialize

# progress(stage, done, total, current table)
Progress = Callable[[str, int, int, str], None]


def build_lake(
    csv_dir: str,
    lake_dir: str,
    name: str,
    description: str = "",
    progress: Optional[Progress] = None,
) -> Dict:
    """Build a lake cache from every *.csv in `csv_dir` and return its lake.json content.

    Tables that cannot be read or profiled are skipped and listed under "skipped".
    """
    progress = progress or (lambda *_: None)
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not paths:
        raise ValueError(f"No CSV file found in {csv_dir}")

    profiler = Profiler()
    fingerprints: Dict[str, Dict] = {}
    texts: List[str] = []
    names: List[str] = []
    skipped: List[str] = []

    # ── Read + profile + fingerprint + serialise, one table at a time ───────
    t0 = time.perf_counter()
    for i, path in enumerate(paths):
        table = os.path.basename(path)
        progress("profiling", i, len(paths), table)
        try:
            df = pd.read_csv(path, low_memory=False)
            profile = profiler.profile(path, df, table)
        except Exception:  # noqa: BLE001 — a bad file must not stop the build
            profile = None
        if profile is None:
            skipped.append(table)
            continue
        fingerprints[table] = build_fingerprint(profile)
        texts.append(serialize(df, table, profile))
        names.append(table)
    profile_s = time.perf_counter() - t0
    progress("profiling", len(paths), len(paths), "")
    if not names:
        raise ValueError("None of the CSV files could be profiled.")

    # ── Encode + FAISS index ────────────────────────────────────────────────
    progress("embedding", 0, 1, f"{len(names)} tables")
    t0 = time.perf_counter()
    cfg = LightweightConfig(cache_dir=lake_dir)
    semantic = SemanticIndex(cfg).build(texts, names)
    embed_s = time.perf_counter() - t0
    progress("embedding", 1, 1, "")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(lake_dir, exist_ok=True)
    semantic.save()
    with open(fingerprints_path(cfg), "wb") as f:
        pickle.dump(fingerprints, f)
    meta = {
        "name": name,
        "description": description,
        "n_tables": len(names),
        "build_profile_s": round(profile_s, 1),
        "build_embedding_s": round(embed_s, 1),
        "build_total_s": round(profile_s + embed_s, 1),
        "skipped": skipped,
    }
    with open(os.path.join(lake_dir, "lake.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
