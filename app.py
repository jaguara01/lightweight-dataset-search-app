"""
app.py
------
Streamlit front end for the lightweight method: pick a data lake, pick a query (a table
of that lake or an uploaded CSV), and see the tables most similar to it.

    streamlit run 02_app/app.py

The lakes hold no tables, only their metaprofiles and embedding index (see README.md).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time

import pandas as pd
import streamlit as st

from engine.builder import build_lake
from engine.config import LAKES_DIR, make_config
from engine.searcher import BisSearcher

TOP_N = 10

LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img", "AbgCVsSL_400x400.png")
THESIS_INFO = """
**Author**  
Alexis Andre L Vendrix

**Thesis supervisor**  
Marc Maynou Yelamos

**Thesis co-supervisor**  
Sergi Nadal Francesch

**Degree**  
Master's Degree in Data Science

**Master's thesis**  
Facultat d'Informàtica de Barcelona (FIB)  
Universitat Politècnica de Catalunya (UPC) - BarcelonaTech
"""

# Report vocabulary for the result columns.
LABELS = {
    "rank": "Rank",
    "candidate_table": "Table",
    "similarity_score": "Similarity",
    "joinability_score": "Joinability",
    "table_semantic": "Table semantic",
    "struct_sim": "Structural similarity",
    "join_columns": "Join columns",
    "n_join_keys": "Join keys",
}
# Header tooltips: a short definition of each column.
HELP = {
    "rank": "Position of the table, ordered by similarity.",
    "candidate_table": "A table of the data lake.",
    "similarity_score": "How useful the table is to combine with the query: "
                        "√(joinability × (0.7 · table semantic + 0.3 · structural similarity)). "
                        "High only if the tables can be joined and are about the same thing.",
    "joinability_score": "Can the two tables be joined? Estimated containment of the query "
                         "column values in a candidate column (MinHash sketches), weighted by "
                         "type and length compatibility, combined over the 3 strongest keys.",
    "table_semantic": "Topic: cosine of the two table embeddings (name, columns, sample "
                      "values). Reads no value overlap.",
    "struct_sim": "Shape: cosine of the column role, cardinality and value-length "
                  "histograms of the two tables.",
    "join_columns": "Query column ↔ column of this table, with the pair's joinability. "
                    "Lists every join key (pairs of at least 0.5); if there is none, the best "
                    "weaker pair.",
    "n_join_keys": "Number of query columns whose best match has a joinability of at least 0.5.",
}
COLUMNS = list(LABELS)

# Plain names for the search phases timed by the engine.
STEP_NAMES = {
    "Reconstruct query": "stored profile",
    "Query CSV read": "reading",
    "Query profile + fingerprint": "profiling",
    "Serialise + encode": "embedding",
    "Score fingerprints": "scoring",
}
SCORE_COLUMNS = ["similarity_score", "joinability_score", "table_semantic", "struct_sim"]


# ── Data ────────────────────────────────────────────────────────────────────


def available_lakes() -> dict[str, dict]:
    lakes = {}
    os.makedirs(LAKES_DIR, exist_ok=True)
    for key in sorted(os.listdir(LAKES_DIR)):
        meta_path = os.path.join(LAKES_DIR, key, "lake.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as fh:
                lakes[key] = json.load(fh)
    return lakes


def cache_size_mb(lake: str) -> float:
    """On-disk size of what the lake stores: profiles, embedding index, mapping."""
    folder = os.path.join(LAKES_DIR, lake)
    return sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)
               if f != "lake.json") / 1e6


def seconds(x: float | None) -> str:
    return "n/a" if x is None else (f"{x * 1000:.0f} ms" if x < 1 else f"{x:.1f} s")


def engine_version() -> float:
    """Latest modification time of the engine code, so an edit invalidates the cache."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")
    return max(os.path.getmtime(os.path.join(root, f))
               for root, _, files in os.walk(folder) for f in files if f.endswith(".py"))


def lake_version(lake: str) -> float:
    """Modification time of the lake's profiles, so a rebuilt lake is reloaded."""
    return os.path.getmtime(os.path.join(LAKES_DIR, lake, "fingerprints.pkl"))


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "lake"


@st.cache_resource(show_spinner="Loading the lake index…")
def load_searcher(lake: str, version: float, lake_ver: float) -> BisSearcher:
    # `version` and `lake_ver` are only part of the cache key: a changed engine or a
    # rebuilt lake reloads the index.
    import importlib

    from engine import scoring, searcher as searcher_mod

    importlib.reload(scoring)
    importlib.reload(searcher_mod)
    return searcher_mod.BisSearcher(make_config(lake)).load()


def display(df: pd.DataFrame) -> pd.DataFrame:
    return df[COLUMNS].rename(columns=LABELS)


def column_config() -> dict:
    """Score columns as bars, and a header tooltip on every column."""
    cfg = {}
    for c in COLUMNS:
        if c in SCORE_COLUMNS:
            cfg[LABELS[c]] = st.column_config.ProgressColumn(
                LABELS[c], help=HELP[c], min_value=0.0, max_value=1.0, format="%.3f")
        elif c in ("rank", "n_join_keys"):
            cfg[LABELS[c]] = st.column_config.NumberColumn(LABELS[c], help=HELP[c])
        else:
            cfg[LABELS[c]] = st.column_config.TextColumn(
                LABELS[c], help=HELP[c], width="large" if c == "join_columns" else None)
    return cfg


# ── Page ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Lightweight dataset search", layout="wide")
st.title("Lightweight dataset search")
st.caption(
    "Ranks the tables of a data lake by how useful they are to combine with a query table. "
    "**Joinability** estimates whether the two tables share a join key, from MinHash "
    "sketches of their columns. **Similarity** combines it with the topic and the shape of "
    "the tables: √(joinability × (0.7 · table semantic + 0.3 · structural similarity))."
)

lakes = available_lakes()
if "pending_lake" in st.session_state:
    # A lake was just built: select it in the sidebar.
    st.session_state["lake_select"] = st.session_state.pop("pending_lake")

with st.sidebar:
    st.header("Data lake")
    if lakes:
        lake = st.selectbox("Compare against", list(lakes), key="lake_select",
                            format_func=lambda k: lakes[k]["name"])
        st.caption(f"{lakes[lake]['n_tables']} tables. {lakes[lake]['description']}")
        st.caption("Only the profiles and the embedding index are stored, not the tables.")
    else:
        lake = None
        st.caption("No lake yet: build one in the **Build a lake** tab.")

    st.divider()
    st.markdown(THESIS_INFO)
    if os.path.isfile(LOGO):
        st.image(LOGO, width=160)

tab_search, tab_build = st.tabs(["Search", "Build a lake"])

# ── Search ──────────────────────────────────────────────────────────────────

with tab_search:
    if lake is None:
        st.info("No lake to search. Build one in the **Build a lake** tab.")
    else:
        searcher = load_searcher(lake, engine_version(), lake_version(lake))

        st.subheader("Query")
        source = st.radio("Query table", ["A table of the lake", "Upload a CSV"], horizontal=True)

        query_name, uploaded = None, None
        if source == "A table of the lake":
            query_name = st.selectbox("Table", sorted(searcher.embeddings), index=None,
                                      placeholder="Type to search a table name")
        else:
            uploaded = st.file_uploader("CSV file", type="csv")

        ready = query_name is not None or uploaded is not None
        if st.button("Search", type="primary", disabled=not ready):
            try:
                t0 = time.perf_counter()
                if query_name is not None:
                    result = searcher.search_indexed(query_name, top_n=None)
                    label = query_name
                else:
                    df = pd.read_csv(uploaded, low_memory=False)
                    if df.empty or len(df.columns) == 0:
                        raise ValueError("The file has no rows or no columns.")
                    with st.spinner("Profiling and encoding the file…"):
                        result = searcher.search(df, top_n=None, query_name=uploaded.name)
                    label = uploaded.name
                st.session_state["result"] = (lake, label, result, time.perf_counter() - t0,
                                              list(searcher.last_search_timings))
            except Exception as e:  # noqa: BLE001 — shown to the user, not raised
                st.session_state.pop("result", None)
                st.error(f"The query could not be processed: {e}")

        stored = st.session_state.get("result")
        if stored and not set(COLUMNS) <= set(stored[2].columns):
            # A result computed by an older version of the engine: drop it.
            st.session_state.pop("result", None)
            stored = None
        if stored and stored[0] == lake:
            _, label, result, elapsed, timings = stored
            meta = lakes[lake]
            st.subheader(f"Most similar tables to {label} ({meta['name']})")

            st.metric("Query time", seconds(elapsed),
                      help="This query, from the click to the ranking. For an uploaded file it "
                           "includes profiling and embedding the file.")
            steps = [f"{STEP_NAMES.get(name, name.lower())} {seconds(t)}" for name, t, _ in timings]
            st.caption(f"{len(result)} tables ranked · " + " · ".join(steps))
            st.caption(
                f"**{meta['name']}:** {meta['n_tables']} tables · "
                f"cache {cache_size_mb(lake):.2f} MB · "
                f"profiling {seconds(meta.get('build_profile_s'))} · "
                f"embedding {seconds(meta.get('build_embedding_s'))} (built once, offline)",
                help="Cache: everything stored for the lake (column profiles with MinHash sketches, "
                     "table embeddings, name mapping); no table is stored. Profiling: reading every "
                     "table and computing its column profiles. Embedding: encoding every table with "
                     "MiniLM and building the FAISS index, model loading included.",
            )

            st.dataframe(display(result.head(TOP_N)), hide_index=True, width="stretch",
                         column_config=column_config())
            st.caption("Hover a column header for its definition.")

            with st.expander(f"Show the full ranking ({len(result)} tables)"):
                st.dataframe(display(result), hide_index=True, width="stretch",
                             column_config=column_config())
                st.download_button(
                    "Download the ranking (CSV)",
                    result.to_csv(index=False).encode(),
                    file_name=f"ranking_{os.path.splitext(label)[0]}.csv",
                    mime="text/csv",
                )

# ── Build a lake ────────────────────────────────────────────────────────────

with tab_build:
    if st.session_state.pop("reset_build_form", False):
        # The previous build succeeded: start from an empty form.
        for k in ("build_folder", "build_name", "build_description"):
            st.session_state[k] = ""
    st.subheader("Build a lake from CSV files")
    st.caption(
        "Every CSV is profiled (column statistics and MinHash sketches) and described by a "
        "table embedding. Only these are kept: the tables themselves are not copied. "
        "Expect about 0.1–0.2 s per table, plus a few seconds to load the embedding model."
    )
    build_source = st.radio("Tables", ["A folder on this machine", "Upload CSV files"],
                            horizontal=True, key="build_source")
    csv_dir, files = None, []
    if build_source == "A folder on this machine":
        folder = st.text_input("Folder path", placeholder="/path/to/folder/with/csv/files",
                               key="build_folder",
                               help="A folder readable by the machine that runs the app. "
                                    "Every *.csv file directly inside it is used.")
        if folder:
            folder = os.path.expanduser(folder.strip())
            if not os.path.isdir(folder):
                st.error("This folder does not exist.")
            else:
                n_csv = len([f for f in os.listdir(folder) if f.lower().endswith(".csv")])
                st.caption(f"{n_csv} CSV files found.")
                csv_dir = folder if n_csv else None
    else:
        files = st.file_uploader("CSV files", type="csv", accept_multiple_files=True)

    name = st.text_input("Lake name", placeholder="e.g. My data lake", key="build_name")
    description = st.text_input("Description (optional)", key="build_description")
    key = slugify(name) if name else None
    replace = False
    if key and key in lakes:
        replace = st.checkbox(f"Replace the existing lake “{lakes[key]['name']}”")

    ready = bool(name) and (csv_dir is not None or bool(files)) and (key not in lakes or replace)
    if st.button("Build the lake", type="primary", disabled=not ready):
        bar = st.progress(0.0, text="Starting…")

        def on_progress(stage: str, done: int, total: int, table: str) -> None:
            if stage == "profiling":
                bar.progress(0.9 * done / total,
                             text=f"Profiling {done}/{total} tables" + (f": {table}" if table else ""))
            else:
                bar.progress(0.9 + 0.1 * done / total, text="Computing the table embeddings…")

        work = tempfile.mkdtemp(prefix=".building-", dir=LAKES_DIR)
        upload_dir = None
        try:
            if files:
                upload_dir = tempfile.mkdtemp(prefix="lake-upload-")
                for f in files:
                    with open(os.path.join(upload_dir, os.path.basename(f.name)), "wb") as out:
                        out.write(f.getbuffer())
            meta = build_lake(upload_dir or csv_dir, work, name=name,
                              description=description, progress=on_progress)
            final = os.path.join(LAKES_DIR, key)
            if os.path.isdir(final):
                shutil.rmtree(final)
            os.replace(work, final)
            bar.progress(1.0, text="Done")
            st.session_state["build_done"] = (name, meta)
            st.session_state["pending_lake"] = key
            st.session_state["reset_build_form"] = True
            st.rerun()
        except Exception as e:  # noqa: BLE001 — shown to the user, not raised
            bar.empty()
            st.error(f"The lake could not be built: {e}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
            if upload_dir:
                shutil.rmtree(upload_dir, ignore_errors=True)

    done = st.session_state.get("build_done")
    if done:
        built_name, meta = done
        st.success(
            f"**{built_name}** is ready and selected in the sidebar: {meta['n_tables']} tables, "
            f"profiling {seconds(meta['build_profile_s'])}, "
            f"embedding {seconds(meta['build_embedding_s'])}."
        )
        skipped = meta.get("skipped") or []
        if skipped:
            what = "file could not be read or profiled and was" if len(skipped) == 1 else \
                "files could not be read or profiled and were"
            st.warning(f"{len(skipped)} {what} skipped: " + ", ".join(skipped))
