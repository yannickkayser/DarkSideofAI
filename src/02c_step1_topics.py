"""
02c_step1_topics.py
===================
Nelson (2020) Step 1 — extension: Topic Modelling & Step 2 Sampling

Bridges Step 1 (computational pattern detection) to Step 2 (interpretive
close reading) by identifying latent topics in the corpus and using them
to build a principled sampling strategy.

Pipeline:
  1. LDA topic model (sklearn) on the full corpus — extracts K topics
  2. Per-topic audience profile: how much does each topic appear in
     B2B (client) vs B2W (worker) pages?  Classifies topics as
     client-leaning, worker-leaning, or shared.
  3. PCA on the document-topic matrix — reduces K dimensions to 2–3
     principal components for visualisation and clustering.
  4. Hypothesis-stratified sampling for Step 2: for each hypothesis
     (H1a visibility, H1b automation, H1c hypervisibility), identifies
     the 2-3 most relevant topics by term overlap, then samples top
     pages per audience scored by topic weight × collocate divergence ×
     hypothesis term density.  This produces ~50-60 pages with explicit
     theoretical justification for every selection.

Outputs written to four SQLite tables:
  - topic_terms            : top terms per topic + coherence
  - document_topics        : per-page topic assignments + PCA coords
  - topic_audience_profile : per-topic B2B vs B2W prominence
  - step2_sample           : ranked page_ids for Step 2 close reading

Prerequisites:
  - 01_prepare.py  (corpus_view must exist)
  - 02_step1_frequency.py  (cooccurrence_results used for divergence scoring)

Usage:
    python3 src/02c_step1_topics.py
"""

import sqlite3
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH        = "data/scraping.db"
N_TOPICS       = 40         # LDA topics to extract (tune between 30-50)
N_TOP_TERMS    = 20         # top terms per topic to store
N_PCA_DIMS     = 3          # PCA components to keep
MAX_ITER       = 50         # LDA iterations (increase for final run)
RANDOM_STATE   = 42
MIN_DF         = 5          # minimum document frequency for vectoriser
MAX_DF_FRAC    = 0.85       # maximum document fraction for vectoriser
MIN_TOKEN_COUNT  = 30       # minimum tokens for a page to enter the model

# Shared-topic threshold: a topic is "shared" if neither audience accounts
# for more than this fraction of the topic's total weight.
SHARED_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Hypothesis-stratified sampling config
# ---------------------------------------------------------------------------
# For each hypothesis, the sampling strategy identifies the 2-3 most
# relevant topics (by term overlap with hypothesis vocabulary), then
# samples the top pages from each audience within those topics.
# This gives ~15-20 pages per hypothesis with clear theoretical
# justification for every selection.

HYPOTHESIS_TERMS = {
    "H1a_visibility": {
        "terms": {"worker", "labour", "task", "job", "pay", "earn", "payment",
                  "work", "annotator", "labeller", "moderator", "freelance",
                  "gig", "contractor", "wage", "income", "employment"},
        "description": "Labour visibility — explicit references to human labour",
        "n_topics": 3,       # top N topics per hypothesis
        "n_pages_per_topic_per_audience": 5,
    },
    "H1b_automation": {
        "terms": {"autonomous", "machine", "automate", "intelligent", "automation",
                  "model", "algorithm", "pipeline", "scalable", "engine",
                  "deploy", "inference", "prediction", "neural", "llm"},
        "description": "Automation myth — framing as autonomous/intelligent systems",
        "n_topics": 3,
        "n_pages_per_topic_per_audience": 5,
    },
    "H1c_hypervisibility": {
        "terms": {"human", "quality", "oversight", "annotation", "label",
                  "datum", "accuracy", "review", "expert", "curate",
                  "human-in-the-loop", "verification", "audit", "check"},
        "description": "Strategic hypervisibility — human labour as quality feature",
        "n_topics": 3,
        "n_pages_per_topic_per_audience": 5,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — sklearn / numpy may need to be installed
# ---------------------------------------------------------------------------

def _import_ml():
    """Import ML dependencies with a helpful error if missing."""
    try:
        from sklearn.decomposition import LatentDirichletAllocation, PCA
        from sklearn.feature_extraction.text import CountVectorizer
        import numpy as np
        return LatentDirichletAllocation, PCA, CountVectorizer, np
    except ImportError as e:
        log.error(
            "This script requires scikit-learn and numpy.\n"
            "  pip install scikit-learn numpy"
        )
        raise SystemExit(1) from e


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_output_tables(conn: sqlite3.Connection):
    conn.executescript("""
        DROP TABLE IF EXISTS topic_terms;
        DROP TABLE IF EXISTS document_topics;
        DROP TABLE IF EXISTS topic_audience_profile;
        DROP TABLE IF EXISTS step2_sample;

        CREATE TABLE topic_terms (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id    INTEGER NOT NULL,
            topic_label TEXT,               -- optional human label (filled later)
            term        TEXT NOT NULL,
            weight      REAL NOT NULL,       -- term weight in topic
            rank        INTEGER NOT NULL,    -- 1 = highest weight
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE document_topics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id         INTEGER NOT NULL,
            domain          TEXT,
            audience        TEXT,
            dominant_topic   INTEGER NOT NULL,  -- topic with highest weight
            topic_weight    REAL NOT NULL,      -- weight of dominant topic
            topic_vector    TEXT,               -- full topic vector as JSON
            pca_1           REAL,
            pca_2           REAL,
            pca_3           REAL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE topic_audience_profile (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id        INTEGER NOT NULL,
            topic_label     TEXT,
            avg_weight_client REAL,     -- mean topic weight across client pages
            avg_weight_worker REAL,     -- mean topic weight across worker pages
            client_share    REAL,       -- client_sum / (client_sum + worker_sum)
            category        TEXT,       -- 'client_leaning' | 'worker_leaning' | 'shared'
            n_dominant_client INTEGER,  -- pages where this is the dominant topic (client)
            n_dominant_worker INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE step2_sample (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id         INTEGER NOT NULL,
            url             TEXT,
            domain          TEXT,
            audience        TEXT,
            dominant_topic  INTEGER,
            topic_weight    REAL,
            sampling_reason TEXT,        -- why this page was selected
            collocate_divergence REAL,   -- PMI profile divergence score (if available)
            priority_rank   INTEGER,     -- 1 = most analytically interesting
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_dt_page
            ON document_topics(page_id);
        CREATE INDEX IF NOT EXISTS idx_dt_topic
            ON document_topics(dominant_topic);
        CREATE INDEX IF NOT EXISTS idx_tap_cat
            ON topic_audience_profile(category);
        CREATE INDEX IF NOT EXISTS idx_s2_rank
            ON step2_sample(priority_rank);
    """)
    conn.commit()
    log.info("Output tables created.")


# ---------------------------------------------------------------------------
# Step 1: Load corpus — return page metadata + token strings for sklearn
# ---------------------------------------------------------------------------

def load_exclusions(conn: sqlite3.Connection) -> tuple:
    """Load excluded page IDs and terms from 01_prepare_additions tables."""
    excluded_pages = set()
    excluded_terms = set()

    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                     "AND name='excluded_pages'").fetchone():
        excluded_pages = {
            r[0] for r in conn.execute("SELECT page_id FROM excluded_pages").fetchall()
        }
        log.info(f"  Loaded {len(excluded_pages)} excluded pages.")

    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                     "AND name='excluded_terms'").fetchone():
        excluded_terms = {
            r[0] for r in conn.execute("SELECT term FROM excluded_terms").fetchall()
        }
        log.info(f"  Loaded {len(excluded_terms)} excluded terms.")

    return excluded_pages, excluded_terms


def load_corpus(conn: sqlite3.Connection) -> tuple:
    """
    Returns:
      docs     : list of str — space-joined token strings per page
      metadata : list of dict — {page_id, domain, audience, url, token_count}
      excluded_terms : set — terms to exclude (passed to vectoriser stop_words)
    """
    log.info("Loading corpus from corpus_view...")

    excluded_pages, excluded_terms = load_exclusions(conn)

    rows = conn.execute(f"""
        SELECT page_id, url, audience, domain, unigrams, token_count
        FROM corpus_view
        WHERE audience IN ('client', 'worker')
          AND token_count >= {MIN_TOKEN_COUNT}
    """).fetchall()

    docs     = []
    metadata = []
    skipped  = 0

    for row in rows:
        if row["page_id"] in excluded_pages:
            skipped += 1
            continue

        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        if not unigrams:
            continue

        # Filter excluded terms before passing to vectoriser
        if excluded_terms:
            unigrams = [t for t in unigrams if t not in excluded_terms]
            if not unigrams:
                continue

        # sklearn CountVectorizer expects space-separated strings
        docs.append(" ".join(unigrams))
        metadata.append({
            "page_id":     row["page_id"],
            "url":         row["url"],
            "domain":      row["domain"],
            "audience":    row["audience"],
            "token_count": row["token_count"],
        })

    log.info(f"  {len(docs)} pages loaded (min {MIN_TOKEN_COUNT} tokens, "
             f"{skipped} excluded pages skipped, "
             f"{len(excluded_terms)} terms filtered).")
    return docs, metadata, excluded_terms


# ---------------------------------------------------------------------------
# Step 2: Fit LDA
# ---------------------------------------------------------------------------

def fit_lda(docs, np, CountVectorizer, LatentDirichletAllocation,
            excluded_terms=None):
    """
    Vectorise → LDA → return model, vectoriser, document-topic matrix.
    excluded_terms are passed as stop_words to prevent boilerplate/noise
    from forming their own topics.
    """
    log.info("Vectorising corpus...")
    stop_words = list(excluded_terms) if excluded_terms else None
    vectoriser = CountVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF_FRAC,
        stop_words=stop_words,
        # tokens are already clean unigrams — no extra preprocessing
        token_pattern=r"(?u)\S+",
    )
    dtm = vectoriser.fit_transform(docs)
    vocab = vectoriser.get_feature_names_out()
    log.info(f"  DTM shape: {dtm.shape[0]} docs × {dtm.shape[1]} terms")

    log.info(f"Fitting LDA with {N_TOPICS} topics (max_iter={MAX_ITER})...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        learning_method="batch",
        n_jobs=-1,
    )
    doc_topic_matrix = lda.fit_transform(dtm)  # shape: (n_docs, N_TOPICS)

    log.info(f"  LDA perplexity: {lda.perplexity(dtm):.2f}")
    log.info(f"  Log-likelihood: {lda.score(dtm):.2f}")

    return lda, vectoriser, doc_topic_matrix, vocab


# ---------------------------------------------------------------------------
# Step 3: Extract topic terms
# ---------------------------------------------------------------------------

def extract_topic_terms(lda, vocab, np) -> list[dict]:
    """Top N_TOP_TERMS per topic."""
    results = []
    for topic_id, component in enumerate(lda.components_):
        top_indices = component.argsort()[::-1][:N_TOP_TERMS]
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "topic_id":    topic_id,
                "topic_label": None,   # to be filled manually later
                "term":        vocab[idx],
                "weight":      round(float(component[idx]), 6),
                "rank":        rank,
            })
    return results


# ---------------------------------------------------------------------------
# Step 4: Document-topic assignments + PCA
# ---------------------------------------------------------------------------

def compute_document_topics(doc_topic_matrix, metadata, np, PCA) -> list[dict]:
    """
    For each page: dominant topic, full vector, and PCA coordinates.
    """
    log.info(f"Running PCA ({N_PCA_DIMS} components) on document-topic matrix...")
    pca = PCA(n_components=N_PCA_DIMS, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(doc_topic_matrix)

    explained = pca.explained_variance_ratio_
    log.info(f"  Explained variance: " +
             "  ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(explained)))
    log.info(f"  Total explained: {sum(explained):.3f}")

    results = []
    for i, meta in enumerate(metadata):
        topic_vec = doc_topic_matrix[i]
        dominant  = int(topic_vec.argmax())

        results.append({
            "page_id":        meta["page_id"],
            "domain":         meta["domain"],
            "audience":       meta["audience"],
            "dominant_topic": dominant,
            "topic_weight":   round(float(topic_vec[dominant]), 6),
            "topic_vector":   json.dumps([round(float(v), 6) for v in topic_vec]),
            "pca_1":          round(float(pca_coords[i, 0]), 6),
            "pca_2":          round(float(pca_coords[i, 1]), 6),
            "pca_3":          round(float(pca_coords[i, 2]), 6) if N_PCA_DIMS >= 3 else None,
        })

    return results


# ---------------------------------------------------------------------------
# Step 5: Topic audience profiles
# ---------------------------------------------------------------------------

def compute_topic_profiles(doc_topic_matrix, metadata, np) -> list[dict]:
    """
    For each topic: mean weight in client pages vs worker pages.
    Classify as client_leaning, worker_leaning, or shared.
    """
    client_mask = [m["audience"] == "client" for m in metadata]
    worker_mask = [m["audience"] == "worker" for m in metadata]

    client_matrix = doc_topic_matrix[client_mask]
    worker_matrix = doc_topic_matrix[worker_mask]

    # Dominant topic counts
    client_dominant = client_matrix.argmax(axis=1) if client_matrix.shape[0] > 0 else []
    worker_dominant = worker_matrix.argmax(axis=1) if worker_matrix.shape[0] > 0 else []

    results = []
    for t in range(N_TOPICS):
        avg_c = float(client_matrix[:, t].mean()) if client_matrix.shape[0] > 0 else 0
        avg_w = float(worker_matrix[:, t].mean()) if worker_matrix.shape[0] > 0 else 0
        total = avg_c + avg_w
        c_share = avg_c / total if total > 0 else 0.5

        if c_share > SHARED_THRESHOLD:
            cat = "client_leaning"
        elif c_share < (1 - SHARED_THRESHOLD):
            cat = "worker_leaning"
        else:
            cat = "shared"

        n_dom_c = int(sum(1 for d in client_dominant if d == t))
        n_dom_w = int(sum(1 for d in worker_dominant if d == t))

        results.append({
            "topic_id":           t,
            "topic_label":        None,
            "avg_weight_client":  round(avg_c, 6),
            "avg_weight_worker":  round(avg_w, 6),
            "client_share":       round(c_share, 4),
            "category":           cat,
            "n_dominant_client":  n_dom_c,
            "n_dominant_worker":  n_dom_w,
        })

    return results


# ---------------------------------------------------------------------------
# Step 6: Collocate divergence scoring
# ---------------------------------------------------------------------------

def compute_collocate_divergence(conn: sqlite3.Connection) -> dict:
    """
    For each focus term in cooccurrence_results (from 02_step1_frequency.py),
    compute how different its collocate profile is between client and worker.

    Divergence = 1 - cosine_similarity(PMI_vector_client, PMI_vector_worker)

    Returns {focus_term: divergence_score}
    """
    log.info("Computing collocate divergence from 02 co-occurrence data...")

    # Check if cooccurrence_results exists
    table_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cooccurrence_results'"
    ).fetchone()
    if not table_check:
        log.warning("  cooccurrence_results table not found — "
                     "run 02_step1_frequency.py first. Skipping divergence scoring.")
        return {}

    rows = conn.execute("""
        SELECT focus_term, audience, collocate, pmi
        FROM cooccurrence_results
        WHERE comparison = 'cross_platform'
    """).fetchall()

    if not rows:
        log.warning("  No cross_platform co-occurrence data found.")
        return {}

    # Build PMI vectors per (focus_term, audience)
    profiles = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        profiles[r["focus_term"]][r["audience"]][r["collocate"]] = r["pmi"]

    divergences = {}
    for term, aud_data in profiles.items():
        if "client" not in aud_data or "worker" not in aud_data:
            continue

        c_vec = aud_data["client"]
        w_vec = aud_data["worker"]
        all_collocates = set(c_vec) | set(w_vec)

        if len(all_collocates) < 3:
            continue

        # Cosine similarity on PMI vectors
        dot  = sum(c_vec.get(c, 0) * w_vec.get(c, 0) for c in all_collocates)
        mag_c = math.sqrt(sum(c_vec.get(c, 0) ** 2 for c in all_collocates))
        mag_w = math.sqrt(sum(w_vec.get(c, 0) ** 2 for c in all_collocates))

        if mag_c == 0 or mag_w == 0:
            cos_sim = 0.0
        else:
            cos_sim = dot / (mag_c * mag_w)

        divergences[term] = round(1.0 - cos_sim, 6)

    log.info(f"  Divergence scores for {len(divergences)} focus terms.")

    # Log most divergent
    by_div = sorted(divergences.items(), key=lambda x: x[1], reverse=True)
    log.info("  Most DIVERGENT collocate profiles (same word, different framing):")
    for term, div in by_div[:15]:
        log.info(f"    {term:<25} divergence={div:.4f}")

    return divergences


# ---------------------------------------------------------------------------
# Step 7: Build Step 2 sampling table
# ---------------------------------------------------------------------------

def compute_topic_hypothesis_relevance(
    topic_terms_list: list[dict],
) -> dict:
    """
    For each hypothesis, compute a relevance score for each topic based on
    the overlap between the topic's top terms and the hypothesis vocabulary.

    Returns: {hypothesis_key: [(topic_id, overlap_score, matching_terms), ...]}
    sorted by overlap descending.
    """
    # Build topic → top terms mapping
    topic_top = defaultdict(set)
    for r in topic_terms_list:
        if r["rank"] <= 15:   # consider top 15 terms per topic
            topic_top[r["topic_id"]].add(r["term"])

    result = {}
    for hyp_key, hyp_config in HYPOTHESIS_TERMS.items():
        hyp_terms = hyp_config["terms"]
        scored = []
        for topic_id, top_terms in topic_top.items():
            overlap = top_terms & hyp_terms
            if overlap:
                # Score: number of matching terms weighted by their rank
                score = len(overlap)
                scored.append((topic_id, score, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        result[hyp_key] = scored

    return result


def build_step2_sample(
    doc_topics: list[dict],
    topic_profiles: list[dict],
    topic_terms_list: list[dict],
    divergences: dict,
    metadata: list[dict],
    conn: sqlite3.Connection,
) -> list[dict]:
    """
    HYPOTHESIS-STRATIFIED sampling strategy.

    For each hypothesis (H1a, H1b, H1c):
      1. Identify the 2-3 most relevant topics (by term overlap with
         hypothesis vocabulary)
      2. From those topics, sample top pages per audience, scored by
         topic weight × collocate divergence

    This ensures coverage of each hypothesis domain with clear
    theoretical justification for every selection.
    """
    log.info("Building hypothesis-stratified Step 2 sampling table...")

    # --- Compute topic-hypothesis relevance ---
    relevance = compute_topic_hypothesis_relevance(topic_terms_list)

    for hyp_key, scored_topics in relevance.items():
        hyp_config = HYPOTHESIS_TERMS[hyp_key]
        log.info(f"  {hyp_key} ({hyp_config['description']}):")
        for topic_id, score, terms in scored_topics[:5]:
            profile = next((p for p in topic_profiles if p["topic_id"] == topic_id), {})
            cat = profile.get("category", "?")
            log.info(f"    Topic {topic_id} [{cat}]  overlap={score}  "
                     f"terms: {', '.join(sorted(terms))}")

    # --- Build metadata lookup ---
    meta_by_page = {m["page_id"]: m for m in metadata}

    # --- Group document-topic entries by topic ---
    by_topic = defaultdict(list)
    for dt in doc_topics:
        by_topic[dt["dominant_topic"]].append(dt)

    # --- Page-level divergence scoring ---
    page_terms_cache = {}

    def get_page_terms(page_id):
        if page_id not in page_terms_cache:
            row = conn.execute(
                "SELECT unigrams FROM corpus_view WHERE page_id = ?", (page_id,)
            ).fetchone()
            if row and row["unigrams"]:
                page_terms_cache[page_id] = set(json.loads(row["unigrams"]))
            else:
                page_terms_cache[page_id] = set()
        return page_terms_cache[page_id]

    def score_page(dt, hyp_terms):
        """Score a page by topic weight, collocate divergence, and hypothesis term density."""
        page_id = dt["page_id"]
        terms = get_page_terms(page_id)

        # Collocate divergence: average divergence of focus terms on this page
        if divergences:
            matching_divs = [divergences[t] for t in terms if t in divergences]
            avg_div = (sum(matching_divs) / len(matching_divs)) if matching_divs else 0
        else:
            avg_div = 0

        # Hypothesis term density: fraction of page terms that are hypothesis-relevant
        hyp_count = len(terms & hyp_terms)
        hyp_density = hyp_count / len(terms) if terms else 0

        # Combined: topic_weight × (1 + divergence) × (1 + hyp_density × 10)
        combined = dt["topic_weight"] * (1 + avg_div) * (1 + hyp_density * 10)

        return {
            "page_id":              page_id,
            "url":                  meta_by_page.get(page_id, {}).get("url", ""),
            "domain":               dt["domain"],
            "audience":             dt["audience"],
            "dominant_topic":       dt["dominant_topic"],
            "topic_weight":         dt["topic_weight"],
            "collocate_divergence": round(avg_div, 6),
            "hyp_density":          round(hyp_density, 6),
            "combined_score":       combined,
        }

    results = []
    seen_pages = set()   # avoid duplicate sampling across hypotheses

    for hyp_key, scored_topics in relevance.items():
        hyp_config = HYPOTHESIS_TERMS[hyp_key]
        hyp_terms  = hyp_config["terms"]
        n_topics   = hyp_config["n_topics"]
        n_per      = hyp_config["n_pages_per_topic_per_audience"]

        # Select top N topics for this hypothesis
        selected_topics = scored_topics[:n_topics]
        if not selected_topics:
            log.warning(f"  No relevant topics for {hyp_key} — skipping.")
            continue

        log.info(f"  Sampling for {hyp_key}: topics "
                 f"{[t[0] for t in selected_topics]}")

        for topic_id, overlap_score, matching_terms in selected_topics:
            candidates = by_topic.get(topic_id, [])
            if not candidates:
                continue

            scored = [score_page(dt, hyp_terms) for dt in candidates]
            scored.sort(key=lambda x: x["combined_score"], reverse=True)

            for aud in ("client", "worker"):
                aud_cands = [s for s in scored
                             if s["audience"] == aud
                             and s["page_id"] not in seen_pages]
                for s in aud_cands[:n_per]:
                    seen_pages.add(s["page_id"])
                    results.append({
                        "page_id":              s["page_id"],
                        "url":                  s["url"],
                        "domain":               s["domain"],
                        "audience":             s["audience"],
                        "dominant_topic":       s["dominant_topic"],
                        "topic_weight":         s["topic_weight"],
                        "sampling_reason":      (f"{hyp_key}_topic_{topic_id}_"
                                                 f"overlap={overlap_score}_"
                                                 f"terms={','.join(sorted(matching_terms))}"),
                        "collocate_divergence": s["collocate_divergence"],
                        "priority_rank":        0,   # set below
                    })

    # Global ranking by combined analytical interest
    results.sort(key=lambda x: (x["topic_weight"] * (1 + x["collocate_divergence"])),
                 reverse=True)
    for i, r in enumerate(results):
        r["priority_rank"] = i + 1

    # Summary
    hyp_counts = Counter(r["sampling_reason"].split("_topic_")[0] for r in results)
    aud_counts = Counter(r["audience"] for r in results)
    log.info(f"  {len(results)} pages selected for Step 2:")
    log.info(f"    By hypothesis: {dict(hyp_counts)}")
    log.info(f"    By audience:   {dict(aud_counts)}")

    return results


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_topic_terms(conn, results):
    conn.executemany("""
        INSERT INTO topic_terms (topic_id, topic_label, term, weight, rank)
        VALUES (:topic_id, :topic_label, :term, :weight, :rank)
    """, results)
    conn.commit()


def save_document_topics(conn, results):
    conn.executemany("""
        INSERT INTO document_topics
            (page_id, domain, audience, dominant_topic, topic_weight,
             topic_vector, pca_1, pca_2, pca_3)
        VALUES
            (:page_id, :domain, :audience, :dominant_topic, :topic_weight,
             :topic_vector, :pca_1, :pca_2, :pca_3)
    """, results)
    conn.commit()


def save_topic_profiles(conn, results):
    conn.executemany("""
        INSERT INTO topic_audience_profile
            (topic_id, topic_label, avg_weight_client, avg_weight_worker,
             client_share, category, n_dominant_client, n_dominant_worker)
        VALUES
            (:topic_id, :topic_label, :avg_weight_client, :avg_weight_worker,
             :client_share, :category, :n_dominant_client, :n_dominant_worker)
    """, results)
    conn.commit()


def save_step2_sample(conn, results):
    conn.executemany("""
        INSERT INTO step2_sample
            (page_id, url, domain, audience, dominant_topic, topic_weight,
             sampling_reason, collocate_divergence, priority_rank)
        VALUES
            (:page_id, :url, :domain, :audience, :dominant_topic, :topic_weight,
             :sampling_reason, :collocate_divergence, :priority_rank)
    """, results)
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("02c_step1_topics.py — Topic Modelling & Step 2 Sampling")
    log.info("=" * 60)

    # Import ML dependencies
    LatentDirichletAllocation, PCA, CountVectorizer, np = _import_ml()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Verify corpus_view
    view_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()
    if not view_check:
        raise RuntimeError("corpus_view not found — run 01_prepare.py first.")

    init_output_tables(conn)

    # --- Load ---
    docs, metadata, excluded_terms = load_corpus(conn)

    # --- LDA ---
    log.info("-" * 60)
    log.info("TOPIC MODELLING (LDA)")
    log.info("-" * 60)
    lda, vectoriser, doc_topic_matrix, vocab = fit_lda(
        docs, np, CountVectorizer, LatentDirichletAllocation,
        excluded_terms=excluded_terms
    )

    # --- Topic terms ---
    topic_terms = extract_topic_terms(lda, vocab, np)

    # Log top terms per topic
    for t in range(N_TOPICS):
        terms = [r["term"] for r in topic_terms if r["topic_id"] == t and r["rank"] <= 8]
        log.info(f"  Topic {t:>2}: {', '.join(terms)}")

    # --- Document-topic assignments + PCA ---
    log.info("-" * 60)
    log.info("DOCUMENT-TOPIC ASSIGNMENTS + PCA")
    log.info("-" * 60)
    doc_topics = compute_document_topics(doc_topic_matrix, metadata, np, PCA)

    # --- Topic audience profiles ---
    log.info("-" * 60)
    log.info("TOPIC AUDIENCE PROFILES")
    log.info("-" * 60)
    topic_profiles = compute_topic_profiles(doc_topic_matrix, metadata, np)

    cats = Counter(p["category"] for p in topic_profiles)
    for cat, n in cats.most_common():
        log.info(f"  {cat:<20} {n} topics")

    for p in topic_profiles:
        terms = [r["term"] for r in topic_terms
                 if r["topic_id"] == p["topic_id"] and r["rank"] <= 5]
        log.info(f"  Topic {p['topic_id']:>2} [{p['category']:<16}]  "
                 f"client={p['avg_weight_client']:.4f}  "
                 f"worker={p['avg_weight_worker']:.4f}  "
                 f"share={p['client_share']:.2f}  "
                 f"terms: {', '.join(terms)}")

    # --- Collocate divergence (from 02 outputs) ---
    log.info("-" * 60)
    log.info("COLLOCATE DIVERGENCE (from 02_step1_frequency.py)")
    log.info("-" * 60)
    divergences = compute_collocate_divergence(conn)

    # --- Step 2 sampling ---
    log.info("-" * 60)
    log.info("STEP 2 SAMPLING STRATEGY")
    log.info("-" * 60)
    sample = build_step2_sample(doc_topics, topic_profiles, topic_terms,
                                divergences, metadata, conn)

    # Log sample summary
    if sample:
        sample_aud = Counter(s["audience"] for s in sample)
        sample_top = Counter(s["dominant_topic"] for s in sample)
        log.info(f"  By audience: {dict(sample_aud)}")
        log.info(f"  By topic:    {dict(sample_top.most_common(10))}")
        log.info(f"  Top 10 pages for Step 2 close reading:")
        for s in sample[:10]:
            log.info(f"    rank={s['priority_rank']:>3}  "
                     f"page={s['page_id']:<6}  "
                     f"{s['audience']:<8}  "
                     f"topic={s['dominant_topic']:<3}  "
                     f"tw={s['topic_weight']:.3f}  "
                     f"div={s['collocate_divergence']:.3f}  "
                     f"{s['domain']}")

    # -----------------------------------------------------------------------
    # Save everything
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Saving results to database...")

    save_topic_terms(conn, topic_terms)
    log.info(f"  topic_terms            : {len(topic_terms):,} rows")

    save_document_topics(conn, doc_topics)
    log.info(f"  document_topics        : {len(doc_topics):,} rows")

    save_topic_profiles(conn, topic_profiles)
    log.info(f"  topic_audience_profile : {len(topic_profiles)} rows")

    save_step2_sample(conn, sample)
    log.info(f"  step2_sample           : {len(sample):,} rows")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("TOPIC MODELLING & SAMPLING COMPLETE")
    log.info("Query examples:")
    log.info("  -- Shared topics (appear in both B2B and B2W):")
    log.info("  SELECT topic_id, avg_weight_client, avg_weight_worker, client_share")
    log.info("  FROM topic_audience_profile")
    log.info("  WHERE category = 'shared'")
    log.info("  ORDER BY client_share;")
    log.info("")
    log.info("  -- Top terms for a specific topic:")
    log.info("  SELECT term, weight FROM topic_terms")
    log.info("  WHERE topic_id = 5 ORDER BY rank LIMIT 20;")
    log.info("")
    log.info("  -- Step 2 sample (ranked by analytical interest):")
    log.info("  SELECT s.priority_rank, s.page_id, s.audience, s.domain,")
    log.info("         s.dominant_topic, s.collocate_divergence, c.url")
    log.info("  FROM step2_sample s")
    log.info("  JOIN corpus_view c ON c.page_id = s.page_id")
    log.info("  ORDER BY s.priority_rank LIMIT 30;")
    log.info("")
    log.info("  -- PCA coordinates for visualisation:")
    log.info("  SELECT page_id, audience, domain, pca_1, pca_2, dominant_topic")
    log.info("  FROM document_topics;")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
