"""
03_step1_topics.py
==================
Nelson (2020) Step 1 — Topic Modelling, Sampling Strategy, and STM Export

Consolidates src/02c_step1_topics.py  (LDA + Step 2 sampling)
         and src/02d_step1_stm_export.py (STM CSV export for R).

Pipeline position:
  Stage 3 — Topic Modelling (run after 02_step1_analysis.py)
  Prerequisites:
    01_prepare_corpus.py  (creates corpus_view — now includes segments column)
    02_step1_analysis.py  (creates cooccurrence_results, excluded_* tables)
  Next step:
    04_step1_figures.py   (visualisations: PCA scatter, topic profiles, etc.)
    Step 2 close reading: step2_sample table guides 05_step2_export.py

What this script does:
──────────────────────
Part A — LDA Topic Modelling:

  1. LDA topic model (sklearn LatentDirichletAllocation) on the full
     corpus.  N_TOPICS latent topics, each characterised by a probability
     distribution over the vocabulary.  N_TOPICS = 40 is the default
     after testing 20/30/40/50 and inspecting topic coherence and
     separation (see RUN_DIAGNOSTICS mode below).

  2. LDA quality diagnostics (when RUN_DIAGNOSTICS = True):
     Fits LDA for each topic count in DIAGNOSTIC_N_TOPICS_RANGE and
     computes three complementary quality metrics:
       a. Perplexity: measures how well the model predicts held-out data.
          Lower = better.  Use the elbow to select N_TOPICS.
       b. UMass coherence: average log co-occurrence of top-word pairs.
          Higher (less negative) = more coherent topics.  Computed from
          the binary DTM — no gensim dependency required.
       c. FREX (topic exclusivity): fraction of top-K terms that are NOT
          among the top-K terms of any other topic.  High FREX =
          well-differentiated topics.
     Results are saved to lda_diagnostics table for inspection.

  3. Per-topic audience profile: for each topic, what fraction of its
     total weight comes from client pages vs worker pages?  Topics are
     classified as client_leaning (>65% client), worker_leaning (>65%
     worker), or shared.

  4. PCA on the document-topic matrix: reduces the N_TOPICS-dimensional
     topic space to 2–3 principal components for visualisation.  PC1
     typically separates client from worker pages.

     PCA quality checks:
       a. Scree data: explained variance ratios logged for inspection.
       b. Topic loadings: which topics drive PC1 and PC2.
       c. Audience separability test: logistic regression on PC1+PC2
          predicts audience using 5-fold cross-validation.  Accuracy
          above ~75% confirms register distinction is recoverable.
       d. Tail investigation: pages in the high-PC1 tail (top 5%)
          identified and logged by domain and topic.

  5. Hypothesis-stratified sampling for Step 2:
     For each hypothesis (H1a visibility, H1b automation, H1c
     hypervisibility + H2–H5b extensions), identifies the most
     topic-relevant LDA topics (by term overlap with HYPOTHESIS_TERMS
     vocabulary), then samples top pages per audience within those
     topics.  Pages are scored by:
       combined = topic_weight × (1 + collocate_divergence) × (1 + hyp_density × 10)
     Domains below MIN_PAGES_PER_DOMAIN are excluded from sampling
     candidates but remain in the LDA model.

Part B — STM Export:

  6. Exports the corpus to CSV files for Structural Topic Modelling in
     RStudio (stm package):
       output/step_1/stm/corpus_export.csv    — tokenised text, one row per page
       output/step_1/stm/metadata_export.csv  — covariates only
       output/step_1/stm/export_summary.txt   — human-readable data description

Output tables written to data/scraping.db:
  topic_terms            : top N_TOP_TERMS per topic + weights + rank
  document_topics        : per-page dominant topic, weight, PCA coords
  topic_audience_profile : per-topic B2B/B2W balance and category
  step2_sample           : ranked page_ids for Step 2 close reading
  lda_diagnostics        : per-n_topics quality metrics (RUN_DIAGNOSTICS mode)

Configuration to tune:
  N_TOPICS              — start with 30–50; increase until topics are coherent
  MAX_ITER              — increase to 100 for final runs (slow but better model)
  RUN_DIAGNOSTICS       — set True ONCE for multi-topic evaluation; then False
  HYPOTHESIS_TERMS      — review and update if analytical focus shifts
  MIN_PAGES_PER_DOMAIN  — minimum pages for a domain to appear in sampling

Usage:
    python3 src2/03_step1_topics.py
"""

import csv
import json
import logging
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH      = "data/scraping_2.db"
OUTPUT_DIR   = Path("STMAnalysis/output/step_1/stm")

# ── LDA hyperparameters ──────────────────────────────────────────────────
N_TOPICS       = 40       # LDA topics to extract (tune between 30–50)
N_TOP_TERMS    = 20       # top terms per topic stored in topic_terms
N_PCA_DIMS     = 3        # PCA components to keep (3 stored, 2 plotted)
MAX_ITER       = 50       # LDA iterations (increase to 100 for final run)
RANDOM_STATE   = 42       # fixed seed for reproducibility
MIN_DF         = 5        # minimum document frequency for CountVectorizer
MAX_DF_FRAC    = 0.85     # terms in >85% of docs are removed
MIN_TOKEN_COUNT = 30      # pages with fewer tokens are excluded from LDA

# ── Domain quality / PCA filter ──────────────────────────────────────────
# Applies ONLY to Step 2 sampling; all pages still enter LDA.
MIN_PAGES_PER_DOMAIN = 5

# Domains excluded from PCA fitting (kept in LDA).
# www.sama.com, mindy-support.com, scale.com have template-heavy content
# that dominates PC1 and collapses it into a platform-identity axis rather
# than the audience-register axis we want to visualise.
PCA_EXCLUDE_DOMAINS = {}

# ── LDA diagnostic mode ───────────────────────────────────────────────────
# Set True ONCE to run multi-topic evaluation (~30–60 min).
# Set False for all subsequent analysis runs.
RUN_DIAGNOSTICS           = False
DIAGNOSTIC_N_TOPICS_RANGE = [20, 30, 40, 50]
FREX_TOP_K                = 10    # top K terms per topic for FREX calculation

# Shared-topic threshold: a topic is "shared" if neither audience accounts
# for more than SHARED_THRESHOLD of total topic weight.
SHARED_THRESHOLD = 0.65

# ── STM export config ─────────────────────────────────────────────────────
STM_MIN_TOKEN_COUNT = 30  # minimum tokens per page after exclusion filtering

# Artifact / noise terms that dominated a calendar-noise LDA topic (T5:
# task/february/june).  Also includes UI strings from B2W page scrapers.
# These are merged with DB-loaded excluded_terms before LDA vectorisation
# AND before STM export — so they cannot appear in any topic.
EXTRA_STOP_WORDS = {
    # Calendar noise (job posting dates embedded in page text)
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # UI / navigational noise (cookie banners, footer elements)
    "cookie", "faq", "subscribe", "website", "youtube",
    # Foreign language fragments from non-English pages that slipped through
    "de", "en", "esp", "la", "du", "và", "có", "viêc",
}

# ── Hypothesis-stratified sampling ───────────────────────────────────────
# HYPOTHESIS_TERMS defines the analytical vocabulary for each hypothesis.
# Used to:
#   (a) Identify which LDA topics are most relevant to each hypothesis.
#   (b) Score candidate pages by hypothesis term density.
#
# H1a — Labour visibility gap:
#   Workers systematically rendered invisible in B2B communications.
# H1b — Automation myth:
#   B2B frames AI outputs as autonomous to obscure human labour.
# H1c — Strategic hypervisibility:
#   Human labour foregrounded in B2B as a quality/accuracy signal.
# H2  — Flexibilised control paradox (autonomy rhetoric vs algorithmic control)
# H3  — Resource framing / entrepreneurial subjectivity
# H4  — Alienation from others / community vocabulary
# H5  — Algorithmic coloniality (geographic stratification)
HYPOTHESIS_TERMS = {
    "H1a_visibility": {
        "terms": {
            "worker", "labour", "task", "job", "pay", "earn", "payment",
            "work", "annotator", "labeller", "moderator", "freelance",
            "gig", "contractor", "wage", "income", "employment",
        },
        "description": "Labour visibility — explicit references to human labour",
        "n_topics": 3,
        "n_pages_per_topic_per_audience": 5,
    },
    "H1b_automation": {
        "terms": {
            "autonomous", "machine", "automate", "intelligent", "automation",
            "model", "algorithm", "pipeline", "scalable", "engine",
            "deploy", "inference", "prediction", "neural", "llm",
        },
        "description": "Automation myth — framing as autonomous/intelligent systems",
        "n_topics": 3,
        "n_pages_per_topic_per_audience": 5,
    },
    "H1c_hypervisibility": {
        "terms": {
            "human", "quality", "oversight", "annotation", "label",
            "datum", "accuracy", "review", "expert", "curate",
            "human-in-the-loop", "verification", "audit", "check",
        },
        "description": "Strategic hypervisibility — human labour as quality feature",
        "n_topics": 3,
        "n_pages_per_topic_per_audience": 5,
    },
    "H2_flexibility": {
        "terms": {
            "flexible", "freedom", "schedule", "own", "choose", "anywhere",
            "remote", "anytime", "control", "independent", "autonomy",
            "manage", "availability", "track", "monitor", "score", "rate",
        },
        "description": "Flexibilised control — autonomy rhetoric vs algorithmic monitoring",
        "n_topics": 2,
        "n_pages_per_topic_per_audience": 4,
    },
    "H3_resource": {
        "terms": {
            "earn", "income", "opportunity", "grow", "career", "skill",
            "invest", "talent", "community", "platform", "marketplace",
            "passive", "side", "hustle", "gig", "freelance", "entrepreneur",
        },
        "description": "Resource framing / entrepreneurial subjectivity",
        "n_topics": 2,
        "n_pages_per_topic_per_audience": 4,
    },
    "H4_alienation": {
        "terms": {
            "community", "connect", "collaborate", "team", "together",
            "support", "member", "belong", "social", "network", "peer",
            "alone", "isolated", "individual", "solo",
        },
        "description": "Alienation from others — community framing in B2W",
        "n_topics": 2,
        "n_pages_per_topic_per_audience": 4,
    },
    "H5_coloniality": {
        "terms": {
            "global", "africa", "asia", "latin", "develop", "emerging",
            "local", "region", "impact", "source", "diversity", "inclusion",
            "market", "geographic", "country", "north", "south",
        },
        "description": "Algorithmic coloniality — geographic/regional framing",
        "n_topics": 2,
        "n_pages_per_topic_per_audience": 4,
    },
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy ML imports
# ---------------------------------------------------------------------------

def _import_ml():
    """
    Import scikit-learn and numpy lazily with a helpful error message.

    These are heavy dependencies not needed by other pipeline scripts,
    so they are imported only when this script runs.

    Returns:
        Tuple of (LatentDirichletAllocation, PCA, CountVectorizer,
                  LogisticRegression, cross_val_score, numpy)
    """
    try:
        from sklearn.decomposition import LatentDirichletAllocation, PCA
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np
        return (LatentDirichletAllocation, PCA, CountVectorizer,
                LogisticRegression, cross_val_score, np)
    except ImportError as e:
        log.error(
            "This script requires scikit-learn and numpy.\n"
            "  pip install scikit-learn numpy --break-system-packages"
        )
        raise SystemExit(1) from e


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_output_tables(conn: sqlite3.Connection):
    """
    Create output tables, dropping previous versions for clean re-runs.

    topic_terms            : vocabulary of each LDA topic
    document_topics        : per-page topic assignment + PCA coordinates
    topic_audience_profile : B2B vs B2W balance per topic
    step2_sample           : sampling table guiding Step 2 close reading
    lda_diagnostics        : multi-run quality metrics (RUN_DIAGNOSTICS only)

    lda_diagnostics persists across runs (only dropped if RUN_DIAGNOSTICS
    is True) so diagnostic results from an expensive run are not lost.
    """
    conn.executescript("""
        DROP TABLE IF EXISTS topic_terms;
        DROP TABLE IF EXISTS document_topics;
        DROP TABLE IF EXISTS topic_audience_profile;
        DROP TABLE IF EXISTS step2_sample;

        CREATE TABLE topic_terms (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id    INTEGER NOT NULL,
            topic_label TEXT,                 -- filled manually after inspection
            term        TEXT    NOT NULL,
            weight      REAL    NOT NULL,     -- from lda.components_
            rank        INTEGER NOT NULL,     -- 1 = highest weight term
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE document_topics (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id        INTEGER NOT NULL,
            domain         TEXT,
            audience       TEXT,
            dominant_topic INTEGER NOT NULL,  -- argmax of topic weight vector
            topic_weight   REAL    NOT NULL,  -- weight of dominant topic [0,1]
            topic_vector   TEXT,              -- full N-dim topic dist as JSON
            pca_1          REAL,              -- PC1 coordinate (NULL if PCA-excluded)
            pca_2          REAL,
            pca_3          REAL,
            created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE topic_audience_profile (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id          INTEGER NOT NULL,
            topic_label       TEXT,
            avg_weight_client REAL,       -- mean topic weight across client pages
            avg_weight_worker REAL,       -- mean topic weight across worker pages
            client_share      REAL,       -- client_sum / (client_sum + worker_sum)
            category          TEXT,       -- client_leaning | worker_leaning | shared
            n_dominant_client INTEGER,    -- pages where this is dominant (client)
            n_dominant_worker INTEGER,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE step2_sample (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id              INTEGER NOT NULL,
            url                  TEXT,
            domain               TEXT,
            audience             TEXT,
            dominant_topic       INTEGER,
            topic_weight         REAL,
            sampling_reason      TEXT,     -- hypothesis + topic + overlap terms
            collocate_divergence REAL,     -- 1 - cosine(PMI_B2B, PMI_B2W)
            priority_rank        INTEGER,  -- 1 = most analytically interesting
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_dt_page   ON document_topics(page_id);
        CREATE INDEX IF NOT EXISTS idx_dt_topic  ON document_topics(dominant_topic);
        CREATE INDEX IF NOT EXISTS idx_tap_cat   ON topic_audience_profile(category);
        CREATE INDEX IF NOT EXISTS idx_s2_rank   ON step2_sample(priority_rank);
    """)
    conn.commit()

    if RUN_DIAGNOSTICS:
        conn.executescript("""
            DROP TABLE IF EXISTS lda_diagnostics;
            CREATE TABLE lda_diagnostics (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                n_topics            INTEGER NOT NULL,
                perplexity          REAL,    -- lower is better
                log_likelihood      REAL,    -- higher is better
                avg_umass_coherence REAL,    -- higher (less negative) is better
                avg_frex            REAL,    -- [0,1] higher = better separated topics
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        log.info("lda_diagnostics table (re)created for diagnostic run.")
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lda_diagnostics (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                n_topics            INTEGER NOT NULL,
                perplexity          REAL,
                log_likelihood      REAL,
                avg_umass_coherence REAL,
                avg_frex            REAL,
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    log.info("Output tables created.")


# ---------------------------------------------------------------------------
# Exclusion helpers (shared between LDA and STM export)
# ---------------------------------------------------------------------------

def load_exclusions(conn: sqlite3.Connection) -> tuple:
    """
    Load excluded page IDs and terms; merge EXTRA_STOP_WORDS.

    Returns:
        excluded_pages : set of int page_ids to skip entirely
        excluded_terms : set of str terms to filter from token lists
                         AND pass as stop_words to CountVectorizer
    """
    excluded_pages = set()
    excluded_terms = set()

    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='excluded_pages'").fetchone():
        excluded_pages = {
            r[0] for r in conn.execute(
                "SELECT page_id FROM excluded_pages"
            ).fetchall()
        }
        log.info(f"  Loaded {len(excluded_pages)} excluded pages.")

    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='excluded_terms'").fetchone():
        excluded_terms = {
            r[0] for r in conn.execute(
                "SELECT term FROM excluded_terms"
            ).fetchall()
        }
        log.info(f"  Loaded {len(excluded_terms)} excluded terms "
                 f"(+ {len(EXTRA_STOP_WORDS)} EXTRA_STOP_WORDS).")

    excluded_terms = excluded_terms | EXTRA_STOP_WORDS
    return excluded_pages, excluded_terms


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(conn: sqlite3.Connection) -> tuple:
    """
    Load the corpus for LDA modelling.

    Uses only unigrams for LDA (bigrams make topic interpretation harder;
    LDA already learns co-occurrence patterns).  The segments column
    (added by src2/00_preprocess.py) is present in corpus_view but is NOT
    used here — LDA works on bag-of-words, not sentence sequences.

    Returns:
        docs              : list of str — space-joined unigram sequences
        metadata          : list of dict — {page_id, url, domain, audience,
                            token_count}
        excluded_terms    : set of str — passed to CountVectorizer
        domain_page_counts: dict {domain: n_pages}
    """
    log.info("Loading corpus from corpus_view...")
    excluded_pages, excluded_terms = load_exclusions(conn)

    rows = conn.execute(f"""
        SELECT page_id, url, audience, domain, unigrams, token_count
        FROM corpus_view
        WHERE audience IN ('client', 'worker')
          AND token_count >= {MIN_TOKEN_COUNT}
    """).fetchall()

    docs               = []
    metadata           = []
    skipped            = 0
    domain_page_counts = Counter()

    for row in rows:
        if row["page_id"] in excluded_pages:
            skipped += 1
            continue

        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        if not unigrams:
            continue

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
        domain_page_counts[row["domain"]] += 1

    sparse = {d for d, n in domain_page_counts.items() if n < MIN_PAGES_PER_DOMAIN}
    log.info(f"  {len(docs)} pages loaded ({skipped} excluded, "
             f"{len(excluded_terms)} terms filtered, "
             f"{len(sparse)} sparse domains excluded from sampling).")

    return docs, metadata, excluded_terms, domain_page_counts


# ---------------------------------------------------------------------------
# LDA: fit model
# ---------------------------------------------------------------------------

def fit_lda(docs, np, CountVectorizer, LatentDirichletAllocation,
            excluded_terms=None, n_topics=None):
    """
    Vectorise corpus and fit LDA topic model.

    Uses:
      - CountVectorizer with MIN_DF / MAX_DF_FRAC thresholds.
        token_pattern=r"(?u)\\S+" accepts underscore-joined compound forms.
      - LatentDirichletAllocation with batch learning and n_jobs=-1.

    Returns:
        lda, vectoriser, dtm, doc_topic_matrix, vocab
    """
    if n_topics is None:
        n_topics = N_TOPICS

    log.info("Vectorising corpus...")
    stop_words = list(excluded_terms) if excluded_terms else None
    vectoriser = CountVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF_FRAC,
        stop_words=stop_words,
        token_pattern=r"(?u)\S+",   # tokens already clean; don't split on _
    )
    dtm   = vectoriser.fit_transform(docs)
    vocab = vectoriser.get_feature_names_out()
    log.info(f"  DTM: {dtm.shape[0]} docs × {dtm.shape[1]} terms")

    log.info(f"Fitting LDA with {n_topics} topics (max_iter={MAX_ITER})...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        learning_method="batch",
        n_jobs=-1,
    )
    doc_topic_matrix = lda.fit_transform(dtm)

    log.info(f"  Perplexity    : {lda.perplexity(dtm):.2f}  (lower = better)")
    log.info(f"  Log-likelihood: {lda.score(dtm):.2f}  (higher = better)")

    return lda, vectoriser, dtm, doc_topic_matrix, vocab


# ---------------------------------------------------------------------------
# LDA: quality diagnostics
# ---------------------------------------------------------------------------

def compute_umass_coherence(dtm, lda_components, vocab, np, n_top=15) -> float:
    """
    Compute mean UMass coherence across all topics.

    UMass coherence (Mimno et al., 2011) measures whether a topic's top
    words tend to co-occur in the same documents.  Computed entirely from
    the training corpus — no external reference required.

    Formula per topic: (2/M(M-1)) × Σ_{i>j} log((D(wi,wj)+1) / D(wj))

    Returns: float ≤ 0.  Typical range −10 to −1; higher is better.
    """
    bin_dtm = (dtm > 0).astype(np.float32)

    topic_coherences = []
    for topic_vec in lda_components:
        top_idx  = topic_vec.argsort()[::-1][:n_top]
        top_cols = np.asarray(bin_dtm[:, top_idx].todense())
        doc_freq = top_cols.sum(axis=0)

        total = 0.0
        count = 0
        for i in range(n_top):
            for j in range(i):
                d_wij = float(top_cols[:, i].dot(top_cols[:, j]))
                d_wj  = float(doc_freq[j])
                if d_wj > 0:
                    total += math.log((d_wij + 1.0) / d_wj)
                    count += 1

        if count > 0:
            topic_coherences.append(total / count)

    return float(np.mean(topic_coherences)) if topic_coherences else float("nan")


def compute_topic_frex(lda_components, np, k=None) -> float:
    """
    Compute mean FREX (top-K exclusivity) across all topics.

    For each topic: fraction of top-K terms NOT in any other topic's top-K.
    1.0 = perfectly separated; 0.0 = all terms shared.
    Typical range for a good 40-topic model: 0.5–0.8.
    """
    if k is None:
        k = FREX_TOP_K
    n_topics = lda_components.shape[0]
    top_sets = [
        set(lda_components[t].argsort()[::-1][:k])
        for t in range(n_topics)
    ]
    frex_scores = []
    for t, top_t in enumerate(top_sets):
        others = set()
        for t2, top_t2 in enumerate(top_sets):
            if t2 != t:
                others |= top_t2
        frex_scores.append(len(top_t - others) / k)
    return float(np.mean(frex_scores))


def run_lda_diagnostics(docs, np, CountVectorizer, LatentDirichletAllocation,
                        excluded_terms, conn):
    """
    Fit LDA for each n_topics in DIAGNOSTIC_N_TOPICS_RANGE and save metrics.

    Expensive (~30–60 min).  Gated behind RUN_DIAGNOSTICS = True.
    Inspect lda_diagnostics table afterwards, then set N_TOPICS to the
    elbow where perplexity stops improving and coherence is still acceptable.
    """
    log.info("=" * 60)
    log.info(f"LDA DIAGNOSTICS — testing topic counts: {DIAGNOSTIC_N_TOPICS_RANGE}")
    log.info("  This may take 30–60 min.  Set RUN_DIAGNOSTICS=False afterwards.")
    log.info("=" * 60)

    rows = []
    for n_topics in DIAGNOSTIC_N_TOPICS_RANGE:
        log.info(f"  Fitting n_topics={n_topics}...")
        lda, _, dtm, _, vocab = fit_lda(
            docs, np, CountVectorizer, LatentDirichletAllocation,
            excluded_terms=excluded_terms, n_topics=n_topics
        )
        coherence = compute_umass_coherence(dtm, lda.components_, vocab, np)
        frex      = compute_topic_frex(lda.components_, np)
        rows.append({
            "n_topics":            n_topics,
            "perplexity":          round(lda.perplexity(dtm), 4),
            "log_likelihood":      round(lda.score(dtm), 4),
            "avg_umass_coherence": round(coherence, 6),
            "avg_frex":            round(frex, 6),
        })
        log.info(f"    n_topics={n_topics:>3}  "
                 f"perplexity={rows[-1]['perplexity']:>10.2f}  "
                 f"coherence={coherence:>8.4f}  frex={frex:.4f}")

    conn.executemany("""
        INSERT INTO lda_diagnostics
            (n_topics, perplexity, log_likelihood, avg_umass_coherence, avg_frex)
        VALUES (:n_topics, :perplexity, :log_likelihood, :avg_umass_coherence, :avg_frex)
    """, rows)
    conn.commit()
    log.info("Diagnostics saved. SQL: SELECT * FROM lda_diagnostics ORDER BY n_topics;")


# ---------------------------------------------------------------------------
# LDA: extract topic terms
# ---------------------------------------------------------------------------

def extract_topic_terms(lda, vocab, np) -> list:
    """
    Extract top N_TOP_TERMS per topic from lda.components_.

    topic_label is left as None — fill manually after inspecting topics
    (use SQL: UPDATE topic_terms SET topic_label='...' WHERE topic_id=N).
    """
    results = []
    for topic_id, component in enumerate(lda.components_):
        top_indices = component.argsort()[::-1][:N_TOP_TERMS]
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                "topic_id":    topic_id,
                "topic_label": None,
                "term":        vocab[idx],
                "weight":      round(float(component[idx]), 6),
                "rank":        rank,
            })
    return results


# ---------------------------------------------------------------------------
# LDA: document-topic assignments + PCA
# ---------------------------------------------------------------------------

def compute_document_topics(doc_topic_matrix, metadata, np, PCA) -> tuple:
    """
    Assign dominant topic to each page and compute PCA coordinates.

    PCA is fitted only on pages NOT in PCA_EXCLUDE_DOMAINS to avoid
    outlier domains dominating early principal components.  Excluded pages
    still receive NULL pca_1/2/3 so they remain in all downstream outputs.

    Returns:
        doc_topic_rows : list of dicts aligned with metadata
        pca_model      : fitted PCA object (for quality checks downstream)
    """
    pca_keep = [m["domain"] not in PCA_EXCLUDE_DOMAINS for m in metadata]
    pca_idx  = [i for i, keep in enumerate(pca_keep) if keep]
    n_excl   = len(metadata) - len(pca_idx)

    if n_excl:
        excl_domains = {metadata[i]["domain"]
                        for i in range(len(metadata)) if not pca_keep[i]}
        log.info(f"  PCA domain filter: {n_excl} pages excluded {excl_domains}")
        log.info(f"  PCA fitted on {len(pca_idx)} pages "
                 f"({len(pca_idx)/len(metadata)*100:.1f}% of corpus).")

    pca_matrix = doc_topic_matrix[pca_idx]
    pca = PCA(n_components=N_PCA_DIMS, random_state=RANDOM_STATE)
    pca_sub = pca.fit_transform(pca_matrix)

    # Put coordinates back into a full-size array; excluded rows stay NaN.
    pca_coords = np.full((len(metadata), N_PCA_DIMS), np.nan)
    for new_i, orig_i in enumerate(pca_idx):
        pca_coords[orig_i] = pca_sub[new_i]

    # Scree data
    explained = pca.explained_variance_ratio_
    log.info("  Explained variance (scree):")
    for i, v in enumerate(explained):
        log.info(f"    PC{i+1} = {v:.4f}  ({v*100:.1f}%)")
    log.info(f"  Total ({N_PCA_DIMS} PCs): {sum(explained):.4f}")

    # Topic loadings on PC1 and PC2
    log.info("  Topic loadings on PC1 (top 5):")
    for tid, load in sorted(enumerate(pca.components_[0]),
                            key=lambda x: abs(x[1]), reverse=True)[:5]:
        log.info(f"    Topic {tid:>3}: {load:+.4f}")
    if N_PCA_DIMS >= 2:
        log.info("  Topic loadings on PC2 (top 5):")
        for tid, load in sorted(enumerate(pca.components_[1]),
                                key=lambda x: abs(x[1]), reverse=True)[:5]:
            log.info(f"    Topic {tid:>3}: {load:+.4f}")

    def _f(v, dim):
        val = pca_coords[v, dim]
        return None if np.isnan(val) else round(float(val), 6)

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
            "pca_1":          _f(i, 0),
            "pca_2":          _f(i, 1) if N_PCA_DIMS >= 2 else None,
            "pca_3":          _f(i, 2) if N_PCA_DIMS >= 3 else None,
        })

    return results, pca


# ---------------------------------------------------------------------------
# PCA: quality checks
# ---------------------------------------------------------------------------

def run_pca_audience_test(pca_coords, metadata, np,
                          LogisticRegression, cross_val_score) -> float:
    """
    Test audience separability in PCA space via logistic regression (5-fold CV).

    If PC1+PC2 encode audience register, accuracy should be well above
    chance (~50%).  Above ~75% = independent statistical validation that
    audience is a dominant structural axis — the core Step 1 claim.

    Returns: mean cross-validation accuracy (0–1).
    """
    log.info("  PCA audience separability test (logistic regression, 5-fold CV)...")
    X = pca_coords[:, :2]
    y = np.array([1 if m["audience"] == "client" else 0 for m in metadata])

    if len(set(y)) < 2:
        log.warning("  Only one audience class — cannot run separability test.")
        return float("nan")

    clf    = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    mean_acc = float(scores.mean())
    std_acc  = float(scores.std())

    log.info(f"    Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    log.info(f"    Fold scores: {[round(s, 3) for s in scores]}")

    if mean_acc >= 0.80:
        log.info("    ✓ Strong separability (≥80%): audience register is recoverable from PC1+PC2.")
    elif mean_acc >= 0.65:
        log.info("    ~ Moderate separability (65–80%).")
    else:
        log.info("    ✗ Weak separability (<65%). Investigate topic structure or PCA loadings.")

    return mean_acc


def log_pca_tail(pca_coords, metadata, doc_topic_matrix, np,
                 tail_pct=0.05) -> list:
    """
    Identify and log pages in the high-PC1 tail.

    The tail often reveals outlier domains with concentrated vocabulary
    that dominate PC1.  Useful for detecting whether the axis is driven
    by genuine register differences or platform-specific content.

    Returns: list of dicts for tail pages {page_id, url, domain, audience,
             pca_1, dominant_topic}.
    """
    pc1_vals  = pca_coords[:, 0]
    threshold = np.percentile(pc1_vals, (1 - tail_pct) * 100)
    tail_mask = pc1_vals > threshold

    log.info(f"  PC1 tail ({tail_pct*100:.0f}%, threshold={threshold:.4f}): "
             f"{tail_mask.sum()} pages")

    tail_pages = []
    for i in np.where(tail_mask)[0]:
        meta = metadata[i]
        tail_pages.append({
            "page_id":        meta["page_id"],
            "url":            meta["url"],
            "domain":         meta["domain"],
            "audience":       meta["audience"],
            "pca_1":          round(float(pc1_vals[i]), 4),
            "dominant_topic": int(doc_topic_matrix[i].argmax()),
        })
    tail_pages.sort(key=lambda x: x["pca_1"], reverse=True)

    tail_domains  = Counter(p["domain"]   for p in tail_pages)
    tail_audience = Counter(p["audience"] for p in tail_pages)
    tail_topics   = Counter(p["dominant_topic"] for p in tail_pages)
    log.info(f"    Audience: {dict(tail_audience)}")
    log.info(f"    Top domains: {tail_domains.most_common(10)}")
    log.info(f"    Top topics:  {tail_topics.most_common(5)}")

    return tail_pages


# ---------------------------------------------------------------------------
# Topic audience profiles
# ---------------------------------------------------------------------------

def compute_topic_profiles(doc_topic_matrix, metadata, np) -> list:
    """
    Compute per-topic B2B vs B2W balance.

    For each topic:
      avg_weight_client : mean weight across client pages
      avg_weight_worker : mean weight across worker pages
      client_share      : client_sum / (client_sum + worker_sum)
      category          : client_leaning | worker_leaning | shared
      n_dominant_*      : pages where this is the dominant topic
    """
    client_mask   = np.array([m["audience"] == "client" for m in metadata])
    worker_mask   = np.array([m["audience"] == "worker" for m in metadata])
    client_matrix = doc_topic_matrix[client_mask]
    worker_matrix = doc_topic_matrix[worker_mask]

    client_dominant = (client_matrix.argmax(axis=1)
                       if client_matrix.shape[0] > 0 else [])
    worker_dominant = (worker_matrix.argmax(axis=1)
                       if worker_matrix.shape[0] > 0 else [])

    results = []
    n_topics = doc_topic_matrix.shape[1]
    for t in range(n_topics):
        avg_c  = float(client_matrix[:, t].mean()) if client_matrix.shape[0] > 0 else 0
        avg_w  = float(worker_matrix[:, t].mean()) if worker_matrix.shape[0] > 0 else 0
        total  = avg_c + avg_w
        c_share = avg_c / total if total > 0 else 0.5

        if c_share > SHARED_THRESHOLD:
            cat = "client_leaning"
        elif c_share < (1 - SHARED_THRESHOLD):
            cat = "worker_leaning"
        else:
            cat = "shared"

        results.append({
            "topic_id":           t,
            "topic_label":        None,
            "avg_weight_client":  round(avg_c, 6),
            "avg_weight_worker":  round(avg_w, 6),
            "client_share":       round(c_share, 4),
            "category":           cat,
            "n_dominant_client":  int(sum(1 for d in client_dominant if d == t)),
            "n_dominant_worker":  int(sum(1 for d in worker_dominant if d == t)),
        })
    return results


# ---------------------------------------------------------------------------
# Collocate divergence (from 02_step1_analysis.py co-occurrence data)
# ---------------------------------------------------------------------------

def compute_collocate_divergence(conn: sqlite3.Connection) -> dict:
    """
    Score how differently each focus term is framed in B2B vs B2W.

    divergence = 1 - cosine(PMI_client_profile, PMI_worker_profile)
    0 = identical collocate profiles; 1 = completely different framing.

    Only uses cross_platform comparison (all client vs all worker).

    Returns: dict {focus_term: divergence_score}; empty if table missing.
    """
    log.info("Computing collocate divergence from cooccurrence_results...")

    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='cooccurrence_results'"
    ).fetchone():
        log.warning("  cooccurrence_results not found — run 02_step1_analysis.py first.")
        return {}

    rows = conn.execute("""
        SELECT focus_term, audience, collocate, pmi
        FROM cooccurrence_results
        WHERE comparison = 'cross_platform'
    """).fetchall()

    if not rows:
        log.warning("  No cross_platform co-occurrence data found.")
        return {}

    profiles = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        profiles[r["focus_term"]][r["audience"]][r["collocate"]] = r["pmi"]

    divergences = {}
    for term, aud_data in profiles.items():
        if "client" not in aud_data or "worker" not in aud_data:
            continue
        c_vec = aud_data["client"]
        w_vec = aud_data["worker"]
        all_c = set(c_vec) | set(w_vec)
        if len(all_c) < 3:
            continue

        dot   = sum(c_vec.get(c, 0) * w_vec.get(c, 0) for c in all_c)
        mag_c = math.sqrt(sum(v ** 2 for v in c_vec.values()))
        mag_w = math.sqrt(sum(v ** 2 for v in w_vec.values()))
        cos_sim = (dot / (mag_c * mag_w)) if (mag_c and mag_w) else 0.0
        divergences[term] = round(1.0 - cos_sim, 6)

    log.info(f"  Divergence scores for {len(divergences)} focus terms.")
    by_div = sorted(divergences.items(), key=lambda x: x[1], reverse=True)
    log.info("  Most divergent (same word, different collocate profile):")
    for term, div in by_div[:15]:
        log.info(f"    {term:<25} divergence={div:.4f}")

    return divergences


# ---------------------------------------------------------------------------
# Step 2 sampling
# ---------------------------------------------------------------------------

def compute_topic_hypothesis_relevance(topic_terms_list: list) -> dict:
    """
    Score each LDA topic by overlap with each hypothesis vocabulary.

    Returns:
        {hyp_key: [(topic_id, overlap_score, matching_terms), ...]}
        Sorted by overlap_score descending.
    """
    topic_top = defaultdict(set)
    for r in topic_terms_list:
        if r["rank"] <= 15:
            topic_top[r["topic_id"]].add(r["term"])

    result = {}
    for hyp_key, hyp_config in HYPOTHESIS_TERMS.items():
        scored = []
        for topic_id, top_terms in topic_top.items():
            overlap = top_terms & hyp_config["terms"]
            if overlap:
                scored.append((topic_id, len(overlap), overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        result[hyp_key] = scored
    return result


def build_step2_sample(doc_topics: list, topic_profiles: list,
                       topic_terms_list: list, divergences: dict,
                       metadata: list, domain_page_counts: dict,
                       conn: sqlite3.Connection) -> list:
    """
    Build hypothesis-stratified Step 2 sampling table.

    Domain quality filter: domains with <MIN_PAGES_PER_DOMAIN pages are
    excluded from sampling candidates (but remain in LDA).

    Selection: for each hypothesis, identify n_topics most relevant LDA
    topics; sample top n_pages_per_topic_per_audience pages per (topic,
    audience) by combined score:
        combined = topic_weight × (1 + avg_divergence) × (1 + hyp_density × 10)

    Deduplication: a page already selected by one hypothesis is not
    re-selected by another (avoids over-representing boundary pages).

    Returns: list of dicts with page_id, url, domain, audience,
             dominant_topic, topic_weight, sampling_reason,
             collocate_divergence, priority_rank.
    """
    log.info("Building Step 2 sampling table...")

    sparse_domains = {
        d for d, n in domain_page_counts.items() if n < MIN_PAGES_PER_DOMAIN
    }
    if sparse_domains:
        log.info(f"  Excluding {len(sparse_domains)} sparse domains: {sorted(sparse_domains)}")

    relevance = compute_topic_hypothesis_relevance(topic_terms_list)
    for hyp_key, scored_topics in relevance.items():
        desc = HYPOTHESIS_TERMS[hyp_key]["description"]
        log.info(f"  {hyp_key} — {desc}:")
        for topic_id, score, terms in scored_topics[:4]:
            cat = next(
                (p["category"] for p in topic_profiles if p["topic_id"] == topic_id), "?"
            )
            log.info(f"    Topic {topic_id} [{cat}] overlap={score} "
                     f"terms: {', '.join(sorted(terms))}")

    meta_by_page = {m["page_id"]: m for m in metadata}
    by_topic = defaultdict(list)
    n_excl   = 0
    for dt in doc_topics:
        if dt["domain"] in sparse_domains:
            n_excl += 1
            continue
        by_topic[dt["dominant_topic"]].append(dt)
    if n_excl:
        log.info(f"  {n_excl} pages excluded by sparse domain filter.")

    # Cache page tokens for hypothesis density computation
    page_terms_cache = {}

    def get_page_terms(page_id):
        if page_id not in page_terms_cache:
            row = conn.execute(
                "SELECT unigrams FROM corpus_view WHERE page_id = ?", (page_id,)
            ).fetchone()
            page_terms_cache[page_id] = (
                set(json.loads(row["unigrams"])) if row and row["unigrams"] else set()
            )
        return page_terms_cache[page_id]

    def score_page(dt, hyp_terms):
        page_id  = dt["page_id"]
        terms    = get_page_terms(page_id)
        if divergences:
            matching_divs = [divergences[t] for t in terms if t in divergences]
            avg_div = (sum(matching_divs) / len(matching_divs)) if matching_divs else 0
        else:
            avg_div = 0
        hyp_count   = len(terms & hyp_terms)
        hyp_density = hyp_count / len(terms) if terms else 0
        combined    = dt["topic_weight"] * (1 + avg_div) * (1 + hyp_density * 10)
        return {
            "page_id":              page_id,
            "url":                  meta_by_page.get(page_id, {}).get("url", ""),
            "domain":               dt["domain"],
            "audience":             dt["audience"],
            "dominant_topic":       dt["dominant_topic"],
            "topic_weight":         dt["topic_weight"],
            "collocate_divergence": round(avg_div, 6),
            "combined_score":       combined,
        }

    results    = []
    seen_pages = set()

    for hyp_key, scored_topics in relevance.items():
        hyp_config = HYPOTHESIS_TERMS[hyp_key]
        hyp_terms  = hyp_config["terms"]
        n_top_t    = hyp_config["n_topics"]
        n_per      = hyp_config["n_pages_per_topic_per_audience"]

        selected_topics = scored_topics[:n_top_t]
        if not selected_topics:
            log.warning(f"  No relevant topics for {hyp_key} — skipping.")
            continue

        for topic_id, overlap_score, matching_terms in selected_topics:
            candidates = by_topic.get(topic_id, [])
            if not candidates:
                continue
            scored = sorted(
                [score_page(dt, hyp_terms) for dt in candidates],
                key=lambda x: x["combined_score"],
                reverse=True,
            )
            for aud in ("client", "worker"):
                aud_cands = [
                    s for s in scored
                    if s["audience"] == aud and s["page_id"] not in seen_pages
                ]
                for s in aud_cands[:n_per]:
                    seen_pages.add(s["page_id"])
                    results.append({
                        "page_id":              s["page_id"],
                        "url":                  s["url"],
                        "domain":               s["domain"],
                        "audience":             s["audience"],
                        "dominant_topic":       s["dominant_topic"],
                        "topic_weight":         s["topic_weight"],
                        "sampling_reason":      (
                            f"{hyp_key}_topic_{topic_id}"
                            f"_overlap={overlap_score}"
                            f"_terms={','.join(sorted(matching_terms))}"
                        ),
                        "collocate_divergence": s["collocate_divergence"],
                        "priority_rank":        0,
                    })

    # Global ranking by combined analytical interest
    results.sort(
        key=lambda x: x["topic_weight"] * (1 + x["collocate_divergence"]),
        reverse=True,
    )
    for i, r in enumerate(results):
        r["priority_rank"] = i + 1

    hyp_counts = Counter(r["sampling_reason"].split("_topic_")[0] for r in results)
    aud_counts = Counter(r["audience"] for r in results)
    log.info(f"  {len(results)} pages selected for Step 2:")
    log.info(f"    By hypothesis: {dict(hyp_counts)}")
    log.info(f"    By audience:   {dict(aud_counts)}")

    return results


# ---------------------------------------------------------------------------
# Save LDA results to DB
# ---------------------------------------------------------------------------

def save_lda_results(conn, topic_terms, doc_topics, topic_profiles, sample):
    """Insert all LDA output tables in a single transaction."""
    conn.executemany("""
        INSERT INTO topic_terms (topic_id, topic_label, term, weight, rank)
        VALUES (:topic_id, :topic_label, :term, :weight, :rank)
    """, topic_terms)

    conn.executemany("""
        INSERT INTO document_topics
            (page_id, domain, audience, dominant_topic, topic_weight,
             topic_vector, pca_1, pca_2, pca_3)
        VALUES
            (:page_id, :domain, :audience, :dominant_topic, :topic_weight,
             :topic_vector, :pca_1, :pca_2, :pca_3)
    """, doc_topics)

    conn.executemany("""
        INSERT INTO topic_audience_profile
            (topic_id, topic_label, avg_weight_client, avg_weight_worker,
             client_share, category, n_dominant_client, n_dominant_worker)
        VALUES
            (:topic_id, :topic_label, :avg_weight_client, :avg_weight_worker,
             :client_share, :category, :n_dominant_client, :n_dominant_worker)
    """, topic_profiles)

    conn.executemany("""
        INSERT INTO step2_sample
            (page_id, url, domain, audience, dominant_topic, topic_weight,
             sampling_reason, collocate_divergence, priority_rank)
        VALUES
            (:page_id, :url, :domain, :audience, :dominant_topic, :topic_weight,
             :sampling_reason, :collocate_divergence, :priority_rank)
    """, sample)

    conn.commit()


# ===========================================================================
# Part B — STM Export
# ===========================================================================

def export_stm(conn: sqlite3.Connection) -> dict:
    """
    Export corpus to CSV files for STM in RStudio.

    Applies three exclusion layers:
      1. excluded_pages table (manual flagging)
      2. excluded_terms table + EXTRA_STOP_WORDS (vocabulary noise)
      3. STM_MIN_TOKEN_COUNT filter (pages too short after term removal)

    Writes:
      output/step_1/stm/corpus_export.csv   — page_id, audience, domain, tokens
      output/step_1/stm/metadata_export.csv — page_id, audience, domain,
                                              company_id, platform_type, hq_region

    Returns stats dict for summary writing.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    excluded_pages, excluded_terms = load_exclusions(conn)

    rows = conn.execute("""
        SELECT  cv.page_id,
                cv.audience,
                cv.domain,
                cv.company_id,
                cv.platform_type,
                cv.hq_region,
                cv.unigrams,
                cv.token_count
        FROM    corpus_view cv
        WHERE   cv.audience IN ('client', 'worker', 'both')
          AND   cv.token_count >= ?
        ORDER   BY cv.page_id
    """, (STM_MIN_TOKEN_COUNT,)).fetchall()

    log.info(f"  STM export: {len(rows)} rows from corpus_view (pre-filter)")

    corpus_path = OUTPUT_DIR / "corpus_export.csv"
    meta_path   = OUTPUT_DIR / "metadata_export.csv"

    stats = {
        "total_rows": len(rows), "skipped_excluded": 0, "skipped_short": 0,
        "written": 0, "client": 0, "worker": 0, "both": 0,
        "domains": set(), "total_tokens": 0,
    }

    with open(corpus_path, "w", newline="", encoding="utf-8") as f_corp, \
         open(meta_path,   "w", newline="", encoding="utf-8") as f_meta:

        corp = csv.writer(f_corp, quoting=csv.QUOTE_ALL)
        meta = csv.writer(f_meta, quoting=csv.QUOTE_ALL)
        corp.writerow(["page_id", "audience", "domain", "tokens"])
        meta.writerow(["page_id", "audience", "domain",
                       "company_id", "platform_type", "hq_region"])

        for row in rows:
            pid = row["page_id"]
            if pid in excluded_pages:
                stats["skipped_excluded"] += 1
                continue

            tokens = json.loads(row["unigrams"]) if row["unigrams"] else []
            tokens = [t for t in tokens if t not in excluded_terms]

            if len(tokens) < STM_MIN_TOKEN_COUNT:
                stats["skipped_short"] += 1
                continue

            aud    = row["audience"]
            domain = row["domain"]
            corp.writerow([pid, aud, domain, " ".join(tokens)])
            meta.writerow([pid, aud, domain,
                           row["company_id"]    or "",
                           row["platform_type"] or "",
                           row["hq_region"]     or ""])

            stats["written"]      += 1
            stats[aud]            += 1
            stats["domains"].add(domain)
            stats["total_tokens"] += len(tokens)

    return stats


def write_stm_summary(stats: dict, excluded_terms: set):
    """Write a plain-text summary file for reading in RStudio."""
    summary_path = OUTPUT_DIR / "export_summary.txt"
    lines = [
        "DarkSideofAI — STM Export Summary",
        "=" * 50,
        "",
        "FILES",
        f"  corpus_export.csv     {stats['written']} pages",
        f"  metadata_export.csv   {stats['written']} pages",
        "",
        "CORPUS STATISTICS",
        f"  Total pages written   : {stats['written']}",
        f"  Client-facing (B2B)   : {stats['client']}  "
        f"({stats['client']/max(stats['written'],1)*100:.1f}%)",
        f"  Worker-facing (B2W)   : {stats['worker']}  "
        f"({stats['worker']/max(stats['written'],1)*100:.1f}%)",
        f"  Both audiences        : {stats['both']}  "
        f"({stats['both']/max(stats['written'],1)*100:.1f}%)",
        f"  Unique domains        : {len(stats['domains'])}",
        f"  Total tokens          : {stats['total_tokens']:,}",
        f"  Mean tokens per page  : {stats['total_tokens']//max(stats['written'],1):,}",
        "",
        "EXCLUSION FILTERS",
        f"  Manually excluded pages          : {stats['skipped_excluded']}",
        f"  Too short after term removal     : {stats['skipped_short']}",
        f"  Total exclusion vocabulary       : {len(excluded_terms)} terms",
        "",
        "TOKENISATION STATUS",
        "  Tokens are lemmatised and lowercased (spaCy, en_core_web_lg).",
        "  English stopwords already removed (spaCy default list).",
        "  Month names and UI noise removed by EXTRA_STOP_WORDS filter.",
        "  Co-occurrence computed within sentence boundaries (src2 fix).",
        "  Do NOT re-stem, re-lemmatise, or re-apply stopwords in R.",
        "",
        "AUDIENCE LABELS",
        "  'client' = B2B platform (addresses businesses buying AI services)",
        "  'worker' = B2W platform (addresses human annotators seeking work)",
        "  'both'   = platform addresses both audiences.",
        "  Labels from platforms config table — NOT URL-matched.",
        "",
        "HANDLING 'BOTH' PAGES IN R",
        "  Option A — Exclude (recommended for clean binary contrast):",
        "    corpus <- corpus[corpus$audience != 'both', ]",
        "    meta   <- meta[meta$audience   != 'both', ]",
        "  Option B — Keep as third factor level (robustness check):",
        "    meta$audience <- factor(meta$audience,",
        "                           levels = c('client','both','worker'))",
        "",
        "LOADING IN R",
        "    library(stm)",
        "    corpus <- read.csv('output/step_1/stm/corpus_export.csv')",
        "    meta   <- read.csv('output/step_1/stm/metadata_export.csv')",
        "    meta$audience <- factor(meta$audience, levels = c('client','worker'))",
        "    processed <- textProcessor(",
        "        corpus$tokens, metadata = meta,",
        "        lowercase = FALSE, removestopwords = FALSE,",
        "        removenumbers = FALSE, removepunctuation = FALSE, stem = FALSE,",
        "        wordLengths = c(2, Inf))",
        "    out <- prepDocuments(processed$documents, processed$vocab,",
        "                         processed$meta, lower.thresh = 5)",
        "    stm_model <- stm(",
        "        documents  = out$documents, vocab = out$vocab,",
        "        K          = 35,",
        "        prevalence = ~ audience, content = ~ audience,",
        "        data       = out$meta, init.type = 'Spectral', seed = 42)",
        "",
        "PAIRED DOMAINS (company_id covariate)",
        "  appen.com    <-> crowdgen.com     (company_id = appen)",
        "  scale.com    <-> remotasks.com    (company_id = scale)",
        "  toloka.ai    <-> mindrift.ai      (company_id = toloka)",
        "  centific.com <-> oneforma.com     (company_id = centific)",
        "  labelbox.com <-> alignerr.com     (company_id = labelbox)",
        "",
        "ADDITIONAL COVARIATES",
        "  company_id    — links paired domains",
        "  platform_type — crowd_market | enterprise_bpo | impact_sourcing",
        "  hq_region     — north | south  (Global North / Global South HQ)",
        "",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  STM summary → {summary_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    """
    Orchestrate topic modelling and STM export.

    Part A — LDA:
      0. [Optional] LDA diagnostics (RUN_DIAGNOSTICS = True)
      1. Fit LDA (N_TOPICS)
      2. UMass coherence + FREX for chosen model
      3. Extract topic terms
      4. Document-topic assignments + PCA
      5. PCA quality checks (separability test, PC1 tail)
      6. Topic audience profiles
      7. Collocate divergence (from 02_step1_analysis.py)
      8. Step 2 hypothesis-stratified sampling
      9. Save all results to DB

    Part B — STM Export:
      10. Export corpus to CSV for RStudio STM analysis
      11. Write export_summary.txt
    """
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("03_step1_topics.py — Topic Modelling, Sampling & STM Export")
    log.info("=" * 60)

    # Verify prerequisites
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    view_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()
    if not view_check:
        log.error("corpus_view not found — run 01_prepare_corpus.py first.")
        sys.exit(1)

    # Verify segments column present (ensures src2/00_preprocess.py was run)
    seg_check = conn.execute("PRAGMA table_info(pages_tfidf)").fetchall()
    seg_cols  = {row["name"] for row in seg_check}
    if "segments" not in seg_cols:
        log.error(
            "pages_tfidf.segments column missing.\n"
            "  Run python3 src2/00_preprocess.py first (not src/preprocess.py)."
        )
        sys.exit(1)

    # Import ML dependencies
    (LatentDirichletAllocation, PCA, CountVectorizer,
     LogisticRegression, cross_val_score, np) = _import_ml()

    init_output_tables(conn)

    # ── Load corpus ──────────────────────────────────────────────────────
    docs, metadata, excluded_terms, domain_page_counts = load_corpus(conn)

    # ── [Optional] LDA diagnostics ───────────────────────────────────────
    if RUN_DIAGNOSTICS:
        run_lda_diagnostics(
            docs, np, CountVectorizer, LatentDirichletAllocation,
            excluded_terms, conn
        )
        log.info("Diagnostics complete.")
        log.info("  Inspect: SELECT * FROM lda_diagnostics ORDER BY n_topics;")
        log.info("  Then set RUN_DIAGNOSTICS = False, choose N_TOPICS, and re-run.")
        conn.close()
        return

    # ── Part A: LDA ───────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info(f"LDA — N_TOPICS={N_TOPICS}  MAX_ITER={MAX_ITER}")
    log.info("-" * 60)

    lda, vectoriser, dtm, doc_topic_matrix, vocab = fit_lda(
        docs, np, CountVectorizer, LatentDirichletAllocation,
        excluded_terms=excluded_terms
    )

    # Single-model quality check
    coherence = compute_umass_coherence(dtm, lda.components_, vocab, np)
    frex      = compute_topic_frex(lda.components_, np)
    log.info(f"  UMass coherence : {coherence:.4f}  (higher = better topics)")
    log.info(f"  Mean FREX       : {frex:.4f}  (higher = better separation)")

    # Topic terms
    topic_terms = extract_topic_terms(lda, vocab, np)
    log.info("-" * 60)
    log.info("TOPIC TERMS (top 8 per topic):")
    for t in range(N_TOPICS):
        terms = [r["term"] for r in topic_terms if r["topic_id"] == t and r["rank"] <= 8]
        log.info(f"  Topic {t:>2}: {', '.join(terms)}")

    # Document-topic assignments + PCA
    log.info("-" * 60)
    log.info("DOCUMENT-TOPIC ASSIGNMENTS + PCA")
    log.info("-" * 60)
    doc_topics, pca_model = compute_document_topics(
        doc_topic_matrix, metadata, np, PCA
    )

    # Build full PCA coordinate array for quality checks
    pca_coords_full = np.array([
        [r["pca_1"] if r["pca_1"] is not None else np.nan,
         r["pca_2"] if r["pca_2"] is not None else np.nan,
         r["pca_3"] if r["pca_3"] is not None else 0.0]
        for r in doc_topics
    ])
    valid_mask          = ~np.isnan(pca_coords_full[:, 0])
    pca_coords          = pca_coords_full[valid_mask]
    metadata_pca        = [m for m, v in zip(metadata, valid_mask) if v]
    doc_topic_matrix_pca = doc_topic_matrix[valid_mask]

    log.info(f"  PCA quality checks on {int(valid_mask.sum())} pages "
             f"({len(metadata) - int(valid_mask.sum())} PCA-excluded).")

    # PCA quality checks
    log.info("-" * 60)
    log.info("PCA QUALITY CHECKS")
    log.info("-" * 60)
    sep_accuracy = run_pca_audience_test(
        pca_coords, metadata_pca, np, LogisticRegression, cross_val_score
    )
    tail_pages = log_pca_tail(
        pca_coords, metadata_pca, doc_topic_matrix_pca, np
    )

    # Topic audience profiles
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
                 f"c={p['avg_weight_client']:.4f}  w={p['avg_weight_worker']:.4f}  "
                 f"share={p['client_share']:.2f}  {', '.join(terms)}")

    # Collocate divergence
    log.info("-" * 60)
    log.info("COLLOCATE DIVERGENCE")
    log.info("-" * 60)
    divergences = compute_collocate_divergence(conn)

    # Step 2 sampling
    log.info("-" * 60)
    log.info("STEP 2 SAMPLING")
    log.info("-" * 60)
    sample = build_step2_sample(
        doc_topics, topic_profiles, topic_terms,
        divergences, metadata, domain_page_counts, conn
    )
    if sample:
        log.info("  Top 10 pages for Step 2 close reading:")
        for s in sample[:10]:
            log.info(f"    rank={s['priority_rank']:>3}  page={s['page_id']:<6}  "
                     f"{s['audience']:<8}  topic={s['dominant_topic']:<3}  "
                     f"tw={s['topic_weight']:.3f}  div={s['collocate_divergence']:.3f}  "
                     f"{s['domain']}")

    # Save to DB
    log.info("-" * 60)
    log.info("Saving LDA results to database...")
    save_lda_results(conn, topic_terms, doc_topics, topic_profiles, sample)
    log.info(f"  topic_terms            : {len(topic_terms):,}")
    log.info(f"  document_topics        : {len(doc_topics):,}")
    log.info(f"  topic_audience_profile : {len(topic_profiles)}")
    log.info(f"  step2_sample           : {len(sample):,}")

    # ── Part B: STM Export ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STM EXPORT")
    log.info("=" * 60)
    _, exc_terms_for_summary = load_exclusions(conn)
    stm_stats = export_stm(conn)
    write_stm_summary(stm_stats, exc_terms_for_summary)

    log.info(f"  Pages written    : {stm_stats['written']}")
    log.info(f"  Client (B2B)     : {stm_stats['client']}")
    log.info(f"  Worker (B2W)     : {stm_stats['worker']}")
    log.info(f"  Both             : {stm_stats['both']}")
    log.info(f"  Domains          : {len(stm_stats['domains'])}")
    log.info(f"  Total tokens     : {stm_stats['total_tokens']:,}")
    log.info(f"  Output dir       : {OUTPUT_DIR.resolve()}")

    conn.close()

    # ── Final summary ─────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("COMPLETE — summary:")
    log.info(f"  LDA model        : {N_TOPICS} topics, {MAX_ITER} iterations")
    log.info(f"  UMass coherence  : {coherence:.4f}")
    log.info(f"  FREX             : {frex:.4f}")
    log.info(f"  PCA separability : {sep_accuracy:.3f} accuracy (5-fold CV)")
    log.info(f"  PC1 tail pages   : {len(tail_pages)}")
    log.info(f"  Step 2 sample    : {len(sample)} pages")
    log.info(f"  STM export       : {stm_stats['written']} pages → {OUTPUT_DIR}")
    log.info("")
    log.info("Useful queries:")
    log.info("  -- Shared topics:")
    log.info("  SELECT topic_id, client_share FROM topic_audience_profile")
    log.info("  WHERE category = 'shared' ORDER BY client_share;")
    log.info("")
    log.info("  -- Top terms for a topic:")
    log.info("  SELECT term, weight FROM topic_terms")
    log.info("  WHERE topic_id = 5 ORDER BY rank LIMIT 20;")
    log.info("")
    log.info("  -- Step 2 sample:")
    log.info("  SELECT priority_rank, page_id, audience, domain,")
    log.info("         dominant_topic, collocate_divergence, sampling_reason")
    log.info("  FROM step2_sample ORDER BY priority_rank LIMIT 30;")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
