"""
02c_step1_topics.py
===================
Nelson (2020) Step 1 — extension: Topic Modelling & Step 2 Sampling

Pipeline position:
  Stage 2c — Topic Modelling and Sampling Strategy (run after
  02_step1_frequency.py; can run in parallel with 02b)
  Prerequisites:
    01_prepare.py           (corpus_view)
    01_prepare_additions.py (excluded pages / terms)
    02_step1_frequency.py   (cooccurrence_results for divergence scoring)
  Next step:
    03b_visualise_distinctiveness_topics.py (figures 9, 10, 11, 12)
    Step 2 close reading (the step2_sample table produced here guides
    which pages to read)

What this script does:
  Bridges Step 1 (computational pattern detection) to Step 2 (interpretive
  close reading) by:

  1. LDA topic model (sklearn LatentDirichletAllocation) on the full
     corpus.  Extracts N_TOPICS latent topics, each characterised by a
     probability distribution over the vocabulary.  N_TOPICS = 40 was
     chosen after testing 20/30/40/50 and inspecting topic coherence and
     separation (see RUN_DIAGNOSTICS mode below).

  2. LDA quality diagnostics (when RUN_DIAGNOSTICS = True):
     Fits LDA for each topic count in DIAGNOSTIC_N_TOPICS_RANGE and
     computes three complementary quality metrics:
       a. Perplexity (sklearn .perplexity()): measures how well the model
          predicts held-out data.  Lower = better.  Use the elbow in
          the perplexity curve to select N_TOPICS.
       b. UMass coherence: for each topic's top words [w1..wM], average
          log((D(wi,wj)+1) / D(wj)) over all ordered pairs.  D(wj) =
          number of documents containing wj; D(wi,wj) = number containing
          both.  Higher (less negative) = more coherent topics.  Computed
          from the binary DTM — no gensim dependency required.
       c. FREX (topic exclusivity): fraction of each topic's top-K terms
          that are NOT among the top-K terms of any other topic.  High
          FREX = well-differentiated, non-overlapping topics.
     All results are saved to lda_diagnostics table for inspection.

  3. Per-topic audience profile: for each topic, what fraction of its
     total weight comes from client pages vs worker pages?  Topics are
     classified as client_leaning (>65% client), worker_leaning (>65%
     worker), or shared.  The threshold SHARED_THRESHOLD = 0.65 is
     configurable.

  4. PCA on the document-topic matrix: reduces the 40-dimensional topic
     space to 2–3 principal components for visualisation.  PC1 typically
     separates client from worker pages, providing visual evidence that
     audience is a dominant structural axis of the corpus.

     PCA quality checks:
       a. Scree data: explained variance ratios for all N_PCA_DIMS
          components, logged for inspection.  An L-shaped scree plot
          (PC1 captures much more than PC2) indicates that one dimension
          dominates — analytically important to investigate rather than
          treat as noise.
       b. Topic loadings: which topics drive PC1 and PC2?  High loadings
          reveal the semantic content of each principal axis.
       c. Audience separability test: logistic regression on PC1+PC2
          predicts audience (client/worker) using 5-fold cross-validation.
          Accuracy above ~75% confirms that audience register is
          recoverable from the topic-reduced representation alone —
          independent statistical validation of the register distinction.
       d. Tail investigation: pages in the high-PC1 tail (top 5%) are
          identified and logged by domain and topic.  Useful for detecting
          domain-specific outliers (e.g. autonomous vehicles sector) that
          drive the PC1 axis.

  5. Hypothesis-stratified sampling for Step 2:
     For each hypothesis (H1a visibility, H1b automation, H1c
     hypervisibility), identifies the 2–3 most topic-relevant LDA topics
     (by term overlap with HYPOTHESIS_TERMS vocabulary), then samples the
     top pages per audience within those topics.  Pages are scored by:
       combined = topic_weight × (1 + collocate_divergence) × (1 + hyp_density × 10)
     where:
       topic_weight         = LDA weight of the dominant topic for the page
       collocate_divergence = how differently the page's key terms are
                              framed in B2B vs B2W (from 02 co-occurrence)
       hyp_density          = fraction of page tokens that are hypothesis-
                              relevant terms
     Domains with fewer than MIN_PAGES_PER_DOMAIN scraped pages are excluded
     from sampling candidates: their vocabulary is too sparse to be reliable
     platform-level representatives.
     This produces ~50–60 pages with explicit theoretical justification
     for every selection (stored in step2_sample.sampling_reason).

Input (from data/scraping.db):
  corpus_view          — unigrams, audience, domain, token_count
  excluded_pages       — pages to skip
  excluded_terms       — terms to filter out AND pass as stop_words
                         to sklearn CountVectorizer
  cooccurrence_results — from 02_step1_frequency.py; used to compute
                         per-page collocate divergence scores
  domain_quality       — (optional) from 02b_step1_distinctiveness.py;
                         if present, domain page-count filter uses its
                         n_pages column instead of recomputing

Output tables written to data/scraping.db:
  topic_terms            : top N_TOP_TERMS per topic + weights + rank
  document_topics        : per-page dominant topic, weight, PCA coords
  topic_audience_profile : per-topic B2B/B2W balance and category
  step2_sample           : ranked page_ids for Step 2 close reading
  lda_diagnostics        : (when RUN_DIAGNOSTICS=True) per-n_topics
                           perplexity, coherence and FREX metrics

Output used by:
  03b_visualise_distinctiveness_topics.py
    fig9_pca_scatter         — PCA scatter coloured by audience
    fig10_topic_audience_profile — per-topic B2B/B2W bar chart
    fig11_collocate_divergence   — divergence ranking chart
    fig12_step2_sample_map       — PCA with sampled pages highlighted
  Thesis Step 2 methodology: step2_sample drives the close-reading corpus
  exported by 04_step2_export.py

Configuration to tune:
  N_TOPICS              — start with 30-50; increase until topics are coherent
  MAX_ITER              — increase to 100 for final runs (slow but better model)
  RUN_DIAGNOSTICS       — set True ONCE to run multi-topic evaluation;
                          set False for normal production runs (saves time)
  DIAGNOSTIC_N_TOPICS_RANGE — topic counts to test in diagnostic mode
  HYPOTHESIS_TERMS      — review and update if analytical focus shifts
  MIN_PAGES_PER_DOMAIN  — minimum scraped pages for a domain to appear in
                          Step 2 sampling candidates (default 5)

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
N_TOP_TERMS    = 20         # top terms per topic to store in topic_terms
N_PCA_DIMS     = 3          # PCA components to keep (3 stored, 2 plotted)
MAX_ITER       = 50         # LDA iterations (increase to 100 for final run)
RANDOM_STATE   = 42         # fixed seed for reproducibility
MIN_DF         = 5          # minimum document frequency for CountVectorizer
MAX_DF_FRAC    = 0.85       # maximum document fraction for CountVectorizer
                             # (removes terms in >85% of documents — likely
                             # function words not caught by stop_words)
MIN_TOKEN_COUNT  = 30       # minimum tokens for a page to enter the model
                             # (short pages produce noisy topic assignments)

# ---------------------------------------------------------------------------
# Domain quality filter
# ---------------------------------------------------------------------------
# Domains with fewer scraped pages than this threshold are excluded from
# Step 2 sampling candidates.  Their vocabulary distribution is unreliable:
# 1–4 pages cannot represent a platform's communicative register.
# This mirrors the filter in 02b_step1_distinctiveness.py.
# Note: the filter applies ONLY to sampling — all pages still enter LDA,
# since pooled modelling is robust to sparse individual domains.
MIN_PAGES_PER_DOMAIN = 5

# ---------------------------------------------------------------------------
# PCA domain filter
# ---------------------------------------------------------------------------
# Domains listed here are EXCLUDED from PCA fitting but kept in LDA.
# Why separate the two? LDA is a pooled bag-of-words model: 300 Sama pages
# among 5,916 contribute normally to topic estimation.  PCA operates on
# the document-topic matrix and is sensitive to outlier domains whose
# content is far from the corpus mean — they dominate early principal
# components and collapse the axis into a platform-identity dimension
# rather than the audience-register dimension we want to visualise.
#
# www.sama.com: template-heavy service pages (description/duration/session/
# store vocabulary captured by Topic 6) dominate PC1 with loading +0.98.
# All 296 pages in the top-5% PC1 tail are from this one domain.
# Excluding them lets PC1 recover the B2B/B2W register axis.
#
# Add any domain string here to exclude it from PCA (and from the PCA
# quality checks that follow it: audience separability test, tail log).
# The domain_quality table in 02b records the page counts for reference.
PCA_EXCLUDE_DOMAINS = {"www.sama.com", "mindy-support.com", "scale.com"}

# ---------------------------------------------------------------------------
# LDA diagnostic mode
# ---------------------------------------------------------------------------
# Set True ONCE to run the multi-topic evaluation (expensive: ~30–60 min).
# Results are saved to lda_diagnostics and inspected manually.
# Set False for all subsequent analysis runs.
RUN_DIAGNOSTICS          = False
DIAGNOSTIC_N_TOPICS_RANGE = [20, 30, 40, 50]
FREX_TOP_K               = 10   # top K terms per topic for FREX calculation

# Shared-topic threshold: a topic is "shared" if neither audience accounts
# for more than this fraction of the topic's total weight.
# 0.65 means: if client_share is between 0.35 and 0.65, the topic is shared.
# Lower values = more topics classified as shared.
SHARED_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# Hypothesis-stratified sampling config
# ---------------------------------------------------------------------------
# HYPOTHESIS_TERMS defines the analytical vocabulary for each of the three
# core hypotheses.  These sets are used to:
#   (a) Identify which LDA topics are most relevant to each hypothesis
#       (by term overlap with topic_terms).
#   (b) Score candidate pages by hypothesis term density.
#
# H1a — Labour visibility gap:
#   Workers in AI data labour are systematically rendered invisible in B2B
#   communications.  Labour vocabulary (worker, task, pay) should appear
#   less frequently or be framed more abstractly in B2B texts.
#
# H1b — Automation myth:
#   B2B communications frame AI outputs as autonomous / intelligent to
#   obscure the human labour behind them.  Automation vocabulary
#   (autonomous, pipeline, deploy) should be B2B-distinctive.
#
# H1c — Strategic hypervisibility:
#   When human labour is mentioned in B2B texts, it is foregrounded as a
#   quality/accuracy signal (human-in-the-loop, expert reviewers).  This
#   is strategically visible — human labour as product feature, not worker.
HYPOTHESIS_TERMS = {
    "H1a_visibility": {
        "terms": {"worker", "labour", "task", "job", "pay", "earn", "payment",
                  "work", "annotator", "labeller", "moderator", "freelance",
                  "gig", "contractor", "wage", "income", "employment"},
        "description": "Labour visibility — explicit references to human labour",
        "n_topics": 3,       # top N topics per hypothesis for sampling
        "n_pages_per_topic_per_audience": 5,  # pages sampled per topic per audience
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

# ---------------------------------------------------------------------------
# Artifact / noise terms to exclude from LDA vocabulary.
# These are added to excluded_terms before vectorisation so they cannot
# contribute to any topic.  Add month names, platform-specific noise,
# and any other terms identified as artifacts during topic inspection.
# ---------------------------------------------------------------------------
EXTRA_STOP_WORDS = {
    # Calendar noise (from job posting timestamps)
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # Other known artifacts (extend as needed after topic inspection)
    "cookie", "faq", "subscribe", "website", "youtube",
    "de", "en", "esp", "la", "du", "và", "có", "viêc"
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
    """
    Import scikit-learn and numpy with a helpful error message if missing.

    These are heavy dependencies not required by other pipeline scripts,
    so they are imported lazily (only when this script runs) rather than
    at module level.

    Returns:
        Tuple of (LatentDirichletAllocation, PCA, CountVectorizer,
                  LogisticRegression, cross_val_score, numpy)

    Raises:
        SystemExit if scikit-learn or numpy is not installed.
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
            "  pip install scikit-learn numpy"
        )
        raise SystemExit(1) from e


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_output_tables(conn: sqlite3.Connection):
    """
    Create output tables, dropping previous versions for clean re-runs.

    Tables created:
      topic_terms            : vocabulary of each LDA topic
      document_topics        : per-page topic assignment + PCA coordinates
      topic_audience_profile : B2B vs B2W balance per topic
      step2_sample           : sampling table guiding Step 2 close reading
      lda_diagnostics        : multi-run quality metrics (populated only
                               when RUN_DIAGNOSTICS = True)

    The lda_diagnostics table persists across runs (DROP only if
    RUN_DIAGNOSTICS is True) so diagnostic results from a single expensive
    run are not lost when re-running in normal mode.
    """
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
            weight      REAL NOT NULL,       -- term weight in topic (from LDA components_)
            rank        INTEGER NOT NULL,    -- 1 = highest weight term
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE document_topics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id         INTEGER NOT NULL,
            domain          TEXT,
            audience        TEXT,
            dominant_topic   INTEGER NOT NULL,  -- topic with highest weight for this page
            topic_weight    REAL NOT NULL,      -- weight of dominant topic [0,1]
            topic_vector    TEXT,               -- full 40-dim topic distribution as JSON
            pca_1           REAL,               -- PC1 coordinate
            pca_2           REAL,               -- PC2 coordinate
            pca_3           REAL,               -- PC3 coordinate
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
            n_dominant_client INTEGER,  -- pages where this is dominant topic (client)
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
            sampling_reason TEXT,        -- encodes hypothesis + topic + overlap terms
            collocate_divergence REAL,   -- divergence of key terms' PMI profiles
            priority_rank   INTEGER,     -- 1 = most analytically interesting
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_dt_page   ON document_topics(page_id);
        CREATE INDEX IF NOT EXISTS idx_dt_topic  ON document_topics(dominant_topic);
        CREATE INDEX IF NOT EXISTS idx_tap_cat   ON topic_audience_profile(category);
        CREATE INDEX IF NOT EXISTS idx_s2_rank   ON step2_sample(priority_rank);
    """)
    conn.commit()

    # lda_diagnostics persists across runs; only drop/recreate if diagnostics mode
    if RUN_DIAGNOSTICS:
        conn.executescript("""
            DROP TABLE IF EXISTS lda_diagnostics;
            CREATE TABLE lda_diagnostics (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                n_topics            INTEGER NOT NULL,
                perplexity          REAL,    -- lower = better
                log_likelihood      REAL,    -- higher = better
                avg_umass_coherence REAL,    -- higher (less negative) = better
                avg_frex            REAL,    -- [0,1] higher = better-separated topics
                created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        log.info("lda_diagnostics table (re)created for diagnostic run.")
    else:
        # Ensure the table exists but leave any prior data intact
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
# Step 1: Load corpus
# ---------------------------------------------------------------------------

def load_exclusions(conn: sqlite3.Connection) -> tuple:
    """
    Load excluded page IDs and terms from 01_prepare_additions tables.

    Gracefully returns empty sets if the tables do not exist yet.

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
    Load the corpus for LDA modelling.

    Returns documents as space-joined token strings (expected by sklearn
    CountVectorizer) along with per-page metadata.

    Unlike the frequency analysis in 02_step1_frequency.py, this function
    uses ONLY unigrams for LDA (bigrams are sparse and make topic
    interpretation harder; LDA already learns multi-word patterns from
    co-occurrence).

    Domain page counts are computed here so that downstream functions
    (notably build_step2_sample) can filter sparse domains from sampling
    candidates without re-querying the database.

    Args:
        conn: Open SQLite connection.

    Returns:
        docs             : list of str — space-joined unigram sequences
        metadata         : list of dict — {page_id, url, domain, audience,
                           token_count} aligned with docs
        excluded_terms   : set of str — passed to CountVectorizer stop_words
                           so excluded terms cannot contribute to LDA topics
        domain_page_counts: dict {domain: n_pages} — used by
                           build_step2_sample to exclude sparse domains
    """
    log.info("Loading corpus from corpus_view...")

    excluded_pages, excluded_terms = load_exclusions(conn)
    # Merge script-level artifact terms into the excluded set
    excluded_terms = excluded_terms | EXTRA_STOP_WORDS
    log.info(f"  EXTRA_STOP_WORDS: {len(EXTRA_STOP_WORDS)} additional terms merged into exclusions.")

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
        domain_page_counts[row["domain"]] += 1

    log.info(f"  {len(docs)} pages loaded (min {MIN_TOKEN_COUNT} tokens, "
             f"{skipped} excluded pages skipped, "
             f"{len(excluded_terms)} terms filtered).")

    sparse = {d for d, n in domain_page_counts.items() if n < MIN_PAGES_PER_DOMAIN}
    log.info(f"  Domain page counts: {len(domain_page_counts)} domains total, "
             f"{len(sparse)} sparse (<{MIN_PAGES_PER_DOMAIN} pages, will be excluded "
             f"from Step 2 sampling candidates).")
    if sparse:
        log.info(f"  Sparse domains: {sorted(sparse)}")

    return docs, metadata, excluded_terms, domain_page_counts


# ---------------------------------------------------------------------------
# Step 2: Fit LDA (single run)
# ---------------------------------------------------------------------------

def fit_lda(docs, np, CountVectorizer, LatentDirichletAllocation,
            excluded_terms=None, n_topics=None):
    """
    Vectorise corpus and fit LDA topic model.

    Two-stage process:
      1. CountVectorizer: converts docs (space-joined strings) to a
         document-term matrix (DTM).  MIN_DF=5 removes terms appearing
         in fewer than 5 documents; MAX_DF_FRAC=0.85 removes terms in
         more than 85% of documents.  excluded_terms are also passed as
         stop_words so they cannot contribute to any topic.
         token_pattern=r"(?u)\S+" accepts tokens with underscores and
         special characters (lemmatised compound forms).
      2. LDA: fits n_topics topics on the DTM using batch learning.
         n_jobs=-1 uses all CPU cores for parallel computation.

    After fitting, logs perplexity and log-likelihood for model evaluation.
    Lower perplexity = better model fit.  Compare across N_TOPICS values
    when tuning.

    Args:
        docs              : list of str from load_corpus().
        np                : numpy module (passed in to avoid re-import).
        CountVectorizer   : sklearn class.
        LatentDirichletAllocation: sklearn class.
        excluded_terms    : set of terms to pass as stop_words.
        n_topics          : number of topics (defaults to N_TOPICS global).

    Returns:
        lda              : fitted LDA model
        vectoriser       : fitted CountVectorizer
        dtm              : sparse DTM (n_docs × vocab_size) — used by
                           coherence computation
        doc_topic_matrix : ndarray shape (n_docs, n_topics)
        vocab            : ndarray of vocabulary terms
    """
    if n_topics is None:
        n_topics = N_TOPICS

    log.info("Vectorising corpus...")
    stop_words = list(excluded_terms) if excluded_terms else None
    vectoriser = CountVectorizer(
        min_df=MIN_DF,
        max_df=MAX_DF_FRAC,
        stop_words=stop_words,
        # tokens are already clean unigrams — no extra preprocessing
        token_pattern=r"(?u)\S+",
    )
    dtm   = vectoriser.fit_transform(docs)
    vocab = vectoriser.get_feature_names_out()
    log.info(f"  DTM shape: {dtm.shape[0]} docs × {dtm.shape[1]} terms")

    log.info(f"Fitting LDA with {n_topics} topics (max_iter={MAX_ITER})...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        learning_method="batch",
        n_jobs=-1,
    )
    doc_topic_matrix = lda.fit_transform(dtm)  # shape: (n_docs, n_topics)

    perplexity = lda.perplexity(dtm)
    ll         = lda.score(dtm)
    log.info(f"  LDA perplexity: {perplexity:.2f}")
    log.info(f"  Log-likelihood: {ll:.2f}")

    return lda, vectoriser, dtm, doc_topic_matrix, vocab


# ---------------------------------------------------------------------------
# LDA quality diagnostics
# ---------------------------------------------------------------------------

def compute_umass_coherence(dtm, lda_components, vocab, np,
                            n_top=15) -> float:
    """
    Compute mean UMass coherence across all topics.

    UMass coherence (Mimno et al., 2011) measures whether the top words
    in a topic tend to co-occur in the same documents.  Unlike word2vec
    or pointwise coherence measures, it is computed entirely from the
    training corpus and requires no external reference.

    Formula for a single topic with top words [w1..wM]:
      Coherence = (2 / M(M-1)) × Σ_{i>j} log( (D(wi,wj) + 1) / D(wj) )

    where:
      D(wj)    = number of documents containing term wj
      D(wi,wj) = number of documents containing both wi and wj
      +1 smoothing avoids log(0) when the pair never co-occurs

    The formula is computed from a binary version of the DTM (1 if term
    appears in document, 0 otherwise).  No gensim dependency is required.

    Interpretation:
      0.0  = perfect coherence (all top-word pairs always co-occur)
      −3.0 to −1.0 = typical range for meaningful topics
      < −10 = likely noise topics (words rarely co-occur)

    Args:
        dtm            : sparse DTM from CountVectorizer.fit_transform()
        lda_components : lda.components_ array (n_topics × vocab_size)
        vocab          : vocabulary array from CountVectorizer
        np             : numpy module
        n_top          : number of top words per topic to use (default 15)

    Returns:
        Mean UMass coherence across all topics (float, ≤ 0).
    """
    # Binary document-term matrix (1 if term present, 0 otherwise)
    # Using float32 for memory efficiency
    bin_dtm = (dtm > 0).astype(np.float32)
    n_docs  = bin_dtm.shape[0]

    topic_coherences = []

    for topic_vec in lda_components:
        # Top n_top term indices by topic weight
        top_idx = topic_vec.argsort()[::-1][:n_top]

        # Document frequency of each top term
        # bin_dtm is sparse; .toarray() slices only the needed columns
        top_cols = np.asarray(bin_dtm[:, top_idx].todense())  # (n_docs, n_top)
        doc_freq = top_cols.sum(axis=0)  # (n_top,) — D(wj)

        total = 0.0
        count = 0
        for i in range(n_top):
            for j in range(i):         # j < i; pair (wi, wj) with i > j
                # D(wi, wj) = dot product of binary column vectors
                d_wij = float(top_cols[:, i].dot(top_cols[:, j]))
                d_wj  = float(doc_freq[j])
                if d_wj > 0:
                    total += math.log((d_wij + 1.0) / d_wj)
                    count += 1

        if count > 0:
            topic_coherences.append(total / count)

    if not topic_coherences:
        return float("nan")

    mean_coh = float(np.mean(topic_coherences))
    return mean_coh


def compute_topic_frex(lda_components, np, k=None) -> float:
    """
    Compute mean FREX (Frequency + Exclusivity) across all topics.

    FREX as used here is a simplified exclusivity metric: for each topic,
    what fraction of its top-K terms appear ONLY in that topic's top-K
    list and in no other topic?  A high score means topics are well
    separated — each topic owns its most distinctive vocabulary.

    Full FREX (Roberts et al., 2014) combines frequency and exclusivity
    harmonically; this lighter version focuses on top-K exclusivity
    only, which is sufficient for model selection and methodological
    reporting.

    Interpretation:
      1.0 = all top-K terms are unique to their topic (ideal)
      0.0 = all top-K terms appear in every topic (useless)
      Typical range for a good 40-topic model: 0.5–0.8

    Args:
        lda_components : lda.components_ array (n_topics × vocab_size)
        np             : numpy module
        k              : top K terms per topic (defaults to FREX_TOP_K)

    Returns:
        Mean FREX score across all topics (float, 0–1).
    """
    if k is None:
        k = FREX_TOP_K

    n_topics = lda_components.shape[0]
    # Build set of top-K term indices per topic
    top_sets = [
        set(lda_components[t].argsort()[::-1][:k])
        for t in range(n_topics)
    ]

    frex_scores = []
    for t, top_t in enumerate(top_sets):
        # Union of all other topics' top-K terms
        others = set()
        for t2, top_t2 in enumerate(top_sets):
            if t2 != t:
                others |= top_t2
        # Fraction of this topic's terms NOT in any other topic's top-K
        exclusive = top_t - others
        frex_scores.append(len(exclusive) / k)

    return float(np.mean(frex_scores))


def run_lda_diagnostics(docs, np, CountVectorizer, LatentDirichletAllocation,
                         excluded_terms, conn):
    """
    Fit LDA for each topic count in DIAGNOSTIC_N_TOPICS_RANGE and save
    quality metrics to the lda_diagnostics table.

    This is an expensive operation (multiple full LDA fits).  It is gated
    behind RUN_DIAGNOSTICS = True and intended to be run once during model
    selection, not as part of routine analysis.

    Metrics saved per topic count:
      perplexity          : lower is better (model fit on training data)
      log_likelihood      : higher is better
      avg_umass_coherence : higher (less negative) = more coherent topics
      avg_frex            : higher = better-separated topics

    After running, inspect the lda_diagnostics table and choose N_TOPICS
    at the elbow point where perplexity stops improving substantially and
    coherence is still acceptable.

    Args:
        docs         : list of str from load_corpus().
        np           : numpy module.
        CountVectorizer, LatentDirichletAllocation : sklearn classes.
        excluded_terms : set of terms for stop_words.
        conn         : open SQLite connection.
    """
    log.info("=" * 60)
    log.info("LDA DIAGNOSTICS: running topic count search "
             f"{DIAGNOSTIC_N_TOPICS_RANGE}")
    log.info("  This may take 30–60 minutes. Set RUN_DIAGNOSTICS=False "
             "for subsequent runs.")
    log.info("=" * 60)

    rows = []
    for n_topics in DIAGNOSTIC_N_TOPICS_RANGE:
        log.info(f"  Fitting LDA with {n_topics} topics...")
        lda, vectoriser, dtm, _, vocab = fit_lda(
            docs, np, CountVectorizer, LatentDirichletAllocation,
            excluded_terms=excluded_terms, n_topics=n_topics
        )

        perplexity = lda.perplexity(dtm)
        ll         = lda.score(dtm)

        log.info(f"    Computing UMass coherence...")
        coherence  = compute_umass_coherence(dtm, lda.components_, vocab, np)
        log.info(f"    Computing FREX...")
        frex       = compute_topic_frex(lda.components_, np)

        rows.append({
            "n_topics":            n_topics,
            "perplexity":          round(perplexity, 4),
            "log_likelihood":      round(ll, 4),
            "avg_umass_coherence": round(coherence, 6),
            "avg_frex":            round(frex, 6),
        })

        log.info(f"    n_topics={n_topics:>3}  "
                 f"perplexity={perplexity:>10.2f}  "
                 f"coherence={coherence:>8.4f}  "
                 f"frex={frex:.4f}")

    conn.executemany("""
        INSERT INTO lda_diagnostics
            (n_topics, perplexity, log_likelihood, avg_umass_coherence, avg_frex)
        VALUES (:n_topics, :perplexity, :log_likelihood, :avg_umass_coherence, :avg_frex)
    """, rows)
    conn.commit()

    log.info("LDA diagnostics saved to lda_diagnostics table.")
    log.info("  Interpretation:")
    log.info("  - Choose N_TOPICS at the 'elbow' where perplexity stops")
    log.info("    improving substantially but coherence is still acceptable.")
    log.info("  - High FREX (>0.5) = topics are well separated.")
    log.info("  SQL to inspect: SELECT * FROM lda_diagnostics ORDER BY n_topics;")


# ---------------------------------------------------------------------------
# Step 3: Extract topic terms
# ---------------------------------------------------------------------------

def extract_topic_terms(lda, vocab, np) -> list:
    """
    Extract the top N_TOP_TERMS terms per topic.

    Uses lda.components_ (shape: n_topics × vocab_size) — each row is
    the unnormalised topic distribution over the vocabulary.  Higher
    component values = higher topic weight for that term.

    The topic_label column is left as None — it can be filled manually
    after inspecting the topic terms.  For the thesis, topic labels were
    assigned manually to the most analytically relevant topics.

    Args:
        lda   : fitted LDA model.
        vocab : vocabulary array from CountVectorizer.
        np    : numpy module.

    Returns:
        List of dicts with keys: topic_id, topic_label (None), term,
        weight, rank.  Length = N_TOPICS × N_TOP_TERMS.
    """
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

def compute_document_topics(doc_topic_matrix, metadata, np, PCA) -> list:
    """
    Compute per-page dominant topic and PCA coordinates, with diagnostics.

    For each page:
      dominant_topic = argmax of the topic weight vector
      topic_weight   = the weight of the dominant topic
      topic_vector   = full topic distribution stored as JSON (useful
                       for future dimensionality reduction experiments)
      pca_1/2/3      = coordinates in PCA-reduced topic space

    PCA is fitted on the document-topic matrix (n_docs × N_TOPICS) to
    project documents into a 2-3 dimensional space for visualisation.
    PC1 typically separates client from worker pages.

    Diagnostics logged:
      - Explained variance ratio per component (scree data)
      - Topic loadings on PC1 and PC2 (top contributing topics)

    Args:
        doc_topic_matrix : ndarray (n_docs, N_TOPICS) from LDA.
        metadata         : list of per-page dicts from load_corpus.
        np               : numpy module.
        PCA              : sklearn PCA class.

    Returns:
        Tuple: (doc_topic_rows, pca_model)
        doc_topic_rows : list of dicts with document-topic data aligned
                         with metadata.
        pca_model      : fitted PCA object (passed to run_pca_audience_test
                         and log_pca_tail).
    """
    # ---------------------------------------------------------------
    # PCA domain filter: exclude outlier domains before fitting.
    # Excluded pages still receive NULL pca_1/2/3 in document_topics
    # so they remain in all LDA outputs and in Step 2 sampling.
    # ---------------------------------------------------------------
    pca_keep   = [m["domain"] not in PCA_EXCLUDE_DOMAINS for m in metadata]
    pca_idx    = [i for i, keep in enumerate(pca_keep) if keep]
    n_excl     = len(metadata) - len(pca_idx)

    if n_excl:
        excl_domains = {metadata[i]["domain"]
                        for i in range(len(metadata)) if not pca_keep[i]}
        log.info(f"  PCA domain filter: excluding {n_excl} pages from "
                 f"{excl_domains}  (PCA_EXCLUDE_DOMAINS).")
        log.info(f"  PCA fitted on {len(pca_idx)} pages "
                 f"({len(pca_idx)/len(metadata)*100:.1f}% of corpus).")

    pca_matrix = doc_topic_matrix[pca_idx]   # shape: (n_eligible, N_TOPICS)

    log.info(f"Running PCA ({N_PCA_DIMS} components) on "
             f"{len(pca_idx)}-page filtered matrix...")
    pca = PCA(n_components=N_PCA_DIMS, random_state=RANDOM_STATE)
    pca_sub = pca.fit_transform(pca_matrix)   # shape: (n_eligible, N_PCA_DIMS)

    # Place coords back into a full-size array; excluded rows stay NaN.
    pca_coords = np.full((len(metadata), N_PCA_DIMS), np.nan)
    for new_i, orig_i in enumerate(pca_idx):
        pca_coords[orig_i] = pca_sub[new_i]

    # --- Scree data ---
    explained = pca.explained_variance_ratio_
    log.info("  Explained variance (scree data) — filtered corpus:")
    for i, v in enumerate(explained):
        log.info(f"    PC{i+1} = {v:.4f}  ({v*100:.1f}%)")
    log.info(f"  Total explained (first {N_PCA_DIMS} PCs): {sum(explained):.4f}")

    # --- Topic loadings on PC1 and PC2 ---
    log.info("  Topic loadings on PC1 (top 5 by absolute value):")
    pc1_loadings = sorted(enumerate(pca.components_[0]),
                          key=lambda x: abs(x[1]), reverse=True)
    for topic_id, loading in pc1_loadings[:5]:
        log.info(f"    Topic {topic_id:>3}: loading = {loading:+.4f}")

    if N_PCA_DIMS >= 2:
        log.info("  Topic loadings on PC2 (top 5 by absolute value):")
        pc2_loadings = sorted(enumerate(pca.components_[1]),
                              key=lambda x: abs(x[1]), reverse=True)
        for topic_id, loading in pc2_loadings[:5]:
            log.info(f"    Topic {topic_id:>3}: loading = {loading:+.4f}")

    # Helper: NaN → None for SQLite storage
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
# PCA quality checks
# ---------------------------------------------------------------------------

def run_pca_audience_test(pca_coords, metadata, np, LogisticRegression,
                          cross_val_score) -> float:
    """
    Test whether audience (client vs worker) is recoverable from PC1+PC2.

    Methodology:
      A logistic regression classifier is trained on the 2D PCA
      representation (PC1, PC2 only) and evaluated using 5-fold
      stratified cross-validation.  The mean accuracy is reported.

    Why this matters:
      If PC1+PC2 encode audience register, a simple linear classifier
      should achieve well above chance (~50%).  Accuracy above ~75%
      provides independent statistical validation that audience is a
      dominant structural axis of the corpus — the core claim of the
      Step 1 analysis.  This is methodologically stronger than relying
      solely on visual inspection of the scatter plot.

    The test uses only PC1 and PC2 (not PC3 or topic vectors) to avoid
    overfitting: 2 features × ~N pages is a very easy problem for
    logistic regression, so performance reflects genuine structure in
    the data rather than model capacity.

    Args:
        pca_coords       : ndarray (n_docs, N_PCA_DIMS) — full PCA matrix.
        metadata         : list of per-page dicts from load_corpus.
        np               : numpy module.
        LogisticRegression, cross_val_score : sklearn objects.

    Returns:
        Mean cross-validation accuracy (0–1).
    """
    log.info("  PCA audience separability test (logistic regression, 5-fold CV)...")

    # Features: PC1 and PC2 only
    X = pca_coords[:, :2]
    # Labels: 1 for client, 0 for worker
    y = np.array([1 if m["audience"] == "client" else 0 for m in metadata])

    if len(set(y)) < 2:
        log.warning("  Only one audience class — cannot run separability test.")
        return float("nan")

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    mean_acc = float(scores.mean())
    std_acc  = float(scores.std())

    log.info(f"    Mean accuracy = {mean_acc:.3f} ± {std_acc:.3f}")
    log.info(f"    Individual fold scores: {[round(s, 3) for s in scores]}")

    if mean_acc >= 0.80:
        log.info("    ✓ Strong separability: audience register is recoverable "
                 "from PC1+PC2 alone (≥80%).")
    elif mean_acc >= 0.65:
        log.info("    ~ Moderate separability: some audience structure in "
                 "PC1+PC2 (65–80%).")
    else:
        log.info("    ✗ Weak separability: audience is not well separated "
                 "by PC1+PC2 (<65%). Investigate topic structure or PCA loadings.")

    return mean_acc


def log_pca_tail(pca_coords, metadata, doc_topic_matrix, np,
                 tail_pct=0.05) -> list:
    """
    Identify and log pages in the high-PC1 tail.

    The PC1 axis often shows a long tail of pages with unusually high
    PC1 values.  These outliers drive the L-shape / spike visible in the
    PCA scatter plot.  Investigating them reveals whether they represent:
      (a) A genuinely distinctive sub-corpus (e.g. autonomous vehicles
          sector with concentrated technical vocabulary) that is analytically
          interesting and should be discussed in the thesis;
      (b) Data quality issues (scraping artefacts, duplicate pages) that
          should be excluded.

    The tail is defined as pages above the (1 - tail_pct) percentile of
    PC1 values (default: top 5%).

    Args:
        pca_coords       : ndarray (n_docs, N_PCA_DIMS) — full PCA matrix.
        metadata         : list of per-page dicts from load_corpus.
        doc_topic_matrix : ndarray (n_docs, N_TOPICS) — used to identify
                           dominant topics for tail pages.
        np               : numpy module.
        tail_pct         : fraction defining the tail (default 0.05 = top 5%).

    Returns:
        List of dicts for tail pages:
        {page_id, url, domain, audience, pca_1, dominant_topic}
    """
    pc1_vals = pca_coords[:, 0]
    threshold = np.percentile(pc1_vals, (1 - tail_pct) * 100)
    tail_mask = pc1_vals > threshold

    log.info(f"  PCA tail investigation (top {tail_pct*100:.0f}% of PC1 "
             f"values, threshold = {threshold:.4f}):")
    log.info(f"    {tail_mask.sum()} pages above PC1 = {threshold:.4f}")

    tail_pages = []
    for i in np.where(tail_mask)[0]:
        meta = metadata[i]
        dominant = int(doc_topic_matrix[i].argmax())
        tail_pages.append({
            "page_id":       meta["page_id"],
            "url":           meta["url"],
            "domain":        meta["domain"],
            "audience":      meta["audience"],
            "pca_1":         round(float(pc1_vals[i]), 4),
            "dominant_topic": dominant,
        })

    tail_pages.sort(key=lambda x: x["pca_1"], reverse=True)

    # Summary by domain and audience
    tail_domains  = Counter(p["domain"]   for p in tail_pages)
    tail_audience = Counter(p["audience"] for p in tail_pages)
    tail_topics   = Counter(p["dominant_topic"] for p in tail_pages)

    log.info(f"    Audience: {dict(tail_audience)}")
    log.info(f"    Top domains: {tail_domains.most_common(10)}")
    log.info(f"    Top topics:  {tail_topics.most_common(5)}")

    log.info("    Top 10 tail pages:")
    for p in tail_pages[:10]:
        log.info(f"      pca1={p['pca_1']:>7.4f}  "
                 f"{p['audience']:<8}  topic={p['dominant_topic']:<3}  "
                 f"{p['domain']}")

    return tail_pages


# ---------------------------------------------------------------------------
# Step 5: Topic audience profiles
# ---------------------------------------------------------------------------

def compute_topic_profiles(doc_topic_matrix, metadata, np) -> list:
    """
    Compute per-topic B2B vs B2W balance.

    For each topic, computes:
      avg_weight_client : mean topic weight across client pages
      avg_weight_worker : mean topic weight across worker pages
      client_share      : client_sum / (client_sum + worker_sum)
      category          : 'client_leaning' | 'worker_leaning' | 'shared'
      n_dominant_client : count of client pages where this is the
                          dominant topic (argmax)
      n_dominant_worker : same for worker pages

    The client_share and category are used in:
      - fig10 to visualise topic audience balance
      - Step 2 sampling to select pages from hypothesis-relevant topics

    Args:
        doc_topic_matrix : ndarray (n_docs, N_TOPICS) from LDA.
        metadata         : list of per-page dicts from load_corpus.
        np               : numpy module.

    Returns:
        List of N_TOPICS dicts, one per topic.
    """
    client_mask = np.array([m["audience"] == "client" for m in metadata])
    worker_mask = np.array([m["audience"] == "worker" for m in metadata])

    client_matrix = doc_topic_matrix[client_mask]
    worker_matrix = doc_topic_matrix[worker_mask]

    # Dominant topic counts per audience
    client_dominant = client_matrix.argmax(axis=1) if client_matrix.shape[0] > 0 else []
    worker_dominant = worker_matrix.argmax(axis=1) if worker_matrix.shape[0] > 0 else []

    results = []
    n_topics = doc_topic_matrix.shape[1]
    for t in range(n_topics):
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
    Compute how differently each focus term is framed in B2B vs B2W.

    Uses the PMI profiles stored in cooccurrence_results by
    02_step1_frequency.py.  For each focus term, computes the cosine
    similarity between its B2B PMI profile (vector of collocate PMI
    scores in client texts) and its B2W PMI profile.

    divergence = 1 - cosine_similarity(PMI_client, PMI_worker)
    = 0 means identical collocate profiles (term used the same way)
    = 1 means completely different collocate profiles (term used
         very differently by B2B vs B2W)

    Terms with high divergence are analytically productive for Step 2:
    the same word is doing different rhetorical work in each register.
    For example, "human" in B2B texts may collocate with "oversight" and
    "quality", while in B2W texts it collocates with "task" and "work".

    Only the cross_platform comparison is used (all client vs all worker),
    not within-pair comparisons.

    Returns:
        Dict {focus_term: divergence_score}
        Empty dict if cooccurrence_results table does not exist yet.
    """
    log.info("Computing collocate divergence from 02 co-occurrence data...")

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

        dot   = sum(c_vec.get(c, 0) * w_vec.get(c, 0) for c in all_collocates)
        mag_c = math.sqrt(sum(c_vec.get(c, 0) ** 2 for c in all_collocates))
        mag_w = math.sqrt(sum(w_vec.get(c, 0) ** 2 for c in all_collocates))

        if mag_c == 0 or mag_w == 0:
            cos_sim = 0.0
        else:
            cos_sim = dot / (mag_c * mag_w)

        divergences[term] = round(1.0 - cos_sim, 6)

    log.info(f"  Divergence scores for {len(divergences)} focus terms.")

    by_div = sorted(divergences.items(), key=lambda x: x[1], reverse=True)
    log.info("  Most DIVERGENT collocate profiles (same word, different framing):")
    for term, div in by_div[:15]:
        log.info(f"    {term:<25} divergence={div:.4f}")

    return divergences


# ---------------------------------------------------------------------------
# Step 7: Build Step 2 sampling table
# ---------------------------------------------------------------------------

def compute_topic_hypothesis_relevance(topic_terms_list: list) -> dict:
    """
    Score each LDA topic by its overlap with each hypothesis vocabulary.

    For each hypothesis in HYPOTHESIS_TERMS, calculates how many of the
    topic's top 15 terms appear in the hypothesis vocabulary set.  Topics
    with more overlapping terms are more relevant to that hypothesis.

    Args:
        topic_terms_list: list of dicts from extract_topic_terms().

    Returns:
        Dict {hypothesis_key: [(topic_id, overlap_score, matching_terms), ...]}
        Sorted by overlap_score descending.
    """
    # Build topic → top terms mapping (top 15 terms for matching)
    topic_top = defaultdict(set)
    for r in topic_terms_list:
        if r["rank"] <= 15:
            topic_top[r["topic_id"]].add(r["term"])

    result = {}
    for hyp_key, hyp_config in HYPOTHESIS_TERMS.items():
        hyp_terms = hyp_config["terms"]
        scored = []
        for topic_id, top_terms in topic_top.items():
            overlap = top_terms & hyp_terms
            if overlap:
                score = len(overlap)
                scored.append((topic_id, score, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        result[hyp_key] = scored

    return result


def build_step2_sample(
    doc_topics: list,
    topic_profiles: list,
    topic_terms_list: list,
    divergences: dict,
    metadata: list,
    domain_page_counts: dict,
    conn: sqlite3.Connection,
) -> list:
    """
    Build the hypothesis-stratified Step 2 sampling table.

    Domain quality filter:
      Domains with fewer than MIN_PAGES_PER_DOMAIN scraped pages are
      excluded from sampling candidates.  These domains contribute
      unreliable vocabulary signals — their presence in a sample would
      mean the analyst reads pages from platforms that could not be
      adequately scraped.  The filter is logged so the decision is
      transparent.

      Note: all pages (including sparse domains) still enter the LDA
      model.  The filter only affects which pages are offered as Step 2
      reading candidates.

    Selection strategy:
      For each hypothesis (H1a, H1b, H1c):
        1. Identify the n_topics most topic-relevant LDA topics (by term
           overlap with hypothesis vocabulary).
        2. From those topics, score all candidate pages by:
             combined = topic_weight × (1 + avg_div) × (1 + hyp_density × 10)
        3. Take the top n_pages_per_topic_per_audience pages for each
           (topic, audience) combination.
        4. Skip pages already selected by a previous hypothesis to avoid
           duplication.

    After selection, all results are globally ranked by combined
    analytical interest:
      rank_score = topic_weight × (1 + collocate_divergence)

    Args:
        doc_topics          : list from compute_document_topics().
        topic_profiles      : list from compute_topic_profiles().
        topic_terms_list    : list from extract_topic_terms().
        divergences         : dict from compute_collocate_divergence().
        metadata            : list from load_corpus().
        domain_page_counts  : dict {domain: n_pages} from load_corpus().
        conn                : open SQLite connection (for page term lookup).

    Returns:
        List of dicts with keys: page_id, url, domain, audience,
        dominant_topic, topic_weight, sampling_reason,
        collocate_divergence, priority_rank.
    """
    log.info("Building hypothesis-stratified Step 2 sampling table...")

    # Domain quality filter: identify sparse domains
    sparse_domains = {
        d for d, n in domain_page_counts.items()
        if n < MIN_PAGES_PER_DOMAIN
    }
    if sparse_domains:
        log.info(f"  Excluding {len(sparse_domains)} sparse domains from "
                 f"sampling candidates (< {MIN_PAGES_PER_DOMAIN} pages): "
                 f"{sorted(sparse_domains)}")

    # Compute topic-hypothesis relevance
    relevance = compute_topic_hypothesis_relevance(topic_terms_list)

    for hyp_key, scored_topics in relevance.items():
        hyp_config = HYPOTHESIS_TERMS[hyp_key]
        log.info(f"  {hyp_key} ({hyp_config['description']}):")
        for topic_id, score, terms in scored_topics[:5]:
            profile = next((p for p in topic_profiles if p["topic_id"] == topic_id), {})
            cat = profile.get("category", "?")
            log.info(f"    Topic {topic_id} [{cat}]  overlap={score}  "
                     f"terms: {', '.join(sorted(terms))}")

    # Build metadata lookup by page_id
    meta_by_page = {m["page_id"]: m for m in metadata}

    # Group document-topic entries by dominant topic;
    # filter out pages from sparse domains immediately
    by_topic = defaultdict(list)
    n_excluded = 0
    for dt in doc_topics:
        if dt["domain"] in sparse_domains:
            n_excluded += 1
            continue
        by_topic[dt["dominant_topic"]].append(dt)

    if n_excluded:
        log.info(f"  {n_excluded} pages excluded from sampling pool "
                 f"(sparse domain filter).")

    # Page-level term cache
    page_terms_cache = {}

    def get_page_terms(page_id):
        """Fetch and cache the unigrams for a page."""
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
        """
        Score a page by combined analytical interest.

        combined = topic_weight × (1 + avg_divergence) × (1 + hyp_density × 10)

        The 10× multiplier on hyp_density gives strong weight to pages
        that directly use the hypothesis vocabulary, ensuring the sample
        covers the terms the thesis is testing.
        """
        page_id = dt["page_id"]
        terms = get_page_terms(page_id)

        if divergences:
            matching_divs = [divergences[t] for t in terms if t in divergences]
            avg_div = (sum(matching_divs) / len(matching_divs)) if matching_divs else 0
        else:
            avg_div = 0

        hyp_count   = len(terms & hyp_terms)
        hyp_density = hyp_count / len(terms) if terms else 0

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

    results   = []
    seen_pages = set()

    for hyp_key, scored_topics in relevance.items():
        hyp_config = HYPOTHESIS_TERMS[hyp_key]
        hyp_terms  = hyp_config["terms"]
        n_topics   = hyp_config["n_topics"]
        n_per      = hyp_config["n_pages_per_topic_per_audience"]

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
                        "priority_rank":        0,
                    })

    # Global ranking
    results.sort(key=lambda x: (x["topic_weight"] * (1 + x["collocate_divergence"])),
                 reverse=True)
    for i, r in enumerate(results):
        r["priority_rank"] = i + 1

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
    """Insert topic term rows into topic_terms table."""
    conn.executemany("""
        INSERT INTO topic_terms (topic_id, topic_label, term, weight, rank)
        VALUES (:topic_id, :topic_label, :term, :weight, :rank)
    """, results)
    conn.commit()


def save_document_topics(conn, results):
    """Insert document-topic assignments into document_topics table."""
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
    """Insert topic audience profiles into topic_audience_profile table."""
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
    """Insert Step 2 sampling results into step2_sample table."""
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
    """
    Orchestrate the topic modelling and Step 2 sampling pipeline.

    Re-run safe: all output tables are dropped and recreated at the start.

    Pipeline steps:
      0.  [Optional] LDA diagnostics — multi-topic quality search
          (RUN_DIAGNOSTICS = True only; expensive, run once)
      1.  LDA topic model (chosen N_TOPICS)
      2.  UMass coherence + FREX for chosen model (quick single-run check)
      3.  Extract topic terms
      4.  Document-topic assignments + PCA
      5.  PCA quality checks:
            a. Audience separability test (logistic regression CV)
            b. PC1 tail investigation
      6.  Topic audience profiles
      7.  Collocate divergence (from 02_step1_frequency.py)
      8.  Step 2 sampling (with domain quality filter)
      9.  Save all results
    """
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("02c_step1_topics.py — Topic Modelling & Step 2 Sampling")
    log.info("=" * 60)

    # Import ML dependencies
    (LatentDirichletAllocation, PCA, CountVectorizer,
     LogisticRegression, cross_val_score, np) = _import_ml()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Verify corpus_view
    view_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()
    if not view_check:
        raise RuntimeError("corpus_view not found — run 01_prepare.py first.")

    init_output_tables(conn)

    # Load
    docs, metadata, excluded_terms, domain_page_counts = load_corpus(conn)

    # -----------------------------------------------------------------------
    # [Optional] LDA diagnostics
    # -----------------------------------------------------------------------
    if RUN_DIAGNOSTICS:
        run_lda_diagnostics(
            docs, np, CountVectorizer, LatentDirichletAllocation,
            excluded_terms, conn
        )
        log.info("Diagnostics complete. Inspect lda_diagnostics table, "
                 "then set RUN_DIAGNOSTICS = False and re-run with chosen N_TOPICS.")
        conn.close()
        return

    # -----------------------------------------------------------------------
    # LDA — main model
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("TOPIC MODELLING (LDA)")
    log.info(f"  N_TOPICS = {N_TOPICS}")
    log.info("-" * 60)
    lda, vectoriser, dtm, doc_topic_matrix, vocab = fit_lda(
        docs, np, CountVectorizer, LatentDirichletAllocation,
        excluded_terms=excluded_terms
    )

    # Single-model coherence and FREX (quick check for chosen N_TOPICS)
    log.info("  Computing UMass coherence for chosen model...")
    coherence = compute_umass_coherence(dtm, lda.components_, vocab, np)
    frex      = compute_topic_frex(lda.components_, np)
    log.info(f"  UMass coherence  = {coherence:.4f}  "
             f"(typical range −10 to 0; higher is better)")
    log.info(f"  Mean FREX        = {frex:.4f}  "
             f"(0–1; fraction of top-{FREX_TOP_K} terms unique per topic)")

    # Topic terms
    topic_terms = extract_topic_terms(lda, vocab, np)

    # Log top terms per topic for manual inspection
    log.info("-" * 60)
    log.info("TOPIC TERMS (top 8 per topic):")
    for t in range(N_TOPICS):
        terms = [r["term"] for r in topic_terms if r["topic_id"] == t and r["rank"] <= 8]
        log.info(f"  Topic {t:>2}: {', '.join(terms)}")

    # -----------------------------------------------------------------------
    # Document-topic assignments + PCA
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("DOCUMENT-TOPIC ASSIGNMENTS + PCA")
    log.info("-" * 60)
    doc_topics, pca_model = compute_document_topics(
        doc_topic_matrix, metadata, np, PCA
    )

    # PCA coordinates array — full size (excluded pages have pca_1 = None → NaN)
    pca_coords_full = np.array([
        [r["pca_1"] if r["pca_1"] is not None else np.nan,
         r["pca_2"] if r["pca_2"] is not None else np.nan,
         r["pca_3"] if r["pca_3"] is not None else 0.0]
        for r in doc_topics
    ])

    # Quality checks run only on pages that have valid PCA coordinates
    # (i.e. domains not in PCA_EXCLUDE_DOMAINS).
    valid_mask          = ~np.isnan(pca_coords_full[:, 0])
    pca_coords          = pca_coords_full[valid_mask]
    metadata_pca        = [m for m, v in zip(metadata, valid_mask) if v]
    doc_topic_matrix_pca = doc_topic_matrix[valid_mask]

    n_valid = int(valid_mask.sum())
    n_excl  = len(metadata) - n_valid
    if n_excl:
        log.info(f"  PCA quality checks run on {n_valid} pages "
                 f"({n_excl} PCA-excluded pages skipped).")

    # -----------------------------------------------------------------------
    # PCA quality checks
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("PCA QUALITY CHECKS")
    log.info("-" * 60)

    # a. Audience separability
    sep_accuracy = run_pca_audience_test(
        pca_coords, metadata_pca, np, LogisticRegression, cross_val_score
    )

    # b. PC1 tail investigation
    tail_pages = log_pca_tail(pca_coords, metadata_pca, doc_topic_matrix_pca, np)

    # -----------------------------------------------------------------------
    # Topic audience profiles
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Collocate divergence
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("COLLOCATE DIVERGENCE (from 02_step1_frequency.py)")
    log.info("-" * 60)
    divergences = compute_collocate_divergence(conn)

    # -----------------------------------------------------------------------
    # Step 2 sampling
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("STEP 2 SAMPLING STRATEGY")
    log.info("-" * 60)
    sample = build_step2_sample(
        doc_topics, topic_profiles, topic_terms,
        divergences, metadata, domain_page_counts, conn
    )

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

    log.info("=" * 60)
    log.info("TOPIC MODELLING & SAMPLING COMPLETE")
    log.info("")
    log.info("Quality summary:")
    log.info(f"  LDA perplexity    : see log above")
    log.info(f"  UMass coherence   : {coherence:.4f}  (higher = better topics)")
    log.info(f"  FREX              : {frex:.4f}  (higher = better separation)")
    log.info(f"  PCA separability  : {sep_accuracy:.3f}  accuracy (logistic CV)")
    log.info(f"  PC1 tail pages    : {len(tail_pages)} pages in top 5%")
    log.info("")
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
    log.info("         s.dominant_topic, s.collocate_divergence, s.sampling_reason")
    log.info("  FROM step2_sample s")
    log.info("  ORDER BY s.priority_rank LIMIT 30;")
    log.info("")
    log.info("  -- LDA diagnostics (if RUN_DIAGNOSTICS was run):")
    log.info("  SELECT n_topics, perplexity, avg_umass_coherence, avg_frex")
    log.info("  FROM lda_diagnostics ORDER BY n_topics;")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
