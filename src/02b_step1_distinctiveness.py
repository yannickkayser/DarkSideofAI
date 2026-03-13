"""
02b_step1_distinctiveness.py
============================
Nelson (2020) Step 1 — extension: Distinctiveness & Overlap

Complements 02_step1_frequency.py with three additional analyses:

  1. Company-level distinctiveness
     Jensen-Shannon Divergence (JSD) and cosine similarity between every
     pair of domains.  Answers: how similar/different is each company's
     language from every other company's?

  2. Aggregate B2B-vs-B2W distance
     A single JSD score summarising the overall vocabulary distance
     between client-facing and worker-facing subcorpora — the "one number"
     that captures how distinct the two registers are.

  3. Term exclusivity scores
     For each term: in what fraction of B2B platforms does it appear,
     vs what fraction of B2W platforms?  Produces a continuous
     exclusivity index from –1 (exclusively worker) to +1 (exclusively
     client), plus a categorical label.  Words near 0 are "shared" —
     interesting for discourse analysis precisely because the same term
     gets different collocate frames on each side.

Revision notes (v2):
  - Filters excluded pages and excluded terms from 01_prepare_additions.py
  - JSD computed on HIGH-VARIANCE terms only (terms whose relative
    frequency variance across domains exceeds the median).  This strips
    out the shared-vocabulary baseline that compressed the JSD range
    in v1, letting genuinely distinctive terms drive the distance metric.

Outputs written to three SQLite tables:
  - distinctiveness_matrix   : domain×domain JSD + cosine
  - aggregate_distance       : single-row B2B-vs-B2W JSD
  - term_exclusivity         : per-term exclusivity index and category

Prerequisites:
  - 01_prepare.py  (corpus_view must exist)
  - 01_prepare_additions.py  (excluded_pages, excluded_terms — optional but recommended)

Usage:
    python3 src/02b_step1_distinctiveness.py
"""

import sqlite3
import json
import math
import logging
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH       = "data/scraping.db"
MIN_TERM_FREQ = 5          # ignore terms below this total corpus frequency
# Exclusivity thresholds — a term is "exclusive" to one audience if its
# prevalence ratio exceeds these.  Prevalence = fraction of platforms in
# that audience where the term occurs at least once.
EXCLUSIVITY_THRESHOLD = 0.70   # high end
SHARED_BAND           = 0.25   # terms within ±0.25 of centre are "shared"

# High-variance vocabulary: only terms above median cross-domain variance
# are used for JSD.  This prevents the large shared-vocabulary baseline
# from compressing all JSD values into a narrow band.
HIGH_VARIANCE_PERCENTILE = 50  # keep terms above this percentile of variance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_output_tables(conn: sqlite3.Connection):
    conn.executescript("""
        DROP TABLE IF EXISTS distinctiveness_matrix;
        DROP TABLE IF EXISTS aggregate_distance;
        DROP TABLE IF EXISTS term_exclusivity;

        CREATE TABLE distinctiveness_matrix (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_a        TEXT NOT NULL,
            audience_a      TEXT,
            domain_b        TEXT NOT NULL,
            audience_b      TEXT,
            jsd             REAL NOT NULL,   -- Jensen-Shannon Divergence [0,1]
            cosine_sim      REAL NOT NULL,   -- cosine similarity [0,1]
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE aggregate_distance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison      TEXT NOT NULL,    -- 'b2b_vs_b2w'
            jsd             REAL NOT NULL,
            cosine_sim      REAL NOT NULL,
            n_client_tokens INTEGER,
            n_worker_tokens INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE term_exclusivity (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            term            TEXT NOT NULL,
            term_type       TEXT NOT NULL,    -- 'unigram' or 'bigram'
            prevalence_client REAL,           -- fraction of client platforms with term
            prevalence_worker REAL,           -- fraction of worker platforms with term
            exclusivity_index REAL,           -- +1 = only client, -1 = only worker
            category        TEXT,             -- 'client_exclusive'|'worker_exclusive'|'shared'|'leaning_client'|'leaning_worker'
            total_freq      INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_dm_pair
            ON distinctiveness_matrix(domain_a, domain_b);
        CREATE INDEX IF NOT EXISTS idx_te_category
            ON term_exclusivity(category, exclusivity_index);
    """)
    conn.commit()
    log.info("Output tables created.")


# ---------------------------------------------------------------------------
# Corpus loading (reuses corpus_view from 01_prepare.py)
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


def load_corpus(conn: sqlite3.Connection) -> dict:
    """
    Returns:
      'platform': {domain: {'audience': str, 'freq': Counter, 'n_tokens': int}}
      'cross':    {'client': (Counter, n_tokens), 'worker': (Counter, n_tokens)}
    """
    log.info("Loading corpus from corpus_view...")

    excluded_pages, excluded_terms = load_exclusions(conn)

    rows = conn.execute("""
        SELECT page_id, audience, domain, unigrams, bigrams, token_count
        FROM corpus_view
        WHERE audience IN ('client', 'worker')
          AND token_count >= 10
    """).fetchall()

    log.info(f"  Loaded {len(rows)} pages (before exclusions).")

    # Per-domain accumulators
    platform_freq   = defaultdict(Counter)
    platform_tokens = defaultdict(int)
    platform_aud    = {}

    # Cross-platform accumulators
    cross_freq   = defaultdict(Counter)
    cross_tokens = defaultdict(int)

    skipped = 0
    for row in rows:
        if row["page_id"] in excluded_pages:
            skipped += 1
            continue

        audience = row["audience"]
        domain   = row["domain"]

        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        bigrams  = json.loads(row["bigrams"])  if row["bigrams"]  else []

        # Filter excluded terms
        if excluded_terms:
            unigrams = [t for t in unigrams if t not in excluded_terms]
            bigrams  = [t for t in bigrams  if t not in excluded_terms]

        tokens   = unigrams + bigrams
        n_uni    = len(unigrams)

        platform_freq[domain].update(tokens)
        platform_tokens[domain] += n_uni
        platform_aud[domain]     = audience

        cross_freq[audience].update(tokens)
        cross_tokens[audience] += n_uni

    if skipped:
        log.info(f"  Skipped {skipped} excluded pages.")

    platform = {
        d: {
            "audience": platform_aud[d],
            "freq":     platform_freq[d],
            "n_tokens": platform_tokens[d],
        }
        for d in platform_freq
    }

    cross = {
        aud: (cross_freq[aud], cross_tokens[aud])
        for aud in cross_freq
    }

    log.info(f"  {len(platform)} domains loaded.")
    return {"platform": platform, "cross": cross}


# ---------------------------------------------------------------------------
# Helper: probability distribution from a Counter
# ---------------------------------------------------------------------------

def to_prob_dist(freq: Counter, vocab: set) -> dict:
    """
    Convert a frequency counter into a probability distribution over vocab.
    Uses additive (Laplace) smoothing with alpha=1 to handle zero counts,
    which is necessary for well-defined KL divergence.
    """
    alpha = 1
    total = sum(freq.get(t, 0) for t in vocab) + alpha * len(vocab)
    return {t: (freq.get(t, 0) + alpha) / total for t in vocab}


# ---------------------------------------------------------------------------
# Analysis 1: Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

def kl_divergence(p: dict, q: dict) -> float:
    """KL(P || Q) — both dists must share the same key set."""
    return sum(p[t] * math.log(p[t] / q[t]) for t in p if p[t] > 0 and q[t] > 0)


def jsd(p: dict, q: dict) -> float:
    """Jensen-Shannon Divergence — symmetric, bounded [0, ln2]."""
    m = {t: 0.5 * (p[t] + q[t]) for t in p}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def jsd_normalised(p: dict, q: dict) -> float:
    """JSD normalised to [0, 1] by dividing by ln(2)."""
    return jsd(p, q) / math.log(2)


# ---------------------------------------------------------------------------
# Analysis 2: Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(freq_a: Counter, freq_b: Counter, vocab: set) -> float:
    """Cosine similarity on raw frequency vectors over shared vocab."""
    dot = sum(freq_a.get(t, 0) * freq_b.get(t, 0) for t in vocab)
    mag_a = math.sqrt(sum(freq_a.get(t, 0) ** 2 for t in vocab))
    mag_b = math.sqrt(sum(freq_b.get(t, 0) ** 2 for t in vocab))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Analysis 3: Term exclusivity
# ---------------------------------------------------------------------------

def compute_term_exclusivity(platform: dict) -> list[dict]:
    """
    For each term, compute:
      prevalence_client = fraction of client platforms where term appears
      prevalence_worker = fraction of worker platforms where term appears
      exclusivity_index = prevalence_client - prevalence_worker  (in [-1, +1])

    A term with exclusivity_index near +1 appears only on client platforms;
    near -1 only on worker platforms; near 0 it is "shared".
    """
    client_domains = [d for d, v in platform.items() if v["audience"] == "client"]
    worker_domains = [d for d, v in platform.items() if v["audience"] == "worker"]
    n_client = len(client_domains)
    n_worker = len(worker_domains)

    if n_client == 0 or n_worker == 0:
        log.warning("  Cannot compute exclusivity — need both audiences.")
        return []

    # Gather global vocabulary + total freq
    global_freq = Counter()
    for d, v in platform.items():
        global_freq.update(v["freq"])

    # For each term, count how many client/worker platforms contain it
    term_client_count = Counter()
    term_worker_count = Counter()

    for d in client_domains:
        for t in platform[d]["freq"]:
            term_client_count[t] += 1
    for d in worker_domains:
        for t in platform[d]["freq"]:
            term_worker_count[t] += 1

    results = []
    for term, total in global_freq.items():
        if total < MIN_TERM_FREQ:
            continue

        prev_c = term_client_count.get(term, 0) / n_client
        prev_w = term_worker_count.get(term, 0) / n_worker
        excl   = prev_c - prev_w   # +1 = only client, -1 = only worker

        # Categorise
        if excl > EXCLUSIVITY_THRESHOLD:
            cat = "client_exclusive"
        elif excl < -EXCLUSIVITY_THRESHOLD:
            cat = "worker_exclusive"
        elif abs(excl) <= SHARED_BAND:
            cat = "shared"
        elif excl > 0:
            cat = "leaning_client"
        else:
            cat = "leaning_worker"

        results.append({
            "term":              term,
            "term_type":         "bigram" if "_" in term else "unigram",
            "prevalence_client": round(prev_c, 4),
            "prevalence_worker": round(prev_w, 4),
            "exclusivity_index": round(excl, 4),
            "category":          cat,
            "total_freq":        total,
        })

    results.sort(key=lambda x: abs(x["exclusivity_index"]), reverse=True)
    log.info(f"  {len(results):,} terms scored for exclusivity.")
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("02b_step1_distinctiveness.py — Distinctiveness & Overlap")
    log.info("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Verify corpus_view
    view_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()
    if not view_check:
        raise RuntimeError("corpus_view not found — run 01_prepare.py first.")

    init_output_tables(conn)
    corpus = load_corpus(conn)
    platform = corpus["platform"]

    # -----------------------------------------------------------------------
    # 1. Domain-level pairwise distinctiveness matrix
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("DOMAIN-LEVEL DISTINCTIVENESS MATRIX")
    log.info("-" * 60)

    domains = sorted(platform.keys())
    n_domains = len(domains)

    # Build shared vocabulary (terms occurring in at least 2 domains)
    term_domain_count = Counter()
    for d in domains:
        for t in platform[d]["freq"]:
            term_domain_count[t] += 1
    base_vocab = {t for t, c in term_domain_count.items() if c >= 2 and
                  sum(platform[d]["freq"].get(t, 0) for d in domains) >= MIN_TERM_FREQ}
    log.info(f"  Base vocabulary size: {len(base_vocab):,} terms")

    # HIGH-VARIANCE FILTERING: compute relative-frequency variance per term
    # across domains, keep only terms above the median.  This strips out
    # the shared baseline that compressed JSD in v1.
    import numpy as _np
    term_variances = {}
    for t in base_vocab:
        rel_freqs = []
        for d in domains:
            n_tok = platform[d]["n_tokens"]
            freq  = platform[d]["freq"].get(t, 0)
            rel_freqs.append(freq / n_tok if n_tok > 0 else 0)
        term_variances[t] = _np.var(rel_freqs)

    variance_threshold = _np.percentile(
        list(term_variances.values()), HIGH_VARIANCE_PERCENTILE
    )
    vocab = {t for t, v in term_variances.items() if v >= variance_threshold}
    log.info(f"  High-variance vocabulary (p{HIGH_VARIANCE_PERCENTILE}): "
             f"{len(vocab):,} terms  (variance threshold: {variance_threshold:.2e})")

    matrix_rows = []
    for i, da in enumerate(domains):
        pa = to_prob_dist(platform[da]["freq"], vocab)
        for db in domains[i + 1:]:
            pb = to_prob_dist(platform[db]["freq"], vocab)

            jsd_val = jsd_normalised(pa, pb)
            cos_val = cosine_similarity(platform[da]["freq"],
                                        platform[db]["freq"], vocab)

            matrix_rows.append({
                "domain_a":   da,
                "audience_a": platform[da]["audience"],
                "domain_b":   db,
                "audience_b": platform[db]["audience"],
                "jsd":        round(jsd_val, 6),
                "cosine_sim": round(cos_val, 6),
            })

    log.info(f"  {len(matrix_rows)} domain pairs computed.")

    # Log most/least similar pairs
    by_jsd = sorted(matrix_rows, key=lambda x: x["jsd"])
    log.info("  Most SIMILAR domain pairs (lowest JSD):")
    for r in by_jsd[:5]:
        log.info(f"    {r['domain_a']:<25} ↔ {r['domain_b']:<25}  "
                 f"JSD={r['jsd']:.4f}  cos={r['cosine_sim']:.4f}  "
                 f"[{r['audience_a']} vs {r['audience_b']}]")
    log.info("  Most DIFFERENT domain pairs (highest JSD):")
    for r in by_jsd[-5:]:
        log.info(f"    {r['domain_a']:<25} ↔ {r['domain_b']:<25}  "
                 f"JSD={r['jsd']:.4f}  cos={r['cosine_sim']:.4f}  "
                 f"[{r['audience_a']} vs {r['audience_b']}]")

    # -----------------------------------------------------------------------
    # 2. Aggregate B2B vs B2W distance
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("AGGREGATE B2B vs B2W DISTANCE")
    log.info("-" * 60)

    cross = corpus["cross"]
    if "client" in cross and "worker" in cross:
        client_freq, n_client = cross["client"]
        worker_freq, n_worker = cross["worker"]

        # Vocabulary: all terms with sufficient frequency in either subcorpus
        agg_vocab = {t for t in (set(client_freq) | set(worker_freq))
                     if client_freq.get(t, 0) + worker_freq.get(t, 0) >= MIN_TERM_FREQ}

        pc = to_prob_dist(client_freq, agg_vocab)
        pw = to_prob_dist(worker_freq, agg_vocab)

        agg_jsd = jsd_normalised(pc, pw)
        agg_cos = cosine_similarity(client_freq, worker_freq, agg_vocab)

        log.info(f"  JSD (B2B vs B2W)    = {agg_jsd:.6f}")
        log.info(f"  Cosine (B2B vs B2W) = {agg_cos:.6f}")
        log.info(f"  Client tokens: {n_client:,}  Worker tokens: {n_worker:,}")
        log.info(f"  Vocabulary size: {len(agg_vocab):,}")

        agg_row = {
            "comparison":      "b2b_vs_b2w",
            "jsd":             round(agg_jsd, 6),
            "cosine_sim":      round(agg_cos, 6),
            "n_client_tokens": n_client,
            "n_worker_tokens": n_worker,
        }
    else:
        log.warning("  Missing one audience — skipping aggregate distance.")
        agg_row = None

    # -----------------------------------------------------------------------
    # 3. Term exclusivity scores
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("TERM EXCLUSIVITY")
    log.info("-" * 60)

    exclusivity = compute_term_exclusivity(platform)

    # Category summary
    cats = Counter(r["category"] for r in exclusivity)
    for cat, n in cats.most_common():
        log.info(f"  {cat:<20} {n:>6} terms")

    # Log top exclusive terms in each direction
    client_excl = [r for r in exclusivity if r["category"] == "client_exclusive"]
    worker_excl = [r for r in exclusivity if r["category"] == "worker_exclusive"]
    shared      = [r for r in exclusivity if r["category"] == "shared"]

    log.info(f"  Top CLIENT-exclusive terms (appear almost only on B2B platforms):")
    for r in client_excl[:15]:
        log.info(f"    {r['term']:<30} excl={r['exclusivity_index']:>+.3f}  "
                 f"prev_c={r['prevalence_client']:.2f}  "
                 f"prev_w={r['prevalence_worker']:.2f}  "
                 f"freq={r['total_freq']}")

    log.info(f"  Top WORKER-exclusive terms (appear almost only on B2W platforms):")
    for r in worker_excl[:15]:
        log.info(f"    {r['term']:<30} excl={r['exclusivity_index']:>+.3f}  "
                 f"prev_c={r['prevalence_client']:.2f}  "
                 f"prev_w={r['prevalence_worker']:.2f}  "
                 f"freq={r['total_freq']}")

    log.info(f"  Top SHARED terms (appear on both B2B and B2W platforms):")
    shared_by_freq = sorted(shared, key=lambda x: x["total_freq"], reverse=True)
    for r in shared_by_freq[:15]:
        log.info(f"    {r['term']:<30} excl={r['exclusivity_index']:>+.3f}  "
                 f"prev_c={r['prevalence_client']:.2f}  "
                 f"prev_w={r['prevalence_worker']:.2f}  "
                 f"freq={r['total_freq']}")

    # -----------------------------------------------------------------------
    # Save everything
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Saving results to database...")

    conn.executemany("""
        INSERT INTO distinctiveness_matrix
            (domain_a, audience_a, domain_b, audience_b, jsd, cosine_sim)
        VALUES
            (:domain_a, :audience_a, :domain_b, :audience_b, :jsd, :cosine_sim)
    """, matrix_rows)
    log.info(f"  distinctiveness_matrix : {len(matrix_rows):,} rows")

    if agg_row:
        conn.execute("""
            INSERT INTO aggregate_distance
                (comparison, jsd, cosine_sim, n_client_tokens, n_worker_tokens)
            VALUES
                (:comparison, :jsd, :cosine_sim, :n_client_tokens, :n_worker_tokens)
        """, agg_row)
        log.info(f"  aggregate_distance    : 1 row")

    conn.executemany("""
        INSERT INTO term_exclusivity
            (term, term_type, prevalence_client, prevalence_worker,
             exclusivity_index, category, total_freq)
        VALUES
            (:term, :term_type, :prevalence_client, :prevalence_worker,
             :exclusivity_index, :category, :total_freq)
    """, exclusivity)
    log.info(f"  term_exclusivity      : {len(exclusivity):,} rows")

    conn.commit()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("DISTINCTIVENESS ANALYSIS COMPLETE")
    log.info("Query examples:")
    log.info("  -- Most similar cross-audience domain pairs (B2B↔B2W):")
    log.info("  SELECT domain_a, domain_b, jsd, cosine_sim")
    log.info("  FROM distinctiveness_matrix")
    log.info("  WHERE audience_a != audience_b")
    log.info("  ORDER BY jsd ASC LIMIT 10;")
    log.info("")
    log.info("  -- Terms shared across both audiences (Step 2 candidates):")
    log.info("  SELECT term, exclusivity_index, prevalence_client, prevalence_worker")
    log.info("  FROM term_exclusivity")
    log.info("  WHERE category = 'shared'")
    log.info("  ORDER BY total_freq DESC LIMIT 50;")
    log.info("")
    log.info("  -- Aggregate B2B vs B2W distance:")
    log.info("  SELECT * FROM aggregate_distance;")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
