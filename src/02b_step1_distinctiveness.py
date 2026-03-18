"""
02b_step1_distinctiveness.py
============================
Nelson (2020) Step 1 — extension: Distinctiveness & Overlap

Pipeline position:
  Stage 2b — Distinctiveness Analysis (run after 02_step1_frequency.py,
  can run in parallel with 02c_step1_topics.py)
  Prerequisites: 01_prepare.py (corpus_view), 01_prepare_additions.py
  Next step:     03b_visualise_distinctiveness_topics.py (figures 7–12)

What this script does:
  Complements 02_step1_frequency.py with three additional perspectives
  on vocabulary divergence between B2B and B2W registers:

  1. Company-level distinctiveness matrix
     For every pair of QUALIFIED domains (see MIN_PAGES_PER_DOMAIN),
     computes Jensen-Shannon Divergence (JSD) and cosine similarity over
     the high-variance vocabulary.  Answers: "How linguistically
     similar/different is each platform from every other?"

  2. Aggregate B2B-vs-B2W distance
     A single JSD and cosine score summarising the overall vocabulary
     distance between the client and worker subcorpora, computed over ALL
     domains (no page-count filter applied here since both audiences are
     pooled into large corpora).

  3. Term exclusivity scores
     For each term: in what fraction of QUALIFIED B2B platforms does it
     appear (prevalence_client) and in what fraction of QUALIFIED B2W
     platforms (prevalence_worker)?  The difference gives an
     exclusivity_index from +1 (only on B2B platforms) to -1 (only on
     B2W platforms).

Domain quality filter (MIN_PAGES_PER_DOMAIN = 5):
  Platform-level analyses treat each domain as an equal unit.  A domain
  represented by only 1-2 scraped pages is NOT a valid unit because:

    1. Its vocabulary distribution is unreliable (one page cannot
       represent a site's full register).
    2. It contributes a binary 0/1 to prevalence counts with the same
       weight as a 100-page site, distorting exclusivity scores.
    3. Its JSD relative to other domains reflects data sparsity, not
       genuine linguistic difference — a sparse domain appears more
       "distinctive" because its vocabulary is incomplete.

  Excluded domains remain in the CROSS-PLATFORM AGGREGATE (pooled with
  all other pages — their marginal token contribution is negligible).
  All exclusion decisions are recorded in the domain_quality table.

  Examples of affected domains: mturk.com (redirected, single page
  captured), telusinternational.com (B2B — robots.txt blocked most
  pages), similar scraping failures.

Revision notes:
  - v2: applies exclusion filtering from 01_prepare_additions.py;
        JSD computed on HIGH-VARIANCE terms only.
  - v3: adds MIN_PAGES_PER_DOMAIN quality gate + domain_quality table.

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
MIN_TERM_FREQ = 5

# --- Domain quality gate ---
# Minimum pages a domain must have to be included in the PLATFORM-LEVEL
# analyses (distinctiveness matrix, term exclusivity).  Rationale: see
# module docstring above.  Set to 1 to disable the filter.
MIN_PAGES_PER_DOMAIN = 5

EXCLUSIVITY_THRESHOLD = 0.70
SHARED_BAND           = 0.25
HIGH_VARIANCE_PERCENTILE = 50

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
    """
    Create output tables, dropping previous versions for clean re-runs.

    New in v3: domain_quality table records per-domain page counts and
    whether the domain was included in or excluded from platform-level
    analyses.  This table is the audit trail for the quality filter and
    should be referenced in the thesis methodology section.
    """
    conn.executescript("""
        DROP TABLE IF EXISTS distinctiveness_matrix;
        DROP TABLE IF EXISTS aggregate_distance;
        DROP TABLE IF EXISTS term_exclusivity;
        DROP TABLE IF EXISTS domain_quality;

        CREATE TABLE distinctiveness_matrix (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_a        TEXT NOT NULL,
            audience_a      TEXT,
            domain_b        TEXT NOT NULL,
            audience_b      TEXT,
            jsd             REAL NOT NULL,
            cosine_sim      REAL NOT NULL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE aggregate_distance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison      TEXT NOT NULL,
            jsd             REAL NOT NULL,
            cosine_sim      REAL NOT NULL,
            n_client_tokens INTEGER,
            n_worker_tokens INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE term_exclusivity (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            term              TEXT NOT NULL,
            term_type         TEXT NOT NULL,
            prevalence_client REAL,
            prevalence_worker REAL,
            exclusivity_index REAL,
            category          TEXT,
            total_freq        INTEGER,
            created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE domain_quality (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            domain               TEXT NOT NULL,
            audience             TEXT,
            n_pages              INTEGER NOT NULL,
            n_tokens             INTEGER NOT NULL,
            included_in_matrix   INTEGER NOT NULL,  -- 1=yes, 0=excluded
            exclusion_reason     TEXT,              -- NULL if included
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_dm_pair    ON distinctiveness_matrix(domain_a, domain_b);
        CREATE INDEX IF NOT EXISTS idx_te_cat     ON term_exclusivity(category, exclusivity_index);
        CREATE INDEX IF NOT EXISTS idx_dq_domain  ON domain_quality(domain);
    """)
    conn.commit()
    log.info("Output tables created.")


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_exclusions(conn):
    excluded_pages = set()
    excluded_terms = set()
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='excluded_pages'").fetchone():
        excluded_pages = {r[0] for r in conn.execute(
            "SELECT page_id FROM excluded_pages").fetchall()}
        log.info(f"  Loaded {len(excluded_pages)} excluded pages.")
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='excluded_terms'").fetchone():
        excluded_terms = {r[0] for r in conn.execute(
            "SELECT term FROM excluded_terms").fetchall()}
        log.info(f"  Loaded {len(excluded_terms)} excluded terms.")
    return excluded_pages, excluded_terms


def load_corpus(conn):
    """
    Load corpus from corpus_view, tracking per-domain page counts.

    The 'n_pages' key added to each platform entry is used by
    apply_domain_quality_filter() to exclude sparsely-sampled domains
    from the pairwise matrix and term exclusivity analyses.

    The cross-platform aggregate ('cross' key) includes ALL domains
    regardless of page count — pooling makes individual domain sparsity
    inconsequential.
    """
    log.info("Loading corpus from corpus_view...")
    excluded_pages, excluded_terms = load_exclusions(conn)

    rows = conn.execute("""
        SELECT page_id, audience, domain, unigrams, bigrams, token_count
        FROM corpus_view
        WHERE audience IN ('client', 'worker')
          AND token_count >= 10
    """).fetchall()

    log.info(f"  {len(rows)} pages (before exclusions).")

    platform_freq   = defaultdict(Counter)
    platform_tokens = defaultdict(int)
    platform_pages  = defaultdict(int)      # NEW: page count per domain
    platform_aud    = {}
    cross_freq      = defaultdict(Counter)
    cross_tokens    = defaultdict(int)
    skipped = 0

    for row in rows:
        if row["page_id"] in excluded_pages:
            skipped += 1
            continue
        audience = row["audience"]
        domain   = row["domain"]
        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        bigrams  = json.loads(row["bigrams"])  if row["bigrams"]  else []
        if excluded_terms:
            unigrams = [t for t in unigrams if t not in excluded_terms]
            bigrams  = [t for t in bigrams  if t not in excluded_terms]
        tokens = unigrams + bigrams
        n_uni  = len(unigrams)
        platform_freq[domain].update(tokens)
        platform_tokens[domain] += n_uni
        platform_pages[domain]  += 1
        platform_aud[domain]     = audience
        cross_freq[audience].update(tokens)
        cross_tokens[audience]  += n_uni

    if skipped:
        log.info(f"  Skipped {skipped} excluded pages.")

    platform = {
        d: {
            "audience": platform_aud[d],
            "freq":     platform_freq[d],
            "n_tokens": platform_tokens[d],
            "n_pages":  platform_pages[d],
        }
        for d in platform_freq
    }

    # Log page-count distribution — shows which domains are sparse
    sorted_domains = sorted(platform.items(), key=lambda x: x[1]["n_pages"])
    log.info(f"  {len(platform)} domains loaded. Page counts (ascending):")
    for d, v in sorted_domains:
        flag = "  ⚠ BELOW THRESHOLD" if v["n_pages"] < MIN_PAGES_PER_DOMAIN else ""
        log.info(f"    {d:<35} {v['n_pages']:>4} pages  [{v['audience']}]{flag}")

    return {
        "platform": platform,
        "cross": {aud: (cross_freq[aud], cross_tokens[aud]) for aud in cross_freq},
    }


def apply_domain_quality_filter(platform):
    """
    Apply the MIN_PAGES_PER_DOMAIN filter and produce a quality audit table.

    WHY this belongs here and not at load time:
      The cross-platform aggregate analysis (JSD between all B2B vs all
      B2W pages pooled) should include every available page — even one
      page from MTurk contributes valid vocabulary signal when pooled
      with thousands of other pages.  The filter should only apply when
      each domain is treated as an EQUAL UNIT (as in the pairwise matrix
      and term exclusivity calculations).  Separating load from filtering
      makes this distinction explicit.

    Returns:
        filtered_platform : dict with only domains meeting the threshold.
        quality_rows      : list of domain_quality rows for ALL domains.
    """
    filtered = {}
    quality_rows = []
    for d, v in platform.items():
        n = v["n_pages"]
        included = n >= MIN_PAGES_PER_DOMAIN
        reason = None if included else (
            f"n_pages={n} < MIN_PAGES_PER_DOMAIN={MIN_PAGES_PER_DOMAIN}; "
            f"insufficient pages for reliable platform-level vocabulary representation"
        )
        if included:
            filtered[d] = v
        quality_rows.append({
            "domain": d, "audience": v["audience"],
            "n_pages": n, "n_tokens": v["n_tokens"],
            "included_in_matrix": int(included),
            "exclusion_reason": reason,
        })

    n_excl = len(platform) - len(filtered)
    if n_excl:
        excl = [d for d, v in platform.items() if v["n_pages"] < MIN_PAGES_PER_DOMAIN]
        log.info(f"  EXCLUDED from platform analyses ({n_excl} domains): {excl}")
    log.info(f"  INCLUDED in platform analyses: {len(filtered)} domains")
    return filtered, quality_rows


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def to_prob_dist(freq, vocab):
    """Laplace-smoothed probability distribution over vocab."""
    alpha = 1
    total = sum(freq.get(t, 0) for t in vocab) + alpha * len(vocab)
    return {t: (freq.get(t, 0) + alpha) / total for t in vocab}


def kl_divergence(p, q):
    return sum(p[t] * math.log(p[t] / q[t]) for t in p if p[t] > 0 and q[t] > 0)


def jsd(p, q):
    m = {t: 0.5 * (p[t] + q[t]) for t in p}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def jsd_normalised(p, q):
    return jsd(p, q) / math.log(2)


def cosine_similarity(freq_a, freq_b, vocab):
    dot   = sum(freq_a.get(t, 0) * freq_b.get(t, 0) for t in vocab)
    mag_a = math.sqrt(sum(freq_a.get(t, 0) ** 2 for t in vocab))
    mag_b = math.sqrt(sum(freq_b.get(t, 0) ** 2 for t in vocab))
    return dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0.0


# ---------------------------------------------------------------------------
# Term exclusivity
# ---------------------------------------------------------------------------

def compute_term_exclusivity(platform):
    """
    Compute prevalence-based exclusivity using QUALIFIED domains only.

    Prevalence = fraction of platforms (within one audience) where the
    term appears at least once.  Uses filtered platform dict — domains
    with < MIN_PAGES_PER_DOMAIN pages are excluded so that sparse sites
    cannot disproportionately inflate or deflate a term's apparent
    exclusivity.
    """
    client_domains = [d for d, v in platform.items() if v["audience"] == "client"]
    worker_domains = [d for d, v in platform.items() if v["audience"] == "worker"]
    n_c, n_w = len(client_domains), len(worker_domains)
    if n_c == 0 or n_w == 0:
        log.warning("  Cannot compute exclusivity — need both audiences.")
        return []
    log.info(f"  Exclusivity over {n_c} client + {n_w} worker qualified domains.")

    global_freq = Counter()
    for v in platform.values():
        global_freq.update(v["freq"])

    term_c = Counter()
    term_w = Counter()
    for d in client_domains:
        for t in platform[d]["freq"]:
            term_c[t] += 1
    for d in worker_domains:
        for t in platform[d]["freq"]:
            term_w[t] += 1

    results = []
    for term, total in global_freq.items():
        if total < MIN_TERM_FREQ:
            continue
        pc   = term_c.get(term, 0) / n_c
        pw   = term_w.get(term, 0) / n_w
        excl = pc - pw
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
            "term": term,
            "term_type": "bigram" if "_" in term else "unigram",
            "prevalence_client": round(pc, 4),
            "prevalence_worker": round(pw, 4),
            "exclusivity_index": round(excl, 4),
            "category": cat,
            "total_freq": total,
        })

    results.sort(key=lambda x: abs(x["exclusivity_index"]), reverse=True)
    log.info(f"  {len(results):,} terms scored.")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("02b_step1_distinctiveness.py — Distinctiveness & Overlap")
    log.info("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if not conn.execute("SELECT name FROM sqlite_master WHERE type='view' "
                         "AND name='corpus_view'").fetchone():
        raise RuntimeError("corpus_view not found — run 01_prepare.py first.")

    init_output_tables(conn)
    corpus = load_corpus(conn)
    platform_all = corpus["platform"]

    # --- Domain quality filter ---
    log.info("-" * 60)
    log.info(f"DOMAIN QUALITY FILTER (MIN_PAGES_PER_DOMAIN = {MIN_PAGES_PER_DOMAIN})")
    log.info("-" * 60)
    platform, quality_rows = apply_domain_quality_filter(platform_all)

    conn.executemany("""
        INSERT INTO domain_quality
            (domain, audience, n_pages, n_tokens, included_in_matrix, exclusion_reason)
        VALUES (:domain, :audience, :n_pages, :n_tokens,
                :included_in_matrix, :exclusion_reason)
    """, quality_rows)
    conn.commit()
    log.info(f"  domain_quality: {len(quality_rows)} rows saved.")

    # --- Build high-variance vocabulary (filtered domains only) ---
    log.info("-" * 60)
    log.info("DOMAIN-LEVEL DISTINCTIVENESS MATRIX")
    log.info("-" * 60)

    import numpy as _np
    domains = sorted(platform.keys())
    log.info(f"  Matrix for {len(domains)} qualified domains.")

    term_domain_count = Counter()
    for d in domains:
        for t in platform[d]["freq"]:
            term_domain_count[t] += 1
    base_vocab = {t for t, c in term_domain_count.items() if c >= 2 and
                  sum(platform[d]["freq"].get(t, 0) for d in domains) >= MIN_TERM_FREQ}
    log.info(f"  Base vocabulary: {len(base_vocab):,} terms")

    term_variances = {}
    for t in base_vocab:
        rel_freqs = [platform[d]["freq"].get(t, 0) / platform[d]["n_tokens"]
                     if platform[d]["n_tokens"] > 0 else 0
                     for d in domains]
        term_variances[t] = _np.var(rel_freqs)

    var_threshold = _np.percentile(list(term_variances.values()), HIGH_VARIANCE_PERCENTILE)
    vocab = {t for t, v in term_variances.items() if v >= var_threshold}
    log.info(f"  High-variance vocab (p{HIGH_VARIANCE_PERCENTILE}): "
             f"{len(vocab):,} terms  (threshold={var_threshold:.2e})")

    # Pairwise matrix
    matrix_rows = []
    for i, da in enumerate(domains):
        pa = to_prob_dist(platform[da]["freq"], vocab)
        for db in domains[i + 1:]:
            pb = to_prob_dist(platform[db]["freq"], vocab)
            matrix_rows.append({
                "domain_a":   da, "audience_a": platform[da]["audience"],
                "domain_b":   db, "audience_b": platform[db]["audience"],
                "jsd":        round(jsd_normalised(pa, pb), 6),
                "cosine_sim": round(cosine_similarity(platform[da]["freq"],
                                                      platform[db]["freq"], vocab), 6),
            })

    # Audience block JSD comparison (key thesis argument)
    cross_aud = [r for r in matrix_rows if r["audience_a"] != r["audience_b"]]
    intra_c   = [r for r in matrix_rows
                 if r["audience_a"] == "client" and r["audience_b"] == "client"]
    intra_w   = [r for r in matrix_rows
                 if r["audience_a"] == "worker" and r["audience_b"] == "worker"]
    if cross_aud and intra_c and intra_w:
        avg_x = sum(r["jsd"] for r in cross_aud)  / len(cross_aud)
        avg_c = sum(r["jsd"] for r in intra_c)    / len(intra_c)
        avg_w = sum(r["jsd"] for r in intra_w)    / len(intra_w)
        log.info("  Block JSD averages (thesis argument: cross-aud > intra-aud):")
        log.info(f"    Intra-B2B  JSD = {avg_c:.4f}  (n={len(intra_c)} pairs)")
        log.info(f"    Intra-B2W  JSD = {avg_w:.4f}  (n={len(intra_w)} pairs)")
        log.info(f"    Cross-aud  JSD = {avg_x:.4f}  (n={len(cross_aud)} pairs)")
        ratio = avg_x / max(avg_c, avg_w)
        log.info(f"    Cross/Intra ratio = {ratio:.2f}  "
                 f"({'audience IS a dominant structural axis' if ratio > 1.0 else 'CAUTION: cross <= intra'})")

    by_jsd = sorted(matrix_rows, key=lambda x: x["jsd"])
    log.info("  Most SIMILAR pairs (lowest JSD):")
    for r in by_jsd[:5]:
        log.info(f"    {r['domain_a']:<25} ↔ {r['domain_b']:<25}  "
                 f"JSD={r['jsd']:.4f}  [{r['audience_a']} vs {r['audience_b']}]")
    log.info("  Most DIFFERENT pairs (highest JSD):")
    for r in by_jsd[-5:]:
        log.info(f"    {r['domain_a']:<25} ↔ {r['domain_b']:<25}  "
                 f"JSD={r['jsd']:.4f}  [{r['audience_a']} vs {r['audience_b']}]")

    # --- Aggregate distance (ALL domains pooled) ---
    log.info("-" * 60)
    log.info("AGGREGATE B2B vs B2W DISTANCE (all domains, no page-count filter)")
    log.info("-" * 60)
    cross = corpus["cross"]
    agg_row = None
    if "client" in cross and "worker" in cross:
        cfreq, n_c = cross["client"]
        wfreq, n_w = cross["worker"]
        agg_vocab = {t for t in set(cfreq) | set(wfreq)
                     if cfreq.get(t, 0) + wfreq.get(t, 0) >= MIN_TERM_FREQ}
        pc, pw = to_prob_dist(cfreq, agg_vocab), to_prob_dist(wfreq, agg_vocab)
        agg_jsd = jsd_normalised(pc, pw)
        agg_cos = cosine_similarity(cfreq, wfreq, agg_vocab)
        log.info(f"  JSD     = {agg_jsd:.6f}")
        log.info(f"  Cosine  = {agg_cos:.6f}")
        log.info(f"  Tokens  : {n_c:,} client  {n_w:,} worker")
        agg_row = {"comparison": "b2b_vs_b2w", "jsd": round(agg_jsd, 6),
                   "cosine_sim": round(agg_cos, 6),
                   "n_client_tokens": n_c, "n_worker_tokens": n_w}

    # --- Term exclusivity (filtered domains) ---
    log.info("-" * 60)
    log.info("TERM EXCLUSIVITY (qualified domains only)")
    log.info("-" * 60)
    exclusivity = compute_term_exclusivity(platform)
    cats = Counter(r["category"] for r in exclusivity)
    for cat, n in cats.most_common():
        log.info(f"  {cat:<20} {n:>6} terms")

    for label, filt in [("CLIENT-exclusive", lambda r: r["category"] == "client_exclusive"),
                        ("WORKER-exclusive", lambda r: r["category"] == "worker_exclusive"),
                        ("SHARED (top by freq)", lambda r: r["category"] == "shared")]:
        subset = sorted([r for r in exclusivity if filt(r)],
                        key=lambda r: r["total_freq"], reverse=True)
        log.info(f"  Top {label}:")
        for r in subset[:10]:
            log.info(f"    {r['term']:<30} excl={r['exclusivity_index']:>+.3f}  "
                     f"c={r['prevalence_client']:.2f} w={r['prevalence_worker']:.2f}  "
                     f"freq={r['total_freq']}")

    # --- Save ---
    log.info("-" * 60)
    log.info("Saving to database...")
    conn.executemany("""
        INSERT INTO distinctiveness_matrix
            (domain_a, audience_a, domain_b, audience_b, jsd, cosine_sim)
        VALUES (:domain_a, :audience_a, :domain_b, :audience_b, :jsd, :cosine_sim)
    """, matrix_rows)
    if agg_row:
        conn.execute("""
            INSERT INTO aggregate_distance
                (comparison, jsd, cosine_sim, n_client_tokens, n_worker_tokens)
            VALUES (:comparison, :jsd, :cosine_sim, :n_client_tokens, :n_worker_tokens)
        """, agg_row)
    conn.executemany("""
        INSERT INTO term_exclusivity
            (term, term_type, prevalence_client, prevalence_worker,
             exclusivity_index, category, total_freq)
        VALUES (:term, :term_type, :prevalence_client, :prevalence_worker,
                :exclusivity_index, :category, :total_freq)
    """, exclusivity)
    conn.commit()
    log.info(f"  distinctiveness_matrix : {len(matrix_rows):,} rows")
    log.info(f"  aggregate_distance     : 1 row")
    log.info(f"  term_exclusivity       : {len(exclusivity):,} rows")
    log.info(f"  domain_quality         : {len(quality_rows)} rows")

    log.info("=" * 60)
    log.info("COMPLETE — key queries:")
    log.info("  SELECT * FROM domain_quality WHERE included_in_matrix=0;")
    log.info("  SELECT * FROM aggregate_distance;")
    log.info("  SELECT term, exclusivity_index, prevalence_client, prevalence_worker")
    log.info("    FROM term_exclusivity WHERE category='shared'")
    log.info("    ORDER BY total_freq DESC LIMIT 30;")
    log.info("=" * 60)
    conn.close()


if __name__ == "__main__":
    main()
