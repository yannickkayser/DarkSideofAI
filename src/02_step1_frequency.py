"""
02_step1_frequency.py
=====================
Nelson (2020) Step 1: Pattern Detection

Computes keyness and co-occurrence statistics comparing B2B (client)
vs B2W (worker) language. Runs at two levels simultaneously:

  1. Cross-platform  — all client pages vs all worker pages
  2. Within-pair     — paired domains of the same company
                       (e.g. appen.com vs crowdgen.com)

Outputs written to three SQLite tables:
  - keyness_results       : every term ranked by log-likelihood G²
  - cooccurrence_results  : PMI collocate profiles for top 50 terms
  - platform_term_counts  : per-domain term frequencies for pair analysis

Prerequisites:
  - 01_prepare.py must have been run (corpus_view must exist)
  - 01_prepare_additions.py should have been run to populate
    excluded_pages and excluded_terms; if those tables are absent the
    script degrades gracefully (logs a warning and proceeds unfiltered).

Exclusion filtering (applied at corpus-load time, not display time):
  - Pages listed in excluded_pages are skipped entirely
  - Terms listed in excluded_terms are removed from every page's
    unigrams and bigrams before any frequency or PMI counting
  - THEORY_FOCUS_TERMS are never excluded even if they appear in
    excluded_terms (belt-and-suspenders guard)

Usage:
    python3 src/02_step1_frequency.py
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
DB_PATH        = "data/scraping.db"
TOP_N_COOC     = 150     # compute co-occurrence profiles for top N key terms by LL
MIN_TERM_FREQ  = 5      # ignore terms appearing fewer than N times total
WINDOW_SIZE    = 5      # ±5 token window for co-occurrence
MIN_PMI_COFREQ = 10      # minimum co-occurrence count to compute PMI

# Theoretically motivated terms to include in co-occurrence analysis
# regardless of their LL rank. These are terms central to H1a-c that may
# appear on BOTH sides of the corpus (low keyness) but whose collocate
# profiles are analytically important for Step 2 close reading.
# H1a — labour visibility:         worker, labour, task, job, pay, earn
# H1b — automation myth:           autonomous, machine, automate, intelligent
# H1c — strategic hypervisibility: human, quality, oversight, annotation, label
THEORY_FOCUS_TERMS = {
    # H1a
    "worker", "labour", "task", "job", "pay", "earn",
    # H1b
    "autonomous", "machine", "automate", "intelligent", "automation",
    # H1c
    "human", "quality", "oversight", "annotation", "label",
}

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
    """Create result tables, dropping old ones so re-runs are clean."""
    conn.executescript("""
        DROP TABLE IF EXISTS keyness_results;
        DROP TABLE IF EXISTS cooccurrence_results;
        DROP TABLE IF EXISTS platform_term_counts;

        CREATE TABLE keyness_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison      TEXT NOT NULL,  -- 'cross_platform' or company_id for pairs
            term            TEXT NOT NULL,
            term_type       TEXT NOT NULL,  -- 'unigram' or 'bigram'
            ll_score        REAL NOT NULL,  -- positive = client-distinctive
            freq_client     INTEGER,
            freq_worker     INTEGER,
            rel_freq_client REAL,           -- per 1000 tokens
            rel_freq_worker REAL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE cooccurrence_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison  TEXT NOT NULL,
            audience    TEXT NOT NULL,      -- 'client' or 'worker'
            focus_term  TEXT NOT NULL,      -- the key term being profiled
            collocate   TEXT NOT NULL,      -- the co-occurring term
            pmi         REAL NOT NULL,
            cofreq      INTEGER NOT NULL,   -- raw co-occurrence count
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE platform_term_counts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            domain      TEXT NOT NULL,
            audience    TEXT NOT NULL,
            term        TEXT NOT NULL,
            term_type   TEXT NOT NULL,
            freq        INTEGER NOT NULL,
            rel_freq    REAL NOT NULL,      -- per 1000 tokens
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_keyness_comparison
            ON keyness_results(comparison, ll_score);
        CREATE INDEX IF NOT EXISTS idx_cooc_focus
            ON cooccurrence_results(comparison, audience, focus_term);
        CREATE INDEX IF NOT EXISTS idx_ptc_domain
            ON platform_term_counts(domain, term);
    """)
    conn.commit()
    log.info("Output tables created.")


# ---------------------------------------------------------------------------
# Exclusion loading (mirrors pattern in 02b / 02c)
# ---------------------------------------------------------------------------

def load_exclusions(conn: sqlite3.Connection) -> tuple[set[int], set[str]]:
    """
    Load excluded page IDs and excluded terms from the DB tables created
    by 01_prepare_additions.py.

    Returns:
        excluded_page_ids : set of page_id integers to skip
        excluded_terms    : set of term strings to remove from token lists

    Gracefully returns empty sets if the tables do not exist yet, so the
    script can still run (without cleaning) even if 01_prepare_additions.py
    has not been run.  A warning is logged in that case.
    """
    # Check whether the exclusion tables exist
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    excluded_page_ids: set[int] = set()
    excluded_terms: set[str]    = set()

    if "excluded_pages" not in tables or "excluded_terms" not in tables:
        log.warning(
            "excluded_pages / excluded_terms tables not found. "
            "Run 01_prepare_additions.py first for clean-corpus analysis. "
            "Proceeding without exclusion filtering."
        )
        return excluded_page_ids, excluded_terms

    excluded_page_ids = {
        r[0] for r in conn.execute("SELECT page_id FROM excluded_pages").fetchall()
    }
    # Never exclude theory-focus terms even if they ended up in excluded_terms
    raw_excluded_terms = {
        r[0] for r in conn.execute("SELECT term FROM excluded_terms").fetchall()
    }
    excluded_terms = raw_excluded_terms - THEORY_FOCUS_TERMS

    log.info(
        f"Loaded exclusions: {len(excluded_page_ids)} pages, "
        f"{len(excluded_terms)} terms "
        f"({len(raw_excluded_terms - excluded_terms)} theory-focus terms protected)."
    )
    return excluded_page_ids, excluded_terms


# ---------------------------------------------------------------------------
# Step 1: Load corpus from corpus_view
# ---------------------------------------------------------------------------

def load_corpus(conn: sqlite3.Connection,
                excluded_page_ids: set[int],
                excluded_terms: set[str]) -> dict:
    """
    Load token lists from corpus_view, applying exclusion filters.

    Pages listed in excluded_page_ids are skipped entirely.
    Terms listed in excluded_terms are removed from each page's
    unigram and bigram lists before any downstream counting.

    Returns a dict with:
      'cross': {'client': [[tokens], ...], 'worker': [[tokens], ...]}
      'pairs': {company_id: {'client': [...], 'worker': [...], 'domains': {...}}}
      'platform': {domain: {'audience': str, 'pages': [[tokens], ...]}}
    """
    log.info("Loading corpus from corpus_view...")

    rows = conn.execute("""
        SELECT page_id, audience, company_id, domain,
               unigrams, bigrams, token_count
        FROM corpus_view
        WHERE audience IN ('client', 'worker')
          AND token_count >= 10
    """).fetchall()

    log.info(f"  Raw rows fetched: {len(rows)}")

    # Apply page-level exclusions
    rows = [r for r in rows if r["page_id"] not in excluded_page_ids]
    log.info(f"  After page exclusions: {len(rows)} pages.")

    cross   = defaultdict(list)   # audience → list of token lists
    pairs   = defaultdict(lambda: defaultdict(list))  # company_id → audience → pages
    pair_domains = defaultdict(dict)
    platform = defaultdict(lambda: {"audience": None, "pages": []})

    for row in rows:
        audience   = row["audience"]
        company_id = row["company_id"]
        domain     = row["domain"]

        # Parse token lists
        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        bigrams  = json.loads(row["bigrams"])  if row["bigrams"]  else []

        # Apply term-level exclusions: strip artifact / non-English / boilerplate terms
        unigrams = [t for t in unigrams if t not in excluded_terms]
        bigrams  = [t for t in bigrams  if t not in excluded_terms]

        tokens = unigrams + bigrams

        # Cross-platform: pool all pages by audience
        cross[audience].append(tokens)

        # Within-pair: only include pages from companies with both audiences
        pairs[company_id][audience].append(tokens)
        pair_domains[company_id][audience] = domain

        # Platform-level: per domain
        platform[domain]["audience"] = audience
        platform[domain]["pages"].append(tokens)

    # Filter pairs: keep only company_ids that have BOTH client and worker pages
    valid_pairs = {
        cid: data for cid, data in pairs.items()
        if "client" in data and "worker" in data
    }

    log.info(f"  Cross-platform: {len(cross['client'])} client pages, "
             f"{len(cross['worker'])} worker pages")
    log.info(f"  Valid pairs: {list(valid_pairs.keys())}")
    log.info(f"  Platforms: {len(platform)} domains")

    return {
        "cross":        dict(cross),
        "pairs":        valid_pairs,
        "pair_domains": dict(pair_domains),
        "platform":     dict(platform),
    }


# ---------------------------------------------------------------------------
# Step 2: Keyness analysis (log-likelihood G²)
# ---------------------------------------------------------------------------

def build_freq_table(pages: list[list[str]]) -> tuple[Counter, int]:
    """
    Count term frequencies across all pages in a subcorpus.
    Returns (counter, total_token_count).
    Unigrams and bigrams are already in the combined token list.
    """
    counter     = Counter()
    total_tokens = 0
    for tokens in pages:
        # Count unigrams and bigrams separately for rel_freq normalisation
        unigrams = [t for t in tokens if "_" not in t]
        bigrams  = [t for t in tokens if "_" in t]
        counter.update(unigrams)
        counter.update(bigrams)
        total_tokens += len(unigrams)   # token count based on unigrams only
    return counter, total_tokens


def log_likelihood(o1: int, o2: int, n1: int, n2: int) -> float:
    """
    Compute log-likelihood ratio G² for a single term.
    o1, o2 = observed counts in subcorpus 1 and 2
    n1, n2 = total tokens in subcorpus 1 and 2
    Returns signed G²: positive = overrepresented in subcorpus 1 (client).
    """
    n   = n1 + n2
    o   = o1 + o2
    e1  = n1 * o / n
    e2  = n2 * o / n

    def safe_log(observed, expected):
        if observed == 0 or expected == 0:
            return 0.0
        return observed * math.log(observed / expected)

    g2 = 2 * (safe_log(o1, e1) + safe_log(o2, e2))

    # Sign: positive means overrepresented in subcorpus 1 (client)
    return g2 if o1 / n1 >= o2 / n2 else -g2


def compute_keyness(
    client_pages: list[list[str]],
    worker_pages: list[list[str]],
    comparison_label: str,
) -> list[dict]:
    """
    Compute log-likelihood keyness for all terms.
    Returns list of result dicts ready for DB insertion.
    """
    log.info(f"  Computing keyness for '{comparison_label}'...")

    client_freq, n_client = build_freq_table(client_pages)
    worker_freq, n_worker = build_freq_table(worker_pages)

    log.info(f"    Client: {n_client:,} tokens, {len(client_freq):,} unique terms")
    log.info(f"    Worker: {n_worker:,} tokens, {len(worker_freq):,} unique terms")

    all_terms = set(client_freq.keys()) | set(worker_freq.keys())
    results   = []

    for term in all_terms:
        o1 = client_freq.get(term, 0)
        o2 = worker_freq.get(term, 0)

        # Skip rare terms
        if o1 + o2 < MIN_TERM_FREQ:
            continue

        ll = log_likelihood(o1, o2, n_client, n_worker)

        results.append({
            "comparison":      comparison_label,
            "term":            term,
            "term_type":       "bigram" if "_" in term else "unigram",
            "ll_score":        round(ll, 4),
            "freq_client":     o1,
            "freq_worker":     o2,
            "rel_freq_client": round(1000 * o1 / n_client, 4) if n_client else 0,
            "rel_freq_worker": round(1000 * o2 / n_worker, 4) if n_worker else 0,
        })

    # Sort by absolute LL descending
    results.sort(key=lambda x: abs(x["ll_score"]), reverse=True)
    log.info(f"    {len(results):,} terms above min_freq threshold.")
    return results


# ---------------------------------------------------------------------------
# Step 3: Co-occurrence analysis (PMI)
# ---------------------------------------------------------------------------

def build_cooccurrence_index(
    pages: list[list[str]],
    focus_terms: set[str],
    window: int = WINDOW_SIZE,
) -> tuple[Counter, Counter, int]:
    """
    For each focus term, collect co-occurring terms within ±window tokens.
    Returns:
      cofreq    : Counter of (focus_term, collocate) pairs
      term_freq : Counter of all individual term frequencies
      total     : total tokens processed
    """
    cofreq    = Counter()
    term_freq = Counter()
    total     = 0

    for tokens in pages:
        # Work only on unigrams for co-occurrence to keep windows meaningful
        unigrams = [t for t in tokens if "_" not in t]
        term_freq.update(unigrams)
        total += len(unigrams)

        for i, token in enumerate(unigrams):
            if token not in focus_terms:
                continue
            start = max(0, i - window)
            end   = min(len(unigrams), i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                cofreq[(token, unigrams[j])] += 1

    return cofreq, term_freq, total


def compute_pmi(
    cofreq: Counter,
    term_freq: Counter,
    total: int,
    focus_terms: set[str],
    audience: str,
    comparison_label: str,
) -> list[dict]:
    """
    Compute PMI for all (focus_term, collocate) pairs.
    PMI = log2( P(x,y) / (P(x) * P(y)) )
    """
    results = []

    for (focus, collocate), cf in cofreq.items():
        if focus not in focus_terms:
            continue
        if cf < MIN_PMI_COFREQ:
            continue

        p_joint   = cf / total
        p_focus   = term_freq[focus]   / total
        p_coloc   = term_freq[collocate] / total

        if p_focus == 0 or p_coloc == 0:
            continue

        pmi = math.log2(p_joint / (p_focus * p_coloc))

        results.append({
            "comparison": comparison_label,
            "audience":   audience,
            "focus_term": focus,
            "collocate":  collocate,
            "pmi":        round(pmi, 4),
            "cofreq":     cf,
        })

    return results


def compute_cooccurrence_for_top_terms(
    client_pages: list[list[str]],
    worker_pages: list[list[str]],
    keyness_results: list[dict],
    comparison_label: str,
) -> list[dict]:
    """
    Compute PMI co-occurrence profiles for the top N key terms.
    Profiles are computed separately for client and worker subcorpora.
    """
    # Extract top N terms by absolute LL score
    ll_terms = set(
        r["term"] for r in keyness_results[:TOP_N_COOC]
        if r["term_type"] == "unigram"   # co-occurrence on unigrams only
    )

    # Add theoretically motivated terms regardless of LL rank.
    # These are terms central to H1a-c whose collocate profiles matter for
    # Step 2 even if they are not strongly key (i.e. they appear on both sides).
    # Only include terms that actually appear in the corpus (present in keyness).
    corpus_terms = set(r["term"] for r in keyness_results if r["term_type"] == "unigram")
    theory_terms = THEORY_FOCUS_TERMS & corpus_terms
    missing = THEORY_FOCUS_TERMS - corpus_terms
    if missing:
        log.warning(f"  Theory terms absent from corpus: {missing}")

    top_terms = ll_terms | theory_terms
    log.info(f"  Computing co-occurrence for {len(top_terms)} focus terms "
             f"({len(ll_terms)} by LL + {len(theory_terms)} theory-driven)...")

    all_results = []

    for audience, pages in [("client", client_pages), ("worker", worker_pages)]:
        cofreq, term_freq, total = build_cooccurrence_index(pages, top_terms)
        pmi_results = compute_pmi(
            cofreq, term_freq, total, top_terms, audience, comparison_label
        )
        all_results.extend(pmi_results)
        log.info(f"    {audience}: {len(pmi_results)} PMI pairs computed.")

    return all_results


# ---------------------------------------------------------------------------
# Step 4: Platform-level term counts
# ---------------------------------------------------------------------------

def compute_platform_term_counts(platform: dict) -> list[dict]:
    """
    Compute per-domain term frequencies for within-pair comparisons.
    Enables: appen.com (client) vs crowdgen.com (worker) term-by-term.
    """
    log.info("  Computing platform-level term counts...")
    results = []

    for domain, data in platform.items():
        audience = data["audience"]
        pages    = data["pages"]

        freq, n_tokens = build_freq_table(pages)

        for term, count in freq.items():
            if count < 2:   # minor filter — platform counts can be sparse
                continue
            results.append({
                "domain":    domain,
                "audience":  audience,
                "term":      term,
                "term_type": "bigram" if "_" in term else "unigram",
                "freq":      count,
                "rel_freq":  round(1000 * count / n_tokens, 4) if n_tokens else 0,
            })

    log.info(f"  {len(results):,} platform-term rows computed.")
    return results


# ---------------------------------------------------------------------------
# Step 5: Save to database
# ---------------------------------------------------------------------------

def save_keyness(conn: sqlite3.Connection, results: list[dict]):
    conn.executemany("""
        INSERT INTO keyness_results
            (comparison, term, term_type, ll_score,
             freq_client, freq_worker, rel_freq_client, rel_freq_worker)
        VALUES
            (:comparison, :term, :term_type, :ll_score,
             :freq_client, :freq_worker, :rel_freq_client, :rel_freq_worker)
    """, results)
    conn.commit()


def save_cooccurrence(conn: sqlite3.Connection, results: list[dict]):
    conn.executemany("""
        INSERT INTO cooccurrence_results
            (comparison, audience, focus_term, collocate, pmi, cofreq)
        VALUES
            (:comparison, :audience, :focus_term, :collocate, :pmi, :cofreq)
    """, results)
    conn.commit()


def save_platform_counts(conn: sqlite3.Connection, results: list[dict]):
    conn.executemany("""
        INSERT INTO platform_term_counts
            (domain, audience, term, term_type, freq, rel_freq)
        VALUES
            (:domain, :audience, :term, :term_type, :freq, :rel_freq)
    """, results)
    conn.commit()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_top_terms(results: list[dict], n: int = 20):
    """Print the most distinctive terms in each direction."""
    client_top = [r for r in results if r["ll_score"] > 0][:n]
    worker_top = [r for r in results if r["ll_score"] < 0][:n]

    log.info(f"  Top {n} CLIENT-distinctive terms:")
    for r in client_top:
        log.info(f"    {r['term']:<30} LL={r['ll_score']:>10.2f}  "
                 f"client={r['rel_freq_client']:.2f}‰  "
                 f"worker={r['rel_freq_worker']:.2f}‰")

    log.info(f"  Top {n} WORKER-distinctive terms:")
    for r in worker_top:
        log.info(f"    {r['term']:<30} LL={r['ll_score']:>10.2f}  "
                 f"client={r['rel_freq_client']:.2f}‰  "
                 f"worker={r['rel_freq_worker']:.2f}‰")


def log_top_cooccurrences(results: list[dict], focus_term: str,
                          audience: str, n: int = 10):
    """Print top collocates for a specific focus term and audience."""
    filtered = [
        r for r in results
        if r["focus_term"] == focus_term and r["audience"] == audience
    ]
    filtered.sort(key=lambda x: x["pmi"], reverse=True)
    log.info(f"  Top collocates for '{focus_term}' ({audience}):")
    for r in filtered[:n]:
        log.info(f"    {r['collocate']:<25} PMI={r['pmi']:>6.3f}  freq={r['cofreq']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("02_step1_frequency.py — Keyness and Co-occurrence Analysis")
    log.info("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Verify corpus_view exists
    view_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()
    if not view_check:
        raise RuntimeError("corpus_view not found — run 01_prepare.py first.")

    init_output_tables(conn)

    # --- Load exclusion lists (from 01_prepare_additions.py) ---
    excluded_page_ids, excluded_terms = load_exclusions(conn)

    # --- Load corpus (with exclusions applied at token level) ---
    corpus = load_corpus(conn, excluded_page_ids, excluded_terms)

    all_keyness     = []
    all_cooccurrence = []

    # -----------------------------------------------------------------------
    # Analysis 1: Cross-platform (all client vs all worker)
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("CROSS-PLATFORM COMPARISON")
    log.info("-" * 60)

    client_pages = corpus["cross"].get("client", [])
    worker_pages = corpus["cross"].get("worker", [])

    cross_keyness = compute_keyness(client_pages, worker_pages, "cross_platform")
    log_top_terms(cross_keyness)
    all_keyness.extend(cross_keyness)

    cross_cooc = compute_cooccurrence_for_top_terms(
        client_pages, worker_pages, cross_keyness, "cross_platform"
    )
    # Log co-occurrence profiles for a few theoretically important terms
    for term in ["human", "worker", "quality", "automate", "skill"]:
        if any(r["focus_term"] == term for r in cross_cooc):
            log_top_cooccurrences(cross_cooc, term, "client")
            log_top_cooccurrences(cross_cooc, term, "worker")
    all_cooccurrence.extend(cross_cooc)

    # -----------------------------------------------------------------------
    # Analysis 2: Within-pair comparisons
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("WITHIN-PAIR COMPARISONS")
    log.info("-" * 60)

    for company_id, data in corpus["pairs"].items():
        log.info(f"  Pair: {company_id}")
        c_pages = data.get("client", [])
        w_pages = data.get("worker", [])

        if len(c_pages) < 5 or len(w_pages) < 5:
            log.warning(f"  Skipping {company_id} — too few pages "
                        f"(client={len(c_pages)}, worker={len(w_pages)})")
            continue

        pair_keyness = compute_keyness(c_pages, w_pages, company_id)
        log_top_terms(pair_keyness, n=10)
        all_keyness.extend(pair_keyness)

        pair_cooc = compute_cooccurrence_for_top_terms(
            c_pages, w_pages, pair_keyness, company_id
        )
        all_cooccurrence.extend(pair_cooc)

    # -----------------------------------------------------------------------
    # Analysis 3: Platform-level term counts
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("PLATFORM-LEVEL TERM COUNTS")
    log.info("-" * 60)

    platform_counts = compute_platform_term_counts(corpus["platform"])

    # -----------------------------------------------------------------------
    # Save everything
    # -----------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Saving results to database...")

    save_keyness(conn, all_keyness)
    log.info(f"  keyness_results      : {len(all_keyness):,} rows")

    save_cooccurrence(conn, all_cooccurrence)
    log.info(f"  cooccurrence_results : {len(all_cooccurrence):,} rows")

    save_platform_counts(conn, platform_counts)
    log.info(f"  platform_term_counts : {len(platform_counts):,} rows")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 1 COMPLETE")
    log.info(f"  Corpus filtered: {len(excluded_page_ids)} pages excluded, "
             f"{len(excluded_terms)} terms excluded.")
    log.info("Outputs are in your database. Query examples:")
    log.info("  -- Top client-distinctive terms (cross-platform):")
    log.info("  SELECT term, ll_score, rel_freq_client, rel_freq_worker")
    log.info("  FROM keyness_results")
    log.info("  WHERE comparison='cross_platform' AND ll_score > 0")
    log.info("  ORDER BY ll_score DESC LIMIT 50;")
    log.info("")
    log.info("  -- Co-occurrence profile for 'human' in client texts:")
    log.info("  SELECT collocate, pmi, cofreq")
    log.info("  FROM cooccurrence_results")
    log.info("  WHERE comparison='cross_platform'")
    log.info("    AND audience='client' AND focus_term='human'")
    log.info("  ORDER BY pmi DESC LIMIT 20;")
    log.info("")
    log.info("  -- Within-pair keyness for appen:")
    log.info("  SELECT term, ll_score, rel_freq_client, rel_freq_worker")
    log.info("  FROM keyness_results")
    log.info("  WHERE comparison='appen'")
    log.info("  ORDER BY ll_score DESC LIMIT 30;")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
