"""
02_step1_analysis.py
====================
Nelson (2020) Step 1: Pattern Detection — all statistical analyses.

Merges src/02_step1_frequency.py and src/02b_step1_distinctiveness.py
into a single script.  Both scripts loaded the same corpus, computed
statistics from the same token data, and wrote to the same database.
There was no reason to keep them separate.

What this script computes:

  A. Keyness (log-likelihood G²)
       Which terms are statistically overrepresented in client-facing
       vs worker-facing texts?  Computed at two levels:
         — cross-platform: all client pages vs all worker pages
         — within-pair: paired domains of the same company

  B. Co-occurrence profiles (PMI)
       For each theoretically motivated and top-LL term, what words
       does it typically appear with?  Computed within ±WINDOW_SIZE
       tokens of each sentence (not across the full page).

  C. Platform-level term counts
       Per-domain relative frequencies enabling within-pair comparisons
       (e.g. appen.com vs crowdgen.com term-by-term).

  D. Distinctiveness matrix (JSD + cosine similarity)
       How linguistically similar or different is each platform from
       every other?  Computed over the high-variance vocabulary.
       Uses a domain quality filter (MIN_PAGES_PER_DOMAIN).

  E. Aggregate B2B vs B2W distance
       Single JSD and cosine score summarising overall vocabulary
       distance between the client and worker subcorpora.

  F. Term exclusivity scores
       For each term: on what fraction of client platforms does it
       appear vs worker platforms?  Exclusivity index = difference.

Fixes vs src/:
  — WINDOW_SIZE = 15 (was 5; corrected to match methodology description)
  — Co-occurrence window is sentence-scoped: tokens from different
    sentences are never paired.  This eliminates spurious co-occurrences
    that arose when the window slid across concatenated page sections.
  — Single corpus load shared by all analyses (no redundant DB reads).

Prerequisites:
  01_prepare_corpus.py must have been run:
    — corpus_view must exist (includes segments column)
    — excluded_pages and excluded_terms tables must exist

Usage:
    python3 src2/02_step1_analysis.py
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

DB_PATH                  = "data/scraping_2.db"
TOP_N_COOC               = 150    # co-occurrence profiles for top N terms by LL
MIN_TERM_FREQ            = 5      # ignore terms below this corpus-wide frequency
WINDOW_SIZE              = 5     # ±15 token window (within-sentence only)
MIN_PMI_COFREQ           = 10     # minimum co-occurrence count to store PMI pair
MIN_PAGES_PER_DOMAIN     = 5      # domain quality gate for platform-level analyses
EXCLUSIVITY_THRESHOLD    = 0.70   # prevalence fraction to classify as exclusive
SHARED_BAND              = 0.25   # exclusivity index ≤ this → "shared"
HIGH_VARIANCE_PERCENTILE = 50     # percentile cut for vocabulary used in JSD/cosine

# Theoretically motivated terms always included in co-occurrence analysis
# regardless of LL rank.  Mirror of PROTECTED_TERMS in 01_prepare_corpus.py.
# H1a — Labour visibility gap
# H1b — Automation myth
# H1c — Strategic hypervisibility
THEORY_FOCUS_TERMS = {
    "worker", "labour", "task", "job", "pay", "earn",
    "autonomous", "machine", "automate", "intelligent", "automation",
    "human", "quality", "oversight", "annotation", "label",
    # Additional terms for H2–H4
    "flexible", "flexibility", "freedom", "autonomy",
    "talent", "resource", "contributor", "workforce",
    "community", "collective", "score", "rating", "rank",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# SECTION A — Database initialisation
# ===========================================================================

def init_output_tables(conn: sqlite3.Connection):
    """
    Create all output tables, dropping previous versions for clean re-runs.

    Tables created:
      keyness_results        — LL scores for all terms (cross + pairs)
      cooccurrence_results   — PMI collocate profiles for focus terms
      platform_term_counts   — per-domain term frequencies for pair analysis
      distinctiveness_matrix — pairwise JSD + cosine between qualified domains
      aggregate_distance     — single B2B-vs-B2W JSD + cosine summary
      term_exclusivity       — prevalence-based exclusivity per term
      domain_quality         — audit log for the domain quality filter
    """
    conn.executescript("""
        DROP TABLE IF EXISTS keyness_results;
        DROP TABLE IF EXISTS cooccurrence_results;
        DROP TABLE IF EXISTS platform_term_counts;
        DROP TABLE IF EXISTS distinctiveness_matrix;
        DROP TABLE IF EXISTS aggregate_distance;
        DROP TABLE IF EXISTS term_exclusivity;
        DROP TABLE IF EXISTS domain_quality;

        CREATE TABLE keyness_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison      TEXT NOT NULL,
            term            TEXT NOT NULL,
            term_type       TEXT NOT NULL,
            ll_score        REAL NOT NULL,
            freq_client     INTEGER,
            freq_worker     INTEGER,
            rel_freq_client REAL,
            rel_freq_worker REAL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE cooccurrence_results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison  TEXT NOT NULL,
            audience    TEXT NOT NULL,
            focus_term  TEXT NOT NULL,
            collocate   TEXT NOT NULL,
            pmi         REAL NOT NULL,
            cofreq      INTEGER NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE platform_term_counts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            domain      TEXT NOT NULL,
            audience    TEXT NOT NULL,
            term        TEXT NOT NULL,
            term_type   TEXT NOT NULL,
            freq        INTEGER NOT NULL,
            rel_freq    REAL NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE distinctiveness_matrix (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            domain_a    TEXT NOT NULL,
            audience_a  TEXT,
            domain_b    TEXT NOT NULL,
            audience_b  TEXT,
            jsd         REAL NOT NULL,
            cosine_sim  REAL NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE aggregate_distance (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            comparison       TEXT NOT NULL,
            jsd              REAL NOT NULL,
            cosine_sim       REAL NOT NULL,
            n_client_tokens  INTEGER,
            n_worker_tokens  INTEGER,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            included_in_matrix   INTEGER NOT NULL,
            exclusion_reason     TEXT,
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_keyness_cmp   ON keyness_results(comparison, ll_score);
        CREATE INDEX IF NOT EXISTS idx_cooc_focus    ON cooccurrence_results(comparison, audience, focus_term);
        CREATE INDEX IF NOT EXISTS idx_ptc_domain    ON platform_term_counts(domain, term);
        CREATE INDEX IF NOT EXISTS idx_dm_pair       ON distinctiveness_matrix(domain_a, domain_b);
        CREATE INDEX IF NOT EXISTS idx_te_cat        ON term_exclusivity(category, exclusivity_index);
        CREATE INDEX IF NOT EXISTS idx_dq_domain     ON domain_quality(domain);
    """)
    conn.commit()
    log.info("Output tables created.")


# ===========================================================================
# SECTION B — Exclusion loading
# ===========================================================================

def load_exclusions(conn: sqlite3.Connection) -> tuple[set[int], set[str]]:
    """
    Load excluded page IDs and excluded terms from DB.

    Gracefully returns empty sets if 01_prepare_corpus.py has not been
    run yet (logs a warning).
    """
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    excluded_page_ids: set[int] = set()
    excluded_terms:    set[str] = set()

    if "excluded_pages" not in tables or "excluded_terms" not in tables:
        log.warning(
            "excluded_pages / excluded_terms not found — "
            "run 01_prepare_corpus.py first.  Proceeding unfiltered."
        )
        return excluded_page_ids, excluded_terms

    excluded_page_ids = {
        r[0] for r in conn.execute("SELECT page_id FROM excluded_pages").fetchall()
    }
    raw_excluded_terms = {
        r[0] for r in conn.execute("SELECT term FROM excluded_terms").fetchall()
    }
    # Never exclude theory-focus terms even if they appear in excluded_terms
    excluded_terms = raw_excluded_terms - THEORY_FOCUS_TERMS
    protected_count = len(raw_excluded_terms) - len(excluded_terms)

    log.info(
        f"Exclusions loaded: {len(excluded_page_ids)} pages, "
        f"{len(excluded_terms)} terms "
        f"({protected_count} theory-focus terms protected)."
    )
    return excluded_page_ids, excluded_terms


# ===========================================================================
# SECTION C — Corpus loading
# ===========================================================================

def load_corpus(conn: sqlite3.Connection,
                excluded_page_ids: set[int],
                excluded_terms: set[str]) -> dict:
    """
    Load the full corpus from corpus_view.

    Returns a dict with four data structures, all derived from a single
    DB read for efficiency:

      cross
        {'client': [page_data, ...], 'worker': [page_data, ...]}
        Each page_data = {'segments': [[tok,...], ...], 'tokens': [tok,...]}
        'segments' is the per-sentence token lists (for co-occurrence).
        'tokens'   is the flat unigram+bigram list (for keyness/freq).

      pairs
        {company_id: {'client': [page_data,...], 'worker': [page_data,...]}}
        Only includes companies that have both client and worker pages.

      platform
        {domain: {'audience': str, 'freq': Counter, 'n_tokens': int,
                  'n_pages': int}}
        Per-domain frequency statistics for distinctiveness analysis.
        Includes ALL domains (quality filter applied later).

      pair_domains
        {company_id: {'client': domain, 'worker': domain}}
    """
    log.info("Loading corpus from corpus_view...")

    rows = conn.execute("""
        SELECT page_id, audience, company_id, domain,
               segments, unigrams, bigrams, token_count
        FROM corpus_view
        WHERE audience IN ('client', 'worker')
          AND token_count >= 10
    """).fetchall()

    log.info(f"  {len(rows)} rows fetched from corpus_view.")

    # Apply page-level exclusions
    rows = [r for r in rows if r["page_id"] not in excluded_page_ids]
    log.info(f"  {len(rows)} pages after exclusion filtering.")

    cross        = defaultdict(list)
    pairs        = defaultdict(lambda: defaultdict(list))
    pair_domains = defaultdict(dict)
    platform     = defaultdict(lambda: {"audience": None, "freq": Counter(),
                                        "n_tokens": 0, "n_pages": 0})

    for row in rows:
        audience   = row["audience"]
        company_id = row["company_id"]
        domain     = row["domain"]

        # Load token data
        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        bigrams  = json.loads(row["bigrams"])  if row["bigrams"]  else []
        raw_segs = json.loads(row["segments"]) if row["segments"] else []

        # Apply term-level exclusions to unigrams and bigrams
        if excluded_terms:
            unigrams = [t for t in unigrams if t not in excluded_terms]
            bigrams  = [t for t in bigrams  if t not in excluded_terms]
            raw_segs = [[t for t in seg if t not in excluded_terms]
                        for seg in raw_segs]
            raw_segs = [seg for seg in raw_segs if seg]   # drop empty sents

        flat_tokens = unigrams + bigrams   # used by keyness and freq counts

        page_data = {
            "segments": raw_segs,     # [[sent_toks], [sent_toks], ...]
            "tokens":   flat_tokens,  # [unigrams] + [bigrams]
        }

        # Cross-platform pools all pages by audience
        cross[audience].append(page_data)

        # Within-pair: per company_id
        pairs[company_id][audience].append(page_data)
        pair_domains[company_id][audience] = domain

        # Platform-level: aggregate frequencies per domain
        platform[domain]["audience"]  = audience
        platform[domain]["freq"].update(flat_tokens)
        platform[domain]["n_tokens"] += len(unigrams)    # unigrams only for token count
        platform[domain]["n_pages"]  += 1

    # Discard pairs where only one audience is present
    valid_pairs = {
        cid: data for cid, data in pairs.items()
        if "client" in data and "worker" in data
    }

    log.info(f"  Cross-platform: {len(cross.get('client',[]))} client pages, "
             f"{len(cross.get('worker',[]))} worker pages")
    log.info(f"  Valid pairs: {sorted(valid_pairs.keys())}")
    log.info(f"  Platforms: {len(platform)} domains")

    return {
        "cross":        dict(cross),
        "pairs":        valid_pairs,
        "pair_domains": dict(pair_domains),
        "platform":     dict(platform),
    }


# ===========================================================================
# SECTION D — Keyness analysis (log-likelihood G²)
# ===========================================================================

def build_freq_table(pages: list[dict]) -> tuple[Counter, int]:
    """
    Count term frequencies across all pages.

    Unigrams and bigrams are in each page's 'tokens' list.
    Token count is based on unigrams only (for per-1000-token normalisation).
    """
    counter      = Counter()
    total_tokens = 0
    for page in pages:
        tokens   = page["tokens"]
        unigrams = [t for t in tokens if "_" not in t]
        bigrams  = [t for t in tokens if "_" in t]
        counter.update(unigrams)
        counter.update(bigrams)
        total_tokens += len(unigrams)
    return counter, total_tokens


def log_likelihood(o1: int, o2: int, n1: int, n2: int) -> float:
    """
    Compute signed log-likelihood G² for one term.

    o1, o2 = observed counts in subcorpus 1 (client) and 2 (worker).
    n1, n2 = total tokens in each subcorpus.
    Returns positive G² if overrepresented in client, negative if worker.
    """
    n  = n1 + n2
    o  = o1 + o2
    e1 = n1 * o / n
    e2 = n2 * o / n

    def safe_log(obs, exp):
        return obs * math.log(obs / exp) if obs > 0 and exp > 0 else 0.0

    g2 = 2 * (safe_log(o1, e1) + safe_log(o2, e2))
    return g2 if (o1 / n1) >= (o2 / n2) else -g2


def compute_keyness(client_pages: list[dict],
                    worker_pages: list[dict],
                    comparison_label: str) -> list[dict]:
    """
    Compute LL keyness for all terms above MIN_TERM_FREQ.

    Returns list of result dicts sorted by |LL| descending.
    """
    log.info(f"  Keyness for '{comparison_label}'...")

    client_freq, n_client = build_freq_table(client_pages)
    worker_freq, n_worker = build_freq_table(worker_pages)

    log.info(f"    Client: {n_client:,} tokens, {len(client_freq):,} unique terms")
    log.info(f"    Worker: {n_worker:,} tokens, {len(worker_freq):,} unique terms")

    all_terms = set(client_freq.keys()) | set(worker_freq.keys())
    results   = []

    for term in all_terms:
        o1 = client_freq.get(term, 0)
        o2 = worker_freq.get(term, 0)
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

    results.sort(key=lambda x: abs(x["ll_score"]), reverse=True)
    log.info(f"    {len(results):,} terms above frequency threshold.")
    return results


# ===========================================================================
# SECTION E — Co-occurrence analysis (PMI, sentence-scoped window)
# ===========================================================================

def build_cooccurrence_index(
    pages: list[dict],
    focus_terms: set[str],
    window: int = WINDOW_SIZE,
) -> tuple[Counter, Counter, int]:
    """
    Build co-occurrence index within ±window tokens, sentence-scoped.

    KEY FIX: The original pipeline computed co-occurrence over a flat
    token list for the entire page.  On JS-rendered pages, BeautifulSoup
    concatenates hero text, navigation, feature blocks, and footer into
    one string, so the sliding window produced spurious pairs like
    "work–baby" from tokens in completely unrelated page sections.

    This version iterates over per-sentence token lists (stored in
    page["segments"]).  The window only moves within a single sentence.
    Tokens in different sentences are never paired — they could not have
    co-occurred in a meaningful linguistic context.

    Args:
        pages:        List of page_data dicts (each has 'segments' key).
        focus_terms:  Terms to profile (top LL + theory-focus terms).
        window:       ±token window (default WINDOW_SIZE = 15).

    Returns:
        cofreq     : Counter of (focus_term, collocate) pairs
        term_freq  : Counter of unigram frequencies across all sentences
        total      : total unigrams processed
    """
    cofreq    = Counter()
    term_freq = Counter()
    total     = 0

    for page in pages:
        for segment in page["segments"]:
            # Co-occurrence computed on unigrams only — bigrams would
            # create misleading window spans
            unigrams = [t for t in segment if "_" not in t]
            term_freq.update(unigrams)
            total += len(unigrams)

            for i, token in enumerate(unigrams):
                if token not in focus_terms:
                    continue
                start = max(0, i - window)
                end   = min(len(unigrams), i + window + 1)
                for j in range(start, end):
                    if j != i:
                        cofreq[(token, unigrams[j])] += 1

    return cofreq, term_freq, total


def compute_pmi(cofreq: Counter,
                term_freq: Counter,
                total: int,
                focus_terms: set[str],
                audience: str,
                comparison_label: str) -> list[dict]:
    """
    Compute PMI for all (focus_term, collocate) pairs above MIN_PMI_COFREQ.

    PMI = log2( P(x,y) / (P(x) * P(y)) )
    """
    results = []
    for (focus, collocate), cf in cofreq.items():
        if focus not in focus_terms or cf < MIN_PMI_COFREQ:
            continue
        p_joint = cf / total
        p_focus = term_freq[focus]     / total
        p_coloc = term_freq[collocate] / total
        if p_focus == 0 or p_coloc == 0:
            continue
        results.append({
            "comparison": comparison_label,
            "audience":   audience,
            "focus_term": focus,
            "collocate":  collocate,
            "pmi":        round(math.log2(p_joint / (p_focus * p_coloc)), 4),
            "cofreq":     cf,
        })
    return results


def compute_cooccurrence(client_pages: list[dict],
                         worker_pages: list[dict],
                         keyness_results: list[dict],
                         comparison_label: str) -> list[dict]:
    """
    Compute PMI co-occurrence profiles for the top N key terms and all
    theoretically motivated terms.

    Profiles are computed separately for client and worker subcorpora so
    the discursive neighbourhood of a term can be compared across audiences.
    """
    ll_terms = {
        r["term"] for r in keyness_results[:TOP_N_COOC]
        if r["term_type"] == "unigram"
    }
    corpus_terms = {r["term"] for r in keyness_results if r["term_type"] == "unigram"}
    theory_terms = THEORY_FOCUS_TERMS & corpus_terms
    missing      = THEORY_FOCUS_TERMS - corpus_terms
    if missing:
        log.warning(f"  Theory terms absent from corpus: {missing}")

    focus_terms = ll_terms | theory_terms
    log.info(
        f"  Co-occurrence for {len(focus_terms)} focus terms "
        f"({len(ll_terms)} by LL + {len(theory_terms)} theory-driven)..."
    )

    all_results = []
    for audience, pages in [("client", client_pages), ("worker", worker_pages)]:
        cofreq, term_freq, total = build_cooccurrence_index(pages, focus_terms)
        pmi_results = compute_pmi(
            cofreq, term_freq, total, focus_terms, audience, comparison_label
        )
        all_results.extend(pmi_results)
        log.info(f"    {audience}: {len(pmi_results)} PMI pairs computed.")

    return all_results


# ===========================================================================
# SECTION F — Platform-level term counts
# ===========================================================================

def compute_platform_term_counts(platform: dict) -> list[dict]:
    """
    Per-domain relative frequencies for within-pair comparisons.

    Enables: appen.com (client) vs crowdgen.com (worker) term-by-term.
    """
    results = []
    for domain, data in platform.items():
        audience = data["audience"]
        freq     = data["freq"]
        n_tokens = data["n_tokens"]

        for term, count in freq.items():
            if count < 2:
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


# ===========================================================================
# SECTION G — Distinctiveness (JSD + cosine, high-variance vocabulary)
# ===========================================================================

def select_high_variance_vocab(platform: dict,
                               percentile: int = HIGH_VARIANCE_PERCENTILE) -> set[str]:
    """
    Select the top-percentile terms by variance in relative frequency across
    domains.  Using high-variance vocabulary for JSD and cosine prevents
    ubiquitous function words from dominating the distance measure.
    """
    global_freq: Counter = Counter()
    for data in platform.values():
        global_freq.update(data["freq"])

    all_terms = [t for t, f in global_freq.items() if f >= MIN_TERM_FREQ]
    if not all_terms:
        return set(all_terms)

    # Compute relative frequency per domain for each term
    term_var = {}
    for term in all_terms:
        rel_freqs = []
        for data in platform.values():
            n = data["n_tokens"]
            rel_freqs.append(data["freq"].get(term, 0) / n if n > 0 else 0)
        mean = sum(rel_freqs) / len(rel_freqs)
        term_var[term] = sum((x - mean) ** 2 for x in rel_freqs) / len(rel_freqs)

    # Keep terms above the percentile threshold
    variances = sorted(term_var.values())
    cutoff_idx = int(len(variances) * percentile / 100)
    cutoff_val = variances[cutoff_idx] if cutoff_idx < len(variances) else 0

    selected = {t for t, v in term_var.items() if v >= cutoff_val}
    log.info(
        f"  High-variance vocab: {len(selected)} terms "
        f"(top {100-percentile}th percentile of variance)"
    )
    return selected


def to_prob_dist(freq: Counter, vocab: set[str]) -> dict[str, float]:
    """Laplace-smoothed probability distribution over vocab."""
    alpha = 1
    total = sum(freq.get(t, 0) for t in vocab) + alpha * len(vocab)
    return {t: (freq.get(t, 0) + alpha) / total for t in vocab}


def kl_divergence(p: dict, q: dict) -> float:
    return sum(p[t] * math.log(p[t] / q[t]) for t in p if p[t] > 0 and q[t] > 0)


def jsd_normalised(p: dict, q: dict) -> float:
    """Jensen-Shannon Divergence, normalised to [0, 1]."""
    m = {t: 0.5 * (p[t] + q[t]) for t in p}
    raw = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return raw / math.log(2)


def cosine_similarity(freq_a: Counter,
                      freq_b: Counter,
                      vocab: set[str]) -> float:
    dot   = sum(freq_a.get(t, 0) * freq_b.get(t, 0) for t in vocab)
    mag_a = math.sqrt(sum(freq_a.get(t, 0) ** 2 for t in vocab))
    mag_b = math.sqrt(sum(freq_b.get(t, 0) ** 2 for t in vocab))
    return dot / (mag_a * mag_b) if mag_a > 0 and mag_b > 0 else 0.0


def apply_domain_quality_filter(platform: dict) -> tuple[dict, list[dict]]:
    """
    Apply MIN_PAGES_PER_DOMAIN filter to the platform dict.

    Sparse domains are excluded from PLATFORM-LEVEL analyses (pairwise
    matrix, term exclusivity) but remain in the cross-platform aggregate
    where their marginal token contribution is negligible.

    Returns:
        filtered:      platform dict with only qualified domains.
        quality_rows:  list of domain_quality rows (ALL domains, for audit).
    """
    filtered     = {}
    quality_rows = []

    for d, v in platform.items():
        n        = v["n_pages"]
        included = n >= MIN_PAGES_PER_DOMAIN
        reason   = (
            None if included else
            f"n_pages={n} < MIN_PAGES_PER_DOMAIN={MIN_PAGES_PER_DOMAIN}: "
            f"insufficient pages for reliable vocabulary representation"
        )
        if included:
            filtered[d] = v
        quality_rows.append({
            "domain":             d,
            "audience":           v["audience"],
            "n_pages":            n,
            "n_tokens":           v["n_tokens"],
            "included_in_matrix": int(included),
            "exclusion_reason":   reason,
        })

    excluded_domains = [d for d, v in platform.items()
                        if v["n_pages"] < MIN_PAGES_PER_DOMAIN]
    if excluded_domains:
        log.info(
            f"  Domain quality filter: {len(excluded_domains)} domains excluded "
            f"from platform analyses: {excluded_domains}"
        )
    log.info(f"  Included in platform analyses: {len(filtered)} domains")
    return filtered, quality_rows


def compute_distinctiveness_matrix(platform: dict,
                                   vocab: set[str]) -> list[dict]:
    """
    Compute pairwise JSD and cosine similarity for all qualified domains.
    """
    domains = sorted(platform.keys())
    results = []

    for i, da in enumerate(domains):
        pa = to_prob_dist(platform[da]["freq"], vocab)
        for db in domains[i+1:]:
            pb    = to_prob_dist(platform[db]["freq"], vocab)
            j     = jsd_normalised(pa, pb)
            cos   = cosine_similarity(platform[da]["freq"],
                                      platform[db]["freq"], vocab)
            results.append({
                "domain_a":   da,
                "audience_a": platform[da]["audience"],
                "domain_b":   db,
                "audience_b": platform[db]["audience"],
                "jsd":        round(j, 6),
                "cosine_sim": round(cos, 6),
            })

    log.info(f"  Distinctiveness matrix: {len(results)} domain pairs.")
    return results


def compute_aggregate_distance(cross: dict, vocab: set[str]) -> dict:
    """
    Compute the single B2B-vs-B2W aggregate JSD and cosine.

    This is the summary distance between the two audience sub-registers
    across the full corpus.
    """
    client_freq  = Counter()
    worker_freq  = Counter()
    n_client = 0
    n_worker = 0

    for page in cross.get("client", []):
        tokens   = [t for t in page["tokens"] if "_" not in t]
        client_freq.update(tokens)
        n_client += len(tokens)

    for page in cross.get("worker", []):
        tokens   = [t for t in page["tokens"] if "_" not in t]
        worker_freq.update(tokens)
        n_worker += len(tokens)

    pa  = to_prob_dist(client_freq, vocab)
    pb  = to_prob_dist(worker_freq, vocab)
    j   = jsd_normalised(pa, pb)
    cos = cosine_similarity(client_freq, worker_freq, vocab)

    log.info(f"  Aggregate B2B-vs-B2W: JSD={j:.4f}  cosine_sim={cos:.4f}")
    return {
        "comparison":      "cross_platform_aggregate",
        "jsd":             round(j, 6),
        "cosine_sim":      round(cos, 6),
        "n_client_tokens": n_client,
        "n_worker_tokens": n_worker,
    }


# ===========================================================================
# SECTION H — Term exclusivity
# ===========================================================================

def compute_term_exclusivity(platform: dict) -> list[dict]:
    """
    Compute prevalence-based exclusivity using QUALIFIED domains only.

    For each term:
      prevalence_client = fraction of client platforms where it appears
      prevalence_worker = fraction of worker platforms where it appears
      exclusivity_index = prevalence_client − prevalence_worker
        +1 = exclusively client-facing
        −1 = exclusively worker-facing
         0 = equally distributed

    Category thresholds:
      |exclusivity_index| ≥ EXCLUSIVITY_THRESHOLD  → exclusive (to one side)
      |exclusivity_index| ≤ SHARED_BAND            → shared
      otherwise                                     → leaning (to one side)
    """
    client_domains = [d for d, v in platform.items() if v["audience"] == "client"]
    worker_domains = [d for d, v in platform.items() if v["audience"] == "worker"]
    n_c, n_w = len(client_domains), len(worker_domains)

    if n_c == 0 or n_w == 0:
        log.warning("Cannot compute exclusivity — both audiences required.")
        return []

    log.info(f"  Exclusivity: {n_c} client + {n_w} worker qualified domains.")

    # Build global frequency and per-audience domain counts
    global_freq: Counter = Counter()
    term_c: Counter = Counter()   # domains where term appears (client)
    term_w: Counter = Counter()   # domains where term appears (worker)

    for d in client_domains:
        global_freq.update(platform[d]["freq"])
        for t in platform[d]["freq"]:
            term_c[t] += 1

    for d in worker_domains:
        global_freq.update(platform[d]["freq"])
        for t in platform[d]["freq"]:
            term_w[t] += 1

    results = []
    for term, total in global_freq.items():
        if total < MIN_TERM_FREQ:
            continue
        p_c = term_c.get(term, 0) / n_c
        p_w = term_w.get(term, 0) / n_w
        idx = p_c - p_w

        if abs(idx) <= SHARED_BAND:
            cat = "shared"
        elif idx > 0:
            cat = "client_exclusive" if idx >= EXCLUSIVITY_THRESHOLD else "client_leaning"
        else:
            cat = "worker_exclusive" if abs(idx) >= EXCLUSIVITY_THRESHOLD else "worker_leaning"

        results.append({
            "term":              term,
            "term_type":         "bigram" if "_" in term else "unigram",
            "prevalence_client": round(p_c, 4),
            "prevalence_worker": round(p_w, 4),
            "exclusivity_index": round(idx, 4),
            "category":          cat,
            "total_freq":        total,
        })

    log.info(f"  {len(results):,} exclusivity scores computed.")
    return results


# ===========================================================================
# SECTION I — Persistence helpers
# ===========================================================================

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


def save_distinctiveness(conn: sqlite3.Connection,
                         matrix: list[dict],
                         aggregate: dict,
                         quality: list[dict]):
    conn.executemany("""
        INSERT INTO distinctiveness_matrix
            (domain_a, audience_a, domain_b, audience_b, jsd, cosine_sim)
        VALUES
            (:domain_a, :audience_a, :domain_b, :audience_b, :jsd, :cosine_sim)
    """, matrix)
    conn.execute("""
        INSERT INTO aggregate_distance
            (comparison, jsd, cosine_sim, n_client_tokens, n_worker_tokens)
        VALUES
            (:comparison, :jsd, :cosine_sim, :n_client_tokens, :n_worker_tokens)
    """, aggregate)
    conn.executemany("""
        INSERT INTO domain_quality
            (domain, audience, n_pages, n_tokens,
             included_in_matrix, exclusion_reason)
        VALUES
            (:domain, :audience, :n_pages, :n_tokens,
             :included_in_matrix, :exclusion_reason)
    """, quality)
    conn.commit()


def save_exclusivity(conn: sqlite3.Connection, results: list[dict]):
    conn.executemany("""
        INSERT INTO term_exclusivity
            (term, term_type, prevalence_client, prevalence_worker,
             exclusivity_index, category, total_freq)
        VALUES
            (:term, :term_type, :prevalence_client, :prevalence_worker,
             :exclusivity_index, :category, :total_freq)
    """, results)
    conn.commit()


# ===========================================================================
# SECTION J — Logging summaries
# ===========================================================================

def log_top_keyness(results: list[dict], n: int = 20):
    client_top = [r for r in results if r["ll_score"] > 0][:n]
    worker_top = [r for r in results if r["ll_score"] < 0][:n]
    log.info(f"  Top {n} CLIENT-distinctive terms:")
    for r in client_top:
        log.info(f"    {r['term']:<30} LL={r['ll_score']:>10.2f}  "
                 f"B2B={r['rel_freq_client']:.2f}‰  B2W={r['rel_freq_worker']:.2f}‰")
    log.info(f"  Top {n} WORKER-distinctive terms:")
    for r in worker_top:
        log.info(f"    {r['term']:<30} LL={r['ll_score']:>10.2f}  "
                 f"B2B={r['rel_freq_client']:.2f}‰  B2W={r['rel_freq_worker']:.2f}‰")


def log_top_collocates(results: list[dict], focus_term: str,
                       audience: str, n: int = 10):
    filtered = sorted(
        [r for r in results
         if r["focus_term"] == focus_term and r["audience"] == audience],
        key=lambda x: x["pmi"], reverse=True
    )
    log.info(f"  Top collocates for '{focus_term}' ({audience}):")
    for r in filtered[:n]:
        log.info(f"    {r['collocate']:<25} PMI={r['pmi']:>6.3f}  cofreq={r['cofreq']}")


# ===========================================================================
# SECTION K — Main
# ===========================================================================

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Check corpus_view exists
    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone():
        raise RuntimeError("corpus_view not found — run 01_prepare_corpus.py first.")

    # Check segments column is present (from 00_preprocess.py)
    cols = {row[1] for row in conn.execute(
        "PRAGMA table_info(pages_tfidf)"
    ).fetchall()}
    if "segments" not in cols:
        raise RuntimeError(
            "pages_tfidf.segments column missing — "
            "run 00_preprocess.py (src2 version) first."
        )

    log.info("=" * 60)
    log.info("02_step1_analysis.py — Keyness, Co-occurrence, Distinctiveness")
    log.info(f"  Window size : ±{WINDOW_SIZE} tokens (within-sentence only)")
    log.info(f"  Min term freq: {MIN_TERM_FREQ}")
    log.info(f"  Min PMI cofreq: {MIN_PMI_COFREQ}")
    log.info("=" * 60)

    init_output_tables(conn)

    excluded_page_ids, excluded_terms = load_exclusions(conn)
    corpus = load_corpus(conn, excluded_page_ids, excluded_terms)

    cross        = corpus["cross"]
    pairs        = corpus["pairs"]
    pair_domains = corpus["pair_domains"]
    platform     = corpus["platform"]

    client_pages = cross.get("client", [])
    worker_pages = cross.get("worker", [])

    # -----------------------------------------------------------------------
    # A. Keyness — cross-platform
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("A. Keyness analysis")
    log.info("=" * 60)

    cross_keyness = compute_keyness(client_pages, worker_pages, "cross_platform")
    log_top_keyness(cross_keyness)
    save_keyness(conn, cross_keyness)

    # Keyness for each within-pair comparison
    for cid, data in pairs.items():
        pair_kn = compute_keyness(
            data["client"], data["worker"], comparison_label=cid
        )
        save_keyness(conn, pair_kn)

    log.info(f"Keyness saved: {len(cross_keyness)} cross-platform terms "
             f"+ {len(pairs)} pair comparisons.")

    # -----------------------------------------------------------------------
    # B. Co-occurrence — cross-platform only (pairs can be added if needed)
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("B. Co-occurrence analysis (sentence-scoped, ±%d window)", WINDOW_SIZE)
    log.info("=" * 60)

    cooc_results = compute_cooccurrence(
        client_pages, worker_pages, cross_keyness, "cross_platform"
    )
    save_cooccurrence(conn, cooc_results)

    # Log collocate profiles for key theory terms
    for term in ["human", "work", "worker", "automation", "quality"]:
        for aud in ["client", "worker"]:
            log_top_collocates(cooc_results, term, aud)

    # -----------------------------------------------------------------------
    # C. Platform-level term counts
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("C. Platform-level term counts")
    log.info("=" * 60)

    ptc = compute_platform_term_counts(platform)
    save_platform_counts(conn, ptc)

    # -----------------------------------------------------------------------
    # D + E. Distinctiveness matrix and aggregate distance
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("D. Distinctiveness analysis")
    log.info("=" * 60)

    filtered_platform, quality_rows = apply_domain_quality_filter(platform)

    vocab = select_high_variance_vocab(filtered_platform)

    matrix    = compute_distinctiveness_matrix(filtered_platform, vocab)
    aggregate = compute_aggregate_distance(cross, vocab)

    # -----------------------------------------------------------------------
    # F. Term exclusivity
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("E. Term exclusivity scores")
    log.info("=" * 60)

    exclusivity = compute_term_exclusivity(filtered_platform)

    save_distinctiveness(conn, matrix, aggregate, quality_rows)
    save_exclusivity(conn, exclusivity)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("ANALYSIS COMPLETE")
    log.info(f"  keyness_results       : "
             f"{conn.execute('SELECT COUNT(*) FROM keyness_results').fetchone()[0]:,} rows")
    log.info(f"  cooccurrence_results  : "
             f"{conn.execute('SELECT COUNT(*) FROM cooccurrence_results').fetchone()[0]:,} rows")
    log.info(f"  platform_term_counts  : "
             f"{conn.execute('SELECT COUNT(*) FROM platform_term_counts').fetchone()[0]:,} rows")
    log.info(f"  distinctiveness_matrix: "
             f"{conn.execute('SELECT COUNT(*) FROM distinctiveness_matrix').fetchone()[0]:,} rows")
    log.info(f"  term_exclusivity      : "
             f"{conn.execute('SELECT COUNT(*) FROM term_exclusivity').fetchone()[0]:,} rows")
    log.info("Next steps:")
    log.info("  python3 src2/03_step1_topics.py    (LDA + STM export)")
    log.info("  python3 src2/04_step1_figures.py   (visualisations)")
    log.info("  python3 src2/05_step2_export.py    (close reading corpus)")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
