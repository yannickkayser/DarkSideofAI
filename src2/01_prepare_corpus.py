"""
01_prepare_corpus.py
====================
Stage 1: Corpus preparation — platform metadata, corpus view, and exclusion
tables.

Merges src/01_prepare.py and src/01_prepare_additions.py into a single
script.  Both scripts were always run together and served the same single
purpose: making the database ready for analysis.  There is no reason to
keep them separate.

Pipeline position:
  Run AFTER: 00_preprocess.py (pages_tfidf must exist with segments column)
  Run BEFORE: 02_step1_analysis.py

What this script does (in order):
  1. Parse WEBSITES config → platform metadata (audience, type, pairs).
  2. Create / update the `platforms` table.
  3. Create `corpus_view` — the single authoritative join used by ALL
     analysis scripts.  Includes the new `segments` column from
     00_preprocess.py for use in co-occurrence analysis.
  4. Run corpus diagnostics (pages per platform, pair linkage, token stats).
  5. Create `excluded_pages` — pages to skip in analysis (non-English,
     test pages, etc.).
  6. Create `excluded_terms` — noise terms to strip from token lists
     (boilerplate, single-domain artifacts, non-English words).

Re-run safe: all tables use CREATE IF NOT EXISTS / INSERT OR REPLACE /
DELETE + repopulate, so running this script again always reflects the
latest config and exclusion rules.

Configuration to edit:
  PLATFORM_TYPE_RULES  — map type strings → canonical platform_type
  PAIR_RULES           — link paired domains to the same company_id
  HQ_REGION_RULES      — headquarters region ('north' / 'south')
  EXCLUDED_PAGE_RULES  — URL patterns / domain names / page IDs to skip
  MANUAL_EXCLUSIONS    — known boilerplate terms to always exclude
  GERMAN_MARKERS       — German function words to detect non-English terms

Usage:
    python3 src2/01_prepare_corpus.py
"""

import sqlite3
import json
import re
import logging
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import WEBSITES

DB_PATH = "data/scraping_2.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# SECTION A — Platform metadata configuration
# ===========================================================================

# Maps fragments in the type string → canonical platform_type values.
PLATFORM_TYPE_RULES = [
    ("Algorithmic Crowd Market", "crowd_market"),
    ("Managed Enterprise BPO",   "enterprise_bpo"),
    ("Impact Sourcing",          "impact_sourcing"),
]

# Links paired domains to the same company_id.
# A "pair" = two domains owned by the same company addressing different
# audiences (one B2B client-facing, one B2W worker-facing).
# company_id is the shared key used for within-pair comparisons.
PAIR_RULES = {
    "appen":      "appen",
    "crowdgen":   "appen",
    "toloka":     "toloka",
    "mindrift":   "toloka",
    "centific":   "centific",
    "oneforma":   "centific",
    "labelbox":   "labelbox",
    "alignerr":   "labelbox",
    "scale":      "scale",
    "remotasks":  "scale",
}

# Maps domain fragments → headquarters region.
# 'south' = Global South focus (impact-sourcing firms).
# 'north' = default.
HQ_REGION_RULES = {
    "sama":         "south",
    "imerit":       "south",
    "cloudfactory": "south",
    "defined":      "south",
}


# ===========================================================================
# SECTION B — Exclusion configuration
# ===========================================================================

# Pages to exclude from analysis.
# Each tuple: (match_type, match_value, reason)
# match_type: 'url_contains' | 'domain_exact' | 'page_id'
EXCLUDED_PAGE_RULES = [
    ("url_contains", "/de/",
     "Non-English page (German): detected via German terms in LDA topics "
     "and manual URL inspection"),
]

# Boilerplate anchor terms — co-occurrence with these suggests a term is
# scraping/UI noise rather than substantive content.
BOILERPLATE_ANCHORS = {
    "cookie", "subscribe", "faq", "newsletter", "privacy", "gdpr",
    "consent", "browser", "javascript", "login", "signup", "captcha",
}

# Common German function words — used to identify non-English terms.
# Extend if new German terms appear in LDA topic outputs.
GERMAN_MARKERS = {
    "und", "der", "die", "das", "ist", "für", "mit", "auf", "von",
    "sich", "nicht", "ein", "eine", "auch", "werden", "nach", "bei",
    "durch", "über", "oder", "wie", "noch", "kann", "nur", "alle",
    "mehr", "zum", "zur", "dem", "den", "des", "dass", "wenn", "aber",
    "sie", "wir", "ich", "als",
    # Terms found in previous LDA runs:
    "ueberwinden", "bildbearbeitung", "sicherstellung", "kunst",
}

# Terms that must NEVER be excluded regardless of heuristics.
# Mirrored as THEORY_FOCUS_TERMS in 02_step1_analysis.py.
PROTECTED_TERMS = {
    # H1a — Labour visibility gap
    "worker", "labour", "task", "job", "pay", "earn", "payment",
    "work", "annotator", "labeller", "moderator",
    # H1b — Automation myth
    "autonomous", "machine", "automate", "intelligent", "automation",
    "model", "algorithm",
    # H1c — Strategic hypervisibility
    "human", "quality", "oversight", "annotation", "label", "datum",
    "accuracy", "review",
    # H2 — Flexibilised control
    "flexible", "flexibility", "freedom", "independent", "autonomy",
    "score", "rating", "rank", "performance", "metric",
    # H3a / H3b — Worker identity framing
    "talent", "resource", "contributor", "workforce",
    "community", "collective",
}

# Known boilerplate / artifact terms (identified iteratively from analysis).
MANUAL_EXCLUSIONS = [
    ("cookie",       "boilerplate: cookie consent banner"),
    ("set_cookie",   "boilerplate: cookie consent banner"),
    ("cooky",        "boilerplate: cookie consent variant"),
    ("/hr",          "scraping artifact: URL fragment"),
    ("/hr_remote",   "scraping artifact: URL fragment"),
    ("remote_apply", "scraping artifact: URL fragment"),
    ("faq",          "boilerplate: FAQ section header"),
    ("faq_help",     "boilerplate: FAQ section"),
    ("help_desk",    "boilerplate: support section"),
    ("subscribe",    "boilerplate: newsletter prompt"),
    ("youtube",      "boilerplate: social media link"),
    ("jira",         "scraping artifact: internal tool reference"),
]


# ===========================================================================
# SECTION C — Platform metadata functions
# ===========================================================================

def parse_platform_type(type_str: str) -> str:
    """Map the raw type string to a canonical platform_type value."""
    for fragment, canonical in PLATFORM_TYPE_RULES:
        if fragment.lower() in type_str.lower():
            return canonical
    return "unknown"


def parse_company_id(domain: str) -> str:
    """
    Return the company_id for this domain.

    Links paired domains to the same company for within-pair comparisons.
    Falls back to the domain itself for unmatched (single) domains.
    """
    for fragment, company_id in PAIR_RULES.items():
        if fragment in domain.lower():
            return company_id
    return domain


def parse_hq_region(domain: str) -> str:
    """Return 'south' if domain matches a Global South rule, else 'north'."""
    for fragment, region in HQ_REGION_RULES.items():
        if fragment in domain.lower():
            return region
    return "north"


def build_platforms_records() -> list[dict]:
    """Convert WEBSITES config into platform metadata records."""
    records = []
    for domain, site in WEBSITES.items():
        records.append({
            "domain":        domain,
            "name":          site.get("name", domain),
            "audience":      site.get("audience", "unknown"),
            "platform_type": parse_platform_type(site.get("type", "")),
            "company_id":    parse_company_id(domain),
            "hq_region":     parse_hq_region(domain),
            "type_raw":      site.get("type", ""),
        })
    return records


# ===========================================================================
# SECTION D — Database writes: platforms table + corpus_view
# ===========================================================================

def create_and_populate_platforms(conn: sqlite3.Connection, records: list[dict]):
    """
    Create the platforms table and populate it from WEBSITES config.

    Uses INSERT OR REPLACE so re-running updates stale metadata without
    leaving orphaned rows.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS platforms (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            domain        TEXT UNIQUE NOT NULL,
            name          TEXT,
            audience      TEXT,
            platform_type TEXT,
            company_id    TEXT,
            hq_region     TEXT,
            type_raw      TEXT,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.executemany("""
        INSERT OR REPLACE INTO platforms
            (domain, name, audience, platform_type, company_id, hq_region, type_raw)
        VALUES
            (:domain, :name, :audience, :platform_type, :company_id,
             :hq_region, :type_raw)
    """, records)
    conn.commit()
    log.info(f"Platforms table: {len(records)} records.")


def create_corpus_view(conn: sqlite3.Connection):
    """
    Create (or recreate) corpus_view — the single authoritative join
    used by ALL analysis scripts (02_step1_analysis.py through
    05_step2_export.py).

    Changes vs src/01_prepare.py:
      - Now includes `segments` column from pages_tfidf.  This allows
        02_step1_analysis.py to load per-sentence token lists for the
        segment-aware co-occurrence computation.

    Audience source:
      The view joins platforms.audience (from config) NOT pages_tfidf.audience
      (which is derived from URL matching and unreliable).  This is the
      authoritative audience assignment.

    Override support:
      If the page_audience_override table already exists (populated by
      01_prepare_audience_overrides.py), this script builds the override-aware
      version of corpus_view automatically.  Re-running this script therefore
      never silently loses the URL-based client/worker splits.

    Key columns:
      page_id, url, segments, unigrams, bigrams, token_count
      audience, platform_type, company_id, hq_region, platform_name, domain
    """
    # Check whether the override table exists and has rows.
    has_overrides = conn.execute("""
        SELECT COUNT(*) FROM sqlite_master
        WHERE type='table' AND name='page_audience_override'
    """).fetchone()[0] > 0

    if has_overrides:
        n_overrides = conn.execute(
            "SELECT COUNT(*) FROM page_audience_override"
        ).fetchone()[0]
        has_overrides = n_overrides > 0

    conn.execute("DROP VIEW IF EXISTS corpus_view")

    if has_overrides:
        conn.execute("""
            CREATE VIEW corpus_view AS
            SELECT
                t.page_id,
                t.url,
                t.segments,        -- JSON list of lists: per-sentence token lists
                t.unigrams,        -- JSON flat list: used by keyness analysis
                t.bigrams,         -- JSON list of within-sentence bigrams
                t.token_count,
                COALESCE(o.audience,   pl.audience)   AS audience,
                COALESCE(o.company_id, pl.company_id) AS company_id,
                pl.platform_type,
                pl.hq_region,
                pl.name    AS platform_name,
                w.domain
            FROM pages_tfidf t
            JOIN pages    pg ON pg.id      = t.page_id
            JOIN websites w  ON w.id       = pg.website_id
            JOIN platforms pl ON pl.domain = REPLACE(w.domain, 'www.', '')
            LEFT JOIN page_audience_override o ON o.page_id = t.page_id
        """)
        log.info(
            f"corpus_view created with override support "
            f"({n_overrides} page_audience_override rows applied)."
        )
    else:
        conn.execute("""
            CREATE VIEW corpus_view AS
            SELECT
                t.page_id,
                t.url,
                t.segments,        -- JSON list of lists: per-sentence token lists
                t.unigrams,        -- JSON flat list: used by keyness analysis
                t.bigrams,         -- JSON list of within-sentence bigrams
                t.token_count,
                pl.audience,       -- from config via platforms — authoritative
                pl.platform_type,
                pl.company_id,
                pl.hq_region,
                pl.name    AS platform_name,
                w.domain
            FROM pages_tfidf t
            JOIN pages    pg ON pg.id       = t.page_id
            JOIN websites w  ON w.id        = pg.website_id
            JOIN platforms pl ON pl.domain  = REPLACE(w.domain, 'www.', '')
        """)
        log.info(
            "corpus_view created (no overrides — run "
            "01_prepare_audience_overrides.py to add URL-based splits)."
        )


# ===========================================================================
# SECTION E — Corpus diagnostics
# ===========================================================================

def run_diagnostics(conn: sqlite3.Connection):
    """
    Validate corpus_view and surface data quality issues.

    Logs:
      - Total pages in corpus_view vs pages_tfidf (any unexplained drops)
      - Audience distribution (config-derived, reliable)
      - Pages per platform with token counts
      - Platform pairs (company_id groups with > 1 domain)
      - Corpus-wide token statistics
      - Platform type distribution

    Review this output before proceeding to 02_step1_analysis.py to verify:
      - All expected platforms appear
      - Audience assignments are correct
      - Pairs are linked correctly
      - No large unexplained drops in page count
    """
    log.info("=" * 60)
    log.info("CORPUS DIAGNOSTICS")
    log.info("=" * 60)

    total_raw  = conn.execute("SELECT COUNT(*) FROM pages_tfidf").fetchone()[0]
    total_view = conn.execute("SELECT COUNT(*) FROM corpus_view").fetchone()[0]
    dropped    = total_raw - total_view

    log.info(f"pages_tfidf : {total_raw} rows")
    log.info(f"corpus_view : {total_view} rows")

    if dropped > 0:
        log.warning(
            f"  {dropped} pages dropped — their website has no matching "
            f"entry in platforms.  Check that all scraped domains are in "
            f"the WEBSITES config."
        )

    # Audience distribution (config-derived)
    log.info("-" * 60)
    log.info("Audience distribution (config-derived, authoritative):")
    for row in conn.execute("""
        SELECT audience, COUNT(*) as n FROM corpus_view GROUP BY audience
        ORDER BY n DESC
    """).fetchall():
        pct = 100 * row[1] / total_view if total_view else 0
        log.info(f"  {row[0]:<12} {row[1]:>6} pages ({pct:.1f}%)")

    # Pages per platform
    log.info("-" * 60)
    log.info("Pages per platform:")
    for row in conn.execute("""
        SELECT domain, platform_name, audience, platform_type, company_id,
               hq_region, COUNT(*) as n_pages, SUM(token_count) as total_tokens
        FROM corpus_view
        GROUP BY domain
        ORDER BY platform_type, audience, n_pages DESC
    """).fetchall():
        log.info(
            f"  {row['domain']:<30} "
            f"audience={row['audience']:<8} "
            f"type={row['platform_type']:<18} "
            f"company={row['company_id']:<15} "
            f"region={row['hq_region']:<6} "
            f"pages={row['n_pages']:>5}  tokens={row['total_tokens']:>9,}"
        )

    # Platform pairs
    log.info("-" * 60)
    log.info("Platform pairs (company_id links B2B + B2W domains):")
    pair_rows = conn.execute("""
        SELECT company_id,
               GROUP_CONCAT(domain, ' | ')   AS domains,
               GROUP_CONCAT(audience, ' | ') AS audiences,
               COUNT(DISTINCT domain)         AS n_domains
        FROM platforms
        GROUP BY company_id
        HAVING n_domains > 1
        ORDER BY company_id
    """).fetchall()
    if pair_rows:
        for row in pair_rows:
            log.info(f"  {row['company_id']:<15} → {row['domains']}  "
                     f"[{row['audiences']}]")
    else:
        log.warning("No pairs found — check PAIR_RULES.")

    # Token statistics
    log.info("-" * 60)
    stats = conn.execute("""
        SELECT MIN(token_count) as mn, MAX(token_count) as mx,
               AVG(token_count) as avg, SUM(token_count) as total
        FROM corpus_view
    """).fetchone()
    log.info("Token statistics:")
    log.info(f"  Min={stats['mn']}  Max={stats['mx']}  "
             f"Avg={stats['avg']:.0f}  Total={stats['total']:,}")

    # Platform type distribution
    log.info("-" * 60)
    log.info("Platform type distribution:")
    for row in conn.execute("""
        SELECT platform_type,
               COUNT(DISTINCT domain) as n_platforms,
               COUNT(*) as n_pages
        FROM corpus_view
        GROUP BY platform_type
    """).fetchall():
        log.info(f"  {row['platform_type']:<20} "
                 f"{row['n_platforms']} platforms  {row['n_pages']} pages")

    log.info("=" * 60)


# ===========================================================================
# SECTION F — Exclusion tables
# ===========================================================================

def init_exclusion_tables(conn: sqlite3.Connection):
    """Create excluded_pages and excluded_terms tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS excluded_pages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id          INTEGER,
            url              TEXT,
            domain           TEXT,
            reason           TEXT NOT NULL,
            detection_method TEXT NOT NULL,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS excluded_terms (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            term             TEXT NOT NULL UNIQUE,
            reason           TEXT NOT NULL,
            detection_method TEXT NOT NULL,
            domain_count     INTEGER,
            total_freq       INTEGER,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_et_term ON excluded_terms(term);
    """)
    conn.commit()
    log.info("Exclusion tables ready.")


def populate_excluded_pages(conn: sqlite3.Connection):
    """
    Apply EXCLUDED_PAGE_RULES against corpus_view and write to excluded_pages.

    Clears and repopulates on each run so the table always reflects current
    rules.  Logs each excluded page for audit.
    """
    log.info("Applying page exclusion rules...")
    conn.execute("DELETE FROM excluded_pages")
    count = 0

    for match_type, match_value, reason in EXCLUDED_PAGE_RULES:
        if match_type == "url_contains":
            rows = conn.execute(
                "SELECT page_id, url, domain FROM corpus_view WHERE url LIKE ?",
                [f"%{match_value}%"]
            ).fetchall()
        elif match_type == "domain_exact":
            rows = conn.execute(
                "SELECT page_id, url, domain FROM corpus_view WHERE domain = ?",
                [match_value]
            ).fetchall()
        elif match_type == "page_id":
            rows = conn.execute(
                "SELECT page_id, url, domain FROM corpus_view WHERE page_id = ?",
                [int(match_value)]
            ).fetchall()
        else:
            log.warning(f"Unknown match_type: {match_type}")
            continue

        for r in rows:
            conn.execute("""
                INSERT OR IGNORE INTO excluded_pages
                    (page_id, url, domain, reason, detection_method)
                VALUES (?, ?, ?, ?, 'manual_rule')
            """, [r["page_id"], r["url"], r["domain"], reason])
            count += 1

    conn.commit()
    log.info(f"  {count} pages excluded.")
    for r in conn.execute(
        "SELECT page_id, url, reason FROM excluded_pages"
    ).fetchall():
        log.info(f"    page_id={r['page_id']}  {r['url'][:80]}  — {r['reason']}")


def detect_excluded_terms(conn: sqlite3.Connection):
    """
    Identify artifact, boilerplate, and non-English terms for exclusion.

    Four heuristics applied in order (PROTECTED_TERMS guards all):
      A. MANUAL        — MANUAL_EXCLUSIONS list
      B. NON_ENGLISH   — GERMAN_MARKERS membership + German character pattern
      C. SINGLE_DOMAIN — terms with freq ≥ 5 appearing on only one domain
      D. (Planned)     — boilerplate co-occurrence detection

    All exclusions are auditable: each entry records reason and method.

    Safe re-run behaviour:
      Only auto-detected rows (detection_method != 'manual') are cleared on each
      run.  Rows inserted by add_excluded_terms.py (detection_method = 'manual')
      are preserved, so re-running this script never silently removes the
      hand-curated foreign/noise term list.
    """
    log.info("Detecting artifact / noise terms for exclusion...")
    # Preserve manual entries — only clear auto-detected ones.
    n_manual = conn.execute(
        "SELECT COUNT(*) FROM excluded_terms WHERE detection_method = 'manual'"
    ).fetchone()[0]
    conn.execute(
        "DELETE FROM excluded_terms WHERE detection_method != 'manual'"
    )
    if n_manual:
        log.info(f"  Preserved {n_manual} manual exclusion(s) from add_excluded_terms.py.")

    excluded_page_ids = {
        r[0] for r in conn.execute("SELECT page_id FROM excluded_pages").fetchall()
    }

    rows = conn.execute("""
        SELECT page_id, domain, unigrams FROM corpus_view
        WHERE audience IN ('client', 'worker')
          AND token_count >= 10
    """).fetchall()

    term_freq:    Counter                = Counter()
    term_domains: defaultdict[str, set]  = defaultdict(set)

    for row in rows:
        if row["page_id"] in excluded_page_ids:
            continue
        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        for t in unigrams:
            term_freq[t] += 1
            term_domains[t].add(row["domain"])

    total_domains = len({r["domain"] for r in rows
                         if r["page_id"] not in excluded_page_ids})
    log.info(f"  Vocabulary: {len(term_freq):,} terms across {total_domains} domains")

    exclusions: dict[str, tuple] = {}   # term → (reason, method, n_domains, freq)

    def exclude(term: str, reason: str, method: str):
        """Flag a term for exclusion if it is not in PROTECTED_TERMS."""
        if term in PROTECTED_TERMS:
            return
        if term not in exclusions:
            exclusions[term] = (
                reason, method,
                len(term_domains.get(term, set())),
                term_freq.get(term, 0),
            )

    # --- A. Manual exclusions ---
    for term, reason in MANUAL_EXCLUSIONS:
        exclude(term, reason, "manual")

    # --- B. Non-English: German function words ---
    for term in term_freq:
        if term.lower() in GERMAN_MARKERS:
            exclude(term, "Non-English term (German function word/noun)", "non_english")

    german_char_re = re.compile(r'[äöüßÄÖÜ]')
    for term in term_freq:
        if german_char_re.search(term):
            exclude(term, "Non-English term (German characters)", "non_english")

    # --- C. Single-domain terms (freq ≥ 5, only 1 domain) ---
    for term, freq in term_freq.items():
        if freq >= 5 and len(term_domains[term]) == 1:
            the_domain = list(term_domains[term])[0]
            exclude(
                term,
                f"Single-domain term (only on {the_domain}, freq={freq})",
                "single_domain"
            )

    # Insert all flagged terms
    for term, (reason, method, n_domains, freq) in exclusions.items():
        conn.execute("""
            INSERT OR IGNORE INTO excluded_terms
                (term, reason, detection_method, domain_count, total_freq)
            VALUES (?, ?, ?, ?, ?)
        """, [term, reason, method, n_domains, freq])
    conn.commit()

    # Summary by method
    method_counts = Counter(v[1] for v in exclusions.values())
    log.info(f"  {len(exclusions):,} terms excluded:")
    for method, n in method_counts.most_common():
        log.info(f"    {method:<20} {n:>5} terms")

    # Sample: top 20 by frequency (most visible in analysis outputs)
    log.info("  Sample: top 20 excluded terms by frequency:")
    for term, (reason, method, n_dom, freq) in sorted(
        exclusions.items(), key=lambda x: x[1][3], reverse=True
    )[:20]:
        log.info(
            f"    {term:<30} method={method:<15} "
            f"domains={n_dom}  freq={freq}  — {reason[:55]}"
        )


# ===========================================================================
# SECTION G — Main
# ===========================================================================

def main():
    """
    Run the full corpus preparation sequence:
      1. Platform metadata → platforms table
      2. corpus_view — with override support if page_audience_override exists
      3. Diagnostics
      4. Exclusion tables (pages + terms); manual entries preserved

    REQUIRED run order (run ALL of these when setting up or re-preparing):
      python3 src2/00_preprocess.py
      python3 src2/01_prepare_corpus.py             ← this script
      python3 src/01_prepare_audience_overrides.py  ← URL-based audience splits
      python3 src2/add_excluded_terms.py            ← hand-curated noise terms

    Safe to re-run: this script detects existing overrides and preserves
    manually-added excluded terms, so the output is always the correct
    fully-prepared corpus regardless of re-run order within a session.
    """
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    # Check that pages_tfidf was populated by src2/00_preprocess.py
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    n_tfidf = conn.execute("SELECT COUNT(*) FROM pages_tfidf").fetchone()[0]
    if n_tfidf == 0:
        raise RuntimeError(
            "pages_tfidf is empty — run src2/00_preprocess.py first."
        )

    # Check that the segments column exists — it is added by src2/00_preprocess.py
    # but NOT by src/preprocess.py (the old pipeline).  All analysis scripts in
    # src2/ depend on this column for sentence-boundary-aware co-occurrence.
    tfidf_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(pages_tfidf)").fetchall()
    }
    if "segments" not in tfidf_cols:
        raise RuntimeError(
            "pages_tfidf is missing the 'segments' column.\n"
            "This means the database was preprocessed with src/preprocess.py "
            "(the old pipeline) rather than src2/00_preprocess.py.\n"
            "Fix: run  python3 src2/00_preprocess.py  to re-process the corpus "
            "and populate the segments column before running this script."
        )

    log.info("=" * 60)
    log.info("01_prepare_corpus.py — Corpus Preparation")
    log.info("=" * 60)

    # --- Step 1: Platform metadata ---
    log.info("Parsing WEBSITES config...")
    records = build_platforms_records()
    log.info(f"  {len(records)} platform entries:")
    for r in records:
        log.info(
            f"    {r['domain']:<30} audience={r['audience']:<8} "
            f"type={r['platform_type']:<18} company={r['company_id']:<15} "
            f"region={r['hq_region']}"
        )

    create_and_populate_platforms(conn, records)

    # --- Step 2: corpus_view ---
    create_corpus_view(conn)

    # --- Step 3: Diagnostics ---
    run_diagnostics(conn)

    # --- Step 4: Exclusion tables ---
    log.info("=" * 60)
    log.info("Building exclusion tables...")
    log.info("=" * 60)
    init_exclusion_tables(conn)
    populate_excluded_pages(conn)
    detect_excluded_terms(conn)

    n_exc_pages = conn.execute("SELECT COUNT(*) FROM excluded_pages").fetchone()[0]
    n_exc_terms = conn.execute("SELECT COUNT(*) FROM excluded_terms").fetchone()[0]
    n_manual    = conn.execute(
        "SELECT COUNT(*) FROM excluded_terms WHERE detection_method='manual'"
    ).fetchone()[0]
    n_auto      = n_exc_terms - n_manual

    # Check whether corpus_view has override support
    cv_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()[0]
    has_overrides = "page_audience_override" in cv_sql

    log.info("=" * 60)
    log.info("CORPUS READY")
    log.info(f"  corpus_view audience overrides : {'YES' if has_overrides else 'NO — run 01_prepare_audience_overrides.py'}")
    log.info(f"  Excluded pages : {n_exc_pages}")
    log.info(f"  Excluded terms : {n_exc_terms}  "
             f"(auto={n_auto}, manual={n_manual})")
    if n_manual == 0:
        log.warning(
            "  No manual exclusions found — run add_excluded_terms.py "
            "to add the hand-curated foreign/noise term list."
        )
    log.info("Next step: python3 src2/02_step1_analysis.py")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
