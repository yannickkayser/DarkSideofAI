"""
01_prepare.py
=============
Enriches the scraping database with platform metadata and creates the
canonical corpus view used by all downstream analysis scripts.

Pipeline position:
  Stage 1 — Corpus Preparation (runs ONCE after the full scraping pass
  and preprocessing step are complete)
  Prerequisites: main.py (scraping), preprocess.py (tokenisation)
  Next step:     01_prepare_additions.py (exclusion tables)

What this script does:
  1. Reads WEBSITES config from config/config.py and derives structured
     platform metadata (platform_type, company_id, hq_region) via rule
     tables defined in this file.
  2. Creates and populates the `platforms` table in the DB.
  3. Creates `corpus_view` — the single SQL view that joins
     pages_tfidf (token data from preprocess.py) to platform metadata.
     All scripts from 02 onward query this view instead of rewriting
     the join.
  4. Runs corpus diagnostics and flags any pages that could not be
     joined to a platform (which would make their audience field
     unresolvable).

IMPORTANT — audience source of truth:
  The audience column in pages_tfidf is unreliable.  preprocess.py
  derives audience by matching page URLs against patterns, and pages
  whose URLs do not match any pattern end up as 'unknown'.  This
  script ignores that column entirely.
  All analysis scripts derive audience via:
    pages_tfidf → pages → websites → platforms (from config)
  corpus_view encodes this join once so it is never repeated in
  individual analysis scripts.

Output tables / views written to data/scraping.db:
  platforms   : one row per domain with canonical metadata
  corpus_view : SQL VIEW joining pages_tfidf to platform metadata

Run order:
  python3 src/main.py (scraping)        →
  python3 src/preprocess.py             →
  python3 src/01_prepare.py             ← this script
  python3 src/01_prepare_additions.py   →
  python3 src/02_step1_frequency.py     → ...

Configuration to edit before running:
  DB_PATH         — path to the SQLite database
  PLATFORM_TYPE_RULES — map type strings → canonical platform_type
  PAIR_RULES          — map domain fragments → shared company_id for
                        sites belonging to the same company but
                        addressing different audiences
  HQ_REGION_RULES     — map domain fragments → 'north' | 'south' for
                        headquarters region analysis
"""

import sqlite3
import re
import sys
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root so config is importable — adjust if needed
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent))
from config.config import WEBSITES

# ---------------------------------------------------------------------------
# Config — change DB_PATH to match your setup
# ---------------------------------------------------------------------------
DB_PATH = "data/scraping.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Parse WEBSITES config into structured platform metadata
# ---------------------------------------------------------------------------

# Maps fragments found in the type string to canonical platform_type values.
# These canonical values are used to group sites by business model in
# analysis:
#   crowd_market   — Algorithmic Crowd Market platforms (MTurk-style)
#   enterprise_bpo — Managed Enterprise BPO platforms (B2B outsourcing)
#   impact_sourcing — Impact Sourcing platforms (Global South focus)
# Edit this if your type strings use different wording.
PLATFORM_TYPE_RULES = [
    ("Algorithmic Crowd Market", "crowd_market"),
    ("Managed Enterprise BPO",  "enterprise_bpo"),
    ("Impact Sourcing",         "impact_sourcing"),
]

# Maps domain fragments to company_id for linking pairs.
# A "pair" is two domains owned by the same company that address
# different audiences: one B2B (client-facing) and one B2W
# (worker-facing).  company_id links them so within-pair analyses can
# compare the same company's language across audiences, controlling for
# company-level style variation.
#
# Key: any substring of the domain name (lowercase)
# Value: canonical company identifier string
# Example: appen.com (B2B) and crowdgen.com (B2W) both map to "appen"
PAIR_RULES = {
    "appen":        "appen",
    "crowdgen":     "appen",
    "toloka":      "toloka",
    "mindrift":    "toloka",
    "centific":  "centific",
    "oneforma":  "centific",
    "labelbox":  "labelbox",
    "alignerr":  "labelbox",
    "scale":      "scale",
    "remotasks":  "scale"
}

# Maps domain fragments to headquarters region.
# 'south' = Global South (Africa, South Asia) — impact sourcing focus
# 'north' = Global North (North America, Europe) — default
# This variable is available for potential Global North/South comparison
# analyses (not the primary focus of the H1a-c theses).
HQ_REGION_RULES = {
    "sama":          "south",   # Kenya-focused impact sourcing
    "imerit":        "south",   # India-headquartered
    "cloudfactory":  "south",   # Nepal/Kenya operations
    "defined":       "south",   # Kenya-focused
    # Everything else defaults to "north" below
}


def parse_platform_type(type_str: str) -> str:
    """
    Extract canonical platform_type from the free-text type field in config.

    Iterates PLATFORM_TYPE_RULES in order and returns the canonical value
    for the first matching fragment.  Case-insensitive comparison.

    Args:
        type_str: The raw type string from config (e.g. "Algorithmic
                  Crowd Market — Microtask").

    Returns:
        Canonical platform_type string, or "unknown" if no rule matches.
    """
    for fragment, canonical in PLATFORM_TYPE_RULES:
        if fragment.lower() in type_str.lower():
            return canonical
    return "unknown"


def parse_company_id(domain: str) -> str:
    """
    Return a company_id that links paired domains to the same company.

    Used to group appen.com (B2B) and crowdgen.com (B2W) under the same
    company_id = "appen" so corpus_view can filter:
        WHERE company_id = 'appen'   -- both paired domains
    enabling within-pair comparisons that control for company-level
    language variation.

    Args:
        domain: Domain string as in WEBSITES config (e.g. "crowdgen.com").

    Returns:
        company_id string (e.g. "appen"), or the domain itself if no
        PAIR_RULES entry matches.  Using the domain as fallback means
        single-domain platforms still work in platform-level analyses.
    """
    for fragment, company_id in PAIR_RULES.items():
        if fragment in domain.lower():
            return company_id
    # Default: use domain root (e.g. "scale.com" → "scale.com")
    return domain


def parse_hq_region(domain: str) -> str:
    """
    Return 'north' or 'south' based on domain fragment matching.

    Args:
        domain: Domain string.

    Returns:
        'south' if a HQ_REGION_RULES fragment matches, otherwise 'north'.
    """
    for fragment, region in HQ_REGION_RULES.items():
        if fragment in domain.lower():
            return region
    return "north"


def build_platforms_records() -> list[dict]:
    """
    Convert WEBSITES config into a list of platform metadata dicts.

    Each dict becomes one row in the platforms table.  Logs the derived
    metadata for manual verification before any DB writes so errors in
    PAIR_RULES or PLATFORM_TYPE_RULES can be caught before they
    propagate.

    Returns:
        List of dicts with keys: domain, name, audience, platform_type,
        company_id, hq_region, type_raw.
    """
    records = []
    for domain, site in WEBSITES.items():
        record = {
            "domain":        domain,
            "name":          site.get("name", domain),
            "audience":      site.get("audience", "unknown"),
            "platform_type": parse_platform_type(site.get("type", "")),
            "company_id":    parse_company_id(domain),
            "hq_region":     parse_hq_region(domain),
            "type_raw":      site.get("type", ""),
        }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Step 2: Create platforms table and populate it
# ---------------------------------------------------------------------------

CREATE_PLATFORMS_SQL = """
CREATE TABLE IF NOT EXISTS platforms (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    domain        TEXT UNIQUE NOT NULL,
    name          TEXT,
    audience      TEXT,         -- 'worker' | 'client' | 'both'
    platform_type TEXT,         -- 'crowd_market' | 'enterprise_bpo' | 'impact_sourcing'
    company_id    TEXT,         -- shared key linking paired domains (e.g. 'appen')
    hq_region     TEXT,         -- 'north' | 'south'
    type_raw      TEXT,         -- original type string from config for reference
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# INSERT OR REPLACE so re-running this script updates stale metadata
# without leaving orphaned rows
INSERT_PLATFORM_SQL = """
INSERT OR REPLACE INTO platforms
    (domain, name, audience, platform_type, company_id, hq_region, type_raw)
VALUES
    (:domain, :name, :audience, :platform_type, :company_id, :hq_region, :type_raw)
"""


def create_platforms_table(conn: sqlite3.Connection):
    """Create the platforms table if it does not yet exist."""
    conn.execute(CREATE_PLATFORMS_SQL)
    conn.commit()
    log.info("platforms table ready.")


def populate_platforms(conn: sqlite3.Connection, records: list[dict]):
    """
    Insert or update platform records.

    Uses INSERT OR REPLACE so the script is safely re-runnable — any
    metadata changes in PAIR_RULES or PLATFORM_TYPE_RULES are applied
    on the next run.

    Args:
        records: List of dicts from build_platforms_records().
    """
    conn.executemany(INSERT_PLATFORM_SQL, records)
    conn.commit()
    log.info(f"Inserted/updated {len(records)} platform records.")


# ---------------------------------------------------------------------------
# Step 3: Create corpus_view — used by ALL analysis scripts (02–05)
# ---------------------------------------------------------------------------

def create_corpus_view(conn: sqlite3.Connection):
    """
    Create (or recreate) the canonical corpus_view SQL view.

    corpus_view is the single authoritative source for token data with
    platform metadata.  Every analysis script (02, 02b, 02c, 03, 03b,
    04) queries it directly rather than re-joining the underlying tables.

    Why a view and not a table?
      A view ensures that any new pages added to pages_tfidf (from a
      supplementary scraping run) automatically appear in the corpus
      without needing to re-run this script.  The join is evaluated at
      query time.

    Audience derivation:
      The view joins platforms.audience (from config) rather than
      pages_tfidf.audience (from URL pattern matching in preprocess.py).
      This is critical: pages_tfidf.audience contains many 'unknown'
      values for pages whose URLs did not match the pattern rules.
      The config-derived audience is 100% reliable.

    Key columns:
      page_id        : primary key from pages_tfidf — used to join back
                       to raw text in Export B/C (04_step2_export.py)
      url            : page URL for close reading (Step 2)
      unigrams       : JSON array of lemmatised unigrams (from preprocess.py)
      bigrams        : JSON array of lemmatised bigrams (from preprocess.py)
      token_count    : total unigram count — used for MIN_TOKEN_COUNT
                       filters and relative-frequency normalisation
      audience       : 'client' | 'worker' (from platforms, authoritative)
      platform_type  : 'crowd_market' | 'enterprise_bpo' | 'impact_sourcing'
      company_id     : shared key for paired platforms (e.g. 'appen')
      hq_region      : 'north' | 'south'
      platform_name  : human-readable site name (from config)
      domain         : domain string (e.g. 'appen.com')

    Typical queries:
      SELECT * FROM corpus_view WHERE audience = 'worker'
      SELECT * FROM corpus_view WHERE platform_type = 'crowd_market'
      SELECT * FROM corpus_view WHERE company_id = 'appen'  -- pair analysis
    """
    conn.execute("DROP VIEW IF EXISTS corpus_view")
    conn.execute("""
        CREATE VIEW corpus_view AS
        SELECT
            t.page_id,
            t.url,
            t.unigrams,
            t.bigrams,
            t.token_count,
            -- Audience from config via platforms, not from pages_tfidf
            pl.audience,
            pl.platform_type,
            pl.company_id,
            pl.hq_region,
            pl.name        AS platform_name,
            w.domain
        FROM pages_tfidf t
        JOIN pages    pg ON pg.id        = t.page_id
        JOIN websites w  ON w.id         = pg.website_id
        JOIN platforms pl ON pl.domain   = REPLACE(w.domain, 'www.', '')
    """)
    conn.commit()
    log.info("corpus_view created — all analysis scripts should query this view.")


# ---------------------------------------------------------------------------
# Step 4: Corpus diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(conn: sqlite3.Connection):
    """
    Validate corpus_view and surface potential data quality issues.

    Compares pages_tfidf.audience (unreliable, URL-derived) to
    corpus_view.audience (reliable, config-derived) to quantify how
    many pages had 'unknown' audience that are now correctly resolved.

    Also reports:
      - Pages dropped between pages_tfidf and corpus_view (no matching
        platform in config — these pages have no derivable audience and
        are excluded from all analysis).
      - Pages-per-platform table with token counts.
      - Platform pairs (company_id groups with > 1 domain).
      - Corpus-wide token statistics.
      - Platform type distribution.

    This output should be reviewed manually before proceeding to
    02_step1_frequency.py to verify:
      - All expected platforms appear
      - Audience assignments look correct
      - Pairs are linked correctly
      - No large unexplained drops in page count

    Args:
        conn: Open SQLite connection with corpus_view already created.
    """
    log.info("=" * 60)
    log.info("CORPUS DIAGNOSTICS")
    log.info("=" * 60)

    # --- Before: audience distribution in raw pages_tfidf ---
    total_raw = conn.execute("SELECT COUNT(*) FROM pages_tfidf").fetchone()[0]
    log.info(f"pages_tfidf total rows: {total_raw}")
    log.info("Audience in pages_tfidf (raw, unreliable):")
    for row in conn.execute("""
        SELECT audience, COUNT(*) as n FROM pages_tfidf GROUP BY audience ORDER BY n DESC
    """).fetchall():
        pct = 100 * row[1] / total_raw if total_raw else 0
        flag = "  ← PROBLEM" if row[0] == "unknown" else ""
        log.info(f"  {row[0]:<12} {row[1]:>6} pages ({pct:.1f}%){flag}")

    # --- After: audience from corpus_view (config-derived) ---
    log.info("-" * 60)
    total_view = conn.execute("SELECT COUNT(*) FROM corpus_view").fetchone()[0]
    log.info(f"corpus_view total rows: {total_view}")

    dropped = total_raw - total_view
    if dropped > 0:
        log.warning(
            f"  {dropped} pages dropped from corpus_view — their website has no "
            f"matching entry in platforms. Check that all scraped domains are in WEBSITES config."
        )

    log.info("Audience in corpus_view (config-derived, reliable):")
    for row in conn.execute("""
        SELECT audience, COUNT(*) as n FROM corpus_view GROUP BY audience ORDER BY n DESC
    """).fetchall():
        pct = 100 * row[1] / total_view if total_view else 0
        log.info(f"  {row[0]:<12} {row[1]:>6} pages ({pct:.1f}%)")

    # --- Pages per platform ---
    log.info("-" * 60)
    log.info("Pages per platform:")
    for row in conn.execute("""
        SELECT domain, platform_name, audience, platform_type, company_id, hq_region,
               COUNT(*) as n_pages, SUM(token_count) as total_tokens
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
            f"pages={row['n_pages']:>5}  tokens={row['total_tokens']:>8}"
        )

    # --- Platform pairs ---
    log.info("-" * 60)
    log.info("Platform pairs (company_id links B2B + B2W domains):")
    pair_rows = conn.execute("""
        SELECT company_id,
               GROUP_CONCAT(domain, ' | ')    AS domains,
               GROUP_CONCAT(audience, ' | ')  AS audiences,
               COUNT(DISTINCT domain)          AS n_domains
        FROM platforms
        GROUP BY company_id
        HAVING n_domains > 1
        ORDER BY company_id
    """).fetchall()
    if pair_rows:
        for row in pair_rows:
            log.info(f"  {row['company_id']:<15} → {row['domains']}  [{row['audiences']}]")
    else:
        log.info("  No pairs found — check PAIR_RULES in this script.")

    # --- Token stats ---
    log.info("-" * 60)
    stats = conn.execute("""
        SELECT MIN(token_count) as mn, MAX(token_count) as mx,
               AVG(token_count) as avg, SUM(token_count) as total
        FROM corpus_view
    """).fetchone()
    log.info("Token stats (corpus_view, all audiences):")
    log.info(f"  Min={stats['mn']}  Max={stats['mx']}  "
             f"Avg={stats['avg']:.0f}  Total={stats['total']:,}")

    # --- Platform type distribution ---
    log.info("-" * 60)
    log.info("Platform type distribution:")
    for row in conn.execute("""
        SELECT platform_type, COUNT(DISTINCT domain) as n_platforms,
               COUNT(*) as n_pages
        FROM corpus_view
        GROUP BY platform_type
    """).fetchall():
        log.info(f"  {row['platform_type']:<20} {row['n_platforms']} platforms  {row['n_pages']} pages")

    log.info("=" * 60)
    log.info("Done. If audience distribution and pairs look correct, proceed to 02_step1_frequency.py")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Orchestrate platform enrichment and corpus view creation.

    Exits with FileNotFoundError if the database does not exist — this
    script should only be run after the scraping and preprocessing stages
    have created the DB.

    Steps:
      1. Parse WEBSITES config and log derived metadata for review
      2. Create/update platforms table
      3. Create corpus_view (DROP + CREATE for clean re-runs)
      4. Run diagnostics — review before proceeding to analysis scripts
    """
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("01_prepare.py — Platform enrichment and corpus validation")
    log.info("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Step 1: Parse config and show metadata for manual verification
    log.info("Parsing WEBSITES config...")
    records = build_platforms_records()
    log.info(f"Found {len(records)} platform entries. Verify metadata:")
    for r in records:
        log.info(
            f"  {r['domain']:<30} "
            f"audience={r['audience']:<8} "
            f"type={r['platform_type']:<18} "
            f"company={r['company_id']:<15} "
            f"region={r['hq_region']}"
        )

    # Step 2: Populate platforms table
    create_platforms_table(conn)
    populate_platforms(conn, records)

    # Step 3: Create corpus_view — single join used by all analysis scripts
    create_corpus_view(conn)

    # Step 4: Diagnostics
    run_diagnostics(conn)

    conn.close()
    log.info("Done. Run 01_prepare_additions.py next, then 02_step1_frequency.py.")


if __name__ == "__main__":
    main()
