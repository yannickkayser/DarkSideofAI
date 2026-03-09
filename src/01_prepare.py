"""
01_prepare.py
=============
Enriches the existing SQLite database with a `platforms` table and a
`corpus_view` derived from WEBSITES config, then validates the corpus.

What this script does:
  1. Parses WEBSITES config → platform_type, company_id, hq_region
  2. Creates and populates the `platforms` table
  3. Creates `corpus_view` — the single join all analysis scripts use
  4. Runs corpus diagnostics and flags problems

IMPORTANT — audience source of truth:
  The audience column in pages_tfidf is unreliable (unknown values from
  URL matching failures in preprocess.py). This script ignores it entirely.
  All scripts derive audience from platforms via:
    pages_tfidf → pages → websites → platforms
  corpus_view encodes this join once so no script repeats it.

Run this ONCE before any analysis script.
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
# Edit this if your type strings use different wording.
PLATFORM_TYPE_RULES = [
    ("Algorithmic Crowd Market", "crowd_market"),
    ("Managed Enterprise BPO",  "enterprise_bpo"),
    ("Impact Sourcing",         "impact_sourcing"),
]

# Maps domain fragments to company_id for linking pairs.
# Add your pairs here — key is any substring of the domain, value is a
# short canonical company identifier.
# Example: appen.com and crowdgen.com both map to "appen"
PAIR_RULES = {
    "appen":        "appen",
    "crowdgen":     "appen",
    "toloka":      "toloka",
    "mindrift":    "toloka",     
    "centific":  "centific",
    "oneforma":  "centific",
    "labelbox":  "labelbox",
    "alignerr":  "labelbox"
}

# Maps domain fragments to headquarters region.
# Expand this list to cover all your platforms.
HQ_REGION_RULES = {
    "sama":          "south",   # Kenya-focused impact sourcing
    "imerit":        "south",   # India-headquartered
    "cloudfactory":  "south",   # Nepal/Kenya operations
    "defined":       "south",   # Kenya-focused
    # Everything else defaults to "north" below
}


def parse_platform_type(type_str: str) -> str:
    """Extract canonical platform_type from the free-text type field."""
    for fragment, canonical in PLATFORM_TYPE_RULES:
        if fragment.lower() in type_str.lower():
            return canonical
    return "unknown"


def parse_company_id(domain: str) -> str:
    """
    Return a company_id that links paired domains to the same company.
    Falls back to the domain itself if no pair rule matches.
    """
    for fragment, company_id in PAIR_RULES.items():
        if fragment in domain.lower():
            return company_id
    # Default: use domain root (e.g. "scale.com" → "scale.com")
    return domain


def parse_hq_region(domain: str) -> str:
    """Return 'north' or 'south' based on domain fragment matching."""
    for fragment, region in HQ_REGION_RULES.items():
        if fragment in domain.lower():
            return region
    return "north"


def build_platforms_records() -> list[dict]:
    """
    Convert WEBSITES config into a list of platform metadata dicts.
    Each dict becomes one row in the platforms table.
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

INSERT_PLATFORM_SQL = """
INSERT OR REPLACE INTO platforms
    (domain, name, audience, platform_type, company_id, hq_region, type_raw)
VALUES
    (:domain, :name, :audience, :platform_type, :company_id, :hq_region, :type_raw)
"""


def create_platforms_table(conn: sqlite3.Connection):
    conn.execute(CREATE_PLATFORMS_SQL)
    conn.commit()
    log.info("platforms table ready.")


def populate_platforms(conn: sqlite3.Connection, records: list[dict]):
    conn.executemany(INSERT_PLATFORM_SQL, records)
    conn.commit()
    log.info(f"Inserted/updated {len(records)} platform records.")


# ---------------------------------------------------------------------------
# Step 3: Create corpus_view — used by ALL analysis scripts (02–05)
# ---------------------------------------------------------------------------

def create_corpus_view(conn: sqlite3.Connection):
    """
    A single reusable view that joins pages_tfidf to platform metadata.

    Audience is derived from platforms (config), NOT from pages_tfidf.audience,
    which contains unreliable 'unknown' values from URL-matching failures.

    All analysis scripts query this view directly:
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
    Validates corpus_view and shows what changed vs pages_tfidf.audience.
    Flags any pages that could not be joined to platforms (unresolvable audience).
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
    log.info("Done. Run 02_step1_frequency.py next.")


if __name__ == "__main__":
    main()
