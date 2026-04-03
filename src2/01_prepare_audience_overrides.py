"""
01_prepare_audience_overrides.py
=================================
Splits websites coded as audience='both' into page-level client/worker labels
based on URL path patterns.

Rationale
---------
Some platforms operate a single domain that contains both client-facing content
(addressing businesses) and worker-facing content (addressing annotators) under
different URL paths.  Rather than discarding these pages (BOTH_STRATEGY="exclude")
or treating them as a third category, we can use the URL structure to assign
a binary audience label at the page level.

This creates within-platform pairs — the same company speaking to two different
audiences — which are the analytically strongest evidence for H1a, H1b, H1c
because they eliminate company-level confounders.

URL rules defined in this script
----------------------------------
  clickworker.com
    worker : URL path starts with  /clickworker/
    client : all other pages on    clickworker.com

  prolific.com
    worker : URL path starts with  /participants
    client : all other pages on    prolific.com

  opentrain.ai
    worker : URL path starts with  /become-freelancer/
    client : all other pages on    opentrain.ai

How it works
------------
1. Creates table  page_audience_override  (page_id, audience, matched_rule)
2. For each URL rule, finds matching pages in the pages table and inserts
   an override row with audience = 'worker'
3. For each 'both' domain, inserts audience = 'client' for all remaining
   pages not already assigned a worker override
4. Updates corpus_view to check the override table first:
      COALESCE(override.audience, platform.audience)
   This means:
     - Overridden pages get the URL-based label
     - All other pages keep their existing platform-level label
     - The change is non-destructive — no existing data is modified

company_id for new pairs
------------------------
Pages split from a 'both' domain share the same company_id (the domain name),
making them a within-platform pair identical in structure to the cross-domain
pairs (appen.com ↔ crowdgen.com etc.).

Usage
-----
    python3 src/01_prepare_audience_overrides.py

Run this BEFORE 02d_step1_stm_export.py so the export picks up the new labels.
Re-run the export + STM after running this script.
"""

import sqlite3
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = "data/scraping_2.db"

# URL audience rules.
# Each entry: (company_id, core_domain, worker_path_prefix, rule_name)
#
# worker_path_prefix is matched as: url LIKE 'https://%<core_domain><prefix>%'
# Pages on the domain that do NOT match the worker prefix → client.
#
# company_id is used in corpus_view / metadata_export so paired pages are
# linked.  Use the bare domain name (without www) as a stable identifier.

URL_RULES = [
    {
        "company_id":          "clickworker",
        "core_domain":         "clickworker.com",
        "worker_path_prefix":  "/clickworker/",
        "rule_name":           "clickworker_worker_path",
    },
    {
        "company_id":          "prolific",
        "core_domain":         "prolific.com",
        "worker_path_prefix":  "/participants",
        "rule_name":           "prolific_participants_path",
    },
    {
        "company_id":          "opentrain",
        "core_domain":         "opentrain.ai",
        "worker_path_prefix":  "/become-freelancer/",
        "rule_name":           "opentrain_freelancer_path",
    },
]

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
# Helpers
# ---------------------------------------------------------------------------

def get_corpus_view_definition(conn):
    """Return the current CREATE VIEW statement for corpus_view."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()
    return row[0] if row else None


def create_override_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS page_audience_override (
            page_id      INTEGER PRIMARY KEY,
            audience     TEXT    NOT NULL CHECK(audience IN ('client', 'worker')),
            matched_rule TEXT    NOT NULL,
            company_id   TEXT    NOT NULL,
            FOREIGN KEY (page_id) REFERENCES pages(id)
        )
    """)
    conn.execute("DELETE FROM page_audience_override")   # reset on each run
    conn.commit()
    log.info("  page_audience_override table ready (cleared for fresh run)")


def apply_url_rules(conn):
    """
    For each rule:
      1. Find pages whose URL contains core_domain + worker_path_prefix
      2. Insert as audience='worker'
      3. Find all remaining pages on that domain → audience='client'
    """
    total_worker = 0
    total_client = 0

    for rule in URL_RULES:
        company_id  = rule["company_id"]
        core_domain = rule["core_domain"]
        prefix      = rule["worker_path_prefix"]
        rule_name   = rule["rule_name"]

        log.info(f"\n  Processing: {core_domain}")

        # All pages on this domain (via websites table)
        domain_pages = conn.execute("""
            SELECT p.id, p.url
            FROM   pages p
            JOIN   websites w ON w.id = p.website_id
            WHERE  w.domain LIKE ?
        """, (f"%{core_domain}",)).fetchall()

        if not domain_pages:
            log.warning(f"    No pages found for domain matching '%{core_domain}'")
            continue

        log.info(f"    Total pages on domain: {len(domain_pages)}")

        worker_ids = []
        client_ids = []

        for page_id, url in domain_pages:
            # Normalise: strip scheme and www for path matching
            path_part = url.lower()
            # Match worker prefix anywhere in the URL path
            if (f"{core_domain}{prefix}".lower() in path_part or
                    f"www.{core_domain}{prefix}".lower() in path_part):
                worker_ids.append(page_id)
            else:
                client_ids.append(page_id)

        log.info(f"    Worker pages (path '{prefix}'): {len(worker_ids)}")
        log.info(f"    Client pages (remainder)      : {len(client_ids)}")

        # Insert worker overrides
        conn.executemany("""
            INSERT OR REPLACE INTO page_audience_override
                (page_id, audience, matched_rule, company_id)
            VALUES (?, 'worker', ?, ?)
        """, [(pid, rule_name, company_id) for pid in worker_ids])

        # Insert client overrides (remainder of domain)
        conn.executemany("""
            INSERT OR REPLACE INTO page_audience_override
                (page_id, audience, matched_rule, company_id)
            VALUES (?, 'client', ?, ?)
        """, [(pid, f"{company_id}_client_remainder", company_id) for pid in client_ids])

        total_worker += len(worker_ids)
        total_client += len(client_ids)

    conn.commit()
    log.info(f"\n  Total overrides inserted:")
    log.info(f"    worker : {total_worker}")
    log.info(f"    client : {total_client}")
    log.info(f"    total  : {total_worker + total_client}")


def rebuild_corpus_view(conn):
    existing = get_corpus_view_definition(conn)
    if existing:
        log.info("\n  Existing corpus_view found — rebuilding with override support")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _corpus_view_backup (
                version     INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT DEFAULT (datetime('now')),
                definition  TEXT
            )
        """)
        conn.execute(
            "INSERT INTO _corpus_view_backup (definition) VALUES (?)",
            (existing,)
        )
        conn.execute("DROP VIEW corpus_view")
    else:
        log.info("\n  No existing corpus_view — creating fresh")

    conn.execute("""
        CREATE VIEW corpus_view AS
        SELECT
            t.page_id,
            t.url,
            t.segments,
            t.unigrams,
            t.bigrams,
            t.token_count,
            COALESCE(o.audience,    pl.audience)    AS audience,
            COALESCE(o.company_id,  pl.company_id)  AS company_id,
            pl.platform_type,
            pl.hq_region,
            pl.name  AS platform_name,
            w.domain
        FROM       pages_tfidf  t
        JOIN       pages        pg  ON pg.id    = t.page_id
        JOIN       websites     w   ON w.id     = pg.website_id
        JOIN       platforms    pl  ON pl.domain = REPLACE(w.domain, 'www.', '')
        LEFT JOIN  page_audience_override o ON o.page_id = t.page_id
    """)

    conn.commit()
    log.info("  corpus_view rebuilt with override support.")


def print_summary(conn):
    log.info("\n── Override summary ─────────────────────────────────────────────────────")

    rows = conn.execute("""
        SELECT   o.company_id,
                 o.audience,
                 COUNT(*)        AS n_pages,
                 o.matched_rule
        FROM     page_audience_override o
        GROUP BY o.company_id, o.audience
        ORDER BY o.company_id, o.audience
    """).fetchall()

    current_company = None
    for company_id, audience, n, rule in rows:
        if company_id != current_company:
            log.info(f"\n  {company_id}")
            current_company = company_id
        log.info(f"    {audience:<8}  {n:>4} pages")

    # Verify corpus_view sees the new labels
    log.info("\n── corpus_view audience distribution (after override) ────────────────────")
    dist = conn.execute("""
        SELECT audience, COUNT(*) AS n
        FROM   corpus_view
        GROUP  BY audience
        ORDER  BY audience
    """).fetchall()
    for aud, n in dist:
        log.info(f"  {aud:<10}  {n:>5} pages")

    # New pairs
    log.info("\n── New within-platform pairs ─────────────────────────────────────────────")
    pairs = conn.execute("""
        SELECT   company_id,
                 SUM(CASE WHEN audience='client' THEN 1 ELSE 0 END) AS client_pages,
                 SUM(CASE WHEN audience='worker' THEN 1 ELSE 0 END) AS worker_pages
        FROM     page_audience_override
        GROUP BY company_id
        HAVING   client_pages > 0 AND worker_pages > 0
        ORDER BY company_id
    """).fetchall()

    for company_id, c_pages, w_pages in pairs:
        log.info(f"  {company_id:<20}  client={c_pages}  worker={w_pages}")

    log.info(f"\n  {len(pairs)} new within-platform pair(s) created.")
    log.info("  → Re-run 02d_step1_stm_export.py to pick up the new labels.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        log.error(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Check platforms table exists
    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='platforms'"
    ).fetchone():
        log.error("platforms table not found — run 01_prepare.py first.")
        conn.close()
        return

    log.info("Creating page_audience_override table ...")
    create_override_table(conn)

    log.info("\nApplying URL audience rules ...")
    apply_url_rules(conn)

    log.info("\nRebuilding corpus_view with override support ...")
    rebuild_corpus_view(conn)

    print_summary(conn)
    conn.close()

    log.info("\nDone.")
    log.info("Next steps:")
    log.info("  1. python3 src/02d_step1_stm_export.py   (re-export with new labels)")
    log.info("  2. Re-run STM in RStudio from step 4 onwards")
    log.info("  3. Delete cached out.rds and stm_model.rds before re-running")


if __name__ == "__main__":
    main()
