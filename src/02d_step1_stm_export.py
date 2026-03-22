"""
02d_step1_stm_export.py
========================
Exports the DarkSideofAI corpus from SQLite to two CSV files ready for
Structural Topic Modelling in RStudio (stm package).

Usage
-----
    python3 src/02d_step1_stm_export.py

Output
------
    output/step_1/stm/corpus_export.csv    — one row per page, tokenised text
    output/step_1/stm/metadata_export.csv  — one row per page, covariates only
    output/step_1/stm/export_summary.txt   — human-readable data description
                                             (read this before loading in R)

Prerequisites
-------------
    01_prepare.py           must have been run (creates corpus_view)
    01_prepare_additions.py must have been run (creates excluded_* tables)

==============================================================================
DATA DESCRIPTION — read before loading in RStudio
==============================================================================

SOURCE
------
The data originates from a web scraping project targeting AI data annotation
platforms.  Two types of platforms were scraped:

  client-facing (B2B) — platforms that market AI data services TO businesses.
                         These pages address procurement managers, product
                         owners, and technical leads evaluating vendors.

  worker-facing (B2W) — platforms that recruit and communicate WITH human
                         annotators, labellers, and crowd workers.
                         These pages address people seeking paid micro-task work.

The scraped pages were stored in a SQLite database (data/scraping.db).

TABLE STRUCTURE (relevant tables only)
---------------------------------------
  pages_tfidf   — one row per scraped page.  Contains pre-processed token
                  lists (unigrams, bigrams) stored as JSON arrays.
  pages         — page metadata (url, website_id).
  websites      — one row per scraped domain (domain name, audience label
                  from config).
  platforms     — derived from WEBSITES config in config/config.py.
                  Contains the authoritative audience label, platform_type,
                  company_id (links paired domains), and hq_region.
  corpus_view   — a SQL VIEW joining pages_tfidf → pages → websites →
                  platforms.  This is the single source used by all
                  analysis scripts (02a, 02b, 02c, 02d).

TOKENISATION (performed in preprocess.py, BEFORE this script)
--------------------------------------------------------------
The unigrams column in pages_tfidf contains tokens that have already been
through the following pipeline:

  1. HTML stripping — boilerplate, navigation, and cookie banners removed.
  2. Lowercasing — all tokens are lowercase.
  3. Lemmatisation — tokens are lemmatised (e.g. "running" → "run",
     "annotators" → "annotator").  SpaCy was used for lemmatisation.
  4. Stopword removal — English function words removed via SpaCy's default
     stopword list.
  5. Short token filter — tokens shorter than 2 characters removed.
  6. The result is stored as a JSON array, e.g.:
         ["annotation", "task", "worker", "quality", "model", ...]

AUDIENCE LABELS
---------------
The audience column in corpus_view is derived from the WEBSITES config in
config/config.py, via the platforms table.  It is NOT taken from the
pages_tfidf.audience column, which was found to be unreliable (many
'unknown' values from URL matching failures during scraping).

Each domain is manually assigned in config.py as either:
  "client"  — B2B platform addressing businesses (buyer side)
  "worker"  — B2W platform addressing annotators (labour side)

A small number of platforms carry a "both" audience label — their pages
address both clients and workers (e.g. a platform's generic landing page).
These pages ARE included in this export with audience = "both" so you can
decide how to handle them in R (see HANDLING 'BOTH' PAGES IN R below).

PAIRED DOMAINS
--------------
Some companies operate both a client-facing and a worker-facing domain.
These paired domains are linked via the company_id field:
  appen.com    ↔  crowdgen.com      (company_id = "appen")
  scale.com    ↔  remotasks.com     (company_id = "scale")
  toloka.ai    ↔  mindrift.ai       (company_id = "toloka")
  centific.com ↔  oneforma.com      (company_id = "centific")
  labelbox.com ↔  alignerr.com      (company_id = "labelbox")

These pairs are the most analytically valuable cases: the same company
addressing two different audiences through structurally different language.

EXCLUSION FILTERS APPLIED IN THIS EXPORT
-----------------------------------------
Three exclusion layers are applied before writing to CSV:

  1. excluded_pages (from 01_prepare_additions.py)
     A table of page_ids manually flagged as unsuitable (e.g. login pages,
     404 redirects, non-English pages).  These pages are skipped entirely.

  2. excluded_terms (from 01_prepare_additions.py)
     A vocabulary of terms to remove from token lists before modelling.
     These were identified during initial corpus inspection as platform-
     specific noise (e.g. internal product names, navigational strings).
     They are also passed as stop_words to the STM CountVectorizer equivalent
     so they cannot contribute to any topic.

  3. EXTRA_STOP_WORDS (defined in this script, see config section below)
     Additional artifact terms identified during LDA topic inspection.
     The most important group is CALENDAR NOISE: month names (january,
     february, ..., december) and their abbreviations appear as tokens
     because many B2W platforms display job posting dates inline with page
     content.  These created an artifact topic in the LDA (T5:
     "task, february, june, december") and must be excluded before STM.
     Other artifact types (UI strings, platform names) are also included.

  4. MIN_TOKEN_COUNT filter
     Pages with fewer than 30 tokens AFTER term removal are excluded.
     Short pages produce unstable topic assignments because the LDA/STM
     document representation is too sparse to be reliable.

OUTPUT CSV COLUMNS
------------------
corpus_export.csv:
  page_id   — integer, primary key from pages_tfidf.  Use this to join
               back to any table in scraping.db.
  audience  — "client" or "worker" (from platforms config, reliable).
  domain    — scraped domain (e.g. "imerit.net", "www.microworkers.com").
               Note: www prefix is retained as-is from the database.
  tokens    — space-separated lemmatised tokens, after all exclusion filters.
               This is ready to pass directly to stm::textProcessor() or
               to manual STM document preparation.

metadata_export.csv:
  page_id   — same as corpus_export.csv (join key).
  audience  — same as corpus_export.csv.
  domain    — same as corpus_export.csv.
  company_id — company identifier linking paired domains (see PAIRED DOMAINS
               above).  Useful as an additional covariate in STM.
  platform_type — "crowd_market" | "enterprise_bpo" | "impact_sourcing"
               Describes the business model of the platform.
  hq_region — "north" | "south" (Global North / Global South headquarters).

LOADING IN R
------------
The tokens column contains pre-tokenised, space-separated text.  In R,
use textProcessor() with no internal tokenisation, or build the STM
documents object manually from the token strings.

Recommended R loading approach:

    library(stm)
    corpus <- read.csv("output/step_1/stm/corpus_export.csv")
    meta   <- read.csv("output/step_1/stm/metadata_export.csv")

    # audience as factor — client is the reference level (intercept)
    # Option A: exclude 'both' pages before modelling (cleanest binary contrast)
    corpus <- corpus[corpus$audience != "both", ]
    meta   <- meta[meta$audience   != "both", ]
    meta$audience <- factor(meta$audience, levels = c("client", "worker"))

    # Option B: keep 'both' as a third level (tests whether 'both' pages sit
    # between client and worker in topic space — useful as a robustness check)
    # meta$audience <- factor(meta$audience, levels = c("client", "both", "worker"))

    # Option C: keep 'both' pages but give them a numeric 0.5 weight covariate
    # (only if you want to use audience as a continuous 0/0.5/1 variable)
    # meta$audience_num <- ifelse(meta$audience=="client", 0,
    #                      ifelse(meta$audience=="worker", 1, 0.5))

    # textProcessor expects raw text; tokens are already clean
    # Use lower=FALSE, removestopwords=FALSE, removenumbers=FALSE,
    # removepunctuation=FALSE, stem=FALSE to prevent double-processing
    processed <- textProcessor(
        corpus$tokens,
        metadata         = meta,
        lowercase        = FALSE,
        removestopwords  = FALSE,
        removenumbers    = FALSE,
        removepunctuation = FALSE,
        stem             = FALSE,
        wordLengths      = c(2, Inf)
    )
    out <- prepDocuments(processed$documents, processed$vocab,
                         processed$meta,
                         lower.thresh = 5)   # mirror MIN_DF = 5

    # Fit STM
    stm_model <- stm(
        documents  = out$documents,
        vocab      = out$vocab,
        K          = 35,
        prevalence = ~ audience,
        content    = ~ audience,
        data       = out$meta,
        init.type  = "Spectral",
        seed       = 42
    )

==============================================================================
"""

import sqlite3
import csv
import json
import logging
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH         = "data/scraping.db"
OUTPUT_DIR      = Path("output/step_1/stm")
MIN_TOKEN_COUNT = 30     # minimum tokens per page after exclusion filtering

# Artifact terms identified during LDA inspection (T5: task/february/june/december).
# These are merged with the DB-loaded excluded_terms before exporting.
# Extend this set after each topic model run if new artifacts emerge.
EXTRA_STOP_WORDS = {
    # ── Calendar noise ─────────────────────────────────────────────────────
    # Month names appear in B2W pages because job posting dates are embedded
    # in page content (e.g. "Posted February 2024").  They dominated LDA
    # topic T5 and must be removed.
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
    # ── UI / navigational noise ─────────────────────────────────────────────
    # Strings from cookie banners, navigation bars, footer elements.
    "cookie", "faq", "subscribe", "website", "youtube",
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
# Helpers
# ---------------------------------------------------------------------------

def load_exclusions(conn):
    """Load excluded page IDs and terms; merge EXTRA_STOP_WORDS."""
    excluded_pages = set()
    excluded_terms = set()

    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='excluded_pages'"
    ).fetchone():
        excluded_pages = {
            r[0] for r in conn.execute(
                "SELECT page_id FROM excluded_pages"
            ).fetchall()
        }

    if conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='excluded_terms'"
    ).fetchone():
        excluded_terms = {
            r[0] for r in conn.execute(
                "SELECT term FROM excluded_terms"
            ).fetchall()
        }

    excluded_terms = excluded_terms | EXTRA_STOP_WORDS
    return excluded_pages, excluded_terms


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(conn):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded_pages, excluded_terms = load_exclusions(conn)

    log.info(f"  Excluded pages     : {len(excluded_pages)}")
    log.info(f"  Excluded terms     : {len(excluded_terms)}  "
             f"(DB terms + {len(EXTRA_STOP_WORDS)} EXTRA_STOP_WORDS)")

    # Fetch all eligible pages from corpus_view
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
    """, (MIN_TOKEN_COUNT,)).fetchall()

    log.info(f"  Rows from corpus_view (pre-filter): {len(rows)}")

    corpus_path = OUTPUT_DIR / "corpus_export.csv"
    meta_path   = OUTPUT_DIR / "metadata_export.csv"

    stats = {
        "total_rows":          len(rows),
        "skipped_excluded":    0,
        "skipped_short":       0,
        "written":             0,
        "client":              0,
        "worker":              0,
        "both":                0,
        "domains":             set(),
        "total_tokens":        0,
    }

    with open(corpus_path, "w", newline="", encoding="utf-8") as f_corp, \
         open(meta_path,   "w", newline="", encoding="utf-8") as f_meta:

        corp_writer = csv.writer(f_corp, quoting=csv.QUOTE_ALL)
        meta_writer = csv.writer(f_meta, quoting=csv.QUOTE_ALL)

        corp_writer.writerow(["page_id", "audience", "domain", "tokens"])
        meta_writer.writerow([
            "page_id", "audience", "domain",
            "company_id", "platform_type", "hq_region",
        ])

        for row in rows:
            pid = row["page_id"]

            # Filter 1 — manually excluded pages
            if pid in excluded_pages:
                stats["skipped_excluded"] += 1
                continue

            # Filter 2 — parse and clean token list
            tokens = json.loads(row["unigrams"]) if row["unigrams"] else []
            tokens = [t for t in tokens if t not in excluded_terms]

            # Filter 3 — too short after term removal
            if len(tokens) < MIN_TOKEN_COUNT:
                stats["skipped_short"] += 1
                continue

            aud    = row["audience"]
            domain = row["domain"]

            corp_writer.writerow([pid, aud, domain, " ".join(tokens)])
            meta_writer.writerow([
                pid, aud, domain,
                row["company_id"]    or "",
                row["platform_type"] or "",
                row["hq_region"]     or "",
            ])

            stats["written"]       += 1
            stats[aud]             += 1
            stats["domains"].add(domain)
            stats["total_tokens"]  += len(tokens)

    return stats


def write_summary(stats, excluded_terms):
    """Write a plain-text summary file for easy reading in RStudio."""
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
        f"  Mean tokens per page  : "
        f"{stats['total_tokens']//max(stats['written'],1):,}",
        "",
        "EXCLUSION FILTERS",
        f"  Manually excluded pages (excluded_pages table)  : "
        f"{stats['skipped_excluded']}",
        f"  Pages too short after term removal (<{MIN_TOKEN_COUNT} tokens): "
        f"{stats['skipped_short']}",
        f"  Total exclusion vocabulary                      : "
        f"{len(excluded_terms)} terms",
        "",
        "TOKENISATION STATUS",
        "  Tokens are already lemmatised and lowercased (SpaCy, English model).",
        "  English stopwords already removed (SpaCy default list).",
        "  Month names and UI noise removed by EXTRA_STOP_WORDS filter.",
        "  Do NOT re-stem, re-lemmatise, or re-apply stopwords in R.",
        "",
        "AUDIENCE LABELS",
        "  'client' = B2B platform (addresses businesses buying AI services)",
        "  'worker' = B2W platform (addresses human annotators seeking work)",
        "  'both'   = platform addresses both audiences (e.g. generic landing",
        "             pages, about pages). Included for inspection; handle in R.",
        "  Labels are config-derived (platforms table), not URL-matched.",
        "",
        "HANDLING 'BOTH' PAGES IN R",
        "  Option A — Exclude before fitting (recommended for clean binary contrast):",
        "    corpus <- corpus[corpus$audience != 'both', ]",
        "    meta   <- meta[meta$audience   != 'both', ]",
        "  Option B — Keep as third factor level (robustness / exploratory):",
        "    meta$audience <- factor(meta$audience,",
        "                           levels = c('client','both','worker'))",
        "  Check export_summary.txt for the count of 'both' pages — if it is",
        "  small relative to client/worker, Option A is the safer default.",
        "",
        "LOADING IN R",
        "  See the docstring of 02d_step1_stm_export.py for recommended",
        "  textProcessor() call with all re-processing flags set to FALSE.",
        "",
        "KEY COVARIATE: audience",
        "  Use as both prevalence and content covariate in STM.",
        "    prevalence = ~ audience  -->  which topics appear more per audience",
        "    content    = ~ audience  -->  which words used per topic per audience",
        "  Set reference level: factor(audience, levels=c('client','worker'))",
        "  Positive prevalence estimate = topic more common in WORKER documents.",
        "  Negative prevalence estimate = topic more common in CLIENT documents.",
        "",
        "ADDITIONAL COVARIATES (metadata_export.csv)",
        "  company_id    — links paired domains (e.g. appen.com + crowdgen.com)",
        "  platform_type — crowd_market | enterprise_bpo | impact_sourcing",
        "  hq_region     — north | south  (Global North / Global South HQ)",
        "",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"  Summary    → {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    if not conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone():
        log.error("corpus_view not found — run 01_prepare.py first.")
        sys.exit(1)

    log.info("Exporting corpus for STM ...")
    _, excluded_terms = load_exclusions(conn)
    stats = export(conn)
    conn.close()

    write_summary(stats, excluded_terms)

    log.info("")
    log.info("Export complete.")
    log.info(f"  Pages written  : {stats['written']}")
    log.info(f"  Client (B2B)   : {stats['client']}")
    log.info(f"  Worker (B2W)   : {stats['worker']}")
    log.info(f"  Both           : {stats['both']}")
    log.info(f"  Domains        : {len(stats['domains'])}")
    log.info(f"  Total tokens   : {stats['total_tokens']:,}")
    log.info("")
    log.info(f"  Output dir : {OUTPUT_DIR.resolve()}")
    log.info("  Read export_summary.txt before loading data in RStudio.")


if __name__ == "__main__":
    main()
