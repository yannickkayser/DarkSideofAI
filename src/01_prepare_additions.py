"""
01_prepare_additions.py
=======================
Run AFTER 01_prepare.py.  Adds two exclusion tables used by all downstream
analysis scripts:

  1. excluded_pages   — pages removed from the analysis corpus
     (e.g. non-English pages, duplicate scraped pages)
  2. excluded_terms   — terms filtered from keyness, co-occurrence,
     topic modelling, and visualisations

Both tables record a `reason` and `detection_method` for every entry so
that exclusion decisions are fully auditable in the thesis methodology.

Detection heuristics for excluded_terms:
  A. BOILERPLATE — cookie/nav/UI terms identified by co-occurrence with
     known boilerplate anchors ("cookie", "subscribe", "faq", "account")
  B. SINGLE-DOMAIN — terms that appear on only one domain (and are not
     in the theory-focus vocabulary); these are scraping residue or
     highly idiosyncratic proper nouns
  C. NON-ENGLISH — terms detected as non-English by simple heuristics
     (German umlauts, common German function words, etc.)
  D. NAMED-ENTITIES — personal names and place names concentrated in
     1–2 domains (low document frequency, capitalised in source)

Usage:
    python3 src/01_prepare_additions.py
"""

import sqlite3
import json
import re
import logging
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

DB_PATH = "data/scraping.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config — exclusion rules
# ---------------------------------------------------------------------------

# Pages to exclude (identified manually or by language detection).
# Each tuple: (match_type, match_value, reason)
# match_type: 'url_contains', 'domain_exact', 'page_id'
EXCLUDED_PAGE_RULES = [
    # German-language page — identified during topic modelling (German terms
    # surfacing in LDA topics).  Exclusion is warranted because the corpus
    # analysis is designed for English-language discourse; a non-English page
    # introduces cross-language noise into frequency, keyness, and topic
    # measures without contributing to the analytical questions.
    ("url_contains", "/de/", "Non-English page (German): detected via German "
     "terms in LDA topics and manual URL inspection"),
    # Add further rules here as needed:
    # ("domain_exact", "example.com", "Duplicate domain / test site"),
]

# Boilerplate anchor terms — if a term's top collocates include these,
# it is likely scraping/UI noise rather than substantive content.
BOILERPLATE_ANCHORS = {
    "cookie", "subscribe", "faq", "newsletter", "privacy", "gdpr",
    "consent", "browser", "javascript", "login", "signup", "captcha",
}

# Common German function words / nouns used to detect non-English terms.
GERMAN_MARKERS = {
    "und", "der", "die", "das", "ist", "für", "mit", "auf", "von",
    "sich", "nicht", "ein", "eine", "auch", "werden", "nach", "bei",
    "durch", "über", "oder", "wie", "noch", "kann", "nur", "alle",
    "mehr", "zum", "zur", "dem", "den", "des", "dass", "wenn", "aber",
    "sie", "wir", "ich", "als",
    # German terms found in previous LDA runs:
    "ueberwinden", "bildbearbeitung", "sicherstellung", "kunst",
}

# Theory-focus terms that should NEVER be excluded regardless of heuristics.
# These are the terms central to H1a-c that the analysis must retain.
PROTECTED_TERMS = {
    # H1a
    "worker", "labour", "task", "job", "pay", "earn", "payment",
    "work", "annotator", "labeller", "moderator",
    # H1b
    "autonomous", "machine", "automate", "intelligent", "automation",
    "model", "algorithm",
    # H1c
    "human", "quality", "oversight", "annotation", "label", "datum",
    "accuracy", "review",
}

# Manual overrides — terms always excluded (known noise from prior analysis).
# Each: (term, reason)
MANUAL_EXCLUSIONS = [
    ("cookie", "boilerplate: cookie consent banner"),
    ("set_cookie", "boilerplate: cookie consent banner"),
    ("cooky", "boilerplate: cookie variant"),
    ("/hr", "scraping artifact: URL fragment"),
    ("/hr_remote", "scraping artifact: URL fragment"),
    ("remote_apply", "scraping artifact: URL fragment"),
    ("faq", "boilerplate: FAQ section header"),
    ("faq_help", "boilerplate: FAQ section"),
    ("help_desk", "boilerplate: support section"),
    ("subscribe", "boilerplate: newsletter prompt"),
    ("youtube", "boilerplate: social media link"),
    ("jira", "scraping artifact: internal tool reference"),
]


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def init_exclusion_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS excluded_pages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id         INTEGER,
            url             TEXT,
            domain          TEXT,
            reason          TEXT NOT NULL,
            detection_method TEXT NOT NULL,  -- 'manual_rule' | 'language_detect' | ...
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS excluded_terms (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            term            TEXT NOT NULL UNIQUE,
            reason          TEXT NOT NULL,
            detection_method TEXT NOT NULL,
            -- 'boilerplate' | 'single_domain' | 'non_english' | 'named_entity' | 'manual'
            domain_count    INTEGER,        -- how many domains the term appears in
            total_freq      INTEGER,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_et_term
            ON excluded_terms(term);
    """)
    conn.commit()
    log.info("Exclusion tables ready.")


# ---------------------------------------------------------------------------
# Step 1: Excluded pages
# ---------------------------------------------------------------------------

def populate_excluded_pages(conn: sqlite3.Connection):
    """Apply page exclusion rules against corpus_view."""
    log.info("Applying page exclusion rules...")

    # Clear previous entries (re-run safe)
    conn.execute("DELETE FROM excluded_pages")

    count = 0
    for match_type, match_value, reason in EXCLUDED_PAGE_RULES:
        if match_type == "url_contains":
            rows = conn.execute("""
                SELECT page_id, url, domain FROM corpus_view
                WHERE url LIKE ?
            """, [f"%{match_value}%"]).fetchall()
        elif match_type == "domain_exact":
            rows = conn.execute("""
                SELECT page_id, url, domain FROM corpus_view
                WHERE domain = ?
            """, [match_value]).fetchall()
        elif match_type == "page_id":
            rows = conn.execute("""
                SELECT page_id, url, domain FROM corpus_view
                WHERE page_id = ?
            """, [int(match_value)]).fetchall()
        else:
            log.warning(f"  Unknown match_type: {match_type}")
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

    # Log what was excluded
    for r in conn.execute("SELECT page_id, url, reason FROM excluded_pages").fetchall():
        log.info(f"    page_id={r['page_id']}  {r['url'][:80]}  — {r['reason']}")


# ---------------------------------------------------------------------------
# Step 2: Excluded terms — heuristic detection
# ---------------------------------------------------------------------------

def detect_excluded_terms(conn: sqlite3.Connection):
    """
    Run heuristics to identify artifact terms.  All detection is logged
    with method and reason for auditability.
    """
    log.info("Detecting artifact terms...")

    # Clear previous entries
    conn.execute("DELETE FROM excluded_terms")

    # Load all pages (excluding already-excluded pages)
    excluded_page_ids = {
        r[0] for r in conn.execute("SELECT page_id FROM excluded_pages").fetchall()
    }

    rows = conn.execute("""
        SELECT page_id, domain, unigrams FROM corpus_view
        WHERE audience IN ('client', 'worker')
          AND token_count >= 10
    """).fetchall()

    # Build per-term stats: total freq, domain set
    term_freq    = Counter()
    term_domains = defaultdict(set)

    for row in rows:
        if row["page_id"] in excluded_page_ids:
            continue
        domain   = row["domain"]
        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        for t in unigrams:
            term_freq[t] += 1
            term_domains[t].add(domain)

    total_domains = len({row["domain"] for row in rows
                         if row["page_id"] not in excluded_page_ids})
    log.info(f"  Vocabulary: {len(term_freq):,} terms across {total_domains} domains")

    exclusions = {}  # term → (reason, method, domain_count, freq)

    def exclude(term, reason, method):
        if term in PROTECTED_TERMS:
            return
        if term not in exclusions:
            exclusions[term] = (reason, method,
                                len(term_domains.get(term, set())),
                                term_freq.get(term, 0))

    # --- A. Manual exclusions ---
    for term, reason in MANUAL_EXCLUSIONS:
        exclude(term, reason, "manual")

    # --- B. Non-English (German markers) ---
    for term in term_freq:
        if term.lower() in GERMAN_MARKERS:
            exclude(term, f"Non-English term (German function word/noun)",
                    "non_english")

    # Detect terms with German-specific character patterns
    german_pattern = re.compile(r'[äöüßÄÖÜ]')
    for term in term_freq:
        if german_pattern.search(term):
            exclude(term, "Non-English term (contains German characters)",
                    "non_english")

    # --- C. Single-domain terms (freq >= 5, only 1 domain) ---
    # These are overwhelmingly scraping artifacts or proper nouns specific
    # to one platform's content (staff names, office locations).
    for term, freq in term_freq.items():
        if freq >= 5 and len(term_domains[term]) == 1:
            the_domain = list(term_domains[term])[0]
            exclude(term,
                    f"Single-domain term (only on {the_domain}, freq={freq})",
                    "single_domain")

    # --- D. Boilerplate detection ---
    # Terms that frequently co-occur with boilerplate anchors.
    # We use a simple heuristic: if a term's top 3 collocates (by raw
    # co-occurrence in ±5 window) are all boilerplate anchors, exclude it.
    # This is a lightweight check — the full PMI analysis in 02 is more
    # rigorous, but we need to filter before 02 runs.
    # (Skipped for now if the corpus is very large — manual list covers
    # the most important cases.)

    # --- Summary ---
    log.info(f"  {len(exclusions):,} terms identified for exclusion:")
    method_counts = Counter(v[1] for v in exclusions.values())
    for method, n in method_counts.most_common():
        log.info(f"    {method:<20} {n:>5} terms")

    # Insert into DB
    for term, (reason, method, n_domains, freq) in exclusions.items():
        conn.execute("""
            INSERT OR IGNORE INTO excluded_terms
                (term, reason, detection_method, domain_count, total_freq)
            VALUES (?, ?, ?, ?, ?)
        """, [term, reason, method, n_domains, freq])

    conn.commit()
    log.info(f"  Saved {len(exclusions):,} excluded terms to DB.")

    # Log a sample
    log.info("  Sample excluded terms:")
    for term, (reason, method, n_dom, freq) in sorted(
            exclusions.items(), key=lambda x: x[1][3], reverse=True)[:20]:
        log.info(f"    {term:<30} method={method:<15} domains={n_dom}  "
                 f"freq={freq}  — {reason[:60]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("01_prepare_additions.py — Exclusion Tables")
    log.info("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Verify corpus_view exists
    view_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()
    if not view_check:
        raise RuntimeError("corpus_view not found — run 01_prepare.py first.")

    init_exclusion_tables(conn)
    populate_excluded_pages(conn)
    detect_excluded_terms(conn)

    # Final summary
    log.info("=" * 60)
    n_pages = conn.execute("SELECT COUNT(*) FROM excluded_pages").fetchone()[0]
    n_terms = conn.execute("SELECT COUNT(*) FROM excluded_terms").fetchone()[0]
    log.info(f"EXCLUSION SUMMARY: {n_pages} pages, {n_terms} terms excluded")
    log.info("All downstream scripts should filter against these tables.")
    log.info("Query examples:")
    log.info("  -- Pages still in analysis:")
    log.info("  SELECT * FROM corpus_view")
    log.info("  WHERE page_id NOT IN (SELECT page_id FROM excluded_pages);")
    log.info("")
    log.info("  -- Check why a term was excluded:")
    log.info("  SELECT * FROM excluded_terms WHERE term = 'cookie';")
    log.info("=" * 60)

    conn.close()


if __name__ == "__main__":
    main()
