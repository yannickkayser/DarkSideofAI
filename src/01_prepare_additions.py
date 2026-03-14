"""
01_prepare_additions.py
=======================
Creates and populates two exclusion tables used by all downstream
analysis scripts to filter noise from the corpus.

Pipeline position:
  Stage 1b — Exclusion Tables (run AFTER 01_prepare.py, BEFORE any
  analysis scripts)
  Prerequisites: 01_prepare.py (corpus_view must exist)
  Next step:     02_step1_frequency.py (which calls load_exclusions())

What this script does:
  1. excluded_pages — marks individual pages as outside the analysis
     corpus (e.g. non-English pages, duplicate scraped content).
  2. excluded_terms — marks individual terms as noise to be stripped
     from all token lists before frequency / keyness / PMI counting.

Why these tables matter:
  The scraping pipeline is not perfectly clean.  Two types of noise
  contaminate the corpus:
    a) Page-level: a small number of non-English pages were scraped
       (e.g. a German-language page from a site that serves both English
       and German content).  These pages introduce cross-language noise
       into all frequency, keyness, and topic measures.  They are
       excluded by matching URL patterns or page IDs.
    b) Term-level: scraping JavaScript-heavy pages always picks up some
       boilerplate — cookie consent banners, navigation labels, FAQ
       section headers.  These terms appear at high frequency but carry
       no analytical signal about B2B vs B2W language.  They are
       identified and excluded so they do not distort keyness rankings
       or inflate PMI scores.

Both tables record a `reason` and `detection_method` for every entry so
that all exclusion decisions are auditable in the thesis methodology.

Detection heuristics for excluded_terms:
  A. MANUAL — known boilerplate and scraping artifacts (MANUAL_EXCLUSIONS
     list): cookie, faq, subscribe, youtube, etc.
  B. NON_ENGLISH — German function words (GERMAN_MARKERS set) and terms
     containing German-specific characters (ä, ö, ü, ß).  These were
     identified after LDA topics revealed German terms in initial runs.
  C. SINGLE_DOMAIN — terms that appear on only one domain with frequency
     ≥ 5.  These are typically proper nouns (staff names, office
     locations) or platform-specific UI labels that contribute noise to
     keyness and PMI analyses.
  D. BOILERPLATE — terms whose top collocates are all from BOILERPLATE_ANCHORS
     (NOTE: the D heuristic is noted but not fully implemented; the
     manual list covers the most important cases and the full PMI
     analysis in 02_step1_frequency.py provides more rigorous filtering).

PROTECTED_TERMS guard:
  Terms central to H1a, H1b, and H1c are explicitly protected from all
  exclusion heuristics.  Even if "worker" appeared only on one domain,
  it could never be excluded as a single-domain term.  This prevents
  accidental removal of the vocabulary the thesis is testing.
  The PROTECTED_TERMS set is mirrored in 02_step1_frequency.py
  (THEORY_FOCUS_TERMS), which applies the same guard when loading
  exclusions into the analysis.

Output tables written to data/scraping.db:
  excluded_pages   : page_id, url, domain, reason, detection_method
  excluded_terms   : term, reason, detection_method, domain_count,
                     total_freq

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
# match_type options:
#   'url_contains'  — exclude any page whose URL contains match_value
#   'domain_exact'  — exclude all pages from this domain
#   'page_id'       — exclude a specific page by its integer ID
#
# Examples of when to add entries:
#   - Non-English pages discovered during topic modelling
#   - Test/staging pages accidentally scraped
#   - Duplicate pages served under multiple URLs
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
# Used in the D heuristic (boilerplate co-occurrence detection).
BOILERPLATE_ANCHORS = {
    "cookie", "subscribe", "faq", "newsletter", "privacy", "gdpr",
    "consent", "browser", "javascript", "login", "signup", "captcha",
}

# Common German function words / nouns used to detect non-English terms.
# Extended after each LDA run that surfaced new German terms.
# Add to this list if new German terms appear in topic outputs.
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
# The same set is maintained as THEORY_FOCUS_TERMS in 02_step1_frequency.py
# — any additions here should be mirrored there.
#
# Why a guard is necessary:
#   The single-domain heuristic would exclude "worker" if it only
#   appeared on one platform — which is unlikely but possible for some
#   compound forms or rare variants.  Without this guard, the exclusion
#   pipeline could accidentally remove the core analytical vocabulary.
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
}

# Manual overrides — terms always excluded (known noise from prior analysis).
# Each: (term, reason)
# These were identified iteratively: after each round of analysis, terms
# that clearly did not contribute analytical signal were added here and
# the analysis re-run.
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
    """
    Create excluded_pages and excluded_terms tables if they don't exist.

    Also creates a unique index on excluded_terms.term to prevent
    duplicate entries on re-runs (INSERT OR IGNORE is used in the
    population functions).

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
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
    """
    Apply page exclusion rules from EXCLUDED_PAGE_RULES against corpus_view.

    Clears previous entries first so the table always reflects the
    current rules (re-run safe).  Uses INSERT OR IGNORE to handle any
    edge cases where the same page matches multiple rules.

    Each excluded page is logged with its URL and reason so the exclusion
    decisions are visible in the run log and can be reviewed.

    Args:
        conn: Open SQLite connection.
    """
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
    Run heuristics to identify artifact, noise, and non-English terms.

    Processing steps:
      1. Load all pages from corpus_view (excluding already-excluded pages)
         with audience in ('client', 'worker') and token_count >= 10.
      2. Build per-term statistics:
           term_freq    : total occurrences across all pages
           term_domains : set of domains where each term appears
      3. Apply four detection heuristics (A–D), each guarded by
         PROTECTED_TERMS to preserve the theory-focus vocabulary.
      4. Insert all identified terms into excluded_terms, recording
         the heuristic method and reason.

    Heuristics:
      A. MANUAL    — MANUAL_EXCLUSIONS list (cookie, faq, subscribe, etc.)
      B. NON_ENGLISH — GERMAN_MARKERS membership test + German character
                      pattern (regex for ä/ö/ü/ß/Ä/Ö/Ü)
      C. SINGLE_DOMAIN — terms with freq >= 5 that appear on only one
                         domain.  These are overwhelmingly scraping
                         artifacts (URL fragments, company names in
                         idiosyncratic forms) or proper nouns specific
                         to one platform that add noise to cross-platform
                         comparisons.
      D. BOILERPLATE co-occurrence check (noted but not fully
                         implemented in this version; manual list covers
                         the key cases)

    The exclusion dict maps term → (reason, method, domain_count, freq)
    and is inserted in bulk.  Duplicates are handled with INSERT OR IGNORE
    (first detection method wins if a term matches multiple heuristics).

    Args:
        conn: Open SQLite connection with corpus_view and excluded_pages.
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
        """
        Mark a term for exclusion unless it is in PROTECTED_TERMS.

        The first method to flag a term wins — only one entry per term
        is kept (INSERT OR IGNORE on the term column enforces uniqueness).
        """
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
    # Threshold of freq >= 5 avoids excluding extremely rare terms that
    # are simply low-frequency content words.
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
    # the most important cases.  Revisit after first full run of 02.)

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

    # Log a sample — most frequent excluded terms are the most visible
    # in analysis outputs and most likely to cause confusion if not excluded
    log.info("  Sample excluded terms (top 20 by frequency):")
    for term, (reason, method, n_dom, freq) in sorted(
            exclusions.items(), key=lambda x: x[1][3], reverse=True)[:20]:
        log.info(f"    {term:<30} method={method:<15} domains={n_dom}  "
                 f"freq={freq}  — {reason[:60]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Run page and term exclusion detection, then write results to DB.

    Re-run safe: both tables are cleared and repopulated on each run so
    the latest EXCLUDED_PAGE_RULES and MANUAL_EXCLUSIONS always take
    effect.

    Final summary logs total counts for inclusion in the thesis
    methodology description.
    """
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("01_prepare_additions.py — Exclusion Tables")
    log.info("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Verify corpus_view exists — this script depends on corpus_view being
    # present (created by 01_prepare.py)
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
