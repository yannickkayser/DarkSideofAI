"""
preprocess.py
=============
NLP preprocessing pipeline: converts raw scraped text into lemmatized token
lists suitable for corpus-linguistic analysis.

Pipeline position:
  Stage 1 — Text Preparation
  Reads from : pages (raw text_content from scraper)
  Writes to  : pages_tfidf    — lemmatized unigrams + bigrams per page
               pages_embedding — lightly cleaned text for embedding models

  Run AFTER: main.py / scraper.py (pages table must exist)
  Run BEFORE: 01_prepare.py (which builds corpus_view on top of pages_tfidf)

What this script does:
  1. Loads all pages from the `pages` table that have not yet been processed.
  2. Runs two cleaning passes:
       - clean_raw(): strips HTML, URLs, standalone numbers — for TF-IDF
       - clean_for_embedding(): lighter cleaning preserving sentence structure
  3. Tokenizes and lemmatizes via spaCy (en_core_web_sm) with:
       - stopword removal (except STOPWORD_WHITELIST — see below)
       - company/brand name removal (COMPANY_STOPWORDS)
       - minimum token length of 2 characters
  4. Builds bigrams in two passes:
       Pass 1: counts all bigram candidates across the full corpus
       Pass 2: keeps only bigrams that appear ≥ MIN_BIGRAM_FREQ times
       This prevents rare/idiosyncratic bigrams from inflating the vocabulary.
  5. Writes results in batches (BATCH_SIZE = 200) so the script can
     resume after interruption and handle large databases without exhausting RAM.
  6. Deduplicates: pages flagged in logs/duplicate/duplicate_report.json are
     excluded before processing.  Within each duplicate cluster, the page
     with the longest text_content is kept (most information).

Key NLP decisions:
  STOPWORD_WHITELIST
    spaCy's default stoplist removes function words AND many analytically
    important terms: "work", "control", "power", "agency", "replace".
    The whitelist re-admits these terms specifically because they are
    central to the thesis's theoretical framework (algorithmic management,
    labour displacement, automation rhetoric).

  COMPANY_STOPWORDS
    Platform brand names (appen, toloka, scale, mindrift…) would dominate
    TF-IDF scores on their respective domains without adding analytical
    value — they simply reflect which company's website a page is from.
    These are filtered at the lemmatization stage.  They are derived
    automatically from config.WEBSITES plus a hardcoded list of variants
    that don't appear in the config.

  MIN_BIGRAM_FREQ = 3
    A bigram that appears fewer than 3 times across the whole corpus is
    likely a one-off phrase unique to a single page.  Keeping such bigrams
    adds noise to keyness and co-occurrence analyses without adding signal.
    The threshold of 3 is conservative; standard corpus linguistics practice
    recommends 5–10, but the corpus here is relatively small.

  Audience labelling
    The audience column in pages_tfidf is derived from AUDIENCE_MAP (from
    config.WEBSITES), not from URL pattern matching.  This is important:
    URL-based audience detection produces many 'unknown' values (see
    01_prepare.py docstring).  The config-derived audience is the authoritative
    source used throughout the analysis pipeline.

Output tables:
  pages_tfidf:
    page_id, url, audience, unigrams (JSON list), bigrams (JSON list), token_count
    → Used by all Step 1 analysis scripts via corpus_view

  pages_embedding:
    page_id, url, audience, clean_text (for sentence-transformers),
    tokenized_text (space-separated lemmas, for Word2Vec / fastText)
    → Not used in Step 1; reserved for potential embedding-based Step 3 analysis

Prerequisites:
  pip install spacy scikit-learn
  python -m spacy download en_core_web_sm

Usage:
    python3 src/preprocess.py
"""

import sqlite3
import re
import json
import logging
import traceback
from pathlib import Path
from html.parser import HTMLParser

import spacy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config.config import WEBSITES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH         = "data/scraping.db"   # change to match your actual database path
BATCH_SIZE      = 200            # pages per batch — lower if memory is tight
MIN_BIGRAM_FREQ = 3              # minimum corpus frequency to retain a bigram

# Path to duplicates.json produced by find_duplicates.py.
# Set to None to skip duplicate exclusion entirely.
# Per cluster, the page with the longest text_content is kept; all others excluded.
DUPLICATES_FILE = str(Path(__file__).parent.parent / "logs/duplicate/duplicate_report.json")

# Audience map: derived automatically from config.WEBSITES.
# e.g. {"mindrift.ai": "worker", "appen.com": "client", ...}
# This is the ground-truth audience assignment used in pages_tfidf.
# It is NOT the same as the audience column in pages_tfidf.audience (which
# is unreliable due to URL-matching failures in earlier versions of the script).
AUDIENCE_MAP: dict[str, str] = {
    domain: site["audience"]
    for domain, site in WEBSITES.items()
    if "audience" in site
}

# STOPWORD_WHITELIST: terms that spaCy would normally remove as stopwords
# but that are analytically essential for this thesis.
# Rationale: standard English stopword lists were designed for general IR tasks;
# they systematically remove terms relevant to labour, automation, and AI discourse.
# Whitelisted terms cover: labour relations (work, worker, task, job, role),
# automation discourse (automate, algorithm, model, decision), and critical concepts
# (power, agency, control, surveillance).
STOPWORD_WHITELIST = {
    "work", "worker", "workers", "task", "tasks", "human", "humans",
    "control", "machine", "machines", "skill", "skills", "labor", "labour",
    "job", "jobs", "role", "roles", "replace", "replaces", "replacement",
    "automate", "automated", "automation", "ai", "algorithm", "algorithms",
    "data", "model", "models", "decision", "decisions", "power", "agency",
    "autonomy", "efficiency", "productivity", "manage", "management",
    "monitor", "monitoring", "surveillance", "creative", "creativity",
    "craft", "crafting", "interact", "interaction", "collaborate",
    "collaboration", "assist", "assistance", "augment", "augmentation",
}

# COMPANY_STOPWORDS: brand and product names that inflate TF-IDF without
# adding analytical value.  A term like "appen" appears frequently on appen.com
# simply because the company refers to itself by name — it is not a meaningful
# cross-corpus signal.
# Built automatically from WEBSITES (domain + name parts) + manual extras.
COMPANY_STOPWORDS: set[str] = set()


def _build_company_stopwords() -> set[str]:
    """
    Derive company stopwords from WEBSITES config.

    Splits multi-word names and domain labels into tokens, lowercases them,
    and adds a hardcoded list of variants that config doesn't capture.
    Terms in STOPWORD_WHITELIST are never removed even if they appear in
    a company name (e.g. 'defined.ai' → 'defined', but 'ai' is kept because
    it is in the whitelist).

    Returns:
        Set of lowercase term strings to filter out during lemmatization.
    """
    terms = set()
    for domain, site in WEBSITES.items():
        # Extract domain label parts: "mindrift.ai" → {"mindrift", "ai"}
        for part in domain.replace(".", " ").replace("-", " ").split():
            if len(part) > 2:
                terms.add(part.lower())

        # Extract name parts: "Scale AI" → {"scale"}
        for part in site.get("name", "").replace("-", " ").split():
            if len(part) > 2:
                terms.add(part.lower())

    # Variants not captured by config (product names, platform sub-brands)
    extra = {
        "remotask", "remotasks", "crowdgen", "mindrift", "toloker", "tolokers",
        "alignerr", "oneforma", "mturk", "dataannotation", "surgehq",
        "appen", "sama", "scale", "telus", "prolific", "outlier",
        "cloudfactory", "imerit", "lxt", "defined", "superannotate",
        "mindy", "flipside", "digitaldivide", "humansintheloop", "toloka",
        # Generic legal/corporate suffixes
        "inc", "llc", "ltd", "corp", "gmbh",
    }
    terms.update(extra)

    # Never strip terms that are analytically important
    terms -= STOPWORD_WHITELIST
    return terms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

COMPANY_STOPWORDS = _build_company_stopwords()


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper — avoids external dependencies."""
    def __init__(self):
        super().__init__()
        self._chunks = []

    def handle_data(self, data):
        self._chunks.append(data)

    def get_text(self):
        return " ".join(self._chunks)


def strip_html(html: str) -> str:
    """Remove HTML tags, returning visible text only."""
    s = _HTMLStripper()
    s.feed(html or "")
    return s.get_text()


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

URL_RE   = re.compile(r"https?://\S+|www\.\S+")
NUM_RE   = re.compile(r"\b\d+\b")
SPACE_RE = re.compile(r"\s+")


def clean_raw(text: str) -> str:
    """
    Clean text for TF-IDF / keyness analysis.

    Removes: HTML tags, URLs (not informative for word-level analysis),
    standalone numbers (too common across all pages to be discriminating).

    Args:
        text: Raw scraped text_content from the pages table.

    Returns:
        Cleaned string ready for spaCy tokenization.
    """
    text = strip_html(text)
    text = URL_RE.sub(" ", text)
    text = NUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def clean_for_embedding(text: str) -> str:
    """
    Light cleaning for sentence-transformer input.

    Sentence boundaries and punctuation are preserved because transformer
    models rely on them for contextual representations.  Only HTML and URLs
    are removed.

    Args:
        text: Raw scraped text_content from the pages table.

    Returns:
        Lightly cleaned string suitable for sentence-transformers.
    """
    text = strip_html(text)
    text = URL_RE.sub(" ", text)
    text = NUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# spaCy pipeline
# ---------------------------------------------------------------------------

def load_nlp():
    """
    Load the spaCy pipeline with parser and NER disabled for speed.

    Only the tokenizer and tagger (for lemmatization) are needed.
    Disabling parser and NER cuts processing time by ~60% with no
    effect on lemma quality.

    Returns:
        Loaded spaCy Language object.

    Raises:
        OSError: if en_core_web_sm is not installed.
                 Run: python -m spacy download en_core_web_sm
    """
    log.info("Loading spaCy model (en_core_web_sm)...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        log.info("spaCy model loaded successfully.")
        return nlp
    except OSError:
        log.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise


def tokenize_and_lemmatize(nlp, text: str) -> list[str]:
    """
    Tokenize and lemmatize text, applying stopword and brand filtering.

    Processing steps (in order):
      1. Lowercased processing through spaCy
      2. Whitespace and punctuation tokens removed
      3. Tokens shorter than 2 characters removed
      4. spaCy stopwords removed UNLESS in STOPWORD_WHITELIST
      5. Brand/company terms removed (COMPANY_STOPWORDS)
      6. Lemma is the final form (e.g. 'annotating' → 'annotate')

    Args:
        nlp:  Loaded spaCy Language object.
        text: Clean text string (output of clean_raw).

    Returns:
        List of lemmatized unigram strings for this page.
    """
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip()
        if not lemma or len(lemma) < 2:
            continue
        if token.is_stop and lemma not in STOPWORD_WHITELIST:
            continue
        if lemma in COMPANY_STOPWORDS:
            continue
        tokens.append(lemma)
    return tokens


def make_bigrams(tokens: list[str]) -> list[str]:
    """
    Generate adjacent bigrams as 'word1_word2' strings.

    The underscore separator distinguishes bigrams from unigrams when
    both are stored together in a list (e.g. in pages_tfidf.bigrams).

    Args:
        tokens: Lemmatized unigram list for a single page.

    Returns:
        List of bigram strings.  Length = len(tokens) - 1.
    """
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def compute_bigram_counts(all_bigram_lists: list[list[str]]) -> dict[str, int]:
    """
    Count bigram frequency across all pages in the corpus.

    Used in Pass 1 to identify which bigrams are frequent enough to retain.
    Infrequent bigrams (below MIN_BIGRAM_FREQ) are discarded in Pass 2.

    Args:
        all_bigram_lists: List of per-page bigram lists.

    Returns:
        Dict mapping each bigram string to its total corpus frequency.
    """
    counts: dict[str, int] = {}
    for bigrams in all_bigram_lists:
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_tables(conn: sqlite3.Connection):
    """
    Create pages_tfidf and pages_embedding tables if they don't exist.

    pages_tfidf schema:
      unigrams  — JSON list of lemmatized unigrams for this page
      bigrams   — JSON list of lemmatized bigrams (format: 'word1_word2')
      audience  — derived from AUDIENCE_MAP (config), NOT from URL matching
      token_count — count of unigrams (used by 01_prepare.py to filter short pages)

    pages_embedding schema:
      clean_text     — lightly cleaned full text (for sentence-transformers)
      tokenized_text — space-separated lemmas (for Word2Vec / fastText)
    """
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages_tfidf (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id      INTEGER UNIQUE NOT NULL,
            url          TEXT,
            audience     TEXT,    -- 'worker' or 'client', from config WEBSITES[domain]['audience']
            unigrams     TEXT,    -- JSON list of lemmatized tokens
            bigrams      TEXT,    -- JSON list of filtered bigrams (token_token)
            token_count  INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (page_id) REFERENCES pages(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages_embedding (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id        INTEGER UNIQUE NOT NULL,
            url            TEXT,
            audience       TEXT,   -- 'worker' or 'client', from config WEBSITES[domain]['audience']
            clean_text     TEXT,   -- sentence-transformers input (prose, punctuation intact)
            tokenized_text TEXT,   -- Word2Vec / fastText input (space-separated lemmas)
            processed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (page_id) REFERENCES pages(id)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tfidf_page ON pages_tfidf(page_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emb_page   ON pages_embedding(page_id)")
    conn.commit()
    log.info("Tables pages_tfidf and pages_embedding are ready.")


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def load_excluded_ids(db_path: str, duplicates_file: str | None) -> set[int]:
    """
    Load page_ids to exclude based on near-duplicate detection.

    Reads the cluster report from find_duplicates.py.  Within each cluster,
    the page with the longest text_content is kept; all others are excluded.
    Keeping the longest page ensures maximum vocabulary coverage per cluster.

    Args:
        db_path:          Path to the SQLite database.
        duplicates_file:  Path to duplicate_report.json, or None to skip.

    Returns:
        Set of page_id integers to exclude from processing.
    """
    if not duplicates_file or not Path(duplicates_file).exists():
        if duplicates_file:
            log.warning(f"  Duplicates file not found: {duplicates_file} — skipping exclusion.")
        return set()

    with open(duplicates_file, encoding="utf-8") as f:
        clusters = json.load(f)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    excluded: set[int] = set()
    kept_count = 0

    for cluster in clusters:
        page_ids = [p["page_id"] for p in cluster["pages"]]
        if len(page_ids) < 2:
            continue

        placeholders = ",".join("?" * len(page_ids))
        cursor.execute(
            f"SELECT id, LENGTH(COALESCE(text_content,'')) AS len "
            f"FROM pages WHERE id IN ({placeholders})",
            page_ids,
        )
        lengths = {row[0]: row[1] for row in cursor.fetchall()}

        keep_id = max(lengths, key=lambda pid: lengths.get(pid, 0))
        for pid in page_ids:
            if pid != keep_id:
                excluded.add(pid)
        kept_count += 1

    conn.close()

    log.info(f"  Duplicate exclusion: {kept_count} clusters → "
             f"{len(excluded)} pages excluded, 1 kept per cluster.")
    return excluded


def fetch_unprocessed_ids(conn: sqlite3.Connection,
                          excluded_ids: set[int] = set()) -> list[int]:
    """
    Return IDs of pages not yet in pages_tfidf.

    Excludes duplicate pages (excluded_ids) so they are never processed.
    Ordering by id is important for reproducibility: the same subset is
    processed in the same order on every run.

    Args:
        conn:          Open SQLite connection.
        excluded_ids:  Set of page_ids flagged as duplicates.

    Returns:
        Sorted list of page_ids ready for processing.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id
        FROM   pages p
        WHERE  p.text_content IS NOT NULL
          AND  p.id NOT IN (SELECT page_id FROM pages_tfidf)
        ORDER  BY p.id
    """)
    all_ids = [row[0] for row in cursor.fetchall()]

    if excluded_ids:
        before = len(all_ids)
        all_ids = [i for i in all_ids if i not in excluded_ids]
        log.info(f"  Excluded {before - len(all_ids)} duplicate pages from processing.")

    return all_ids


def audience_from_url(url: str) -> str:
    """
    Derive the audience label for a page by matching its URL against AUDIENCE_MAP.

    This is a fallback mechanism used only in pages_tfidf.  The authoritative
    audience assignment in the analysis pipeline comes from 01_prepare.py's
    join through the platforms table.

    Args:
        url: Full URL of the page.

    Returns:
        'client', 'worker', or 'unknown' if no domain match found.
    """
    for domain, label in AUDIENCE_MAP.items():
        if domain in url:
            return label
    return "unknown"


def fetch_batch(conn: sqlite3.Connection, ids: list[int]) -> list[sqlite3.Row]:
    """
    Fetch a specific list of page rows by id.

    Args:
        conn: Open SQLite connection.
        ids:  List of page_id integers to fetch.

    Returns:
        List of sqlite3.Row objects with id, url, text_content.
    """
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    cursor.execute(
        f"SELECT id, url, text_content FROM pages WHERE id IN ({placeholders})",
        ids,
    )
    return cursor.fetchall()


def process_batch(
    nlp,
    rows: list[sqlite3.Row],
    frequent_bigrams: set[str],
) -> tuple[list[tuple], list[tuple]]:
    """
    Tokenize and clean one batch of pages.

    For each row: clean → tokenize → filter bigrams → assign audience.
    Returns tuples ready for executemany() insertion.

    Args:
        nlp:              Loaded spaCy Language object.
        rows:             List of page rows from fetch_batch.
        frequent_bigrams: Set of bigrams that meet the MIN_BIGRAM_FREQ threshold.

    Returns:
        Tuple of (tfidf_rows, embed_rows) where each is a list of value tuples.
        tfidf_rows: (page_id, url, audience, unigrams_json, bigrams_json, token_count)
        embed_rows: (page_id, url, audience, clean_text, tokenized_text)
    """
    tfidf_rows = []
    embed_rows = []

    for row in rows:
        raw       = clean_raw(row["text_content"])
        clean_emb = clean_for_embedding(row["text_content"])
        tokens    = tokenize_and_lemmatize(nlp, raw)
        bigrams   = [bg for bg in make_bigrams(tokens) if bg in frequent_bigrams]
        audience  = audience_from_url(row["url"])

        if audience == "unknown":
            log.warning(f"    Unknown audience for URL: {row['url']}")

        if len(tokens) < 10:
            log.warning(f"    Low token count ({len(tokens)}) for page id={row['id']} url={row['url']}")

        tfidf_rows.append((
            row["id"], row["url"], audience,
            json.dumps(tokens),
            json.dumps(bigrams),
            len(tokens),
        ))
        embed_rows.append((
            row["id"], row["url"], audience,
            clean_emb,
            " ".join(tokens),
        ))

    return tfidf_rows, embed_rows


def insert_batch(
    conn: sqlite3.Connection,
    tfidf_rows: list[tuple],
    embed_rows: list[tuple],
):
    """
    Insert processed rows into pages_tfidf and pages_embedding.

    Uses INSERT OR REPLACE so re-running the script on already-processed
    pages refreshes their records rather than erroring on unique constraints.

    Args:
        conn:       Open SQLite connection.
        tfidf_rows: Rows for pages_tfidf (output of process_batch).
        embed_rows: Rows for pages_embedding (output of process_batch).
    """
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO pages_tfidf
            (page_id, url, audience, unigrams, bigrams, token_count)
        VALUES (?, ?, ?, ?, ?, ?)
    """, tfidf_rows)
    cursor.executemany("""
        INSERT OR REPLACE INTO pages_embedding
            (page_id, url, audience, clean_text, tokenized_text)
        VALUES (?, ?, ?, ?, ?)
    """, embed_rows)
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(db_path: str, batch_size: int = BATCH_SIZE):
    """
    Run the full two-pass preprocessing pipeline.

    Pass 1: scan all pages to count bigram frequencies.
    Pass 2: tokenize each page, filter bigrams, insert into DB.

    Failed batches in Pass 2 are logged to failed_batches.json and skipped —
    this ensures one malformed page does not abort the entire corpus.

    Args:
        db_path:    Path to the SQLite database.
        batch_size: Number of pages per batch.  Reduce if memory is tight.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    log.info("=" * 60)
    log.info("PREPROCESSING START")
    log.info(f"  Database  : {db_path}")
    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Min bigram frequency: {MIN_BIGRAM_FREQ}")
    log.info(f"  Audience map loaded: {len(AUDIENCE_MAP)} domains")
    log.info(f"  Company stopwords  : {len(COMPANY_STOPWORDS)} terms filtered")
    log.info(f"  (terms: {', '.join(sorted(COMPANY_STOPWORDS)[:10])}...)")
    log.info(f"  Duplicates file    : {DUPLICATES_FILE or 'not set — skipping'}")
    log.info("=" * 60)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_tables(conn)

    nlp = load_nlp()

    excluded_ids = load_excluded_ids(db_path, DUPLICATES_FILE)

    log.info("Scanning for unprocessed pages...")
    all_ids = fetch_unprocessed_ids(conn, excluded_ids)
    total   = len(all_ids)

    if total == 0:
        log.info("No new pages to process. Exiting.")
        conn.close()
        return

    n_batches = (total + batch_size - 1) // batch_size
    log.info(f"Found {total} unprocessed pages → {n_batches} batches of ≤{batch_size}")

    # Pass 1: collect bigram frequencies across ALL pages before filtering
    log.info("-" * 60)
    log.info("Pass 1/2 — collecting bigram frequencies across corpus...")
    bigram_counts: dict[str, int] = {}

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch_ids = all_ids[start : start + batch_size]
        log.info(f"  [Pass 1] Batch {batch_num}/{n_batches}")
        try:
            rows = fetch_batch(conn, batch_ids)
            for row in rows:
                raw    = clean_raw(row["text_content"])
                tokens = tokenize_and_lemmatize(nlp, raw)
                for bg in make_bigrams(tokens):
                    bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        except Exception:
            log.warning(
                f"  [Pass 1] Batch {batch_num}/{n_batches} — error during bigram scan, skipping:\n"
                + traceback.format_exc()
            )

    # Keep only bigrams meeting the frequency threshold
    frequent_bigrams = {bg for bg, c in bigram_counts.items() if c >= MIN_BIGRAM_FREQ}
    log.info(f"  Total unique bigrams found : {len(bigram_counts)}")
    log.info(f"  Bigrams kept (freq ≥ {MIN_BIGRAM_FREQ}): {len(frequent_bigrams)}")
    log.info(f"  Bigrams discarded          : {len(bigram_counts) - len(frequent_bigrams)}")

    # Pass 2: tokenize, filter bigrams, insert into DB
    log.info("-" * 60)
    log.info("Pass 2/2 — tokenizing and inserting into database...")
    failed_batches = []
    inserted_pages = 0

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch_ids = all_ids[start : start + batch_size]
        batch_end = min(start + batch_size, total)
        log.info(f"  [Pass 2] Batch {batch_num}/{n_batches}  (pages {start+1}–{batch_end})")

        try:
            rows = fetch_batch(conn, batch_ids)
            tfidf_rows, embed_rows = process_batch(nlp, rows, frequent_bigrams)
            insert_batch(conn, tfidf_rows, embed_rows)
            inserted_pages += len(tfidf_rows)
            log.info(f"  [Pass 2] Batch {batch_num} done — "
                     f"inserted {len(tfidf_rows)} pages  (total so far: {inserted_pages}/{total})")

        except Exception:
            log.error(
                f"  [Pass 2] Batch {batch_num}/{n_batches} FAILED — skipping:\n"
                + traceback.format_exc()
            )
            failed_batches.append({
                "batch_num": batch_num,
                "page_ids":  batch_ids,
            })

    # Summary
    log.info("=" * 60)
    log.info("PREPROCESSING COMPLETE")
    log.info(f"  Pages inserted successfully : {inserted_pages}/{total}")
    log.info(f"  Batches failed              : {len(failed_batches)}")

    if failed_batches:
        failed_id_count = sum(len(b["page_ids"]) for b in failed_batches)
        log.warning(f"  {failed_id_count} pages skipped due to batch failures.")
        log.warning("  Failed page IDs saved to: failed_batches.json")
        with open("failed_batches.json", "w") as f:
            json.dump(failed_batches, f, indent=2)
    else:
        log.info("  All batches completed successfully — no failures.")

    log.info("=" * 60)
    conn.close()


if __name__ == "__main__":
    process(DB_PATH)
