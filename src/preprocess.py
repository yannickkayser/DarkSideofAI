"""
Preprocessing script for TF-IDF and word embedding analysis.
Reads from existing 'pages' table and writes to two new tables:
    - pages_tfidf    : lemmatized unigrams + bigrams, stopwords removed
    - pages_embedding: lightly cleaned text (sentences preserved) for
                    sentence-transformers AND tokenized text for Word2Vec/fastText

Processes in batches to handle large databases and isolate errors per batch.
A failed batch is logged and skipped — processing continues with the next one.
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

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "scraping.db"
RAW_DATA_DIR = DATA_DIR / "raw"  
LOGS_DIR = PROJECT_ROOT / "logs"

BATCH_SIZE     = 100            # pages per batch — lower if memory is tight
MIN_BIGRAM_FREQ = 3             # minimum corpus frequency to keep a bigram

# Derived from config.WEBSITES — no separate mapping needed.
# e.g. {"mindrift.ai": "worker", "appen.com": "client", ...}
AUDIENCE_MAP: dict[str, str] = {
    domain: site["audience"]
    for domain, site in WEBSITES.items()
    if "audience" in site
}

# Custom whitelist: these words are KEPT even though spaCy would mark them
# as stopwords. Critical for labor/AI discourse analysis.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

class _HTMLStripper(HTMLParser):
    """Minimal HTML stripper — no external dependencies."""
    def __init__(self):
        super().__init__()
        self._chunks = []

    def handle_data(self, data):
        self._chunks.append(data)

    def get_text(self):
        return " ".join(self._chunks)


def strip_html(html: str) -> str:
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
    """For TF-IDF: strip HTML, URLs, standalone numbers."""
    text = strip_html(text)
    text = URL_RE.sub(" ", text)
    text = NUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def clean_for_embedding(text: str) -> str:
    """
    For sentence-transformers: minimal cleaning only.
    Sentence boundaries and punctuation are preserved — the model needs them.
    """
    text = strip_html(text)
    text = URL_RE.sub(" ", text)
    text = NUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# spaCy pipeline
# ---------------------------------------------------------------------------

def load_nlp():
    log.info("Loading spaCy model (en_core_web_sm)...")   # LOG: spaCy startup
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        log.info("spaCy model loaded successfully.")       # LOG: confirm model ready
        return nlp
    except OSError:
        log.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise


def tokenize_and_lemmatize(nlp, text: str) -> list[str]:
    """
    Returns lemmatized unigrams:
      - lowercased
      - stopwords removed (except whitelist terms)
      - punctuation and spaces removed
      - minimum length 2
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
        tokens.append(lemma)
    return tokens


def make_bigrams(tokens: list[str]) -> list[str]:
    """Generate bigrams as 'word1_word2' strings."""
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def compute_bigram_counts(all_bigram_lists: list[list[str]]) -> dict[str, int]:
    """Count bigram frequency across a list of documents."""
    counts: dict[str, int] = {}
    for bigrams in all_bigram_lists:
        for bg in bigrams:
            counts[bg] = counts.get(bg, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_tables(conn: sqlite3.Connection):
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

def fetch_unprocessed_ids(conn: sqlite3.Connection) -> list[int]:
    """Return IDs of all pages not yet in pages_tfidf, ordered for reproducibility."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id
        FROM   pages p
        WHERE  p.text_content IS NOT NULL
          AND  p.id NOT IN (SELECT page_id FROM pages_tfidf)
        ORDER  BY p.id
    """)
    return [row[0] for row in cursor.fetchall()]


def audience_from_url(url: str) -> str:
    """Derive audience label by matching URL against AUDIENCE_MAP domains."""
    for domain, label in AUDIENCE_MAP.items():
        if domain in url:
            return label
    return "unknown"


def fetch_batch(conn: sqlite3.Connection, ids: list[int]) -> list[sqlite3.Row]:
    """Fetch a specific list of page rows by id."""
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    cursor.execute(
        f"SELECT id, url, text_content FROM pages WHERE id IN ({placeholders})",
        ids,
    )
    rows = cursor.fetchall()
    log.debug(f"    Fetched {len(rows)} rows from database.")   # LOG: rows retrieved
    return rows


def process_batch(
    nlp,
    rows: list[sqlite3.Row],
    frequent_bigrams: set[str],
) -> tuple[list[tuple], list[tuple]]:
    """
    Tokenize and clean one batch of rows.
    Returns (tfidf_rows, embed_rows) ready for executemany().
    Raises on per-row errors so the caller can decide to skip or abort.
    """
    tfidf_rows = []
    embed_rows = []

    for row in rows:
        raw       = clean_raw(row["text_content"])
        clean_emb = clean_for_embedding(row["text_content"])
        tokens    = tokenize_and_lemmatize(nlp, raw)
        bigrams   = [bg for bg in make_bigrams(tokens) if bg in frequent_bigrams]
        audience  = audience_from_url(row["url"])

        # LOG: warn if a page's domain is not in AUDIENCE_MAP
        if audience == "unknown":
            log.warning(f"    Unknown audience for URL: {row['url']}")

        # LOG: warn if a page produces very few tokens (possible empty/broken page)
        if len(tokens) < 10:
            log.warning(f"    Low token count ({len(tokens)}) for page id={row['id']} url={row['url']}")

        log.debug(f"    Page id={row['id']} → {len(tokens)} tokens, {len(bigrams)} bigrams, audience={audience}")  # LOG: per-page detail

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
    log.debug(f"    Inserting {len(tfidf_rows)} rows into pages_tfidf and pages_embedding...")  # LOG: pre-insert
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
    log.debug("    Batch committed to database.")   # LOG: confirm commit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(db_path: str, batch_size: int = BATCH_SIZE):
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    # LOG: startup summary
    log.info("=" * 60)
    log.info("PREPROCESSING START")
    log.info(f"  Database  : {db_path}")
    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Min bigram frequency: {MIN_BIGRAM_FREQ}")
    log.info(f"  Audience map loaded: {len(AUDIENCE_MAP)} domains")
    log.info("=" * 60)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_tables(conn)

    nlp = load_nlp()

    # --- Discover all unprocessed page IDs upfront ---
    log.info("Scanning for unprocessed pages...")   # LOG: before ID query
    all_ids = fetch_unprocessed_ids(conn)
    total   = len(all_ids)

    if total == 0:
        log.info("No new pages to process. Exiting.")
        conn.close()
        return

    n_batches = (total + batch_size - 1) // batch_size
    log.info(f"Found {total} unprocessed pages → {n_batches} batches of ≤{batch_size}")

    # --- Pass 1: collect bigram frequencies across ALL pages ---
    log.info("-" * 60)
    log.info("Pass 1/2 — collecting bigram frequencies across corpus...")
    bigram_counts: dict[str, int] = {}

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch_ids = all_ids[start : start + batch_size]
        batch_end = min(start + batch_size, total)
        log.info(f"  [Pass 1] Batch {batch_num}/{n_batches}  (pages {start+1}–{batch_end})")   # LOG: pass 1 batch progress
        try:
            rows = fetch_batch(conn, batch_ids)
            for row in rows:
                raw     = clean_raw(row["text_content"])
                tokens  = tokenize_and_lemmatize(nlp, raw)
                for bg in make_bigrams(tokens):
                    bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
            log.debug(f"  [Pass 1] Batch {batch_num} — running bigram vocab size: {len(bigram_counts)}")  # LOG: vocab growth
        except Exception:
            log.warning(
                f"  [Pass 1] Batch {batch_num}/{n_batches} — error during bigram scan, skipping:\n"
                + traceback.format_exc()
            )

    frequent_bigrams = {bg for bg, c in bigram_counts.items() if c >= MIN_BIGRAM_FREQ}
    # LOG: bigram filter result
    log.info(f"  Total unique bigrams found : {len(bigram_counts)}")
    log.info(f"  Bigrams kept (freq ≥ {MIN_BIGRAM_FREQ}): {len(frequent_bigrams)}")
    log.info(f"  Bigrams discarded          : {len(bigram_counts) - len(frequent_bigrams)}")

    # --- Pass 2: tokenize, filter bigrams, insert ---
    log.info("-" * 60)
    log.info("Pass 2/2 — tokenizing and inserting into database...")
    failed_batches = []
    inserted_pages = 0

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch_ids = all_ids[start : start + batch_size]
        batch_end = min(start + batch_size, total)
        log.info(f"  [Pass 2] Batch {batch_num}/{n_batches}  (pages {start+1}–{batch_end})")  # LOG: pass 2 batch start

        try:
            rows = fetch_batch(conn, batch_ids)
            tfidf_rows, embed_rows = process_batch(nlp, rows, frequent_bigrams)
            insert_batch(conn, tfidf_rows, embed_rows)
            inserted_pages += len(tfidf_rows)
            # LOG: running total after each successful batch
            log.info(f"  [Pass 2] Batch {batch_num} done — inserted {len(tfidf_rows)} pages  (total so far: {inserted_pages}/{total})")

        except Exception:
            log.error(
                f"  [Pass 2] Batch {batch_num}/{n_batches} FAILED — skipping, will log IDs:\n"
                + traceback.format_exc()
            )
            failed_batches.append({
                "batch_num": batch_num,
                "page_ids":  batch_ids,
            })

    # --- Summary ---
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    process(DB_PATH)