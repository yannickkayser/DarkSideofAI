"""
00_preprocess.py
================
NLP preprocessing pipeline: converts raw scraped text into lemmatised token
lists suitable for corpus-linguistic analysis.

Pipeline position:
  Stage 0 — Text Preparation
  Reads from : pages            (raw text_content from scraper)
  Writes to  : pages_tfidf      — unigrams, bigrams, and segments per page
               pages_embedding  — lightly cleaned text for embedding models

  Run AFTER: main.py / scraper.py (pages table must exist)
  Run BEFORE: 01_prepare_corpus.py

Changes from src/preprocess.py
-------------------------------
BUG FIX 1 — Sentence-boundary-aware segmentation (root cause of false
  co-occurrences such as "work–baby"):
    The original script tokenised each page as a single flat token list.
    On JS-rendered pages, BeautifulSoup's get_text() concatenates hero text,
    navigation items, feature blocks, and footer content into one string.
    The ±N token window in the co-occurrence script then paired tokens from
    completely unrelated page sections.
    Fix: spaCy's sentencizer component is added so each page is split into
    sentences before tokenisation.  Per-sentence token lists are stored in
    the new `segments` column.  Co-occurrence in 02_step1_analysis.py is
    computed within sentences only.

BUG FIX 2 — Bigrams formed only within sentences:
    The original make_bigrams() formed all adjacent pairs in the flat list,
    creating cross-boundary bigrams (e.g. "platform_baby" where "platform"
    ended one section and "baby" started the next).
    Fix: bigrams are now generated within each sentence independently.

BUG FIX 3 — clean_for_embedding was identical to clean_raw:
    Both functions had the same implementation despite different documented
    purposes.  clean_for_embedding now preserves numbers and punctuation
    so the stored text is suitable for sentence-transformers and Word2Vec.

BUG FIX 4 — Zero-width and invisible Unicode characters corrupt tokens:
    Several scraped sites (notably mindy-support.com) embed U+200B ZERO WIDTH
    SPACE and related invisible characters as anti-copy-paste protection.
    These silently fragment valid English words into fake tokens: the string
    "ab​il​ity" (with invisible U+200B between characters) produces a unique
    token that never co-occurs with the real lemma "ability".  This created
    thousands of spurious vocabulary items.
    Fix: INVISIBLE_RE is applied at the top of both clean_raw() and
    clean_for_embedding() to strip all zero-width / directional / BOM
    characters before any further processing.

BUG FIX 5 — Non-English pages pollute the vocabulary:
    Several domains host localised page variants (e.g. remoter.me/ua/ in
    Ukrainian, crowdworks.ai with embedded Korean, crowdgen.com with
    multilingual content).  These contribute Korean, Japanese, Chinese,
    Arabic, Russian/Ukrainian, Thai, Khmer, Hindi, and other scripts into
    the term vocabulary, making topic models and keyness results unreliable.
    Two complementary fixes are applied:
      a) Page-level exclusion: pages whose URL contains a non-English
         language path segment (e.g. /ua/, /de/, /ko/) or whose cleaned
         text contains >15% non-ASCII characters are skipped entirely.
         Their pages_tfidf records are deleted if they exist from a
         previous run, ensuring a clean re-run.
      b) Token-level filter: any lemma whose non-ASCII characters are not
         Extended Latin (i.e. contain Cyrillic, CJK, Arabic, Devanagari,
         Thai, etc.) is dropped by _filter_token().  This provides a
         second safety net for mixed-language pages that pass the page-
         level threshold (e.g. a mostly-English page with a few Korean
         characters in a footer).

What this script does:
  1. Loads all pages from the `pages` table.
  2. Screens each page for language: non-English pages are excluded and
     any stale pages_tfidf records for them are deleted.
  3. Applies two cleaning passes:
       clean_raw()           — strips invisible chars, HTML, URLs, standalone
                               numbers (for keyness)
       clean_for_embedding() — strips invisible chars, HTML, and URLs only
                               (preserves sentence structure for embedding)
  4. Tokenises and lemmatises via spaCy (en_core_web_sm + sentencizer):
       - Stopword removal (except STOPWORD_WHITELIST)
       - Company/brand name removal (COMPANY_STOPWORDS)
       - Non-Latin-script token removal
       - Minimum token length: 2 characters
  5. Produces three outputs per page:
       segments    — list of per-sentence token lists (JSON list of lists)
       unigrams    — flat token list (segments flattened; used by keyness)
       bigrams     — adjacent pairs within sentences (frequency-filtered)
  6. Writes results in batches (BATCH_SIZE = 200) for memory efficiency.
  7. Excludes pages flagged in logs/duplicate/duplicate_report.json;
     within each duplicate cluster the longest page is kept.

Key NLP decisions:
  STOPWORD_WHITELIST
    spaCy's default stoplist removes terms analytically central to this
    thesis: "work", "control", "power", "agency", "replace".  These are
    explicitly re-admitted.

  COMPANY_STOPWORDS
    Platform brand names inflate TF-IDF on their respective domains without
    adding analytical signal.  Derived from config.WEBSITES plus a hardcoded
    list of known variants.

  MIN_BIGRAM_FREQ = 3
    Bigrams appearing fewer than 3 times are one-off phrases that add noise.
    Conservative relative to the standard recommendation of 5–10; justified
    by the small corpus size.

  Sentencizer
    Rule-based sentence segmentation (no dependency parser required).
    Splits on sentence-final punctuation.  Fast and adequate for the
    predominantly English promotional/informational text in this corpus.

  FOREIGN_CHAR_THRESHOLD = 0.15
    If more than 15% of non-whitespace characters in a page's cleaned text
    are non-ASCII, the page is treated as non-English and excluded.
    15% is deliberately permissive: a normal English page may contain
    a handful of currency symbols or accented proper nouns; pages with
    genuine non-English content typically exceed 30–40%.

Output columns in pages_tfidf:
  segments    — JSON: [[sent1_tok1, sent1_tok2, ...], [sent2_tok1, ...], ...]
  unigrams    — JSON: [tok1, tok2, ...] (flat, used by keyness analysis)
  bigrams     — JSON: ["word1_word2", ...] (within-sentence pairs, freq-filtered)
  token_count — length of the flat unigrams list

Usage:
    python3 src2/00_preprocess.py
"""

import sqlite3
import re
import json
import logging
import traceback
import unicodedata
from pathlib import Path
from html.parser import HTMLParser

import spacy
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config.config import WEBSITES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH         = "data/scraping_2.db"
BATCH_SIZE      = 200
MIN_BIGRAM_FREQ = 3

DUPLICATES_FILE = str(
    Path(__file__).parent.parent / "logs/duplicate/duplicate_report.json"
)

# Audience map — derived from config, used as a fallback only.
# The authoritative audience assignment comes from 01_prepare_corpus.py
# via the platforms table join in corpus_view.
AUDIENCE_MAP: dict[str, str] = {
    domain: site["audience"]
    for domain, site in WEBSITES.items()
    if "audience" in site
}

# ---------------------------------------------------------------------------
# Stopword whitelist — terms spaCy would remove but that are analytically
# essential for this thesis (labour, automation, and power vocabulary).
# ---------------------------------------------------------------------------
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
    # Additional terms relevant to H2–H5 (flexibility/control paradox,
    # resource framing, entrepreneurial subjectivity, solidarity)
    "flexible", "flexibility", "freedom", "independent", "autonomy",
    "earn", "pay", "payment", "income", "rate", "bonus",
    "talent", "resource", "contributor", "workforce", "pool",
    "community", "collective", "union", "solidarity",
    "score", "rating", "rank", "ranking", "performance", "metric",
    "quality", "accuracy", "trust", "oversight", "review",
}

# ---------------------------------------------------------------------------
# Company stopwords — brand and product names that inflate frequency without
# adding analytical value.  Built from config.WEBSITES + manual extras.
# ---------------------------------------------------------------------------
COMPANY_STOPWORDS: set[str] = set()


def _build_company_stopwords() -> set[str]:
    """
    Derive company stopwords from WEBSITES config.

    Splits multi-word names and domain labels into tokens, lowercases them.
    Terms in STOPWORD_WHITELIST are never removed even if they appear in
    a company name.
    """
    terms: set[str] = set()
    for domain, site in WEBSITES.items():
        for part in domain.replace(".", " ").replace("-", " ").split():
            if len(part) > 2:
                terms.add(part.lower())
        for part in site.get("name", "").replace("-", " ").split():
            if len(part) > 2:
                terms.add(part.lower())

    extra = {
        "remotask", "remotasks", "crowdgen", "mindrift", "toloker", "tolokers",
        "alignerr", "oneforma", "mturk", "dataannotation", "surgehq",
        "appen", "sama", "scale", "telus", "prolific", "outlier",
        "cloudfactory", "imerit", "lxt", "defined", "superannotate",
        "mindy", "flipside", "digitaldivide", "humansintheloop", "toloka",
        "inc", "llc", "ltd", "corp", "gmbh",
    }
    terms.update(extra)
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
    """Minimal HTML stripper — no external dependencies."""
    def __init__(self):
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str):
        self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(self._chunks)


def strip_html(html: str) -> str:
    s = _HTMLStripper()
    s.feed(html or "")
    return s.get_text()


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

URL_RE   = re.compile(r"https?://\S+|www\.\S+")
NUM_RE   = re.compile(r"\b\d+\b")
SPACE_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Fix 4 — Invisible Unicode characters
# Strips zero-width spaces, soft hyphens, directional marks, joiners, and
# the byte-order mark before any other processing.  These characters are
# invisible in all displays but silently corrupt tokenisation by splitting
# words like "ability" into "ab​il​ity" (with U+200B between syllables).
# ---------------------------------------------------------------------------
INVISIBLE_RE = re.compile(
    r"[\u00ad"           # SOFT HYPHEN
    r"\u200b-\u200f"     # ZERO WIDTH SPACE … RIGHT-TO-LEFT MARK
    r"\u202a-\u202e"     # LTR/RTL EMBED, OVERRIDE, POP
    r"\u2060-\u2064"     # WORD JOINER, function application, invisible operators
    r"\u3164"            # HANGUL FILLER (used as anti-copy-paste whitespace)
    r"\uffa0"            # HALFWIDTH HANGUL FILLER
    r"\ufeff]+"          # BYTE ORDER MARK
)

# ---------------------------------------------------------------------------
# Fix 5a — Non-English page detection
#
# URL path segments that indicate a localised (non-English) page variant.
# Pattern matches /xx/ or /xx at end-of-path for common ISO 639-1 codes
# and a few multi-character codes (e.g. /ua/ for Ukrainian).
# ---------------------------------------------------------------------------
LANG_PATH_RE = re.compile(
    r"/(?:"
    r"ua|de|fr|es|ko|ja|zh|ru|vi|th|ar|pt|it|nl|pl|cs|sk|ro|bg|hr|sr|"
    r"uk|tr|hi|id|ms|el|hu|fi|sv|da|no|he|fa|ur|bn|ta|te|ml|kn|si|km|lo|"
    r"ka|am|my|ne|tl|sw|zu|af|sq|mk|bs|lv|lt|et|sl|mt|cy|gl|eu|ca"
    r")(?:/|$)",
    re.IGNORECASE,
)

# Fraction of non-whitespace characters that may be non-ASCII before a page
# is considered non-English.  15% is deliberately permissive (English pages
# may contain a handful of currency symbols or accented proper nouns).
FOREIGN_CHAR_THRESHOLD = 0.15


def clean_raw(text: str) -> str:
    """
    Clean text for TF-IDF / keyness analysis.

    Processing order:
      1. Strip invisible Unicode (Fix 4) — must come first so zero-width
         characters do not survive into the token stream as phantom splits.
      2. Strip HTML tags.
      3. Remove URLs.
      4. Remove standalone numbers (bare digits carry no lexical signal for
         the B2B vs B2W register contrasts of interest).
    """
    text = INVISIBLE_RE.sub("", text)    # Fix 4: invisible chars
    text = strip_html(text)
    text = URL_RE.sub(" ", text)
    text = NUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def clean_for_embedding(text: str) -> str:
    """
    Light cleaning for sentence-transformer / Word2Vec input.

    Sentence boundaries and punctuation are preserved because:
      - Sentence transformers encode contextual meaning using punctuation
        as structural cues.
      - Word2Vec benefits from seeing numbers in context (e.g. "pay $15
        per hour" is more informative than "pay per hour").

    Only invisible characters (Fix 4), HTML, and URLs are removed.
    Numbers and punctuation are kept.
    """
    text = INVISIBLE_RE.sub("", text)    # Fix 4: invisible chars
    text = strip_html(text)
    text = URL_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# spaCy pipeline — with sentencizer for sentence boundary detection
# ---------------------------------------------------------------------------

def load_nlp():
    """
    Load spaCy with parser/NER disabled and sentencizer added.

    Why sentencizer instead of the full parser?
      The parser provides dependency-based sentence segmentation but adds
      ~60% processing overhead.  The sentencizer uses punctuation-based
      rules and is sufficient for the primarily English promotional /
      informational text in this corpus.

    Raises:
        OSError: if en_core_web_sm is not installed.
                 Run: python -m spacy download en_core_web_sm
    """
    log.info("Loading spaCy model (en_core_web_sm + sentencizer)...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        # Add rule-based sentence segmenter (no parser needed)
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        log.info("spaCy model loaded. Pipes: %s", nlp.pipe_names)
        return nlp
    except OSError:
        log.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        raise


# ---------------------------------------------------------------------------
# Fix 5a — Page-level language detection
# ---------------------------------------------------------------------------

def is_foreign_page(url: str, cleaned_text: str) -> bool:
    """
    Return True if this page appears to be non-English and should be skipped.

    Two tests are applied in order (cheapest first):

    1. URL path language code: if the URL contains a path segment matching a
       known ISO 639-1 language code (e.g. /ua/, /de/, /ko/), the page is a
       localised variant and is excluded.  This catches remoter.me/ua/,
       crowdgen.com/de/, etc., without loading the page text at all.

    2. Character ratio: if more than FOREIGN_CHAR_THRESHOLD (15%) of the
       non-whitespace characters in the cleaned text are non-ASCII, the page
       is likely written in a non-Latin-script language (Korean, Japanese,
       Chinese, Arabic, Thai, etc.) or is a densely multilingual page.
       15% is permissive enough to allow pages with a few accented names or
       currency symbols but strict enough to catch genuinely foreign pages
       (which typically show 30–80% non-ASCII).

    Args:
        url          : Page URL string.
        cleaned_text : Text after INVISIBLE_RE and strip_html (before spaCy).

    Returns:
        True → exclude this page.  False → process normally.
    """
    # Test 1: URL language path
    if LANG_PATH_RE.search(url or ""):
        return True

    # Test 2: character ratio on cleaned text
    if cleaned_text:
        non_ws = [c for c in cleaned_text if not c.isspace()]
        if non_ws:
            ratio = sum(1 for c in non_ws if not c.isascii()) / len(non_ws)
            if ratio > FOREIGN_CHAR_THRESHOLD:
                return True

    return False


# ---------------------------------------------------------------------------
# Fix 5b — Token-level non-Latin-script filter
# ---------------------------------------------------------------------------

def _is_latin_token(s: str) -> bool:
    """
    Return True if every non-ASCII character in s belongs to the Latin script.

    Allows:
      - Pure ASCII tokens (all standard English words).
      - Extended-Latin characters: accented vowels (é, ü, ñ), ligatures,
        and other Latin-script variants that appear in English loanwords
        and proper nouns (café, naïve, etc.).

    Rejects tokens containing:
      - Cyrillic (Russian/Ukrainian)
      - CJK / Han (Chinese, Japanese, Korean)
      - Hangul (Korean)
      - Arabic / Hebrew
      - Devanagari (Hindi) / Bengali / Gurmukhi / other Indic scripts
      - Thai / Lao / Khmer / Myanmar / Georgian / Armenian / Sinhala

    These are genuine non-English vocabulary items that should not enter
    the English corpus term list.

    Args:
        s: A lowercased lemma string.

    Returns:
        True if the token is safe to keep.
    """
    for c in s:
        if c.isascii():
            continue
        if "LATIN" in unicodedata.name(c, ""):
            continue
        return False
    return True


def _filter_token(token, lemma: str) -> bool:
    """
    Return True if this token should be KEPT (passes all filters).

    Filters applied (in order):
      1. Skip whitespace and punctuation tokens.
      2. Skip empty or single-character lemmas.
      3. Skip spaCy stopwords unless in STOPWORD_WHITELIST.
      4. Skip brand/company terms in COMPANY_STOPWORDS.
      5. Skip tokens containing non-Latin-script characters (Fix 5b):
         catches any residual Korean/CJK/Cyrillic/Arabic tokens that
         slipped through the page-level language filter.
    """
    if token.is_space or token.is_punct:
        return False
    if not lemma or len(lemma) < 2:
        return False
    if token.is_stop and lemma not in STOPWORD_WHITELIST:
        return False
    if lemma in COMPANY_STOPWORDS:
        return False
    if not _is_latin_token(lemma):          # Fix 5b: non-Latin script
        return False
    return True


def tokenize_into_segments(nlp, text: str) -> list[list[str]]:
    """
    Tokenise text into per-sentence lemma lists.

    Each sentence becomes one list of filtered, lowercased lemmas.
    Empty sentences (after filtering) are dropped.

    This is the KEY fix for the false co-occurrence bug.  By splitting
    into sentences before tokenisation, the co-occurrence window in
    02_step1_analysis.py will only ever pair tokens that appeared in
    the same sentence — not tokens from different page sections that
    happened to be adjacent in the concatenated flat text.

    Args:
        nlp:  Loaded spaCy Language object (with sentencizer).
        text: Cleaned text string (output of clean_raw).

    Returns:
        List of per-sentence token lists.
        Example: [["annotate", "datum", "high", "quality"],
                  ["worker", "earn", "flexible", "schedule"], ...]
    """
    doc = nlp(text.lower())
    segments: list[list[str]] = []

    for sent in doc.sents:
        tokens: list[str] = []
        for token in sent:
            lemma = token.lemma_.strip()
            if _filter_token(token, lemma):
                tokens.append(lemma)
        if tokens:
            segments.append(tokens)

    return segments


def flatten_segments(segments: list[list[str]]) -> list[str]:
    """Flatten per-sentence token lists into a single ordered list."""
    return [token for segment in segments for token in segment]


# ---------------------------------------------------------------------------
# Bigram construction — within sentences only
# ---------------------------------------------------------------------------

def make_bigrams_from_segments(segments: list[list[str]]) -> list[str]:
    """
    Generate adjacent bigrams as 'word1_word2' strings.

    Bigrams are formed only within each sentence (segment).
    This prevents cross-boundary bigrams where the last token of one
    page section is paired with the first token of the next section —
    a source of meaningless bigrams in the original pipeline.

    Args:
        segments: Per-sentence token lists from tokenize_into_segments.

    Returns:
        Flat list of bigram strings from all sentences.
    """
    bigrams: list[str] = []
    for segment in segments:
        for i in range(len(segment) - 1):
            bigrams.append(f"{segment[i]}_{segment[i+1]}")
    return bigrams


def compute_bigram_counts(all_segment_lists: list[list[list[str]]]) -> dict[str, int]:
    """
    Count bigram frequency across all pages in the corpus.

    Used in Pass 1 to identify frequent-enough bigrams to retain.

    Args:
        all_segment_lists: List of per-page segment lists
                           (each page → list of sentences → list of tokens).

    Returns:
        Dict mapping each bigram string to its total corpus frequency.
    """
    counts: dict[str, int] = {}
    for segments in all_segment_lists:
        for bg in make_bigrams_from_segments(segments):
            counts[bg] = counts.get(bg, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def init_tables(conn: sqlite3.Connection):
    """
    Create pages_tfidf and pages_embedding tables.

    pages_tfidf schema additions vs src/preprocess.py:
      segments — JSON list of lists: [[sent_tokens], [sent_tokens], ...]
                 New column; added via ALTER TABLE if the table already
                 exists from a previous run.

    pages_embedding schema:
      clean_for_embedding now produces genuinely different text from
      clean_raw (preserves numbers and punctuation).
    """
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages_tfidf (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id      INTEGER UNIQUE NOT NULL,
            url          TEXT,
            audience     TEXT,
            segments     TEXT,    -- JSON list of lists: per-sentence token lists
            unigrams     TEXT,    -- JSON flat list (segments flattened)
            bigrams      TEXT,    -- JSON list of within-sentence bigrams (freq-filtered)
            token_count  INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (page_id) REFERENCES pages(id)
        )
    """)

    # Add segments column to existing tables that predate this script.
    # ALTER TABLE ADD COLUMN is a no-op if the column already exists in
    # SQLite ≥ 3.37; for older versions we catch the error explicitly.
    for col_def in [
        "ALTER TABLE pages_tfidf ADD COLUMN segments TEXT",
    ]:
        try:
            cursor.execute(col_def)
        except sqlite3.OperationalError:
            pass  # Column already exists — safe to continue

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages_embedding (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id        INTEGER UNIQUE NOT NULL,
            url            TEXT,
            audience       TEXT,
            clean_text     TEXT,    -- for sentence-transformers (punct + numbers kept)
            tokenized_text TEXT,    -- space-separated lemmas for Word2Vec / fastText
            processed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (page_id) REFERENCES pages(id)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tfidf_page ON pages_tfidf(page_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_emb_page   ON pages_embedding(page_id)")
    conn.commit()
    log.info("Tables pages_tfidf and pages_embedding are ready.")


# ---------------------------------------------------------------------------
# Duplicate exclusion
# ---------------------------------------------------------------------------

def load_excluded_ids(db_path: str, duplicates_file: str | None) -> set[int]:
    """
    Load page_ids flagged as near-duplicates by find_duplicates.py.

    Within each cluster, the page with the longest text_content is kept;
    all others are excluded.  Keeping the longest page maximises
    vocabulary coverage per cluster.

    Args:
        db_path:          Path to the SQLite database.
        duplicates_file:  Path to duplicate_report.json, or None to skip.

    Returns:
        Set of page_id integers to exclude from processing.
    """
    if not duplicates_file or not Path(duplicates_file).exists():
        if duplicates_file:
            log.warning(f"Duplicates file not found: {duplicates_file} — skipping.")
        return set()

    with open(duplicates_file, encoding="utf-8") as f:
        clusters = json.load(f)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    excluded: set[int] = set()

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
        excluded.update(pid for pid in page_ids if pid != keep_id)

    conn.close()
    log.info(f"Duplicate exclusion: {len(excluded)} pages excluded.")
    return excluded


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def fetch_all_ids(conn: sqlite3.Connection,
                  excluded_ids: set[int]) -> list[int]:
    """
    Return IDs of ALL pages with text_content, minus excluded duplicates.

    Unlike src/preprocess.py (which fetched only unprocessed pages),
    this script re-processes all pages on every run using INSERT OR REPLACE.
    This ensures the segments column is always consistent with unigrams
    and bigrams, even if the script is run after a schema change.

    Ordered by id for reproducibility.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM pages
        WHERE text_content IS NOT NULL
        ORDER BY id
    """)
    all_ids = [row[0] for row in cursor.fetchall()]

    if excluded_ids:
        before = len(all_ids)
        all_ids = [i for i in all_ids if i not in excluded_ids]
        log.info(f"Excluded {before - len(all_ids)} duplicate pages.")

    return all_ids


def fetch_batch(conn: sqlite3.Connection,
                ids: list[int]) -> list[sqlite3.Row]:
    cursor = conn.cursor()
    placeholders = ",".join("?" * len(ids))
    cursor.execute(
        f"SELECT id, url, text_content FROM pages WHERE id IN ({placeholders})",
        ids,
    )
    return cursor.fetchall()


def audience_from_url(url: str) -> str:
    """
    Fallback audience assignment from URL matching.

    NOTE: This is stored in pages_tfidf.audience but is NOT used by
    any analysis script.  All analysis scripts derive audience via
    corpus_view → platforms table (from config), which is 100% reliable.
    This column is kept only as a convenience for quick ad-hoc queries.
    """
    for domain, label in AUDIENCE_MAP.items():
        if domain in url:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    nlp,
    rows: list[sqlite3.Row],
    frequent_bigrams: set[str],
) -> tuple[list[tuple], list[tuple]]:
    """
    Tokenise one batch of pages into segments, unigrams, and bigrams.

    For each page:
      1. clean_raw()               → strip HTML/URLs/numbers
      2. tokenize_into_segments()  → per-sentence token lists
      3. flatten_segments()        → flat unigram list
      4. make_bigrams_from_segments() → within-sentence bigrams (all)
      5. Filter bigrams by frequent_bigrams set
      6. clean_for_embedding()     → lightly cleaned text (punct/nums kept)

    Args:
        nlp:              Loaded spaCy Language object.
        rows:             Batch of page rows (id, url, text_content).
        frequent_bigrams: Set of bigrams passing MIN_BIGRAM_FREQ threshold.

    Returns:
        (tfidf_rows, embed_rows) — tuples ready for executemany insertion.
    """
    tfidf_rows: list[tuple] = []
    embed_rows: list[tuple] = []

    for row in rows:
        raw       = clean_raw(row["text_content"] or "")
        clean_emb = clean_for_embedding(row["text_content"] or "")
        audience  = audience_from_url(row["url"])

        segments  = tokenize_into_segments(nlp, raw)
        unigrams  = flatten_segments(segments)
        all_bigrams = make_bigrams_from_segments(segments)
        bigrams   = [bg for bg in all_bigrams if bg in frequent_bigrams]

        if audience == "unknown":
            log.debug(f"Unknown audience for URL: {row['url']}")

        if len(unigrams) < 10:
            log.warning(
                f"Low token count ({len(unigrams)}) for "
                f"page id={row['id']} url={row['url']}"
            )

        tfidf_rows.append((
            row["id"], row["url"], audience,
            json.dumps(segments),
            json.dumps(unigrams),
            json.dumps(bigrams),
            len(unigrams),
        ))
        embed_rows.append((
            row["id"], row["url"], audience,
            clean_emb,
            " ".join(unigrams),
        ))

    return tfidf_rows, embed_rows


def insert_batch(
    conn: sqlite3.Connection,
    tfidf_rows: list[tuple],
    embed_rows: list[tuple],
):
    """
    Insert processed rows into pages_tfidf and pages_embedding.

    Uses INSERT OR REPLACE: re-running the script on already-processed
    pages refreshes their records.  This is safe because all data is
    derived deterministically from pages.text_content.
    """
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR REPLACE INTO pages_tfidf
            (page_id, url, audience, segments, unigrams, bigrams, token_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, tfidf_rows)
    cursor.executemany("""
        INSERT OR REPLACE INTO pages_embedding
            (page_id, url, audience, clean_text, tokenized_text)
        VALUES (?, ?, ?, ?, ?)
    """, embed_rows)
    conn.commit()


# ---------------------------------------------------------------------------
# Fix 5a — Foreign page exclusion (corpus-wide scan)
# ---------------------------------------------------------------------------

def detect_and_exclude_foreign_pages(
    conn: sqlite3.Connection,
    all_ids: list[int],
) -> tuple[list[int], int]:
    """
    Scan all candidate pages and remove non-English ones from the processing
    queue.  Also deletes any stale pages_tfidf records for excluded pages
    (from a previous run before this fix was applied).

    Why delete stale records?
      This script uses INSERT OR REPLACE, so if a foreign page already has a
      record from a previous run, it would be overwritten on the next run —
      but only if it is still in the processing queue.  By explicitly deleting
      stale records here, we ensure a clean re-run even if the foreign page
      was processed before this fix existed.

    Args:
        conn    : Open sqlite3.Connection.
        all_ids : Page IDs after duplicate exclusion.

    Returns:
        (filtered_ids, n_excluded) where filtered_ids has foreign pages
        removed and n_excluded is the count that were skipped.
    """
    log.info("Scanning for non-English pages (URL patterns + character ratio)...")

    cursor = conn.cursor()
    foreign_ids: list[int] = []
    kept_ids:    list[int] = []

    # Fetch url + enough text to compute char ratio; no need for full content
    placeholders = ",".join("?" * len(all_ids))
    rows = cursor.execute(
        f"SELECT id, url, SUBSTR(text_content, 1, 4000) AS sample "
        f"FROM pages WHERE id IN ({placeholders})",
        all_ids,
    ).fetchall()

    for row in rows:
        page_id = row[0]
        url     = row[1] or ""
        sample  = INVISIBLE_RE.sub("", row[2] or "")
        sample  = strip_html(sample)
        sample  = URL_RE.sub(" ", sample)

        if is_foreign_page(url, sample):
            foreign_ids.append(page_id)
        else:
            kept_ids.append(page_id)

    n_foreign = len(foreign_ids)
    if n_foreign:
        log.info(f"  Excluding {n_foreign} non-English pages.")
        # Delete stale pages_tfidf rows left over from previous runs
        chunk_size = 500  # avoid hitting SQLite variable limit
        deleted = 0
        for i in range(0, n_foreign, chunk_size):
            chunk = foreign_ids[i: i + chunk_size]
            ph    = ",".join("?" * len(chunk))
            cursor.execute(
                f"DELETE FROM pages_tfidf WHERE page_id IN ({ph})", chunk
            )
            deleted += cursor.rowcount
        conn.commit()
        if deleted:
            log.info(
                f"  Deleted {deleted} stale pages_tfidf record(s) "
                f"for previously-processed foreign pages."
            )
    else:
        log.info("  No non-English pages detected.")

    return kept_ids, n_foreign


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(db_path: str = DB_PATH, batch_size: int = BATCH_SIZE):
    """
    Run the full two-pass preprocessing pipeline.

    Pass 1: Scan all pages to count bigram frequencies corpus-wide.
            Global counting prevents high intra-page repetition from
            inflating a bigram's apparent prevalence.
    Pass 2: Tokenise each page, filter bigrams, insert into DB.

    Failed batches in Pass 2 are logged and skipped to ensure one
    malformed page does not abort the entire corpus.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    log.info("=" * 60)
    log.info("00_preprocess.py — Corpus Preprocessing")
    log.info(f"  Database         : {db_path}")
    log.info(f"  Batch size       : {batch_size}")
    log.info(f"  Min bigram freq  : {MIN_BIGRAM_FREQ}")
    log.info(f"  Audience map     : {len(AUDIENCE_MAP)} domains")
    log.info(f"  Company stopwords: {len(COMPANY_STOPWORDS)} terms")
    log.info(f"  Duplicates file  : {DUPLICATES_FILE or 'not set'}")
    log.info("=" * 60)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_tables(conn)

    nlp = load_nlp()

    excluded_ids = load_excluded_ids(db_path, DUPLICATES_FILE)

    log.info("Loading all page IDs...")
    all_ids = fetch_all_ids(conn, excluded_ids)

    if not all_ids:
        log.info("No pages to process. Exiting.")
        conn.close()
        return

    # ------------------------------------------------------------------
    # Fix 5a: exclude non-English pages before any NLP processing
    # ------------------------------------------------------------------
    all_ids, n_foreign = detect_and_exclude_foreign_pages(conn, all_ids)

    total = len(all_ids)
    if total == 0:
        log.info("No English pages remain after language filtering. Exiting.")
        conn.close()
        return

    n_batches = (total + batch_size - 1) // batch_size
    log.info(
        f"{total} English pages to process "
        f"({n_foreign} non-English excluded) "
        f"→ {n_batches} batches of ≤{batch_size}"
    )

    # ------------------------------------------------------------------
    # Pass 1: corpus-wide bigram frequency count
    # ------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Pass 1/2 — collecting bigram frequencies...")
    bigram_counts: dict[str, int] = {}

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch_ids = all_ids[start: start + batch_size]
        log.info(f"  [Pass 1] Batch {batch_num}/{n_batches}")
        try:
            rows = fetch_batch(conn, batch_ids)
            for row in rows:
                raw      = clean_raw(row["text_content"] or "")
                segments = tokenize_into_segments(nlp, raw)
                for bg in make_bigrams_from_segments(segments):
                    bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        except Exception:
            log.warning(
                f"  [Pass 1] Batch {batch_num} error (skipped):\n"
                + traceback.format_exc()
            )

    frequent_bigrams = {bg for bg, c in bigram_counts.items() if c >= MIN_BIGRAM_FREQ}
    log.info(f"  Unique bigrams found : {len(bigram_counts):,}")
    log.info(f"  Kept (freq ≥ {MIN_BIGRAM_FREQ})  : {len(frequent_bigrams):,}")
    log.info(f"  Discarded            : {len(bigram_counts) - len(frequent_bigrams):,}")

    # ------------------------------------------------------------------
    # Pass 2: tokenise, segment, filter bigrams, insert
    # ------------------------------------------------------------------
    log.info("-" * 60)
    log.info("Pass 2/2 — tokenising and inserting into database...")
    failed_batches: list[dict] = []
    inserted = 0

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch_ids = all_ids[start: start + batch_size]
        batch_end = min(start + batch_size, total)
        log.info(
            f"  [Pass 2] Batch {batch_num}/{n_batches}  "
            f"(pages {start+1}–{batch_end})"
        )
        try:
            rows = fetch_batch(conn, batch_ids)
            tfidf_rows, embed_rows = process_batch(nlp, rows, frequent_bigrams)
            insert_batch(conn, tfidf_rows, embed_rows)
            inserted += len(tfidf_rows)
            log.info(
                f"  [Pass 2] Batch {batch_num} done  "
                f"(total so far: {inserted}/{total})"
            )
        except Exception:
            log.error(
                f"  [Pass 2] Batch {batch_num} FAILED (skipped):\n"
                + traceback.format_exc()
            )
            failed_batches.append({"batch_num": batch_num, "page_ids": batch_ids})

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("PREPROCESSING COMPLETE")
    log.info(f"  Non-English pages excluded   : {n_foreign}")
    log.info(f"  Pages processed successfully : {inserted}/{total}")
    log.info(f"  Batches failed               : {len(failed_batches)}")

    if failed_batches:
        failed_count = sum(len(b["page_ids"]) for b in failed_batches)
        log.warning(f"  {failed_count} pages skipped.")
        failed_path = Path(db_path).parent / "failed_batches.json"
        with open(failed_path, "w") as f:
            json.dump(failed_batches, f, indent=2)
        log.warning(f"  Failed page IDs saved to: {failed_path}")
    else:
        log.info("  All batches completed successfully.")

    log.info("=" * 60)
    log.info("Next step: python3 src2/01_prepare_corpus.py")
    conn.close()


if __name__ == "__main__":
    process()
