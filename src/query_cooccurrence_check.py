"""
query_cooccurrence_check.py
===========================
Diagnostic script for investigating unexpected co-occurring word pairs
found during STM / LDA topic inspection.

Given two (or more) words, this script finds all pages in the corpus
where both words appear, then shows:
  - page metadata (domain, audience, URL)
  - the raw token list (to see word context)
  - the token window around each word (KWIC — Key Word In Context)
  - the dominant LDA topic (from document_topics table, if available)

Usage
-----
    python3 query_cooccurrence_check.py

Edit the QUERY_WORDS list below to investigate different pairs.
"""

import sqlite3
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------

DB_PATH     = "data/scraping.db"
QUERY_WORDS = ["work", "baby"]   # words that co-occur suspiciously
WINDOW      = 10                 # tokens either side for KWIC context
MAX_PAGES   = 30                 # max pages to display

# ---------------------------------------------------------------------------

def kwic(tokens, target, window=8):
    """Return Key-Word-In-Context snippets for target in token list."""
    snippets = []
    for i, tok in enumerate(tokens):
        if tok == target:
            left  = tokens[max(0, i - window): i]
            right = tokens[i + 1: i + 1 + window]
            snippets.append(" ".join(left) + f"  >>>{tok}<<<  " + " ".join(right))
    return snippets


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Check corpus_view exists
    has_view = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name='corpus_view'"
    ).fetchone()

    # Check document_topics table (from LDA run)
    has_topics = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='document_topics'"
    ).fetchone()

    print(f"Co-occurrence diagnostic: {' + '.join(repr(w) for w in QUERY_WORDS)}")
    print("=" * 70)

    if has_view:
        rows = conn.execute("""
            SELECT cv.page_id, cv.domain, cv.audience, cv.unigrams,
                   p.url
            FROM   corpus_view cv
            JOIN   pages p ON p.id = cv.page_id
            ORDER  BY cv.page_id
        """).fetchall()
    else:
        # Fallback without corpus_view
        rows = conn.execute("""
            SELECT pt.page_id, w.domain, pt.unigrams,
                   p.url, 'unknown' AS audience
            FROM   pages_tfidf pt
            JOIN   pages p    ON p.id = pt.page_id
            JOIN   websites w ON w.id = p.website_id
            ORDER  BY pt.page_id
        """).fetchall()

    matches = []
    for row in rows:
        if not row["unigrams"]:
            continue
        tokens = json.loads(row["unigrams"])
        token_set = set(tokens)
        if all(w in token_set for w in QUERY_WORDS):
            matches.append((row, tokens))

    print(f"Pages containing all of {QUERY_WORDS}: {len(matches)}")

    if not matches:
        print("No co-occurrences found. Check spelling (tokens are lemmatised).")
        return

    # Audience breakdown
    audiences = {}
    for row, _ in matches:
        aud = row["audience"] if "audience" in row.keys() else "unknown"
        audiences[aud] = audiences.get(aud, 0) + 1
    print(f"Audience breakdown: {audiences}")
    print()

    for i, (row, tokens) in enumerate(matches[:MAX_PAGES]):
        pid = row["page_id"]
        dom = row["domain"]
        aud = row["audience"] if "audience" in row.keys() else "?"
        url = row["url"] if "url" in row.keys() else "?"

        # Get dominant topic if available
        topic_str = ""
        if has_topics:
            t = conn.execute("""
                SELECT dominant_topic, topic_weight
                FROM   document_topics
                WHERE  page_id = ?
                LIMIT  1
            """, (pid,)).fetchone()
            if t:
                topic_str = f"  dominant_topic=T{t['dominant_topic']} (w={t['topic_weight']:.3f})"

        print(f"[{i+1}] page_id={pid}  domain={dom}  audience={aud}{topic_str}")
        print(f"     url: {url}")

        # KWIC for each query word
        for word in QUERY_WORDS:
            snippets = kwic(tokens, word, window=WINDOW)
            for s in snippets[:2]:    # max 2 snippets per word per page
                print(f"     KWIC '{word}': ...{s}...")

        # Token count and first 30 tokens for quick scan
        print(f"     tokens ({len(tokens)} total): {' '.join(tokens[:30])}{'...' if len(tokens) > 30 else ''}")
        print()

    if len(matches) > MAX_PAGES:
        print(f"  ... {len(matches) - MAX_PAGES} more pages not shown (increase MAX_PAGES)")

    conn.close()


if __name__ == "__main__":
    main()
