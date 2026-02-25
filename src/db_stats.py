"""
db_stats.py
-----------
Reads the scraped-content SQLite database and reports:
  1. Number of websites per category (website_type)
  2. Number of URLs (pages) per website
  3. Word count of the full text corpus per website
  4. Preprocessed corpus stats (pages_tfidf)
  5. Token and bigram counts per website and audience (pages_tfidf)
  6. Embedding table coverage (pages_embedding)

Usage:
    python db_stats.py <path_to_database.db> [output.txt]

Defaults:
    output file → db_stats_report.txt
"""

import sqlite3
import sys
import json
from datetime import datetime


# ── helpers ──────────────────────────────────────────────────────────────────

def word_count(text):
    """Count words in a string, safely."""
    if not text:
        return 0
    return len(text.split())


def run_query(conn, sql, params=()):
    cur = conn.execute(sql, params)
    return cur.fetchall()


def table_exists(conn, table_name):
    """Check if a table exists in the database."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    ).fetchone()
    return row[0] > 0


# ── report sections ──────────────────────────────────────────────────────────

def section_categories(conn):
    """Websites per category (website_type)."""
    rows = run_query(conn, """
        SELECT COALESCE(website_type, 'unknown') AS category,
               COUNT(*) AS num_websites
        FROM websites
        GROUP BY category
        ORDER BY num_websites DESC
    """)

    lines = ["WEBSITES PER CATEGORY", "─" * 40]
    lines.append(f"  {'Category':<25} {'Websites':>9}")
    lines.append("  " + "-" * 36)
    total = 0
    for category, count in rows:
        lines.append(f"  {category:<25} {count:>9}")
        total += count
    lines.append("  " + "-" * 36)
    lines.append(f"  {'TOTAL':<25} {total:>9}")
    return lines


def section_pages_per_website(conn):
    """URL / page count per website."""
    rows = run_query(conn, """
        SELECT w.domain,
               COALESCE(w.website_type, 'unknown') AS category,
               COUNT(p.id) AS num_pages
        FROM websites w
        LEFT JOIN pages p ON p.website_id = w.id
        GROUP BY w.id
        ORDER BY num_pages DESC
    """)

    lines = ["URLS (PAGES) PER WEBSITE", "─" * 60]
    lines.append(f"  {'Domain':<35} {'Category':<15} {'Pages':>7}")
    lines.append("  " + "-" * 58)
    total = 0
    for domain, category, count in rows:
        lines.append(f"  {domain:<35} {category:<15} {count:>7}")
        total += count
    lines.append("  " + "-" * 58)
    lines.append(f"  {'TOTAL':<35} {'':<15} {total:>7}")
    return lines


def section_word_counts(conn):
    """Word count of extracted text content per website."""
    rows = run_query(conn, """
        SELECT w.domain,
               COALESCE(w.website_type, 'unknown') AS category,
               GROUP_CONCAT(p.text_content, ' ') AS corpus
        FROM websites w
        LEFT JOIN pages p ON p.website_id = w.id
        GROUP BY w.id
        ORDER BY w.domain
    """)

    lines = ["WORD COUNT PER WEBSITE (full text corpus)", "─" * 60]
    lines.append(f"  {'Domain':<35} {'Category':<15} {'Words':>10}")
    lines.append("  " + "-" * 62)
    grand_total = 0
    for domain, category, corpus in rows:
        wc = word_count(corpus)
        grand_total += wc
        lines.append(f"  {domain:<35} {category:<15} {wc:>10,}")
    lines.append("  " + "-" * 62)
    lines.append(f"  {'TOTAL':<35} {'':<15} {grand_total:>10,}")
    return lines


def section_tfidf_overview(conn):
    """Overall preprocessed corpus stats from pages_tfidf."""
    if not table_exists(conn, "pages_tfidf"):
        return ["PREPROCESSED CORPUS (pages_tfidf)", "─" * 60,
                "  ⚠  Table not found — run preprocess.py first."]

    # Overall counts
    totals = run_query(conn, """
        SELECT
            COUNT(*)                   AS total_pages,
            SUM(token_count)           AS total_tokens,
            AVG(token_count)           AS avg_tokens_per_page,
            MIN(token_count)           AS min_tokens,
            MAX(token_count)           AS max_tokens
        FROM pages_tfidf
    """)[0]

    # Per-audience breakdown
    aud_rows = run_query(conn, """
        SELECT
            COALESCE(audience, 'unknown')  AS audience,
            COUNT(*)                        AS pages,
            SUM(token_count)                AS tokens,
            AVG(token_count)                AS avg_tokens
        FROM pages_tfidf
        GROUP BY audience
        ORDER BY pages DESC
    """)

    # Bigram count — parse JSON to count bigrams
    bigram_rows = run_query(conn, "SELECT bigrams FROM pages_tfidf WHERE bigrams IS NOT NULL")
    total_bigrams = sum(len(json.loads(r[0])) for r in bigram_rows if r[0])

    lines = ["PREPROCESSED CORPUS OVERVIEW (pages_tfidf)", "─" * 60]
    lines.append(f"  Total pages preprocessed : {totals[0]:>10,}")
    lines.append(f"  Total tokens             : {int(totals[1] or 0):>10,}")
    lines.append(f"  Total bigrams            : {total_bigrams:>10,}")
    lines.append(f"  Avg tokens per page      : {totals[2] or 0:>10.1f}")
    lines.append(f"  Min tokens (single page) : {totals[3] or 0:>10,}")
    lines.append(f"  Max tokens (single page) : {totals[4] or 0:>10,}")
    lines.append("")
    lines.append(f"  {'Audience':<15} {'Pages':>8} {'Tokens':>12} {'Avg tokens/page':>16}")
    lines.append("  " + "-" * 54)
    for aud, pages, tokens, avg in aud_rows:
        lines.append(f"  {aud:<15} {pages:>8,} {int(tokens or 0):>12,} {avg or 0:>16.1f}")

    return lines


def section_tfidf_per_website(conn):
    """Token and bigram counts per website and audience from pages_tfidf."""
    if not table_exists(conn, "pages_tfidf"):
        return []  # already warned in overview section

    rows = run_query(conn, """
        SELECT
            t.url,
            COALESCE(t.audience, 'unknown') AS audience,
            t.token_count,
            t.bigrams
        FROM pages_tfidf t
    """)

    # Aggregate per (domain, audience)
    from collections import defaultdict
    stats = defaultdict(lambda: {"pages": 0, "tokens": 0, "bigrams": 0})

    for url, audience, token_count, bigrams_json in rows:
        # Extract domain from URL
        domain = "unknown"
        for part in url.replace("https://", "").replace("http://", "").split("/"):
            if "." in part:
                domain = part.lstrip("www.")
                break
        key = (domain, audience)
        stats[key]["pages"]   += 1
        stats[key]["tokens"]  += token_count or 0
        stats[key]["bigrams"] += len(json.loads(bigrams_json)) if bigrams_json else 0

    lines = ["TOKEN & BIGRAM COUNTS PER WEBSITE (pages_tfidf)", "─" * 72]
    lines.append(f"  {'Domain':<30} {'Audience':<10} {'Pages':>7} {'Tokens':>10} {'Bigrams':>10} {'Avg tok/pg':>10}")
    lines.append("  " + "-" * 70)

    total_pages = total_tokens = total_bigrams = 0
    for (domain, audience), s in sorted(stats.items(), key=lambda x: -x[1]["tokens"]):
        avg = s["tokens"] / s["pages"] if s["pages"] else 0
        lines.append(
            f"  {domain:<30} {audience:<10} {s['pages']:>7,} "
            f"{s['tokens']:>10,} {s['bigrams']:>10,} {avg:>10.1f}"
        )
        total_pages   += s["pages"]
        total_tokens  += s["tokens"]
        total_bigrams += s["bigrams"]

    lines.append("  " + "-" * 70)
    avg_total = total_tokens / total_pages if total_pages else 0
    lines.append(
        f"  {'TOTAL':<30} {'':<10} {total_pages:>7,} "
        f"{total_tokens:>10,} {total_bigrams:>10,} {avg_total:>10.1f}"
    )
    return lines


def section_embedding_coverage(conn):
    """Coverage and text length stats from pages_embedding."""
    if not table_exists(conn, "pages_embedding"):
        return ["EMBEDDING TABLE (pages_embedding)", "─" * 60,
                "  ⚠  Table not found — run preprocess.py first."]

    rows = run_query(conn, """
        SELECT
            COALESCE(audience, 'unknown')       AS audience,
            COUNT(*)                             AS pages,
            AVG(LENGTH(clean_text))              AS avg_clean_chars,
            AVG(LENGTH(tokenized_text))          AS avg_token_chars,
            SUM(CASE WHEN clean_text IS NULL
                      OR clean_text = '' THEN 1 ELSE 0 END) AS empty_clean,
            SUM(CASE WHEN tokenized_text IS NULL
                      OR tokenized_text = '' THEN 1 ELSE 0 END) AS empty_tokens
        FROM pages_embedding
        GROUP BY audience
        ORDER BY pages DESC
    """)

    # Cross-table coverage: how many pages_tfidf rows also have embedding?
    coverage = run_query(conn, """
        SELECT COUNT(*) FROM pages_tfidf t
        WHERE EXISTS (SELECT 1 FROM pages_embedding e WHERE e.page_id = t.page_id)
    """)[0][0] if table_exists(conn, "pages_tfidf") else "N/A"

    total_tfidf = run_query(conn, "SELECT COUNT(*) FROM pages_tfidf")[0][0] \
        if table_exists(conn, "pages_tfidf") else None

    lines = ["EMBEDDING TABLE COVERAGE (pages_embedding)", "─" * 72]

    if total_tfidf:
        pct = coverage / total_tfidf * 100 if total_tfidf else 0
        lines.append(f"  Pages with both tfidf + embedding : {coverage:,} / {total_tfidf:,}  ({pct:.1f}%)")
    lines.append("")
    lines.append(f"  {'Audience':<12} {'Pages':>8} {'Avg clean chars':>16} {'Avg token chars':>16} {'Empty clean':>12} {'Empty tokens':>13}")
    lines.append("  " + "-" * 80)

    for aud, pages, avg_clean, avg_tok, empty_c, empty_t in rows:
        lines.append(
            f"  {aud:<12} {pages:>8,} {avg_clean or 0:>16.0f} "
            f"{avg_tok or 0:>16.0f} {empty_c:>12,} {empty_t:>13,}"
        )

    lines.append("")
    lines.append("  Note: 'clean_text' = sentence-transformers input (punctuation intact)")
    lines.append("        'tokenized_text' = Word2Vec / fastText input (lemmas only)")
    return lines


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python db_stats.py <database.db> [output.txt]")
        sys.exit(1)

    db_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "db_stats_report.txt"

    conn = sqlite3.connect(db_path)

    report = []
    report.append("=" * 72)
    report.append("  SCRAPED CONTENT — DATABASE STATISTICS REPORT")
    report.append(f"  Database : {db_path}")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 72)

    # Original sections
    for section_fn in [section_categories, section_pages_per_website, section_word_counts]:
        report.append("")
        report.extend(section_fn(conn))

    # Preprocessed table sections
    report.append("")
    report.append("=" * 72)
    report.append("  PREPROCESSED DATA STATISTICS")
    report.append("=" * 72)

    for section_fn in [section_tfidf_overview, section_tfidf_per_website, section_embedding_coverage]:
        report.append("")
        report.extend(section_fn(conn))

    report.append("")
    report.append("=" * 72)
    report.append("  END OF REPORT")
    report.append("=" * 72)

    conn.close()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()