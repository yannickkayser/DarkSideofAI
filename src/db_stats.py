"""
db_stats.py
-----------
Reads the scraped-content SQLite database and reports:
  1. Number of websites per category (website_type)
  2. Number of URLs (pages) per website
  3. Word count of the full text corpus per website

Usage:
    python db_stats.py <path_to_database.db> [output.txt]

Defaults:
    output file → db_stats_report.txt
"""

import sqlite3
import sys
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

    for section_fn in [section_categories, section_pages_per_website, section_word_counts]:
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
