"""
diagnose_cooc4.py
-----------------
For terms missing from cooccurrence_results, checks whether they exist
in keyness_results (i.e. they ARE in the corpus, just not stored as
co-occurrence focus terms).

Also shows the full list of distinct focus_terms stored, so we can
understand what selection criterion was used when the analysis was run.

Usage:
    python3 diagnose_cooc4.py
"""

import sqlite3

DB_PATH = "data/scraping.db"

MISSING = ["labour", "task", "worker", "quality", "oversight",
           "intelligent", "ai", "automation", "create", "automate"]

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

print("=" * 65)
print("Missing terms: presence in keyness_results (cross_platform)")
print("=" * 65)

for term in MISSING:
    row = conn.execute("""
        SELECT term, ll_score, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = 'cross_platform'
          AND term = ? AND term_type = 'unigram'
    """, [term]).fetchone()

    if row:
        direction = "CLIENT-key" if row["ll_score"] > 0 else "WORKER-key"
        print(f"  {term:15s}  IN KEYNESS  ll={row['ll_score']:8.1f}  "
              f"({direction})  "
              f"B2B={row['rel_freq_client']:.3f}‰  B2W={row['rel_freq_worker']:.3f}‰")
    else:
        print(f"  {term:15s}  NOT IN KEYNESS EITHER — absent from corpus or filtered")

print()
print("=" * 65)
print("Total distinct focus terms stored in cooccurrence_results")
print("=" * 65)
n = conn.execute(
    "SELECT COUNT(DISTINCT focus_term) as n FROM cooccurrence_results "
    "WHERE comparison = 'cross_platform'"
).fetchone()["n"]
print(f"  {n} distinct focus terms")

print()
print("=" * 65)
print("Top 40 focus terms by total collocate count (both audiences)")
print("=" * 65)
rows = conn.execute("""
    SELECT focus_term, SUM(1) as n_collocates, MAX(ll.ll_score) as ll
    FROM cooccurrence_results co
    LEFT JOIN keyness_results ll
      ON ll.term = co.focus_term
      AND ll.comparison = 'cross_platform'
      AND ll.term_type = 'unigram'
    WHERE co.comparison = 'cross_platform'
    GROUP BY co.focus_term
    ORDER BY n_collocates DESC
    LIMIT 40
""").fetchall()
for r in rows:
    print(f"  {r['focus_term']:20s}  collocates={r['n_collocates']:5d}  "
          f"ll={r['ll'] or 0:8.1f}")

conn.close()
