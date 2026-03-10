"""
diagnose_cooc2.py
-----------------
Lists all focus terms stored in cooccurrence_results, with collocate counts
and top non-noise collocates, so we can pick the best terms for fig2.

Usage:
    python3 diagnose_cooc2.py
"""

import sqlite3

DB_PATH = "data/scraping.db"

# Same artifact filter as the visualisation script
ARTIFACT_TERMS = {
    "cookie", "set_cookie", "cooky",
    "/hr", "/hr_remote", "remote_apply", "feb", "opportunity_feb",
    "faq", "faq_help", "help_desk", "desk", "subscribe",
    "website", "account", "access", "enable", "microworker", "shall", "youtube",
    "zeynep", "koouchnir", "gavrilov", "unga", "gary", "yalda",
    "monarch", "warhol", "fremont", "pittsburgh", "mpii",
    "experience.with", "rhml", "ead", "cc0", "ft",
    "hole", "overfit", "surprised", "christmas", "morale", "high-quality",
    "slash", "500", "pickup", "loophole", "conceptually", "housing",
    "firefighting", "sidestep", "wary", "downward", "jira", "voluman",
    "squeeze", "retrofit", "yt", "ml",
    # Additional HTML/scraping noise visible in human collocates
    'title="share', "branch", "being",
}

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
ph = ",".join("?" * len(ARTIFACT_TERMS))

print("=" * 65)
print("All focus terms in cooccurrence_results (cross_platform)")
print("=" * 65)

rows = conn.execute("""
    SELECT focus_term, audience, COUNT(*) as n, MAX(cofreq) as max_f
    FROM cooccurrence_results
    WHERE comparison = 'cross_platform'
    GROUP BY focus_term, audience
    ORDER BY focus_term, audience
""").fetchall()

seen = set()
for r in rows:
    key = r["focus_term"]
    if key not in seen:
        seen.add(key)
        print(f"\n  '{key}'")
    print(f"    audience={r['audience']:8s}  collocates={r['n']:5d}  max_cofreq={r['max_f']}")

print("\n" + "=" * 65)
print("Top 6 clean collocates per focus_term × audience (PMI-ranked)")
print("=" * 65)

for focus_term in sorted(seen):
    for audience in ["client", "worker"]:
        top = conn.execute(f"""
            SELECT collocate, pmi, cofreq
            FROM cooccurrence_results
            WHERE comparison = 'cross_platform'
              AND focus_term = ? AND audience = ?
              AND collocate NOT IN ({ph})
            ORDER BY pmi DESC LIMIT 6
        """, [focus_term, audience] + list(ARTIFACT_TERMS)).fetchall()
        if top:
            preview = ", ".join(f"{r['collocate']}(pmi={r['pmi']:.1f},f={r['cofreq']})"
                                for r in top)
            print(f"\n  '{focus_term}'  {audience}: {preview}")

conn.close()
