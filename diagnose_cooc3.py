"""
diagnose_cooc3.py
-----------------
Checks which theoretically motivated candidate terms exist in
cooccurrence_results (cross_platform) and previews their clean collocates.

Candidate groups mirror the three hypotheses:
  H1a — labour visibility:        labour, work, job, task, pay, earn, worker
  H1b — automation myth:          autonomous, automate, machine, intelligent, ai
  H1c — strategic hypervisibility: human, quality, oversight, annotation, label

Usage:
    python3 diagnose_cooc3.py
"""

import sqlite3

DB_PATH = "data/scraping.db"

ARTIFACT_TERMS = {
    "cookie", "set_cookie", "cooky", "/hr", "/hr_remote", "remote_apply",
    "feb", "opportunity_feb", "faq", "faq_help", "help_desk", "desk",
    "subscribe", "website", "account", "access", "enable", "microworker",
    "shall", "youtube", "zeynep", "koouchnir", "gavrilov", "unga", "gary",
    "yalda", "monarch", "warhol", "fremont", "pittsburgh", "mpii",
    "experience.with", "rhml", "ead", "cc0", "ft", "hole", "overfit",
    "surprised", "christmas", "morale", "high-quality", "slash", "500",
    "pickup", "loophole", "conceptually", "housing", "firefighting",
    "sidestep", "wary", "downward", "jira", "voluman", "squeeze",
    "retrofit", "yt", "ml", "deciphering", "trafficking", "recap",
    "ueberwinden", "bildbearbeitung", "sicherstellung", "kunst",
    "human-le", "pto", "generous", "dhanesh", "ramachandram",
    "outlet", "daniela", "braga", "forbe",
}

CANDIDATE_GROUPS = {
    "H1a — Labour visibility":         ["labour", "work", "job", "task", "pay", "earn", "worker"],
    "H1b — Automation myth":           ["autonomous", "automate", "machine", "intelligent", "ai", "automation"],
    "H1c — Strategic hypervisibility": ["human", "quality", "oversight", "annotation", "label", "create"],
}

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
ph = ",".join("?" * len(ARTIFACT_TERMS))

print("=" * 70)
print("Candidate term availability in cooccurrence_results (cross_platform)")
print("=" * 70)

for group, terms in CANDIDATE_GROUPS.items():
    print(f"\n  {group}")
    for term in terms:
        rows = conn.execute("""
            SELECT audience, COUNT(*) as n, MIN(cofreq) as min_f, MAX(cofreq) as max_f
            FROM cooccurrence_results
            WHERE comparison = 'cross_platform' AND focus_term = ?
            GROUP BY audience
        """, [term]).fetchall()

        if not rows:
            print(f"    {term:15s}  -- NOT IN DB --")
            continue

        for r in rows:
            top = conn.execute(f"""
                SELECT collocate, pmi, cofreq
                FROM cooccurrence_results
                WHERE comparison = 'cross_platform'
                  AND focus_term = ? AND audience = ?
                  AND collocate NOT IN ({ph})
                ORDER BY pmi DESC LIMIT 4
            """, [term, r["audience"]] + list(ARTIFACT_TERMS)).fetchall()
            preview = ", ".join(f"{x['collocate']}({x['cofreq']})" for x in top)
            print(f"    {term:15s}  {r['audience']:8s}  "
                  f"n={r['n']:4d}  cofreq {r['min_f']}–{r['max_f']:5d}  "
                  f"top: {preview}")

conn.close()
