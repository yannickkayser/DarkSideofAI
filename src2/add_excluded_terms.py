"""
add_excluded_terms.py
=====================
Inserts non-English / noise terms into the excluded_terms table in the
SQLite database.  Both the Python pipeline and the R STM scripts read this
table to filter vocabulary before analysis.

Run from the project root:
    python3 src2/add_excluded_terms.py

Re-run 01_prepare.R in RStudio afterwards — no full preprocessing needed.
"""

import sqlite3
from pathlib import Path

DB_PATH = "data/scraping_2.db"

# ---------------------------------------------------------------------------
# Terms to exclude
# ---------------------------------------------------------------------------
# These fall into four categories:
#
#  A. Foreign function words (slipped through Latin-script filter because
#     they use accented Latin characters, not Cyrillic/CJK):
#     Norwegian/Swedish: på, så, når, få, får, från, både
#     Spanish:           más, qué, tú, años, cómo
#     French:            être, tâche, données, flexibilité
#     Turkish:           için, çalışma
#     Polish:            się
#     Italian:           più, attività
#
#  B. Foreign-language versions of thesis-relevant words:
#     (qualité, qualità, análisis, español, français) — keep English versions
#
#  C. Place names with diacritics that appear as boilerplate
#     (e.g. platform support-page listings of countries/territories):
#     réunion, barthélemy, curaçao, åland, príncipe, tomé, montréal,
#     méxico, belgië, cómo
#
#  D. Company/location proper nouns with high frequency:
#     büropark (393 occurrences — likely a German business-park name on
#               one specific platform's contact page)
#
# NOTE: 'café' is intentionally NOT excluded — it is a genuine English
# loanword that may appear legitimately in platform text.
# ---------------------------------------------------------------------------

TERMS_TO_EXCLUDE = {
    # A — foreign function words
    "på", "så", "når", "få", "får", "från", "både",   # Scandinavian
    "más", "qué", "tú", "años", "cómo",                # Spanish
    "être", "tâche", "données", "flexibilité",          # French
    "için", "çalışma",                                  # Turkish
    "się",                                             # Polish
    "più", "attività",                                  # Italian

    # B — foreign-language thesis keywords
    "qualité", "qualità", "análisis", "español", "français",

    # C — place names with diacritics (boilerplate country listings)
    "réunion", "barthélemy", "curaçao", "åland", "príncipe",
    "tomé", "montréal", "méxico", "belgië",

    # D — high-frequency proper nouns / location names
    "büropark",

    # Additional from the same diagnostic run
    "são", "côte", "ärztin",
}


def main():
    db = Path(DB_PATH)
    if not db.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}\n"
            "Run this script from the project root."
        )

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Create table if it doesn't exist yet
    cur.execute("""
        CREATE TABLE IF NOT EXISTS excluded_terms (
            term TEXT PRIMARY KEY
        )
    """)

    # Insert — ignore duplicates so this script is safe to re-run
    inserted = 0
    skipped  = 0
    for term in sorted(TERMS_TO_EXCLUDE):
        try:
            cur.execute("INSERT INTO excluded_terms (term) VALUES (?)", (term,))
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()

    # Show current state of the table
    total = cur.execute("SELECT COUNT(*) FROM excluded_terms").fetchone()[0]
    conn.close()

    print(f"  Inserted : {inserted} new terms")
    print(f"  Skipped  : {skipped} already present")
    print(f"  Total    : {total} terms in excluded_terms table")
    print()
    print("  Next step: re-run 01_prepare.R in RStudio")
    print("  (No need to re-run 00_preprocess.py)")


if __name__ == "__main__":
    main()
