"""
03b_import_stm.py
=================
Load the STM results exported by R (04_export.R) back into SQLite so that
the Python figure scripts and any downstream analysis can query them directly
without reading CSV files.

Pipeline position:
  Stage 3b — STM Import (run after 04_export.R in R)
  Reads from: STMAnalysis/output/step_1/stm/*.csv
  Writes to:  scraping_2.db  (four tables + one view)
  Next step:  04b_step1_stm_figures.py  (STM visualisations)

Tables created / replaced:
  stm_theta        — per-document topic proportions (θ matrix)
  stm_topic_terms  — top-N terms per topic (prob + frex / sage columns)
  stm_prevalence   — audience prevalence effects from estimateEffect()
  stm_content      — per-audience word log-odds per topic (if available)

View created / replaced:
  stm_topic_profile — convenience join of stm_theta + stm_prevalence
                       + corpus_view metadata (audience, domain, platform_type)

Usage:
    python3 src2/03b_import_stm.py
"""

import sqlite3
import csv
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH    = "scraping_2.db"
STM_DIR    = Path("STMAnalysis/output/step_1/stm")

THETA_CSV      = STM_DIR / "stm_theta.csv"
TERMS_CSV      = STM_DIR / "stm_topic_terms.csv"
PREVALENCE_CSV = STM_DIR / "stm_prevalence.csv"
CONTENT_CSV    = STM_DIR / "stm_content.csv"   # optional — only with BOTH_STRATEGY=exclude

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(msg: str) -> None:
    log.info("=" * 60)
    log.info(msg)
    log.info("=" * 60)


def require_csv(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV not found: {path}\n"
            "  → Run 04_export.R in R first."
        )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def coerce_float(val: str) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def coerce_int(val: str) -> int | None:
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def coerce_bool(val: str) -> int | None:
    """R writes TRUE/FALSE; SQLite stores as 1/0."""
    if val is None:
        return None
    if val.upper() in ("TRUE", "1"):
        return 1
    if val.upper() in ("FALSE", "0"):
        return 0
    return None

# ---------------------------------------------------------------------------
# Table loaders
# ---------------------------------------------------------------------------

def load_theta(con: sqlite3.Connection, rows: list[dict]) -> int:
    """
    stm_theta: one row per document × topic proportion.

    Columns from R:
      page_id, audience, domain,
      topic_1 … topic_K,   (K dynamic — we pivot to long format)
      dominant_topic, dominant_prop
    """
    con.execute("DROP TABLE IF EXISTS stm_theta")
    con.execute("""
        CREATE TABLE stm_theta (
            page_id        TEXT    NOT NULL,
            audience       TEXT    NOT NULL,
            domain         TEXT,
            topic_id       INTEGER NOT NULL,
            theta          REAL    NOT NULL,
            dominant_topic INTEGER NOT NULL,
            dominant_prop  REAL    NOT NULL
        )
    """)

    # Detect topic columns dynamically (topic_1 … topic_K)
    if not rows:
        return 0
    topic_cols = [c for c in rows[0].keys() if c.startswith("topic_")
                  and c[6:].isdigit()]
    topic_cols.sort(key=lambda c: int(c[6:]))

    records = []
    for row in rows:
        page_id        = row["page_id"]
        audience       = row["audience"]
        domain         = row.get("domain", "")
        dominant_topic = coerce_int(row["dominant_topic"])
        dominant_prop  = coerce_float(row["dominant_prop"])
        for col in topic_cols:
            topic_id = int(col[6:])
            theta    = coerce_float(row[col])
            records.append((
                page_id, audience, domain,
                topic_id, theta,
                dominant_topic, dominant_prop,
            ))

    con.executemany(
        "INSERT INTO stm_theta VALUES (?,?,?,?,?,?,?)",
        records,
    )
    log.info("  stm_theta: %d rows  (%d documents × %d topics)",
             len(records), len(rows), len(topic_cols))

    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_theta_page
        ON stm_theta (page_id)
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_theta_topic
        ON stm_theta (topic_id)
    """)
    return len(records)


def load_terms(con: sqlite3.Connection, rows: list[dict]) -> int:
    """
    stm_topic_terms: top-N terms per topic under two ranking schemes.

    Columns from R:
      topic_id, rank, prob_term, frex_term
    """
    con.execute("DROP TABLE IF EXISTS stm_topic_terms")
    con.execute("""
        CREATE TABLE stm_topic_terms (
            topic_id  INTEGER NOT NULL,
            rank      INTEGER NOT NULL,
            prob_term TEXT,
            frex_term TEXT,
            PRIMARY KEY (topic_id, rank)
        )
    """)

    records = [
        (
            coerce_int(r["topic_id"]),
            coerce_int(r["rank"]),
            r.get("prob_term"),
            r.get("frex_term"),
        )
        for r in rows
    ]
    con.executemany(
        "INSERT OR REPLACE INTO stm_topic_terms VALUES (?,?,?,?)",
        records,
    )
    n_topics = len({r[0] for r in records})
    log.info("  stm_topic_terms: %d rows  (%d topics)", len(records), n_topics)
    return len(records)


def load_prevalence(con: sqlite3.Connection, rows: list[dict]) -> int:
    """
    stm_prevalence: audience prevalence effect per topic from estimateEffect().

    Columns from R:
      topic_id, frex_label, estimate, std_err, ci_lower, ci_upper,
      significant, direction
    """
    con.execute("DROP TABLE IF EXISTS stm_prevalence")
    con.execute("""
        CREATE TABLE stm_prevalence (
            topic_id   INTEGER PRIMARY KEY,
            frex_label TEXT,
            estimate   REAL,
            std_err    REAL,
            ci_lower   REAL,
            ci_upper   REAL,
            significant INTEGER,
            direction  TEXT
        )
    """)

    records = [
        (
            coerce_int(r["topic_id"]),
            r.get("frex_label"),
            coerce_float(r.get("estimate")),
            coerce_float(r.get("std_err")),
            coerce_float(r.get("ci_lower")),
            coerce_float(r.get("ci_upper")),
            coerce_bool(r.get("significant")),
            r.get("direction"),
        )
        for r in rows
    ]
    con.executemany(
        "INSERT OR REPLACE INTO stm_prevalence VALUES (?,?,?,?,?,?,?,?)",
        records,
    )
    n_sig = sum(1 for r in records if r[6] == 1)
    log.info("  stm_prevalence: %d topics  (%d significant)", len(records), n_sig)
    return len(records)


def load_content(con: sqlite3.Connection, rows: list[dict]) -> int:
    """
    stm_content: per-audience word log-odds per topic from sageLabels().

    Columns from R:
      topic_id, audience, rank, term
    """
    con.execute("DROP TABLE IF EXISTS stm_content")
    con.execute("""
        CREATE TABLE stm_content (
            topic_id INTEGER NOT NULL,
            audience TEXT    NOT NULL,
            rank     INTEGER NOT NULL,
            term     TEXT,
            PRIMARY KEY (topic_id, audience, rank)
        )
    """)

    records = [
        (
            coerce_int(r["topic_id"]),
            r["audience"],
            coerce_int(r["rank"]),
            r.get("term"),
        )
        for r in rows
    ]
    con.executemany(
        "INSERT OR REPLACE INTO stm_content VALUES (?,?,?,?)",
        records,
    )
    n_topics = len({r[0] for r in records})
    log.info("  stm_content: %d rows  (%d topics × 2 audiences)", len(records), n_topics)
    return len(records)


def create_topic_profile_view(con: sqlite3.Connection) -> None:
    """
    stm_topic_profile: convenience view joining the dominant-topic per page
    with prevalence effects and corpus metadata.

    Useful for quickly asking:
      - Which topics are client-leaning / worker-leaning?
      - What is the theta of the dominant topic per page?
      - How does topic prevalence vary by platform_type or hq_region?
    """
    con.execute("DROP VIEW IF EXISTS stm_topic_profile")
    con.execute("""
        CREATE VIEW stm_topic_profile AS
        SELECT
            th.page_id,
            th.audience,
            th.domain,
            cv.platform_type,
            cv.hq_region,
            cv.company_id,
            th.topic_id,
            th.theta,
            th.dominant_topic,
            th.dominant_prop,
            pr.frex_label,
            pr.estimate       AS prevalence_estimate,
            pr.significant    AS prevalence_significant,
            pr.direction      AS prevalence_direction
        FROM stm_theta  th
        LEFT JOIN stm_prevalence pr ON pr.topic_id = th.topic_id
        LEFT JOIN corpus_view    cv ON cv.page_id  = th.page_id
        WHERE th.topic_id = th.dominant_topic
    """)
    log.info("  View stm_topic_profile created.")


def print_summary(con: sqlite3.Connection) -> None:
    """Print a quick console overview of the imported data."""
    banner("Import summary")

    k = con.execute("SELECT COUNT(DISTINCT topic_id) FROM stm_theta").fetchone()[0]
    n_docs = con.execute("SELECT COUNT(DISTINCT page_id) FROM stm_theta").fetchone()[0]
    log.info("  K = %d topics  |  %d documents in stm_theta", k, n_docs)

    log.info("")
    log.info("  Prevalence effects (significant topics, ranked by |estimate|):")
    rows = con.execute("""
        SELECT topic_id, frex_label, estimate, direction
        FROM stm_prevalence
        WHERE significant = 1
        ORDER BY ABS(estimate) DESC
    """).fetchall()
    if rows:
        for topic_id, label, est, direction in rows:
            log.info("    T%-3d  %-30s  %+.3f  %s", topic_id, label or "", est or 0, direction or "")
    else:
        log.info("    (none significant)")

    log.info("")
    log.info("  Dominant topic distribution by audience (top 5 topics each):")
    for audience in ("client", "worker"):
        log.info("  [%s]", audience.upper())
        topic_rows = con.execute("""
            SELECT dominant_topic, COUNT(*) AS n
            FROM stm_theta
            WHERE audience = ? AND topic_id = dominant_topic
            GROUP BY dominant_topic
            ORDER BY n DESC
            LIMIT 5
        """, (audience,)).fetchall()
        for tid, n in topic_rows:
            label = con.execute(
                "SELECT frex_label FROM stm_prevalence WHERE topic_id = ?", (tid,)
            ).fetchone()
            label_str = label[0] if label and label[0] else ""
            log.info("    T%-3d  %-28s  %d pages", tid, label_str, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    banner("03b_import_stm.py — Load STM results into SQLite")
    log.info("  Database : %s", Path(DB_PATH).resolve())
    log.info("  STM dir  : %s", STM_DIR.resolve())

    # Check required files
    require_csv(THETA_CSV)
    require_csv(TERMS_CSV)
    require_csv(PREVALENCE_CSV)

    # Check optional content CSV
    has_content = CONTENT_CSV.exists()
    if not has_content:
        log.info("  stm_content.csv not found — Export 4 skipped "
                 "(expected when BOTH_STRATEGY != 'exclude').")

    log.info("")

    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")

        # Check corpus_view exists (produced by 01_prepare_corpus.py)
        cv_exists = con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('view','table') "
            "AND name='corpus_view'"
        ).fetchone()
        if not cv_exists:
            log.warning(
                "corpus_view not found — stm_topic_profile view will have "
                "NULL metadata columns. Run 01_prepare_corpus.py first."
            )

        # ── Load tables ────────────────────────────────────────────────────
        banner("Loading stm_theta")
        load_theta(con, read_csv(THETA_CSV))

        banner("Loading stm_topic_terms")
        load_terms(con, read_csv(TERMS_CSV))

        banner("Loading stm_prevalence")
        load_prevalence(con, read_csv(PREVALENCE_CSV))

        if has_content:
            banner("Loading stm_content")
            load_content(con, read_csv(CONTENT_CSV))

        # ── Create convenience view ────────────────────────────────────────
        banner("Creating stm_topic_profile view")
        create_topic_profile_view(con)

        con.commit()

        # ── Summary ────────────────────────────────────────────────────────
        print_summary(con)

    log.info("")
    log.info("✓  03b_import_stm.py complete.")
    log.info("   Next: python3 src2/04b_step1_stm_figures.py")


if __name__ == "__main__":
    main()
