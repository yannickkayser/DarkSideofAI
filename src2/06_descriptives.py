"""
03_descriptives.py
==================
Computes descriptive statistics for the AI Labour Narrative Corpus.

Pipeline position:
  Run AFTER: 01_prepare_corpus.py (corpus_view + exclusion tables must exist)

Data source:
  corpus_view — the single authoritative join of pages_tfidf + pages +
  websites + platforms, created by 01_prepare_corpus.py.
  Excluded pages (excluded_pages table) are filtered out to match the
  exact corpus used by 02_step1_analysis.py.

  Token data (token_count, unigrams, bigrams) is read directly from
  corpus_view; no re-processing of raw text is needed.

Panels produced:
  A — By audience  (worker / client / both)
  B — By platform type  (crowd_market / enterprise_bpo / impact_sourcing)
  C — Lexical characteristics by audience
        (total tokens, unique tokens, TTR, mean bigrams/page)

Usage (run from project root, same as 01_ and 02_ scripts):
    python3 src2/03_descriptives.py
"""

import sqlite3
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scraping_2.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Canonical display order for audience rows
AUDIENCE_ORDER = ["worker", "client", "both", "Total"]

# Maps internal platform_type codes to display labels
PLATFORM_TYPE_LABELS = {
    "crowd_market":   "Algorithmic Crowd Market",
    "enterprise_bpo": "Managed Enterprise BPO",
    "impact_sourcing": "Impact Sourcing",
    "unknown":        "Other / Unknown",
}


# ===========================================================================
# Database helpers
# ===========================================================================

def open_db(db_path: Path) -> sqlite3.Connection:
    """Open the database and verify all required tables exist."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at: {db_path}\n"
            f"Run from the project root so the path resolves correctly, "
            f"or adjust DB_PATH at the top of this script."
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    required = {"corpus_view", "excluded_pages", "excluded_terms"}
    existing = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        ).fetchall()
    }
    missing = required - existing
    if missing:
        raise RuntimeError(
            f"Missing tables/views: {missing}\n"
            f"Run 01_prepare_corpus.py first."
        )
    return conn


def load_corpus(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load corpus_view, filter excluded pages, and return a DataFrame.

    Columns used:
      page_id, domain, audience, platform_type, company_id, hq_region,
      platform_name, token_count, unigrams (JSON), bigrams (JSON)
    """
    excluded_ids = {
        r[0] for r in conn.execute(
            "SELECT page_id FROM excluded_pages"
        ).fetchall()
    }

    rows = conn.execute("""
        SELECT
            page_id,
            domain,
            audience,
            platform_type,
            company_id,
            hq_region,
            platform_name,
            token_count,
            unigrams,
            bigrams
        FROM corpus_view
        WHERE audience IN ('client', 'worker', 'both')
          AND token_count >= 10
    """).fetchall()

    records = []
    for r in rows:
        if r["page_id"] in excluded_ids:
            continue
        records.append(dict(r))

    df = pd.DataFrame(records)
    log.info(
        f"Corpus loaded: {len(df):,} pages after exclusions "
        f"({len(excluded_ids)} pages excluded)."
    )
    return df


# ===========================================================================
# Panel helpers
# ===========================================================================

def _word_count_stats(group: pd.DataFrame) -> dict:
    """Return mean, SD, and median of token_count for a page group."""
    wc = group["token_count"]
    return {
        "mean_tokens": round(wc.mean()),
        "sd_tokens":   round(wc.std()),
        "median_tokens": round(wc.median()),
    }


def _lexical_stats(group: pd.DataFrame) -> dict:
    """
    Compute lexical statistics from pre-processed JSON columns.

    total_tokens    — sum of token_count (matches what 02_ scripts use)
    unique_tokens   — size of vocabulary across all pages in group
    ttr             — type-token ratio: unique / total
    mean_bigrams    — mean number of within-sentence bigrams per page

    Notes:
      - TTR is computed over the full subcorpus (not averaged per page),
        which is standard for corpus-level comparison.
      - Bigram count per page = len(json.loads(bigrams)), i.e. the number
        of within-sentence adjacent pairs stored by 00_preprocess.py.
    """
    all_tokens: list[str] = []
    bigram_counts: list[int] = []

    for _, row in group.iterrows():
        unigrams = json.loads(row["unigrams"]) if row["unigrams"] else []
        bigrams_list = json.loads(row["bigrams"]) if row["bigrams"] else []
        all_tokens.extend(unigrams)
        bigram_counts.append(len(bigrams_list))

    total_tokens  = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    ttr = round(unique_tokens / total_tokens, 3) if total_tokens else 0.0
    mean_bigrams = round(np.mean(bigram_counts)) if bigram_counts else 0

    return {
        "total_tokens":  total_tokens,
        "unique_tokens": unique_tokens,
        "ttr":           ttr,
        "mean_bigrams":  mean_bigrams,
    }


# ===========================================================================
# Panel A — By audience
# ===========================================================================

def build_panel_a(df: pd.DataFrame) -> pd.DataFrame:
    """
    Page counts, corpus share, and token statistics by audience.

    'both' rows are counted once; Total is the full corpus without
    double-counting.
    """
    rows = []
    total = len(df)

    for audience, group in df.groupby("audience"):
        stats = _word_count_stats(group)
        rows.append({
            "audience":      audience,
            "n_pages":       len(group),
            "pct_corpus":    round(len(group) / total * 100, 1),
            **stats,
        })

    # Total row
    total_stats = _word_count_stats(df)
    rows.append({
        "audience":      "Total",
        "n_pages":       total,
        "pct_corpus":    100.0,
        **total_stats,
    })

    result = pd.DataFrame(rows)
    result["audience"] = pd.Categorical(
        result["audience"], categories=AUDIENCE_ORDER, ordered=True
    )
    return result.sort_values("audience").reset_index(drop=True)


# ===========================================================================
# Panel B — By platform type
# ===========================================================================

def build_panel_b(df: pd.DataFrame) -> pd.DataFrame:
    """
    Page counts and token statistics by platform type.

    n_sites counts distinct domains within each type.
    platform_type labels are mapped to human-readable strings.
    """
    rows = []
    total = len(df)

    for ptype, group in df.groupby("platform_type"):
        label = PLATFORM_TYPE_LABELS.get(ptype, ptype)
        rows.append({
            "platform_type": label,
            "n_sites":       group["domain"].nunique(),
            "n_pages":       len(group),
            "pct_corpus":    round(len(group) / total * 100, 1),
            "mean_tokens":   round(group["token_count"].mean()),
        })

    rows.append({
        "platform_type": "Total",
        "n_sites":       df["domain"].nunique(),
        "n_pages":       total,
        "pct_corpus":    100.0,
        "mean_tokens":   round(df["token_count"].mean()),
    })

    return pd.DataFrame(rows)


# ===========================================================================
# Panel C — Lexical characteristics by audience
# ===========================================================================

def build_panel_c(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vocabulary richness statistics by audience.

    Audience groups are built explicitly so 'both' appears as its own row
    and the Total row covers the full corpus without duplication.
    """
    rows = []

    audience_slices: dict[str, pd.DataFrame] = dict(tuple(df.groupby("audience")))
    audience_slices["Total"] = df

    for label in AUDIENCE_ORDER:
        if label not in audience_slices:
            continue
        stats = _lexical_stats(audience_slices[label])
        rows.append({"audience": label, **stats})

    result = pd.DataFrame(rows)
    result["audience"] = pd.Categorical(
        result["audience"], categories=AUDIENCE_ORDER, ordered=True
    )
    return result.sort_values("audience").reset_index(drop=True)


# ===========================================================================
# Bonus: per-platform breakdown (useful for appendix / audit)
# ===========================================================================

def build_per_platform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-domain page counts and token statistics.

    Sorted by platform_type then audience for easy reading.
    Useful as an appendix table or for spotting corpus imbalance.
    """
    rows = []
    for (domain, audience, ptype), group in df.groupby(
        ["domain", "audience", "platform_type"]
    ):
        rows.append({
            "domain":        domain,
            "audience":      audience,
            "platform_type": PLATFORM_TYPE_LABELS.get(ptype, ptype),
            "n_pages":       len(group),
            "total_tokens":  int(group["token_count"].sum()),
            "mean_tokens":   round(group["token_count"].mean()),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["platform_type", "audience", "n_pages"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


# ===========================================================================
# Main
# ===========================================================================

def main():
    log.info("=" * 60)
    log.info("03_descriptives.py — Corpus Descriptive Statistics")
    log.info("=" * 60)

    conn = open_db(DB_PATH)
    df   = load_corpus(conn)
    conn.close()

    panel_a      = build_panel_a(df)
    panel_b      = build_panel_b(df)
    panel_c      = build_panel_c(df)
    per_platform = build_per_platform(df)

    divider = "=" * 60

    print(f"\n{divider}")
    print("  PANEL A — By audience")
    print(divider)
    print(panel_a.to_string(index=False))

    print(f"\n{divider}")
    print("  PANEL B — By platform type")
    print(divider)
    print(panel_b.to_string(index=False))

    print(f"\n{divider}")
    print("  PANEL C — Lexical characteristics by audience")
    print(divider)
    print(panel_c.to_string(index=False))

    print(f"\n{divider}")
    print("  PER-PLATFORM BREAKDOWN (appendix / audit)")
    print(divider)
    print(per_platform.to_string(index=False))

    log.info("Done.")
    return {
        "panel_a":      panel_a,
        "panel_b":      panel_b,
        "panel_c":      panel_c,
        "per_platform": per_platform,
    }


if __name__ == "__main__":
    main()
