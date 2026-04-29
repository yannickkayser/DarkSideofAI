"""
sample_restructure.py
=====================

Builds a company-centred reading sample from the five within-company
pairs, plus supplementary cross-platform pages from strata 2+3.

The five pairs
--------------
    appen.com (client) ↔ crowdgen.com (worker)         JSD = 0.4102
    scale.com (client) ↔ remotasks.com (worker)        JSD = 0.4085
    labelbox.com (client) ↔ alignerr.com (worker)      JSD = 0.4029
    centific.com (client) ↔ oneforma.com (worker)      JSD = 0.3318
    toloka.ai (client) ↔ mindrift.ai (worker)          JSD = 0.2556

For each pair, the script selects the top-N pages per side ranked by
the strongest STM topic signal, then builds a reading manifest ordered:

    Section A — Within-company cases (pair by pair, worker → client)
    Section B — Cross-platform evidence (strata 2+3 from existing sample)

Usage
-----
    python sample_restructure.py /path/to/scraping_2.db [--pages-per-side N]

    --pages-per-side N   Pages to keep per domain (default: 5).
                         5 pairs × 2 sides × 5 pages = 50 case pages.

Outputs
-------
    outputs/reading_manifest.csv          Full reading list
    outputs/reading_manifest_summary.txt  Compact printout
    step2_sample_structured (DB table)    Queryable from notebook
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


# ─── CLI ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("db", nargs="?", default="scraping_2.db")
parser.add_argument("--pages-per-side", type=int, default=5,
                    help="Pages per domain side (default 5)")
parser.add_argument("--strata23-top", type=int, default=1,
                    help="Top-N per slot for strata 2+3 (default 1)")
args = parser.parse_args()

DB_PATH = Path(args.db)
PAGES_PER_SIDE = args.pages_per_side
STRATA23_TOP = args.strata23_top
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ─── Define the five within-company pairs ──────────────────────────
# Ordered by JSD descending (most divergent = most contrastive first)
PAIRS = [
    {
        "company":       "appen",
        "client_domain": "www.appen.com",
        "worker_domain": "crowdgen.com",
        "jsd":           0.4102,
    },
    {
        "company":       "scale",
        "client_domain": "scale.com",
        "worker_domain": "www.remotasks.com",
        "jsd":           0.4085,
    },
    {
        "company":       "labelbox",
        "client_domain": "labelbox.com",
        "worker_domain": "www.alignerr.com",
        "jsd":           0.4029,
    },
    {
        "company":       "centific",
        "client_domain": "www.centific.com",
        "worker_domain": "www.oneforma.com",
        "jsd":           0.3318,
    },
    {
        "company":       "toloka",
        "client_domain": "toloka.ai",
        "worker_domain": "mindrift.ai",
        "jsd":           0.2556,
    },
]

# Pages already coded by the user — always protected from trimming.
ALREADY_CODED = {9802, 9808, 9811, 9812, 6599}

# Hypothesis coding for STM topics (from the notebook's TOPIC_HYPOTHESIS)
TOPIC_HYPOTHESIS = {
    1:  'H1a',  2:  'H1b',  3:  None, 4:  'H1b',  5:  'H1a',
    6:  'H1c',  7:  'H1b',  8:  None, 9:  'H1b',  10: None,
    11: 'H1a',  12: 'H1c',  13: None, 14: 'H1a',  15: None,
    16: 'H1b',  17: 'H1a',  18: 'H1b', 19: 'H1b',  20: 'H1b',
    21: 'H1b',  22: 'H1a',  23: 'H1b', 24: 'H1b',  25: 'H1b',
}

# ─── Boilerplate topics (mapped to None in TOPIC_HYPOTHESIS) ──────
# Pages whose dominant topic is one of these carry no hypothesis-relevant
# content — they are legal text, cookie banners, or navigational chrome.
# Excluding them from close reading follows Baker (2006, §4.2) and
# Mautner (2009): boilerplate saturates frequency and topic measures
# but contributes nothing to register analysis.
BOILERPLATE_TOPICS = {
    tid for tid, h in TOPIC_HYPOTHESIS.items() if h is None
}   # = {3, 8, 10, 13, 15}

# ─── URL-path patterns that signal non-content pages ──────────────
# Matched case-insensitively against the full URL. Any page whose URL
# contains one of these substrings is excluded from selection (but not
# from the corpus-level STM/keyness aggregates, which already ran).
import re
BOILERPLATE_URL_PATTERNS = re.compile(
    r'(?i)('
    r'terms[-_]of[-_]service|terms[-_]and[-_]conditions|'
    r'privacy[-_]polic|privacy[-_]statement|'
    r'cookie[-_]polic|cookie[-_]notice|'
    r'legal[-_]polic|legal/|/legal|'
    r'data[-_]?process|gdpr|'
    r'acceptable[-_]use|'
    r'modern[-_]slavery|'
    r'imprint|impressum|'
    r'disclaimer|'
    r'/api-reference/|/docs/api|'
    r'/careers/\d|/jobs/\d'
    r')'
)


# ─── Connect ───────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)

# ─── Check which tables exist ──────────────────────────────────────
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print(f"DB tables: {', '.join(sorted(tables))}")

HAS_STM = "stm_theta" in tables
HAS_STEP2 = "step2_sample" in tables
print(f"  stm_theta present: {HAS_STM}")
print(f"  step2_sample present: {HAS_STEP2}")


# ─── Helper: get pages for a domain ───────────────────────────────
def get_domain_pages(domain: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all pages for a domain with token counts and STM theta."""

    # Try multiple domain variants (with/without www.)
    domain_variants = [domain]
    if domain.startswith("www."):
        domain_variants.append(domain[4:])
    else:
        domain_variants.append("www." + domain)

    placeholders = ",".join(["?"] * len(domain_variants))

    # Base page query via websites join
    q_pages = f"""
    SELECT p.id AS page_id,
           p.url,
           w.domain,
           COALESCE(pt.token_count, 0) AS token_count
    FROM   pages p
    JOIN   websites w ON p.website_id = w.id
    LEFT JOIN pages_tfidf pt ON pt.page_id = p.id
    WHERE  w.domain IN ({placeholders})
    """
    pages = pd.read_sql(q_pages, conn, params=domain_variants)
    # Normalise page_id to Python int (avoids int64/object merge errors)
    if not pages.empty:
        pages["page_id"] = pages["page_id"].astype(int)

    if pages.empty:
        print(f"    ⚠ No pages found for {domain} (tried {domain_variants})")
        return pages

    # Add STM theta (dominant topic + max theta)
    if HAS_STM and not pages.empty:
        page_ids = pages["page_id"].tolist()
        if page_ids:
            ph = ",".join(["?"] * len(page_ids))
            q_theta = f"""
            SELECT page_id, topic_id, theta
            FROM   stm_theta
            WHERE  page_id IN ({ph})
            """
            theta = pd.read_sql(q_theta, conn, params=page_ids)
            # Normalise page_id here too
            if not theta.empty:
                theta["page_id"] = theta["page_id"].astype(int)

            if not theta.empty:
                # Get dominant topic per page
                idx = theta.groupby("page_id")["theta"].idxmax()
                dominant = theta.loc[idx, ["page_id", "topic_id", "theta"]].copy()
                dominant["page_id"] = dominant["page_id"].astype(int)
                dominant = dominant.rename(columns={
                    "topic_id": "dominant_topic",
                    "theta": "max_theta"
                })
                pages = pages.merge(dominant, on="page_id", how="left")

                # Get max theta across hypothesis-relevant topics
                theta["hypothesis"] = theta["topic_id"].map(TOPIC_HYPOTHESIS)
                for h in ["H1a", "H1b", "H1c"]:
                    h_theta = (theta[theta["hypothesis"] == h]
                               .groupby("page_id")["theta"]
                               .max()
                               .reset_index()
                               .rename(columns={"theta": f"theta_{h}"}))
                    h_theta["page_id"] = h_theta["page_id"].astype(int)
                    pages = pages.merge(h_theta, on="page_id", how="left")

    # Fill NaN theta columns
    for col in ["dominant_topic", "max_theta", "theta_H1a", "theta_H1b", "theta_H1c"]:
        if col not in pages.columns:
            pages[col] = None

    return pages


# ─── Helper: select top-N pages for a domain ──────────────────────
def select_top_pages(pages: pd.DataFrame, n: int,
                     protected: set[int]) -> pd.DataFrame:
    """Select top-N pages for close reading.

    Selection logic (applied in order):
      1. Always keep protected pages (already coded by user).
      2. Always keep the homepage (first encounter with the register).
      3. EXCLUDE pages whose dominant STM topic is boilerplate
         (T3, T8, T10, T13, T15 — legal text, cookie notices, nav chrome).
      4. EXCLUDE pages whose URL matches boilerplate patterns
         (terms-of-service, privacy-policy, legal/, api-reference, etc.).
      5. Rank remaining candidates by hypothesis-relevant theta:
             best_hyp_theta = max(theta_H1a, theta_H1b, theta_H1c)
         NOT by raw max_theta, which may peak on a boilerplate topic.
      6. Take the top-N from that ranking.

    Methodological justification: Baker (2006, §4.2) and Mautner (2009)
    both exclude formulaic/legal text from close reading on the ground
    that it inflates frequency counts without contributing to the
    register patterns under study. The notebook's Cell 4b applied the
    same filter at the step2_sample level; this function applies it at
    the per-domain selection level.
    """

    if pages.empty:
        return pages

    pages = pages.copy()

    # ── Identify homepage candidates ───────────────────────────
    pages["is_homepage"] = pages["url"].apply(
        lambda u: u.rstrip("/").split("//")[-1].count("/") == 0
        if pd.notna(u) else False
    )

    # ── Tag boilerplate pages ──────────────────────────────────
    pages["is_boilerplate_topic"] = pages["dominant_topic"].apply(
        lambda t: int(t) in BOILERPLATE_TOPICS
        if pd.notna(t) else False
    )
    pages["is_boilerplate_url"] = pages["url"].apply(
        lambda u: bool(BOILERPLATE_URL_PATTERNS.search(str(u)))
        if pd.notna(u) else False
    )
    pages["is_boilerplate"] = (
        pages["is_boilerplate_topic"] | pages["is_boilerplate_url"]
    )

    # ── Compute hypothesis-relevant ranking score ──────────────
    # max(theta_H1a, theta_H1b, theta_H1c) — ignores boilerplate topics
    hyp_cols = ["theta_H1a", "theta_H1b", "theta_H1c"]
    available_hyp = [c for c in hyp_cols if c in pages.columns]
    if available_hyp:
        pages["best_hyp_theta"] = pages[available_hyp].max(axis=1)
    else:
        pages["best_hyp_theta"] = pages.get("max_theta", 0)

    # ── Partition into must-keep vs. candidates ────────────────
    is_protected = pages["page_id"].isin(protected)
    is_home = pages["is_homepage"]

    must_keep = pages[is_protected | is_home].copy()

    candidates = pages[
        ~(is_protected | is_home) & ~pages["is_boilerplate"]
    ].copy()

    # Count what we excluded
    n_excluded = int(pages["is_boilerplate"].sum())
    n_bp_topic = int(pages["is_boilerplate_topic"].sum())
    n_bp_url = int(pages["is_boilerplate_url"].sum())
    if n_excluded > 0:
        print(f"      excluded {n_excluded} boilerplate pages "
              f"({n_bp_topic} by topic, {n_bp_url} by URL)")

    # ── Rank by hypothesis-relevant theta ──────────────────────
    candidates = candidates.sort_values("best_hyp_theta", ascending=False)

    slots = max(0, n - len(must_keep))
    selected = pd.concat([must_keep, candidates.head(slots)], ignore_index=True)
    selected = selected.drop_duplicates(subset="page_id")

    return selected


# ═══════════════════════════════════════════════════════════════════
# SECTION A — Within-company case pages
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"SECTION A — Within-company pairs ({PAGES_PER_SIDE} pages/side)")
print(f"{'='*60}")

case_frames: list[pd.DataFrame] = []

for case_num, pair in enumerate(PAIRS, 1):
    company = pair["company"]
    jsd = pair["jsd"]
    print(f"\n  Case {case_num}: {company} (JSD = {jsd:.4f})")
    print(f"    client: {pair['client_domain']}")
    print(f"    worker: {pair['worker_domain']}")

    for aud_label, domain_key in [("worker", "worker_domain"),
                                   ("client", "client_domain")]:
        domain = pair[domain_key]
        all_pages = get_domain_pages(domain, conn)
        selected = select_top_pages(all_pages, PAGES_PER_SIDE, ALREADY_CODED)
        n_total = len(all_pages)
        n_sel = len(selected)
        print(f"    {aud_label}: {n_sel}/{n_total} pages selected"
              f" (domain={domain})")

        if not selected.empty:
            selected["company"] = company
            selected["audience"] = aud_label
            selected["jsd"] = jsd
            selected["case_number"] = case_num
            selected["case_label"] = f"Case {case_num} — {company}"
            selected["reading_section"] = "A. Within-company cases"
            selected["stratum"] = 1
            case_frames.append(selected)

if case_frames:
    case_all = pd.concat(case_frames, ignore_index=True)
else:
    case_all = pd.DataFrame()

print(f"\n  Section A total: {len(case_all)} pages")


# ═══════════════════════════════════════════════════════════════════
# SECTION B — Strata 2+3 cross-platform pages (from existing sample)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"SECTION B — Cross-platform evidence (strata 2+3)")
print(f"{'='*60}")

case_page_ids = set(case_all["page_id"].tolist()) if not case_all.empty else set()
case_domains = set()
for p in PAIRS:
    case_domains.add(p["client_domain"])
    case_domains.add(p["worker_domain"])
    # Also add without www.
    for key in ["client_domain", "worker_domain"]:
        d = p[key]
        case_domains.add(d)
        if d.startswith("www."):
            case_domains.add(d[4:])
        else:
            case_domains.add("www." + d)

if HAS_STEP2:
    existing = pd.read_sql("SELECT * FROM step2_sample", conn)
    print(f"  Existing step2_sample: {len(existing)} rows")

    # Exclude pages already in Section A (by page_id) and pages from
    # case-company domains (to avoid reading the same firm twice)
    other = existing[
        ~existing["page_id"].isin(case_page_ids) &
        ~existing["domain"].isin(case_domains)
    ].copy()
    print(f"  After excluding case-company domains: {len(other)} rows")

    # Trim stratum 2: top-N per (hypothesis, topic_id, audience)
    s2 = other[other["stratum"] == 2].copy()
    if not s2.empty:
        s2 = s2.sort_values("theta", ascending=False)
        s2 = s2.groupby(["hypothesis", "topic_id", "audience"]).head(STRATA23_TOP)

    # Trim stratum 3: top-N per (hypothesis, focal_term, audience)
    s3 = other[other["stratum"] == 3].copy()
    if not s3.empty:
        s3 = s3.sort_values("rel_freq", ascending=False)
        s3 = s3.groupby(["hypothesis", "focal_term", "audience"]).head(STRATA23_TOP)

    # Other strata (supplementary)
    s_other = other[~other["stratum"].isin([2, 3])]

    section_b = pd.concat([s2, s3, s_other], ignore_index=True)
    section_b = section_b.drop_duplicates(subset="page_id")

    # Add ordering columns
    hyp_rank = {"H1a": 0, "H1b": 1, "H1c": 2}
    aud_rank = {"worker": 0, "client": 1}
    section_b["_hyp"] = section_b["hypothesis"].map(hyp_rank).fillna(3)
    section_b["_aud"] = section_b["audience"].map(aud_rank).fillna(2)
    section_b = section_b.sort_values(
        ["_hyp", "_aud", "stratum"],
        ascending=True
    ).reset_index(drop=True)
    section_b["reading_section"] = "B. Cross-platform evidence"
    section_b["case_label"] = "S2+S3 — " + section_b["hypothesis"].astype(str)

    # Drop helper cols
    section_b = section_b.drop(columns=["_hyp", "_aud"], errors="ignore")

    print(f"  Section B total: {len(section_b)} pages")
else:
    print("  ⚠ No step2_sample table — Section B will be empty.")
    print("    Run the notebook sampling cells first, then re-run this script.")
    section_b = pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# COMBINE AND BUILD READING ORDER
# ═══════════════════════════════════════════════════════════════════

# Section A: ordered by case_number → audience (worker first) → max_theta desc
if not case_all.empty:
    aud_sort = {"worker": 0, "client": 1}
    case_all["_aud"] = case_all["audience"].map(aud_sort).fillna(2)
    case_all = case_all.sort_values(
        ["case_number", "_aud", "max_theta"],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    case_all = case_all.drop(columns=["_aud"], errors="ignore")

combined = pd.concat([case_all, section_b], ignore_index=True)
combined["reading_order"] = range(1, len(combined) + 1)

# Mark already-coded pages
combined["coded"] = combined["page_id"].isin(ALREADY_CODED)


# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"FINAL READING LIST: {len(combined)} pages")
print(f"  Section A (within-company): {len(case_all)} pages")
print(f"  Section B (cross-platform): {len(section_b)} pages")
print(f"{'='*60}")

# Summary by case
print("\nBreakdown:")
for label, grp in combined.groupby("case_label", sort=False):
    aud = grp["audience"].value_counts()
    w = aud.get("worker", 0)
    c = aud.get("client", 0)
    doms = grp["domain"].nunique()
    coded = grp["coded"].sum()
    coded_str = f"  ({coded} already coded)" if coded > 0 else ""
    print(f"  {label}: {len(grp)} pages ({w}W / {c}C, {doms} domains){coded_str}")


# Write CSV manifest
manifest_cols = [
    "reading_order", "reading_section", "case_label", "case_number",
    "company", "domain", "audience", "page_id", "url",
    "token_count", "dominant_topic", "max_theta",
    "theta_H1a", "theta_H1b", "theta_H1c",
    "jsd", "coded",
]
# Only include columns that exist
available = [c for c in manifest_cols if c in combined.columns]
# Also include any columns from step2_sample that we want to preserve
for extra in ["hypothesis", "stratum", "topic_id", "theta",
              "focal_term", "rel_freq", "sampling_reason"]:
    if extra in combined.columns and extra not in available:
        available.append(extra)

manifest = combined[available].copy()
manifest.to_csv(OUT / "reading_manifest.csv", index=False)
print(f"\n  → {OUT / 'reading_manifest.csv'}")

# Write to DB
combined.to_sql("step2_sample_structured", conn, if_exists="replace", index=False)
print(f"  → step2_sample_structured table in DB")


# Print the full manifest
summary_lines: list[str] = []
summary_lines.append(f"{'='*70}")
summary_lines.append(f"READING MANIFEST — {len(combined)} pages")
summary_lines.append(f"{'='*70}")

current_case = None
current_aud = None

for _, row in combined.iterrows():
    cl = row.get("case_label", "")
    if cl != current_case:
        current_case = cl
        current_aud = None
        jsd_str = f"  (JSD = {row['jsd']:.4f})" if pd.notna(row.get("jsd")) else ""
        summary_lines.append(f"\n── {current_case}{jsd_str} ──")

    aud = row.get("audience", "")
    if aud != current_aud:
        current_aud = aud
        domain = row.get("domain", "?")
        summary_lines.append(f"    [{current_aud.upper()} — {domain}]")

    coded_str = " ✓CODED" if row.get("coded") else ""
    theta_str = f"θ={row['max_theta']:.3f}" if pd.notna(row.get("max_theta")) else ""
    tok = int(row["token_count"]) if pd.notna(row.get("token_count")) else 0
    url = str(row.get("url", ""))[:60]

    summary_lines.append(
        f"    {int(row['reading_order']):>3d}. p{int(row['page_id']):<6d} "
        f"{theta_str:10s} {tok:>6d}tok  {url}{coded_str}"
    )

summary_text = "\n".join(summary_lines)
print(summary_text)

(OUT / "reading_manifest_summary.txt").write_text(summary_text, encoding="utf-8")
print(f"\n  → {OUT / 'reading_manifest_summary.txt'}")

conn.close()

print(f"\n{'='*60}")
print("Done.")
print(f"  Read Section A first (5 company pairs, worker → client).")
print(f"  Then Section B (cross-platform strata 2+3).")
print(f"{'='*60}")
