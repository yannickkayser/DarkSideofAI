"""
sampling_diagnostics.py
=======================

Diagnostic script for §4.4 sampling protocol. Runs BEFORE the pair-
selection pipeline and explains why the eligible pool is what it is.
Writes tables and a figure suitable for the thesis appendix.

Usage
-----
    python sampling_diagnostics.py /absolute/path/to/scraping_2.db

Outputs (relative to CWD or OUT env var)
-----------------------------------------
    outputs/diag_audience_coding.csv      Table D.1
    outputs/diag_company_pairability.csv  Table D.2
    outputs/diag_domain_tokens.csv        Table D.3
    outputs/diag_floor_sweep.csv          Table D.4
    outputs/diag_pair_token_ecdf.png      Figure D.1
    outputs/diag_summary.md               One-page narrative summary

Reads only. Writes no rows to the database.
"""

from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scraping_2.db")
OUT = Path(os.environ.get("OUT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)

FLOORS = [250, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]

if not DB_PATH.exists():
    sys.exit(f"scraping database not found at {DB_PATH}")

conn = sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------
# D.1 — audience coding of scraped platforms
# ---------------------------------------------------------------------
q1 = """
SELECT p.audience,
       COUNT(DISTINCT p.domain)     AS n_domains,
       COUNT(DISTINCT p.company_id) AS n_companies
FROM   platforms p
JOIN   websites  w ON w.domain = p.domain
GROUP BY p.audience
ORDER BY n_domains DESC;
"""
d1 = pd.read_sql(q1, conn)
d1.to_csv(OUT / "diag_audience_coding.csv", index=False)
print("\n[D.1] Audience coding of scraped platforms")
print(d1.to_string(index=False))


# ---------------------------------------------------------------------
# D.2 — company-level pairability
# ---------------------------------------------------------------------
q2 = """
SELECT p.company_id,
       SUM(CASE WHEN p.audience = 'worker' THEN 1 ELSE 0 END) AS n_worker,
       SUM(CASE WHEN p.audience = 'client' THEN 1 ELSE 0 END) AS n_client,
       SUM(CASE WHEN p.audience = 'both'   THEN 1 ELSE 0 END) AS n_both
FROM   platforms p
WHERE  EXISTS (SELECT 1 FROM websites w WHERE w.domain = p.domain)
GROUP BY p.company_id;
"""
d2 = pd.read_sql(q2, conn)
d2["pairable_strict"] = (d2["n_worker"] >= 1) & (d2["n_client"] >= 1)
d2["pairable_with_both_split"] = (
    ((d2["n_worker"] + d2["n_both"]) >= 1)
    & ((d2["n_client"] + d2["n_both"]) >= 1)
    & ((d2["n_worker"] + d2["n_client"] + d2["n_both"]) >= 2)
)
d2 = d2.sort_values(
    ["pairable_strict", "pairable_with_both_split", "company_id"],
    ascending=[False, False, True],
)
d2.to_csv(OUT / "diag_company_pairability.csv", index=False)
print(f"\n[D.2] Companies with scraped data: {len(d2)}")
print(f"  strict within-company pairable (worker AND client): {d2['pairable_strict'].sum()}")
print(f"  with 'both'-split fallback:                          {d2['pairable_with_both_split'].sum()}")
print(d2.head(25).to_string(index=False))


# ---------------------------------------------------------------------
# D.3 — token counts per (company, domain, audience)
# ---------------------------------------------------------------------
q3 = """
SELECT   p.company_id,
         p.domain,
         p.audience,
         COALESCE(SUM(pt.token_count), 0) AS tokens,
         COUNT(DISTINCT pg.id)            AS n_pages
FROM     platforms   p
JOIN     websites    w  ON w.domain     = p.domain
LEFT JOIN pages      pg ON pg.website_id = w.id
LEFT JOIN pages_tfidf pt ON pt.page_id    = pg.id
GROUP BY p.company_id, p.domain, p.audience
ORDER BY p.company_id, p.audience;
"""
d3 = pd.read_sql(q3, conn)
d3.to_csv(OUT / "diag_domain_tokens.csv", index=False)
print(f"\n[D.3] Per-domain token counts (n={len(d3)})")
with pd.option_context("display.max_rows", None):
    print(d3.to_string(index=False))
print("\n  summary stats:")
print(d3[["tokens", "n_pages"]].describe().to_string())


# ---------------------------------------------------------------------
# D.4 — floor sensitivity sweep (strict within-company, and with both-split)
# ---------------------------------------------------------------------
def enumerate_pairs(d3: pd.DataFrame, allow_both_split: bool) -> pd.DataFrame:
    if allow_both_split:
        worker_side = d3[d3["audience"].isin(["worker", "both"])].copy()
        client_side = d3[d3["audience"].isin(["client", "both"])].copy()
    else:
        worker_side = d3[d3["audience"] == "worker"].copy()
        client_side = d3[d3["audience"] == "client"].copy()
    merged = worker_side.merge(
        client_side,
        on="company_id",
        suffixes=("_worker", "_client"),
    )
    merged = merged[merged["domain_worker"] != merged["domain_client"]]
    return merged

sweep_rows = []
for allow_both_split in [False, True]:
    merged = enumerate_pairs(d3, allow_both_split)
    for f in FLOORS:
        n_elig = int(
            ((merged["tokens_worker"] >= f) & (merged["tokens_client"] >= f)).sum()
        )
        sweep_rows.append(
            {
                "allow_both_split": allow_both_split,
                "min_tokens": f,
                "n_enumerated": len(merged),
                "n_eligible": n_elig,
            }
        )
d4 = pd.DataFrame(sweep_rows)
d4.to_csv(OUT / "diag_floor_sweep.csv", index=False)
print("\n[D.4] Floor sensitivity sweep")
print(d4.to_string(index=False))


# ---------------------------------------------------------------------
# Figure D.1 — ECDF of min(tokens_worker, tokens_client) across pairs
# ---------------------------------------------------------------------
merged_strict = enumerate_pairs(d3, allow_both_split=False)
merged_split = enumerate_pairs(d3, allow_both_split=True)

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for label, m, style in [
    ("strict worker/client", merged_strict, "-"),
    ("with 'both'-split", merged_split, "--"),
]:
    if len(m) == 0:
        continue
    x = np.sort(
        m[["tokens_worker", "tokens_client"]].min(axis=1).values
    )
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", linestyle=style, label=f"{label} (n={len(m)})")

for f in [1000, 2000, 3000]:
    ax.axvline(f, ls=":", color="black", lw=0.6, alpha=0.6)
    ax.text(f, 0.02, f"{f}", rotation=90, fontsize=8, ha="right", va="bottom")

ax.set_xscale("log")
ax.set_xlabel("min(tokens_worker, tokens_client) [log scale]")
ax.set_ylabel("ECDF across enumerated pairs")
ax.set_title("Token-floor sensitivity across enumerated within-company pairs")
ax.grid(True, which="both", linestyle=":", alpha=0.4)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "diag_pair_token_ecdf.png", dpi=150)
print(f"\n[Figure D.1] written to {OUT / 'diag_pair_token_ecdf.png'}")


# ---------------------------------------------------------------------
# One-page narrative summary for quick reading
# ---------------------------------------------------------------------
strict_at_2000 = int(
    d4[(d4["min_tokens"] == 2000) & (~d4["allow_both_split"])]["n_eligible"].iloc[0]
)
strict_at_1000 = int(
    d4[(d4["min_tokens"] == 1000) & (~d4["allow_both_split"])]["n_eligible"].iloc[0]
)
split_at_2000 = int(
    d4[(d4["min_tokens"] == 2000) & (d4["allow_both_split"])]["n_eligible"].iloc[0]
)
split_at_1000 = int(
    d4[(d4["min_tokens"] == 1000) & (d4["allow_both_split"])]["n_eligible"].iloc[0]
)

summary = f"""# §4.4 sampling diagnostics — summary

## Audience coding (D.1)
{d1.to_string(index=False)}

## Companies with scraped data (D.2)
- total companies: {len(d2)}
- strict pairable (has worker AND client domain): {d2['pairable_strict'].sum()}
- pairable with 'both'-split fallback: {d2['pairable_with_both_split'].sum()}

## Eligibility at key floors (D.4)
|                    | strict | with 'both'-split |
|--------------------|-------:|------------------:|
| floor = 1000 tok   | {strict_at_1000:>6d} | {split_at_1000:>17d} |
| floor = 2000 tok   | {strict_at_2000:>6d} | {split_at_2000:>17d} |

## Reading
- If `strict @ 2000` is the only small number, the bottleneck is the
  token floor; dropping to 1000 typically recovers enough pool.
- If `strict @ 1000` is still small but `split @ 1000` is much larger,
  the bottleneck is audience coding (too many `'both'`-coded domains)
  and the fallback is worth implementing.
- If both columns stay small at every floor, the corpus simply does
  not support a strict 6-pair design; reframe §4.4 around the
  realisable N.
"""
(OUT / "diag_summary.md").write_text(summary)
print(f"\nsummary written to {OUT / 'diag_summary.md'}")

conn.close()
