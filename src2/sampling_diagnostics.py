"""
sampling_diagnostics.py
=======================

Diagnostic script for §4.4 sampling protocol. Runs BEFORE the pair-
selection pipeline and explains why the eligible pool is what it is.

This version does NOT import numpy directly. All array work is done
through pandas (which uses numpy internally but handles its own ABI
compatibility). matplotlib receives plain Python lists, which it
accepts regardless of numpy version.

Compatibility
-------------
    pandas    >= 1.5   (any version that supports .merge / .groupby)
    matplotlib>= 3.5   (any version that accepts Python list inputs)
    numpy     any      (not imported directly)

Usage
-----
    python sampling_diagnostics.py /absolute/path/to/scraping_2.db

Outputs (in ./outputs by default, or set OUT env var)
------------------------------------------------------
    diag_audience_coding.csv      Table D.1
    diag_company_pairability.csv  Table D.2
    diag_domain_tokens.csv        Table D.3
    diag_floor_sweep.csv          Table D.4
    diag_pair_token_ecdf.png      Figure D.1
    diag_summary.md               One-page narrative summary
"""

from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless-safe; must come before pyplot
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DB_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scraping_2.db")
OUT = Path(os.environ.get("OUT", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)

FLOORS: list[int] = [250, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]

if not DB_PATH.exists():
    sys.exit(f"scraping database not found at {DB_PATH}")

print(f"pandas={pd.__version__}  matplotlib={matplotlib.__version__}")
try:
    import numpy as np
    print(f"numpy={np.__version__} (present but not used directly)")
    del np
except Exception:
    print("numpy not importable — that is fine, this script does not use it directly")

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
for col in ("n_worker", "n_client", "n_both"):
    d2[col] = pd.to_numeric(d2[col], errors="coerce").fillna(0).astype(int)

d2["pairable_strict"] = (d2["n_worker"] >= 1) & (d2["n_client"] >= 1)
d2["pairable_with_both_split"] = (
    ((d2["n_worker"] + d2["n_both"]) >= 1)
    & ((d2["n_client"] + d2["n_both"]) >= 1)
    & ((d2["n_worker"] + d2["n_client"] + d2["n_both"]) >= 2)
)
d2 = d2.sort_values(
    ["pairable_strict", "pairable_with_both_split", "company_id"],
    ascending=[False, False, True],
).reset_index(drop=True)
d2.to_csv(OUT / "diag_company_pairability.csv", index=False)

n_strict = int(d2["pairable_strict"].sum())
n_split  = int(d2["pairable_with_both_split"].sum())
print(f"\n[D.2] Companies with scraped data: {len(d2)}")
print(f"  strict within-company pairable (worker AND client): {n_strict}")
print(f"  with 'both'-split fallback:                          {n_split}")
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
FROM     platforms    p
JOIN     websites     w  ON w.domain      = p.domain
LEFT JOIN pages       pg ON pg.website_id = w.id
LEFT JOIN pages_tfidf pt ON pt.page_id    = pg.id
GROUP BY p.company_id, p.domain, p.audience
ORDER BY p.company_id, p.audience;
"""
d3 = pd.read_sql(q3, conn)
d3["tokens"]  = pd.to_numeric(d3["tokens"],  errors="coerce").fillna(0).astype(int)
d3["n_pages"] = pd.to_numeric(d3["n_pages"], errors="coerce").fillna(0).astype(int)
d3.to_csv(OUT / "diag_domain_tokens.csv", index=False)

print(f"\n[D.3] Per-domain token counts (n={len(d3)})")
with pd.option_context("display.max_rows", None):
    print(d3.to_string(index=False))
print("\n  summary stats:")
print(d3[["tokens", "n_pages"]].describe().to_string())


# ---------------------------------------------------------------------
# D.4 — floor sensitivity sweep
# ---------------------------------------------------------------------
def enumerate_pairs(d3: pd.DataFrame, allow_both_split: bool) -> pd.DataFrame:
    if allow_both_split:
        worker = d3[d3["audience"].isin(["worker", "both"])].copy()
        client = d3[d3["audience"].isin(["client", "both"])].copy()
    else:
        worker = d3[d3["audience"] == "worker"].copy()
        client = d3[d3["audience"] == "client"].copy()

    if worker.empty or client.empty:
        return pd.DataFrame(
            columns=["company_id", "domain_worker", "domain_client",
                     "tokens_worker", "tokens_client"]
        )

    merged = worker.merge(client, on="company_id", suffixes=("_worker", "_client"))
    merged = merged[merged["domain_worker"] != merged["domain_client"]].reset_index(drop=True)
    return merged


sweep_rows: list[dict] = []
for allow_both_split in (False, True):
    merged = enumerate_pairs(d3, allow_both_split)
    for f in FLOORS:
        if merged.empty:
            n_elig = 0
        else:
            n_elig = int(
                ((merged["tokens_worker"] >= f) & (merged["tokens_client"] >= f)).sum()
            )
        sweep_rows.append({
            "allow_both_split": allow_both_split,
            "min_tokens":       f,
            "n_enumerated":     len(merged),
            "n_eligible":       n_elig,
        })

d4 = pd.DataFrame(sweep_rows)
d4.to_csv(OUT / "diag_floor_sweep.csv", index=False)
print("\n[D.4] Floor sensitivity sweep")
print(d4.to_string(index=False))


# ---------------------------------------------------------------------
# Figure D.1 — ECDF of min(tokens_worker, tokens_client) across pairs
#
# All arrays are plain Python lists. matplotlib accepts lists directly;
# no numpy required.
# ---------------------------------------------------------------------
merged_strict = enumerate_pairs(d3, allow_both_split=False)
merged_split  = enumerate_pairs(d3, allow_both_split=True)

fig, ax = plt.subplots(figsize=(6.4, 4.0))

for label, m, style in (
    ("strict worker/client", merged_strict, "-"),
    ("with 'both'-split",    merged_split,  "--"),
):
    if len(m) == 0:
        continue
    # compute ECDF using sorted Python list — zero numpy
    min_side: list[float] = sorted(
        float(min(r["tokens_worker"], r["tokens_client"]))
        for _, r in m.iterrows()
    )
    n = len(min_side)
    ecdf_y: list[float] = [(i + 1) / n for i in range(n)]
    ax.step(min_side, ecdf_y, where="post", linestyle=style,
            label=f"{label} (n={n})")

for f in (1000, 2000, 3000):
    ax.axvline(f, ls=":", color="black", lw=0.6, alpha=0.6)
    ax.text(f, 0.02, str(f), rotation=90, fontsize=8, ha="right", va="bottom")

ax.set_xscale("log")
ax.set_xlabel("min(tokens_worker, tokens_client) [log scale]")
ax.set_ylabel("ECDF across enumerated pairs")
ax.set_title("Token-floor sensitivity across enumerated within-company pairs")
ax.grid(True, which="both", linestyle=":", alpha=0.4)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "diag_pair_token_ecdf.png", dpi=150)
plt.close(fig)
print(f"\n[Figure D.1] written to {OUT / 'diag_pair_token_ecdf.png'}")


# ---------------------------------------------------------------------
# One-page narrative summary
# ---------------------------------------------------------------------
def _lookup(d4: pd.DataFrame, floor: int, allow_both: bool) -> int:
    row = d4[(d4["min_tokens"] == floor) & (d4["allow_both_split"] == allow_both)]
    return int(row["n_eligible"].iloc[0]) if not row.empty else 0


strict_at_1000 = _lookup(d4, 1000, False)
strict_at_2000 = _lookup(d4, 2000, False)
split_at_1000  = _lookup(d4, 1000, True)
split_at_2000  = _lookup(d4, 2000, True)

summary = f"""# §4.4 sampling diagnostics — summary

## Audience coding (D.1)

{d1.to_string(index=False)}

## Companies with scraped data (D.2)
- total companies: {len(d2)}
- strict pairable (has worker AND client domain): {n_strict}
- pairable with 'both'-split fallback: {n_split}

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
(OUT / "diag_summary.md").write_text(summary, encoding="utf-8")
print(f"\nsummary written to {OUT / 'diag_summary.md'}")

conn.close()
