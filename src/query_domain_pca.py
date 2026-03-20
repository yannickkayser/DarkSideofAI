"""
query_domain_pca.py
-------------------
Investigates domain-level distribution within the PCA document space.

Answers:
  1. Which domains dominate each audience cluster (B2B / B2W)?
  2. Are domains internally consistent (all pages in same region) or mixed?
  3. Which domains appear in the theoretically charged H1a / H1b / H1c zones?
  4. How many pages per domain fall into each Step 2 sample hypothesis region?

Run from project root:
    python query_domain_pca.py
or specify DB path:
    python query_domain_pca.py --db path/to/scraping.db
"""

import sqlite3
import argparse
import statistics
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DB = "data/scraping.db"

# PCA thresholds — adjust to match your actual data range.
# These divide the PC1 axis into three zones:
#   PC1 < B2W_THRESHOLD  → B2W cluster (H1a zone)
#   PC1 > B2B_THRESHOLD  → B2B cluster (H1b / H1c zone)
#   between              → contested middle
B2W_THRESHOLD = -0.5   # pages strongly B2W
B2B_THRESHOLD =  0.5   # pages strongly B2B

# Minimum pages a domain must have (after PCA filter) to appear in summary
MIN_PAGES = 3

# ── Helpers ───────────────────────────────────────────────────────────────────

def zone(pca1):
    if pca1 is None:
        return "excluded"
    if pca1 < B2W_THRESHOLD:
        return "B2W"
    if pca1 > B2B_THRESHOLD:
        return "B2B"
    return "middle"

def bar(value, total, width=20):
    filled = int(round(value / total * width)) if total else 0
    return "█" * filled + "░" * (width - filled)

# ── Main ──────────────────────────────────────────────────────────────────────

def main(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ── 1. Pull all document_topics rows with PCA coords ──────────────────
    cur.execute("""
        SELECT domain, audience, pca_1, pca_2, dominant_topic
        FROM   document_topics
        ORDER  BY domain, pca_1
    """)
    rows = cur.fetchall()
    print(f"\nTotal documents in document_topics: {len(rows)}")

    # ── 2. Aggregate by domain ────────────────────────────────────────────
    domain_data = defaultdict(lambda: {"pages": [], "audience": None})
    for r in rows:
        d = r["domain"]
        domain_data[d]["pages"].append({
            "pca_1":    r["pca_1"],
            "pca_2":    r["pca_2"],
            "audience": r["audience"],
            "topic":    r["dominant_topic"],
            "zone":     zone(r["pca_1"]),
        })
        domain_data[d]["audience"] = r["audience"]  # last seen (should be uniform per domain)

    # ── 3. Domain consistency analysis ───────────────────────────────────
    print("\n" + "="*70)
    print("DOMAIN CONSISTENCY IN PCA SPACE")
    print("(How coherent is each domain's topic profile?)")
    print("="*70)

    consistent_domains   = []
    mixed_domains        = []

    for domain, data in sorted(domain_data.items()):
        pages = data["pages"]
        if len(pages) < MIN_PAGES:
            continue
        valid = [p for p in pages if p["pca_1"] is not None]
        if not valid:
            continue

        zones = [p["zone"] for p in valid]
        zone_counts = {z: zones.count(z) for z in set(zones)}
        dominant_zone = max(zone_counts, key=zone_counts.get)
        dominant_frac = zone_counts[dominant_zone] / len(valid)

        pca1_vals = [p["pca_1"] for p in valid]
        pca1_mean = statistics.mean(pca1_vals)
        pca1_std  = statistics.stdev(pca1_vals) if len(pca1_vals) > 1 else 0.0

        audience = data["audience"] or "?"

        entry = {
            "domain":         domain,
            "audience":       audience,
            "n_pages":        len(valid),
            "dominant_zone":  dominant_zone,
            "dominant_frac":  dominant_frac,
            "pca1_mean":      pca1_mean,
            "pca1_std":       pca1_std,
            "zone_counts":    zone_counts,
        }

        if dominant_frac >= 0.80:
            consistent_domains.append(entry)
        else:
            mixed_domains.append(entry)

    # Print consistent domains
    print(f"\n{'─'*70}")
    print(f"CONSISTENT DOMAINS  (≥80% pages in same PCA zone)  n={len(consistent_domains)}")
    print(f"{'─'*70}")
    print(f"{'Domain':<35} {'Aud':>4} {'N':>4} {'Zone':>6} {'PCA1 mean':>10} {'std':>6}")
    for e in sorted(consistent_domains, key=lambda x: x["pca1_mean"]):
        print(f"{e['domain']:<35} {e['audience']:>4} {e['n_pages']:>4} "
              f"{e['dominant_zone']:>6} {e['pca1_mean']:>10.3f} {e['pca1_std']:>6.3f}")

    # Print mixed domains
    print(f"\n{'─'*70}")
    print(f"MIXED DOMAINS  (<80% pages in one zone)  n={len(mixed_domains)}")
    print(f"{'─'*70}")
    print(f"{'Domain':<35} {'Aud':>4} {'N':>4} {'B2W':>5} {'mid':>5} {'B2B':>5} {'PCA1 mean':>10} {'std':>6}")
    for e in sorted(mixed_domains, key=lambda x: x["pca1_mean"]):
        b2w = e["zone_counts"].get("B2W", 0)
        mid = e["zone_counts"].get("middle", 0)
        b2b = e["zone_counts"].get("B2B", 0)
        print(f"{e['domain']:<35} {e['audience']:>4} {e['n_pages']:>4} "
              f"{b2w:>5} {mid:>5} {b2b:>5} "
              f"{e['pca1_mean']:>10.3f} {e['pca1_std']:>6.3f}")

    # ── 4. Zone-level domain ranking ─────────────────────────────────────
    print("\n" + "="*70)
    print("DOMAIN DISTRIBUTION BY PCA ZONE")
    print("="*70)

    zone_domains = defaultdict(lambda: defaultdict(int))
    for domain, data in domain_data.items():
        for p in data["pages"]:
            zone_domains[p["zone"]][domain] += 1

    for z in ["B2W", "middle", "B2B"]:
        counts = zone_domains[z]
        total  = sum(counts.values())
        print(f"\n  Zone: {z}  ({total} pages total)")
        print(f"  {'Domain':<35} {'Pages':>6}  {'Share':>6}  {'Bar'}")
        for dom, n in sorted(counts.items(), key=lambda x: -x[1])[:15]:
            share = n / total * 100
            print(f"  {dom:<35} {n:>6}  {share:>5.1f}%  {bar(n, total)}")

    # ── 5. Step 2 sample — domain breakdown ──────────────────────────────
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='step2_sample'")
    if cur.fetchone():
        print("\n" + "="*70)
        print("STEP 2 SAMPLE — DOMAIN BREAKDOWN BY HYPOTHESIS")
        print("="*70)
        # hypothesis is encoded in sampling_reason as prefix: "H1a_topic_11_..."
        cur.execute("""
            SELECT s.domain, s.audience, s.sampling_reason, s.priority_rank,
                   dt.pca_1, dt.pca_2
            FROM   step2_sample s
            LEFT JOIN document_topics dt ON dt.page_id = s.page_id
            ORDER  BY s.sampling_reason, s.domain
        """)
        sample_rows = cur.fetchall()
        hyp_domains = defaultdict(lambda: defaultdict(int))
        for r in sample_rows:
            # Extract hypothesis prefix: "H1a_topic_11_..." → "H1a"
            reason = r["sampling_reason"] or "unknown"
            hyp = reason.split("_topic_")[0] if "_topic_" in reason else reason.split("_")[0]
            hyp_domains[hyp][r["domain"]] += 1

        for hyp, domains in sorted(hyp_domains.items()):
            total = sum(domains.values())
            print(f"\n  Hypothesis: {hyp}  ({total} pages)")
            for dom, n in sorted(domains.items(), key=lambda x: -x[1]):
                print(f"    {dom:<40} {n:>3} page(s)")
    else:
        print("\n  (step2_sample table not found — run 02c_step1_topics.py first)")

    # ── 6. Within-domain variation summary ───────────────────────────────
    print("\n" + "="*70)
    print("WITHIN-DOMAIN VARIATION SUMMARY")
    print("="*70)
    all_entries = consistent_domains + mixed_domains
    if all_entries:
        stds = [e["pca1_std"] for e in all_entries]
        print(f"\n  Domains analysed:          {len(all_entries)}")
        print(f"  Median within-domain std:  {statistics.median(stds):.3f}")
        print(f"  Max within-domain std:     {max(stds):.3f}  ({all_entries[stds.index(max(stds))]['domain']})")
        print(f"  Min within-domain std:     {min(stds):.3f}  ({all_entries[stds.index(min(stds))]['domain']})")
        print(f"\n  Interpretation:")
        print(f"  Low std  → domain is register-consistent (all pages similar tone)")
        print(f"  High std → domain spans multiple registers (mixed-audience platform)")

    conn.close()
    print("\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to scraping.db")
    args = parser.parse_args()
    main(args.db)
