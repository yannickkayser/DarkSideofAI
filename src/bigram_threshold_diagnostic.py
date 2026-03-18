"""
bigram_threshold_diagnostic.py
-------------------------------
Diagnostic: are bigrams unequally distributed across registers at the
count=2 threshold vs count=3+?

Reads from pages_embedding.tokenized_text (space-separated lemmas already
produced by preprocess.py) so no spaCy re-run is needed.  Regenerates ALL
bigrams from those tokens — including count=1 and count=2 which were
discarded by preprocess.py — then compares band statistics across registers.

Pipeline fit:
  Reads from : data/scraping.db → pages_embedding (tokenized_text, audience)
  Writes to  : stdout only (diagnostic, no DB changes)

Run from the project root:
    python3 src/bigram_threshold_diagnostic.py
"""

import sqlite3
import math
from collections import Counter
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH  = "data/scraping.db"   # same path as preprocess.py
SAMPLE_N = 20                   # count=2 bigrams to print as an eyeball check
# ─────────────────────────────────────────────────────────────────────────────


def load_pages(db_path: str) -> list[tuple[str, str]]:
    """Return (audience, tokenized_text) for every row in pages_embedding."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT audience, tokenized_text FROM pages_embedding "
        "WHERE audience IN ('client','worker') AND tokenized_text IS NOT NULL"
    ).fetchall()
    conn.close()
    return rows


def make_bigrams(tokenized_text: str) -> list[str]:
    """Re-generate bigrams from space-separated lemmas (same logic as preprocess.py)."""
    tokens = tokenized_text.split()
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def g2(o11: int, o12: int, total_client: int, total_worker: int) -> float:
    """
    Log-likelihood G² for a single bigram vs all other tokens (2×2 table).
    Returns a signed score: positive = client-distinctive, negative = worker-distinctive.
    """
    o21 = total_client - o11
    o22 = total_worker - o12
    N   = total_client + total_worker
    E11 = (o11 + o12) * (o11 + o21) / N
    E12 = (o11 + o12) * (o12 + o22) / N
    E21 = (o21 + o22) * (o11 + o21) / N
    E22 = (o21 + o22) * (o12 + o22) / N

    def cell(o: int, e: float) -> float:
        return o * math.log(o / e) if o > 0 and e > 0 else 0.0

    score = 2 * (cell(o11, E11) + cell(o12, E12) +
                 cell(o21, E21) + cell(o22, E22))
    # Sign: positive = client-skewed
    rel_client = o11 / max(total_client, 1)
    rel_worker = o12 / max(total_worker, 1)
    return score if rel_client >= rel_worker else -score


def band(n: int) -> str:
    if n == 1:  return "count = 1  (hapax)"
    if n == 2:  return "count = 2"
    if n <= 5:  return "count = 3–5"
    if n <= 10: return "count = 6–10"
    return               "count = 11+"


def median(values: list[float]) -> float:
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def main():
    print(f"Loading pages_embedding from {DB_PATH} …")
    if not Path(DB_PATH).exists():
        print(f"\n[ERROR] Database not found: {DB_PATH}")
        print("Run this script from the project root, or update DB_PATH.")
        return

    rows = load_pages(DB_PATH)
    print(f"  {len(rows):,} pages loaded.\n")

    # ── Count bigrams per register ────────────────────────────────────────────
    client_counts: Counter = Counter()
    worker_counts: Counter = Counter()

    for audience, text in rows:
        bigrams = make_bigrams(text)
        if audience == "client":
            client_counts.update(bigrams)
        else:
            worker_counts.update(bigrams)

    total_client = sum(client_counts.values())
    total_worker = sum(worker_counts.values())
    all_bigrams  = set(client_counts) | set(worker_counts)

    print(f"  Client bigram tokens : {total_client:,}")
    print(f"  Worker bigram tokens : {total_worker:,}")
    print(f"  Unique bigram types  : {len(all_bigrams):,}\n")

    # ── Organise by band ─────────────────────────────────────────────────────
    # Each entry: (bigram, client_count, worker_count, total, g2_score)
    bands: dict[str, list[tuple]] = {}
    for bg in all_bigrams:
        cc   = client_counts[bg]
        wc   = worker_counts[bg]
        tot  = cc + wc
        b    = band(tot)
        score = g2(cc, wc, total_client, total_worker)
        bands.setdefault(b, []).append((bg, cc, wc, tot, score))

    # ── Summary table ─────────────────────────────────────────────────────────
    order = ["count = 1  (hapax)", "count = 2", "count = 3–5",
             "count = 6–10", "count = 11+"]

    print("=" * 74)
    print(f"{'Band':<22} {'Types':>7} {'Client only':>12} {'Worker only':>12} "
          f"{'Med|G²|':>8} {'Max|G²|':>8}")
    print("=" * 74)
    for b in order:
        if b not in bands:
            continue
        entries = bands[b]
        n        = len(entries)
        c_only   = sum(1 for _, cc, wc, *_ in entries if wc == 0)
        w_only   = sum(1 for _, cc, wc, *_ in entries if cc == 0)
        abs_g2   = [abs(e[4]) for e in entries]
        med_g2   = median(abs_g2)
        max_g2   = max(abs_g2)
        print(f"{b:<22} {n:>7,} {c_only:>12,} {w_only:>12,} "
              f"{med_g2:>8.2f} {max_g2:>8.2f}")
    print("=" * 74)
    print("'Client/Worker only' = bigram appears in that register exclusively.")
    print("Med|G²| ≥ 3.84 ≈ p < .05;  higher = more register-distinctive.\n")

    # ── count=2 detail ───────────────────────────────────────────────────────
    b2 = bands.get("count = 2", [])
    if not b2:
        print("No count=2 bigrams found.")
        return

    n2      = len(b2)
    c_only  = [e for e in b2 if e[2] == 0]
    w_only  = [e for e in b2 if e[1] == 0]
    shared  = [e for e in b2 if e[1] > 0 and e[2] > 0]

    print(f"── count=2 band breakdown ───────────────────────────────────────────")
    print(f"  Total types            : {n2:,}")
    print(f"  Appear only in client  : {len(c_only):,}  ({100*len(c_only)/n2:.1f}%)")
    print(f"  Appear only in worker  : {len(w_only):,}  ({100*len(w_only)/n2:.1f}%)")
    print(f"  Appear in both         : {len(shared):,}  ({100*len(shared)/n2:.1f}%)")

    # Compare with the count=3-5 band
    b35 = bands.get("count = 3–5", [])
    if b35:
        n35    = len(b35)
        c35    = sum(1 for e in b35 if e[2] == 0)
        w35    = sum(1 for e in b35 if e[1] == 0)
        print(f"\n  For comparison — count=3–5 band ({n35:,} types):")
        print(f"    Appear only in client  : {c35:,}  ({100*c35/n35:.1f}%)")
        print(f"    Appear only in worker  : {w35:,}  ({100*w35/n35:.1f}%)")

    # ── Eyeball sample ────────────────────────────────────────────────────────
    print(f"\n── Top-{SAMPLE_N} count=2 bigrams by |G²| (check for noise vs signal) ──")
    print(f"{'Bigram':<35} {'Client':>7} {'Worker':>7} {'G²':>8}  Register")
    print("-" * 70)
    sample = sorted(b2, key=lambda e: abs(e[4]), reverse=True)[:SAMPLE_N]
    for bg, cc, wc, tot, score in sample:
        direction = "→ client" if score > 0 else "→ worker"
        print(f"{bg:<35} {cc:>7} {wc:>7} {abs(score):>8.2f}  {direction}")

    # ── Threshold recommendation ──────────────────────────────────────────────
    excl_pct = 100 * (len(c_only) + len(w_only)) / n2
    med2     = median([abs(e[4]) for e in b2])

    print(f"\n── Recommendation ───────────────────────────────────────────────────")
    if excl_pct > 60:
        print(f"  ⚠  {excl_pct:.0f}% of count=2 bigrams appear in only one register.")
        print("     This is characteristic of noise, not stable collocations.")
        print("     Keeping MIN_BIGRAM_FREQ = 3 is well-justified.")
    elif med2 < 3.84:
        print(f"  ⚠  Median G² for count=2 bigrams is {med2:.2f} (below p<.05 threshold).")
        print("     Most carry no reliable keyness signal. Threshold=3 recommended.")
    else:
        print(f"  ✓  count=2 bigrams show G² median of {med2:.2f} — not obviously noisy.")
        print("     Consider running Step 1 at both thresholds and comparing top-50 lists.")
        print("     Pay particular attention to whether new worker-register bigrams appear.")
    print()


if __name__ == "__main__":
    main()
