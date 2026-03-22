"""
query_sample_overview.py
-------------------------
Produces a structured overview of the Step 2 sample:

  Section 1 — Summary by hypothesis
      n pages, domains covered, topics covered, mean PCA position

  Section 2 — Topic coverage
      Which topics appear in the sample, how many times, audience profile,
      top terms — plus what % of total corpus pages for that topic are sampled

  Section 3 — Domain coverage
      Pages per domain per hypothesis, % of that domain's total corpus pages

  Section 4 — Page-level listing
      Full listing sorted by hypothesis → priority rank

Run from project root:
    python query_sample_overview.py
    python query_sample_overview.py --db path/to/scraping.db
"""

import sqlite3
import argparse
from collections import defaultdict

DEFAULT_DB = "data/scraping.db"
TOP_TERMS  = 5   # terms to show per topic in overview

# ── helpers ───────────────────────────────────────────────────────────────────

def pct(n, total):
    return f"{n/total*100:.1f}%" if total else "—"

def bar(v, total, w=15):
    filled = int(round(v / total * w)) if total else 0
    return "█" * filled + "░" * (w - filled)

def hline(w=78):
    print("─" * w)

def section(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)

# ── main ──────────────────────────────────────────────────────────────────────

def main(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    # ── Load sample ──────────────────────────────────────────────────────────
    cur.execute("""
        SELECT  s.page_id, s.domain, s.audience, s.sampling_reason,
                s.priority_rank, s.collocate_divergence,
                s.topic_weight,
                dt.dominant_topic, dt.pca_1, dt.pca_2
        FROM    step2_sample s
        LEFT JOIN document_topics dt ON dt.page_id = s.page_id
        ORDER   BY s.sampling_reason, s.priority_rank
    """)
    sample = [dict(r) for r in cur.fetchall()]

    # Extract hypothesis from sampling_reason
    for r in sample:
        reason = r["sampling_reason"] or ""
        hyp    = reason.split("_topic_")[0] if "_topic_" in reason else reason.split("_")[0]
        for key in ("H1a", "H1b", "H1c"):
            if hyp.startswith(key):
                hyp = key
                break
        r["hyp"] = hyp

    # ── Load topic terms ──────────────────────────────────────────────────────
    cur.execute("""
        SELECT topic_id, term, rank
        FROM   topic_terms
        WHERE  rank <= ?
        ORDER  BY topic_id, rank
    """, (TOP_TERMS,))
    terms_by_topic = defaultdict(list)
    for r in cur.fetchall():
        terms_by_topic[r["topic_id"]].append(r["term"])

    # ── Load corpus-wide topic counts ─────────────────────────────────────────
    cur.execute("""
        SELECT  dominant_topic,
                COUNT(*) AS n_total,
                AVG(CASE WHEN audience = 'client' THEN 1.0 ELSE 0.0 END) AS client_share
        FROM    document_topics
        WHERE   pca_1 IS NOT NULL
        GROUP   BY dominant_topic
    """)
    corpus_topic = {r["dominant_topic"]: dict(r) for r in cur.fetchall()}

    # ── Load corpus-wide domain counts ────────────────────────────────────────
    cur.execute("""
        SELECT domain, COUNT(*) AS n_total
        FROM   document_topics
        WHERE  pca_1 IS NOT NULL
        GROUP  BY domain
    """)
    corpus_domain = {r["domain"]: r["n_total"] for r in cur.fetchall()}

    total_corpus = sum(corpus_domain.values())
    total_sample = len(sample)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Summary by hypothesis
    # ═══════════════════════════════════════════════════════════════════════════
    section("SECTION 1 — SUMMARY BY HYPOTHESIS")

    by_hyp = defaultdict(list)
    for r in sample:
        by_hyp[r["hyp"]].append(r)

    print(f"\n  Total corpus pages : {total_corpus:,}")
    print(f"  Total sample pages : {total_sample}  ({pct(total_sample, total_corpus)} of corpus)")
    print()

    header = f"  {'Hyp':<6} {'N':>4}  {'% corpus':>9}  {'Domains':>7}  {'Topics':>7}  {'Mean PC1':>9}  {'Mean PC2':>9}"
    print(header)
    hline()
    for hyp in ("H1a", "H1b", "H1c"):
        rows   = by_hyp.get(hyp, [])
        n      = len(rows)
        doms   = len({r["domain"] for r in rows})
        tops   = len({r["dominant_topic"] for r in rows if r["dominant_topic"] is not None})
        pc1s   = [r["pca_1"] for r in rows if r["pca_1"] is not None]
        pc2s   = [r["pca_2"] for r in rows if r["pca_2"] is not None]
        mpc1   = f"{sum(pc1s)/len(pc1s):.4f}" if pc1s else "—"
        mpc2   = f"{sum(pc2s)/len(pc2s):.4f}" if pc2s else "—"
        print(f"  {hyp:<6} {n:>4}  {pct(n, total_corpus):>9}  {doms:>7}  {tops:>7}  {mpc1:>9}  {mpc2:>9}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Topic coverage
    # ═══════════════════════════════════════════════════════════════════════════
    section("SECTION 2 — TOPIC COVERAGE IN SAMPLE")

    # Count sampled pages per topic per hypothesis
    topic_hyp_counts = defaultdict(lambda: defaultdict(int))
    for r in sample:
        tid = r["dominant_topic"]
        topic_hyp_counts[tid][r["hyp"]] += 1

    topic_total = {tid: sum(vc.values()) for tid, vc in topic_hyp_counts.items()}
    sorted_topics = sorted(topic_total, key=lambda t: -topic_total[t])

    print()
    print(f"  {'Topic':<8} {'Terms':<35} {'Smpl':>5} {'Corp':>6} {'%Corp':>7} {'H1a':>5} {'H1b':>5} {'H1c':>5} {'Aud':>6}")
    hline()
    for tid in sorted_topics:
        if tid is None:
            continue
        terms  = ", ".join(terms_by_topic.get(tid, [f"T{tid}"]))[:34]
        n_samp = topic_total[tid]
        corp   = corpus_topic.get(tid, {}).get("n_total", 0)
        cshare = corpus_topic.get(tid, {}).get("client_share", 0)
        aud    = "client" if cshare >= 0.65 else ("worker" if cshare <= 0.35 else "mixed")
        h1a    = topic_hyp_counts[tid].get("H1a", 0)
        h1b    = topic_hyp_counts[tid].get("H1b", 0)
        h1c    = topic_hyp_counts[tid].get("H1c", 0)
        print(f"  T{str(tid):<7} {terms:<35} {n_samp:>5} {corp:>6} {pct(n_samp,corp):>7} "
              f"{h1a:>5} {h1b:>5} {h1c:>5} {aud:>6}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Domain coverage
    # ═══════════════════════════════════════════════════════════════════════════
    section("SECTION 3 — DOMAIN COVERAGE IN SAMPLE")

    domain_hyp = defaultdict(lambda: defaultdict(int))
    for r in sample:
        domain_hyp[r["domain"]][r["hyp"]] += 1

    domain_total_samp = {d: sum(v.values()) for d, v in domain_hyp.items()}
    sorted_domains = sorted(domain_total_samp, key=lambda d: -domain_total_samp[d])

    print()
    print(f"  {'Domain':<35} {'Aud':<7} {'Smpl':>5} {'Corp':>6} {'%Corp':>7} {'H1a':>5} {'H1b':>5} {'H1c':>5}")
    hline()

    # Resolve audience per domain from sample
    domain_aud = {}
    for r in sample:
        domain_aud[r["domain"]] = r["audience"] or "?"

    for dom in sorted_domains:
        n_samp = domain_total_samp[dom]
        corp   = corpus_domain.get(dom, 0)
        aud    = domain_aud.get(dom, "?")
        h1a    = domain_hyp[dom].get("H1a", 0)
        h1b    = domain_hyp[dom].get("H1b", 0)
        h1c    = domain_hyp[dom].get("H1c", 0)
        print(f"  {dom:<35} {aud:<7} {n_samp:>5} {corp:>6} {pct(n_samp,corp):>7} "
              f"{h1a:>5} {h1b:>5} {h1c:>5}")

    # Coverage bar
    print()
    print(f"  Domain coverage: {len(sorted_domains)} of {len(corpus_domain)} domains sampled "
          f"({pct(len(sorted_domains), len(corpus_domain))})")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Page-level listing
    # ═══════════════════════════════════════════════════════════════════════════
    section("SECTION 4 — FULL PAGE-LEVEL LISTING")

    for hyp in ("H1a", "H1b", "H1c"):
        rows = sorted(by_hyp.get(hyp, []), key=lambda r: r["priority_rank"] or 999)
        print(f"\n  {hyp}  ({len(rows)} pages)")
        hline()
        print(f"  {'Rank':>4}  {'Domain':<32} {'Topic':>6}  {'Wt':>5}  {'PC1':>8}  {'PC2':>8}  {'Div':>6}")
        for r in rows:
            tid   = r["dominant_topic"]
            twt   = f"{r['topic_weight']:.3f}" if r["topic_weight"] else "—"
            pc1   = f"{r['pca_1']:.4f}"        if r["pca_1"]       else "—"
            pc2   = f"{r['pca_2']:.4f}"        if r["pca_2"]       else "—"
            div   = f"{r['collocate_divergence']:.3f}" if r["collocate_divergence"] else "—"
            rank  = r["priority_rank"] or "—"
            print(f"  {str(rank):>4}  {r['domain']:<32} T{str(tid):<5}  {twt:>5}  {pc1:>8}  {pc2:>8}  {div:>6}")

    conn.close()
    print("\n\nDone.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    args = parser.parse_args()
    main(args.db)
