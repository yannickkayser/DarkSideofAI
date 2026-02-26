"""
find_duplicates.py
------------------
Detects near-duplicate pages in the scraped database using
MinHash + Locality Sensitive Hashing (LSH).

Strategy:
  - Represent each page as a set of character 5-grams (shingles)
  - Use MinHash to create a compact signature per page
  - Use LSH to efficiently find candidate pairs with similarity ≥ THRESHOLD
  - Report duplicates grouped by cluster with their URLs for manual review
  - Writes results to: duplicate_report.txt  and  duplicates.json

Usage:
    python find_duplicates.py [database.db] [output_prefix]

Defaults:
    database     → data/scraping.db  (relative to project root)
    output_prefix → duplicate_report

Install:
    pip install datasketch
"""

import sqlite3
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

from datasketch import MinHash, MinHashLSH

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DB   = str(Path(__file__).parent.parent / "data" / "scraping.db")
THRESHOLD    = 0.95      # Jaccard similarity threshold — 95% = very strict
NUM_PERM     = 128       # MinHash permutations — higher = more accurate, slower
SHINGLE_SIZE = 5         # character n-gram size for shingling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shingling
# ---------------------------------------------------------------------------

def shingle(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """
    Convert text to a set of character k-grams (shingles).
    Character-level shingling is more robust than word-level for
    detecting near-duplicates with minor wording changes.
    """
    text = " ".join(text.lower().split())   # normalise whitespace
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}


def make_minhash(shingles: set[str], num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf8"))
    return m


# ---------------------------------------------------------------------------
# Load pages
# ---------------------------------------------------------------------------

def load_pages(db_path: str) -> list[dict]:
    """Load all pages with non-empty text content from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.url, p.text_content,
               COALESCE(w.domain, 'unknown') AS domain
        FROM   pages p
        LEFT   JOIN websites w ON w.id = p.website_id
        WHERE  p.text_content IS NOT NULL
          AND  LENGTH(p.text_content) > 100
        ORDER  BY p.id
    """)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    log.info(f"Loaded {len(rows):,} pages from database.")
    return rows


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

def find_near_duplicates(pages: list[dict]) -> list[list[dict]]:
    """
    Build MinHash signatures and use LSH to find near-duplicate clusters.
    Returns a list of clusters — each cluster is a list of duplicate pages.
    """
    log.info(f"Building MinHash signatures (num_perm={NUM_PERM}, shingle_size={SHINGLE_SIZE})...")
    lsh       = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    minhashes = {}

    for i, page in enumerate(pages):
        key     = str(page["id"])
        shingles = shingle(page["text_content"])

        if len(shingles) < 5:
            continue   # skip trivially short pages

        mh = make_minhash(shingles)
        minhashes[key] = mh

        try:
            lsh.insert(key, mh)
        except ValueError:
            pass   # duplicate key — already inserted

        if (i + 1) % 500 == 0:
            log.info(f"  Processed {i+1:,}/{len(pages):,} pages...")

    log.info("Querying LSH for near-duplicate candidates...")

    # Build adjacency: page_id → set of near-duplicate page_ids
    adjacency: dict[str, set[str]] = defaultdict(set)
    id_to_page = {str(p["id"]): p for p in pages}

    for key, mh in minhashes.items():
        candidates = lsh.query(mh)
        for candidate in candidates:
            if candidate != key:
                adjacency[key].add(candidate)
                adjacency[candidate].add(key)

    # Union-Find to cluster connected components
    parent = {k: k for k in minhashes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for key, neighbours in adjacency.items():
        for nb in neighbours:
            union(key, nb)

    # Group into clusters — only keep clusters with 2+ pages
    clusters_map: dict[str, list] = defaultdict(list)
    for key in minhashes:
        root = find(key)
        if key in adjacency:   # only include pages that have at least one match
            clusters_map[root].append(id_to_page[key])

    clusters = [c for c in clusters_map.values() if len(c) >= 2]
    clusters.sort(key=len, reverse=True)   # largest clusters first

    log.info(f"Found {len(clusters)} near-duplicate clusters "
             f"({sum(len(c) for c in clusters)} pages involved).")
    return clusters


# ---------------------------------------------------------------------------
# Estimate Jaccard similarity between two pages
# ---------------------------------------------------------------------------

def jaccard_estimate(text_a: str, text_b: str) -> float:
    s_a = shingle(text_a)
    s_b = shingle(text_b)
    if not s_a or not s_b:
        return 0.0
    mh_a = make_minhash(s_a)
    mh_b = make_minhash(s_b)
    return mh_a.jaccard(mh_b)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(clusters: list[list[dict]], out_prefix: str):
    """Write human-readable text report and machine-readable JSON."""

    txt_path  = Path(out_prefix + ".txt")
    json_path = Path(out_prefix + ".json")

    total_pages    = sum(len(c) for c in clusters)
    total_clusters = len(clusters)

    # ── Text report ──
    lines = []
    lines.append("=" * 72)
    lines.append("  NEAR-DUPLICATE PAGE DETECTION REPORT")
    lines.append(f"  Similarity threshold : {THRESHOLD * 100:.0f}%")
    lines.append(f"  Total clusters found : {total_clusters:,}")
    lines.append(f"  Total pages flagged  : {total_pages:,}")
    lines.append("=" * 72)
    lines.append("")
    lines.append("  HOW TO USE THIS REPORT")
    lines.append("  ─" * 36)
    lines.append("  Each cluster below contains pages whose text is ≥95% identical.")
    lines.append("  Review each cluster and decide which page(s) to keep.")
    lines.append("  To exclude a page from preprocessing, note its page_id and add")
    lines.append("  it to an exclusion list before re-running preprocess.py.")
    lines.append("")

    for i, cluster in enumerate(clusters, 1):
        lines.append(f"  CLUSTER {i}  ({len(cluster)} pages)")
        lines.append("  " + "-" * 68)

        # Show pairwise similarity for the first pair as a reference
        if len(cluster) >= 2:
            sim = jaccard_estimate(cluster[0]["text_content"], cluster[1]["text_content"])
            lines.append(f"  Estimated similarity (pages 1–2): {sim*100:.1f}%")

        for j, page in enumerate(cluster, 1):
            snippet = " ".join(page["text_content"].split()[:20]) + "..."
            lines.append(f"  [{j}] page_id={page['id']}  domain={page['domain']}")
            lines.append(f"      URL     : {page['url']}")
            lines.append(f"      Preview : {snippet}")
        lines.append("")

    lines.append("=" * 72)
    lines.append(f"  END OF REPORT — {total_clusters} clusters, {total_pages} pages flagged")
    lines.append("=" * 72)

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Text report saved → {txt_path}")

    # ── JSON export (machine-readable for downstream filtering) ──
    json_data = []
    for i, cluster in enumerate(clusters, 1):
        json_data.append({
            "cluster_id": i,
            "size": len(cluster),
            "pages": [
                {"page_id": p["id"], "url": p["url"], "domain": p["domain"]}
                for p in cluster
            ],
        })

    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    log.info(f"JSON export saved  → {json_path}")
    log.info("Review the text report, then decide which page_ids to exclude.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db_path    = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "duplicate_report"

    if not Path(db_path).exists():
        log.error(f"Database not found: {db_path}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("NEAR-DUPLICATE DETECTION")
    log.info(f"  Database  : {db_path}")
    log.info(f"  Threshold : {THRESHOLD * 100:.0f}% Jaccard similarity")
    log.info(f"  Output    : {out_prefix}.txt / .json")
    log.info("=" * 60)

    pages    = load_pages(db_path)
    clusters = find_near_duplicates(pages)

    if not clusters:
        log.info("No near-duplicates found at the current threshold.")
        return

    write_report(clusters, out_prefix)


if __name__ == "__main__":
    main()
