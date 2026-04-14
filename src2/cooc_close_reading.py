"""
Co-occurrence close-reading diagnostic.

For a given (term1, term2) pair, find every page in which both terms occur,
extract KWIC windows around each hit, group by audience and domain, and
write a markdown report that can be quoted directly in Chapter 4.

Usage (from the project root):
    python cooc_close_reading.py --db scraping_2.db \
        --pair human trafficking \
        --pair worker migrant \
        --pair work baby \
        --out close_readings/

Each pair produces close_readings/cooc_<t1>_<t2>.md with:
    - summary counts by audience and domain
    - one block per page with URL, audience, dominant topic, and KWIC lines
    - a trailing "interpretive note" placeholder for the thesis author to fill
"""

import argparse
import re
import sqlite3
from collections import Counter
from pathlib import Path

WINDOW = 12  # tokens either side of the keyword


def kwic_lines(tokens, target, window=WINDOW, max_hits=4):
    hits = []
    for i, tok in enumerate(tokens):
        if tok == target:
            left = " ".join(tokens[max(0, i - window):i])
            right = " ".join(tokens[i + 1:i + 1 + window])
            hits.append(f"...{left} **>>>{target}<<<** {right}...")
            if len(hits) >= max_hits:
                break
    return hits


def fetch_pages_with_both(conn, t1, t2):
    """Pages where both terms appear in the lemmatised token stream."""
    q = """
    SELECT p.page_id, p.domain, p.url, p.audience, p.tokens,
           COALESCE(t.dominant_topic, '') AS dominant_topic,
           COALESCE(t.dominant_weight, 0.0) AS dominant_weight
    FROM corpus_view p
    LEFT JOIN stm_document_topics t ON p.page_id = t.page_id
    WHERE p.tokens LIKE ? AND p.tokens LIKE ?
    """
    rows = conn.execute(q, (f"%{t1}%", f"%{t2}%")).fetchall()
    # Filter to pages where both are whole tokens (LIKE can hit substrings).
    keep = []
    for r in rows:
        toks = r[4].split()
        if t1 in toks and t2 in toks:
            keep.append(r)
    return keep


def build_report(conn, t1, t2, out_dir: Path):
    pages = fetch_pages_with_both(conn, t1, t2)
    aud_counts = Counter(r[3] for r in pages)
    dom_counts = Counter(r[1] for r in pages)

    lines = []
    lines.append(f"# Co-occurrence close reading: `{t1}` + `{t2}`")
    lines.append("")
    lines.append(f"- **Pages with both terms:** {len(pages)}")
    lines.append(f"- **Audience breakdown:** {dict(aud_counts)}")
    lines.append(f"- **Top domains:** {dom_counts.most_common(5)}")
    lines.append("")
    lines.append("## Per-page passages")
    lines.append("")

    for i, (pid, domain, url, audience, tokens, topic, weight) in enumerate(pages, 1):
        toks = tokens.split()
        k1 = kwic_lines(toks, t1)
        k2 = kwic_lines(toks, t2)
        lines.append(f"### [{i}] `{domain}` — {audience}")
        lines.append(f"- page_id: {pid}")
        lines.append(f"- url: {url}")
        if topic:
            lines.append(f"- dominant topic: {topic} (w={weight:.2f})")
        lines.append("")
        lines.append(f"**KWIC for `{t1}`:**")
        for h in k1:
            lines.append(f"> {h}")
        lines.append("")
        lines.append(f"**KWIC for `{t2}`:**")
        for h in k2:
            lines.append(f"> {h}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Interpretive note (to fill)")
    lines.append("")
    lines.append("- Dominant frame(s):")
    lines.append("- Who occupies the subject position of each keyword?")
    lines.append("- Link to hypothesis (H1a / H1b / H1c):")
    lines.append("- Representative short quote for chapter 4 (<=15 words):")
    lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"cooc_{t1}_{t2}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out, len(pages)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to scraping_2.db")
    ap.add_argument("--pair", action="append", nargs=2, metavar=("T1", "T2"),
                    required=True, help="A term pair. Repeat for more pairs.")
    ap.add_argument("--out", default="close_readings", help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()
    conn = sqlite3.connect(args.db)
    out_dir = Path(args.out)
    for t1, t2 in args.pair:
        path, n = build_report(conn, t1, t2, out_dir)
        print(f"[ok] {t1} + {t2}: {n} pages -> {path}")
    conn.close()


if __name__ == "__main__":
    main()
