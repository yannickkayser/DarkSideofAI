"""
generate_reading_templates.py
=============================

Generates one markdown close-reading template per page in
step2_sample_structured, matching the format of the user's existing
coded files (s1_H1a_client_labelbox_com_p9802.md etc.).

Each template contains:
  - Metadata block (page_id, url, domain, audience, case, STM topic)
  - KWIC lines for all three hypotheses' focal terms
  - Interpretive-note questions for ALL THREE hypotheses (not just
    the one the page was originally sampled for)
  - Frame-label and cross-reference fields

Pages are written in reading-manifest order (Section A company pairs
first, then Section B cross-platform).

Usage
-----
    python generate_reading_templates.py /path/to/scraping_2.db [--outdir DIR]

    --outdir DIR   Where to write the .md files (default: outputs/reading_templates)

Requires step2_sample_structured table (created by sample_restructure.py).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


# ─── CLI ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("db", nargs="?", default="scraping_2.db")
parser.add_argument("--outdir", type=str, default="outputs/reading_templates")
args = parser.parse_args()

DB_PATH = Path(args.db)
TEMPLATE_DIR = Path(args.outdir)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Constants (from the notebook) ─────────────────────────────────

CRITERIA_QUESTIONS = {
    "H1a": [
        "Does the text avoid naming the workers who produce the output?",
        "What agent occupies the subject position of production-related verbs?",
        "Are workers replaced by process-denoting nominals (annotation, labeling, dataset)?",
    ],
    "H1b": [
        "Does automation vocabulary appear where labour vocabulary would be expected?",
        "Is human involvement syntactically absent from product claims?",
        "What verbs frame the AI output — active (produces, delivers) or passive/agentless?",
    ],
    "H1c": [
        "Is human involvement foregrounded as a quality attribute rather than a labour relation?",
        "Does 'human' collocate with evaluative adjectives (irreplaceable, expert, vetted)?",
        "Is labour invoked to justify a price premium rather than to describe a worker's activity?",
    ],
}

FOCAL_TERMS = {
    "H1a": {
        "worker": ["apply", "payment", "earn", "worker", "job"],
        "client": ["workforce", "resource", "supplier", "crowdsourced"],
    },
    "H1b": {
        "worker": ["task", "complete", "submit", "guideline"],
        "client": ["autonomous", "annotation", "automate", "algorithm", "label"],
    },
    "H1c": {
        "worker": [],
        "client": ["human", "vetted", "irreplaceable", "judgement", "oversight"],
    },
}

TOPIC_HYPOTHESIS = {
    1:  'H1a',  2:  'H1b',  3:  None, 4:  'H1b',  5:  'H1a',
    6:  'H1c',  7:  'H1b',  8:  None, 9:  'H1b',  10: None,
    11: 'H1a',  12: 'H1c',  13: None, 14: 'H1a',  15: None,
    16: 'H1b',  17: 'H1a',  18: 'H1b', 19: 'H1b',  20: 'H1b',
    21: 'H1b',  22: 'H1a',  23: 'H1b', 24: 'H1b',  25: 'H1b',
}

KWIC_WINDOW = 15  # tokens either side, matches Step 1


# ─── KWIC helpers (from notebook Cell 10) ──────────────────────────

def get_page_segments(conn: sqlite3.Connection, page_id: int) -> list:
    """Load sentence-segmented tokens for a page."""
    row = conn.execute(
        "SELECT segments FROM pages_tfidf WHERE page_id = ?",
        (int(page_id),)
    ).fetchone()
    if row is None:
        return []
    raw = row[0] if isinstance(row, (tuple, list)) else row["segments"]
    if not raw:
        return []
    segs = json.loads(raw)
    if segs and isinstance(segs[0], list):
        return segs
    # Fallback: treat whole token list as one segment
    row2 = conn.execute(
        "SELECT unigrams FROM pages_tfidf WHERE page_id = ?",
        (int(page_id),)
    ).fetchone()
    if row2:
        raw2 = row2[0] if isinstance(row2, (tuple, list)) else row2["unigrams"]
        if raw2:
            return [json.loads(raw2)]
    return []


def kwic_lines(
    segments: list,
    target: str,
    window: int = KWIC_WINDOW,
    max_hits: int = 4,
) -> list[str]:
    """KWIC extraction within sentence-scoped window."""
    hits: list[str] = []
    target_lower = target.lower()
    for seg in segments:
        for i, tok in enumerate(seg):
            if tok.lower() == target_lower:
                left = seg[max(0, i - window):i]
                right = seg[i + 1:i + 1 + window]
                line = (
                    "..."
                    + " ".join(left)
                    + f" [[[{tok.upper()}]]] "
                    + " ".join(right)
                    + "..."
                )
                hits.append(line)
                if len(hits) >= max_hits:
                    return hits
    return hits


# ─── Connect ───────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row


# ─── Load the structured reading manifest ─────────────────────────
sample = pd.read_sql(
    "SELECT * FROM step2_sample_structured ORDER BY reading_order",
    conn
)
print(f"Loaded step2_sample_structured: {len(sample)} pages")

# Also check if the original step2_sample has extra metadata we want
has_original = bool(conn.execute(
    "SELECT 1 FROM sqlite_master WHERE name='step2_sample'"
).fetchone())
original_meta = {}
if has_original:
    for row in conn.execute(
        "SELECT page_id, hypothesis, stratum, topic_id, theta, "
        "       focal_term, rel_freq, sampling_reason "
        "FROM step2_sample"
    ).fetchall():
        pid = int(row[0]) if row[0] is not None else None
        if pid is not None:
            original_meta[pid] = dict(row)


# ─── Generate one template per page ──────────────────────────────

generated = 0
skipped = 0

for _, pg in sample.iterrows():
    page_id = int(pg["page_id"])
    domain = pg.get("domain", "unknown") or "unknown"
    audience = pg.get("audience", "unknown") or "unknown"
    url = pg.get("url", "")
    company = pg.get("company", "")
    case_label = pg.get("case_label", "")
    reading_section = pg.get("reading_section", "")
    reading_order = int(pg.get("reading_order", 0))
    jsd = pg.get("jsd", None)

    # STM info from the structured table
    dominant_topic = pg.get("dominant_topic", None)
    max_theta = pg.get("max_theta", None)
    theta_h1a = pg.get("theta_H1a", None)
    theta_h1b = pg.get("theta_H1b", None)
    theta_h1c = pg.get("theta_H1c", None)

    # Original sampling metadata (if page was in original step2_sample)
    orig = original_meta.get(page_id, {})
    orig_hypothesis = orig.get("hypothesis", "")
    orig_stratum = orig.get("stratum", pg.get("stratum", ""))
    orig_topic_id = orig.get("topic_id", dominant_topic)
    orig_theta = orig.get("theta", max_theta)
    orig_focal_term = orig.get("focal_term", "")
    orig_rel_freq = orig.get("rel_freq", None)
    orig_sampling_reason = orig.get("sampling_reason", "")

    # Determine primary hypothesis for this page (for filename and header)
    # Use original if available, otherwise derive from dominant topic
    if orig_hypothesis:
        primary_hyp = orig_hypothesis
    elif dominant_topic is not None and pd.notna(dominant_topic):
        try:
            primary_hyp = TOPIC_HYPOTHESIS.get(int(dominant_topic), None) or "general"
        except (ValueError, TypeError):
            primary_hyp = "general"
    else:
        primary_hyp = "general"

    # Build filename matching existing convention
    safe_domain = str(domain).replace(".", "_").replace("/", "_")
    filename = f"s{orig_stratum}_{primary_hyp}_{audience}_{safe_domain}_p{page_id}.md"

    out_path = TEMPLATE_DIR / filename

    # ── Skip files that already exist (preserve prior annotations) ─
    if out_path.exists():
        skipped += 1
        continue

    # ── Gather KWIC lines for ALL hypotheses' focal terms ──────
    segments = get_page_segments(conn, page_id)

    all_kwic: dict[str, dict[str, list[str]]] = {}
    for hyp in ["H1a", "H1b", "H1c"]:
        terms = FOCAL_TERMS.get(hyp, {}).get(audience, [])
        if not terms:
            continue
        hyp_kwic: dict[str, list[str]] = {}
        for term in terms:
            hits = kwic_lines(segments, term)
            if hits:
                hyp_kwic[term] = hits
        if hyp_kwic:
            all_kwic[hyp] = hyp_kwic

    # ── Build markdown ─────────────────────────────────────────
    lines: list[str] = []

    # Header
    lines.append(f"# Close reading: {primary_hyp} | {audience} | {domain}")
    lines.append("")

    # Metadata
    lines.append("## Metadata")
    lines.append(f"- **page_id**: {page_id}")
    lines.append(f"- **url**: {url}")
    lines.append(f"- **domain**: {domain}")
    lines.append(f"- **audience**: {audience}")
    lines.append(f"- **company**: {company}")
    lines.append(f"- **case**: {case_label}")
    lines.append(f"- **reading_order**: {reading_order}")
    lines.append(f"- **reading_section**: {reading_section}")
    if jsd is not None and pd.notna(jsd):
        lines.append(f"- **pair_JSD**: {jsd:.4f}")
    lines.append(f"- **primary_hypothesis**: {primary_hyp}")
    if orig_stratum:
        lines.append(f"- **stratum**: {orig_stratum}")
    if orig_topic_id is not None and pd.notna(orig_topic_id):
        theta_str = f"  (θ = {orig_theta:.4f})" if orig_theta and pd.notna(orig_theta) else ""
        lines.append(f"- **stm_topic**: T{int(orig_topic_id):02d}{theta_str}")
    if orig_focal_term and pd.notna(orig_focal_term) and str(orig_focal_term).strip():
        rf_str = f"  (rel_freq = {orig_rel_freq:.4f})" if orig_rel_freq and pd.notna(orig_rel_freq) else ""
        lines.append(f"- **focal_term**: {orig_focal_term}{rf_str}")
    if orig_sampling_reason and pd.notna(orig_sampling_reason):
        lines.append(f"- **sampling_reason**: {orig_sampling_reason}")

    # Per-hypothesis theta summary
    theta_parts = []
    if theta_h1a is not None and pd.notna(theta_h1a):
        theta_parts.append(f"H1a={theta_h1a:.3f}")
    if theta_h1b is not None and pd.notna(theta_h1b):
        theta_parts.append(f"H1b={theta_h1b:.3f}")
    if theta_h1c is not None and pd.notna(theta_h1c):
        theta_parts.append(f"H1c={theta_h1c:.3f}")
    if theta_parts:
        lines.append(f"- **hypothesis_theta**: {', '.join(theta_parts)}")
    lines.append("")

    # ── KWIC lines (grouped by hypothesis) ─────────────────────
    lines.append("## KWIC lines")
    if all_kwic:
        for hyp in ["H1a", "H1b", "H1c"]:
            if hyp not in all_kwic:
                continue
            lines.append(f"\n### {hyp} focal terms")
            for term, hits in all_kwic[hyp].items():
                lines.append(f"\n#### `{term}`")
                for hit in hits:
                    lines.append(f"> {hit}")
    else:
        lines.append("\n_(no focal-term hits in tokenised text)_")
    lines.append("")

    # ── Interpretive notes for ALL THREE hypotheses ────────────
    lines.append("## Interpretive notes")
    lines.append("")

    for hyp in ["H1a", "H1b", "H1c"]:
        hyp_labels = {"H1a": "Labour visibility gap",
                      "H1b": "Automation myth",
                      "H1c": "Strategic hypervisibility"}
        lines.append(f"### {hyp} — {hyp_labels[hyp]}")
        lines.append("")
        questions = CRITERIA_QUESTIONS[hyp]
        for q in questions:
            lines.append(f"**{q}**")
            lines.append("")
            lines.append("_[Your answer here]_")
            lines.append("")

    # ── Synthesis fields ───────────────────────────────────────
    lines.append("---")
    lines.append("### Representative KWIC line for thesis (≤15 words)")
    lines.append("")
    lines.append("_[Select the single most diagnostically informative line above]_")
    lines.append("")
    lines.append("### Frame label")
    lines.append("")
    lines.append("_[Invisibilisation / Substitution / Renarration / Other]_")
    lines.append("")
    lines.append("### Within-company contrast note")
    lines.append("")
    if company:
        if audience == "worker":
            lines.append(f"_[How does this page's framing compare to {company}'s client-side pages?]_")
        else:
            lines.append(f"_[How does this page's framing compare to {company}'s worker-side pages?]_")
    else:
        lines.append("_[N/A — cross-platform page]_")
    lines.append("")
    lines.append("### Cross-reference")
    lines.append("")
    lines.append("_[Which STM topic / keyness term motivated this selection?]_")

    # Write
    out_path.write_text("\n".join(lines), encoding="utf-8")
    generated += 1

print(f"\n  Generated {generated} new templates → {TEMPLATE_DIR}/")
print(f"  Skipped  {skipped} (file already exists — annotations preserved)")

# ── Print reading order summary ────────────────────────────────
print(f"\n{'='*60}")
print("FILES BY READING ORDER")
print(f"{'='*60}")

current_case = None
current_aud = None
for _, pg in sample.iterrows():
    cl = pg.get("case_label", "")
    aud = pg.get("audience", "")
    domain = pg.get("domain", "")
    page_id = int(pg["page_id"])
    ro = int(pg.get("reading_order", 0))

    if cl != current_case:
        current_case = cl
        current_aud = None
        print(f"\n── {current_case} ──")
    if aud != current_aud:
        current_aud = aud
        print(f"  [{current_aud.upper()} — {domain}]")

    safe_domain = str(domain).replace(".", "_").replace("/", "_")
    orig = original_meta.get(page_id, {})
    hyp = orig.get("hypothesis", "general")
    stratum = orig.get("stratum", pg.get("stratum", ""))
    fname = f"s{stratum}_{hyp}_{aud}_{safe_domain}_p{page_id}.md"
    fpath = TEMPLATE_DIR / fname
    exists_marker = " ✓EXISTS" if fpath.exists() else ""
    print(f"    {ro:>3d}. {fname}{exists_marker}")

conn.close()
print(f"\n{'='*60}")
print("Done. Start with reading_order 1 and work through in sequence.")
print(f"{'='*60}")
