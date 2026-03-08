"""
04_step2_export.py
==================
Produces the Step 2 guided close reading corpus for Nelson (2020)
computational grounded theory.

Step 1 identified WHAT is statistically distinctive. Step 2 asks WHY —
what are the terms actually doing rhetorically? This script exports the
textual material that makes that interpretive work tractable.

Three export types, all as plain-text files:

  Export A — Top pages per key term
    For each theoretically important term, the N pages where it appears
    most densely (highest relative frequency). These are the pages most
    responsible for the term's keyness signal. Reading them reveals the
    discursive context the term habitually appears in.

  Export B — KWIC concordance lines
    Keyword-in-context: every sentence containing the focus term, with
    ±CONTEXT_SENTENCES surrounding sentences, labelled by platform and
    audience. Standard close reading input for corpus linguistics.
    Reveals rhetorical patterns across the full corpus at once.

  Export C — Within-pair page samples
    For each platform pair (appen/crowdgen, toloka/mindrift), a balanced
    sample of pages from both sides. Reading the same company speaking to
    different audiences is the strongest test of audience-driven register
    divergence (controls for company-level style variation).

Output structure:
  outputs/step2/
    A_top_pages/
      client/  <term>.txt
      worker/  <term>.txt
    B_kwic/
      <term>_client.txt
      <term>_worker.txt
    C_within_pair/
      appen_b2b.txt
      appen_b2w.txt
      toloka_b2b.txt
      toloka_b2w.txt

Prerequisites: 01_prepare.py and 02_step1_frequency.py must have been run.

Usage:
    python3 src/04_step2_export.py
"""

import sqlite3
import re
import textwrap
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH     = "data/scraping.db"
OUTPUT_DIR  = Path("outputs/step2")

# Export A: how many pages per term
TOP_PAGES_N = 8

# Export B: sentence window around each keyword hit
CONTEXT_SENTENCES = 2

# Export C: how many pages per platform side in within-pair sample
PAIR_SAMPLE_N = 20

# Terms to export — organised by hypothesis and audience register.
# These are the terms Step 1 identified as theoretically meaningful
# after artifact filtering. Edit this list as your close reading evolves.

CLIENT_TERMS = {
    # H1b — Automation myth
    "autonomous":   "H1b",
    "automate":     "H1b",
    "automation":   "H1b",
    "machine":      "H1b",
    # H1c — Strategic hypervisibility
    "human":        "H1c",
    "quality":      "H1c",
    "oversight":    "H1c",
    "annotation":   "H1c",
    "label":        "H1c",
    "datum":        "H1c",
}

WORKER_TERMS = {
    # H1a — Labour visibility
    "worker":   "H1a",
    "work":     "H1a",
    "job":      "H1a",
    "earn":     "H1a",
    "pay":      "H1a",
    "payment":  "H1a",
    "apply":    "H1a",
    "remote":   "H1a",
    "project":  "H1a",
    "code":     "H1a",
}

# Platform pairs for Export C
PAIRS = [
    ("appen",  "appen.com",    "crowdgen.com"),
    ("toloka", "toloka.ai",    "mindrift.ai"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_page_text(conn, page_id):
    """Retrieve raw text_content from pages table."""
    row = conn.execute(
        "SELECT text_content, url FROM pages WHERE id = ?", [page_id]
    ).fetchone()
    return (row["text_content"] or "", row["url"]) if row else ("", "")


def get_top_pages_for_term(conn, term, audience, n):
    """
    Return the N pages with highest relative frequency for a given term
    in the given audience register, from platform_term_counts joined
    back to pages via corpus_view.
    """
    rows = conn.execute("""
        SELECT
            ptc.domain,
            ptc.rel_freq,
            ptc.freq,
            cv.page_id,
            cv.url,
            cv.platform_name,
            cv.platform_type,
            cv.company_id
        FROM platform_term_counts ptc
        JOIN corpus_view cv
          ON cv.domain = ptc.domain
         AND cv.audience = ptc.audience
        WHERE ptc.term      = ?
          AND ptc.audience  = ?
          AND ptc.term_type = 'unigram'
          AND ptc.rel_freq  > 0
        ORDER BY ptc.rel_freq DESC
        LIMIT ?
    """, [term, audience, n]).fetchall()
    return [dict(r) for r in rows]


def get_all_pages_for_audience(conn, audience):
    """Return all page_ids and metadata for an audience register."""
    rows = conn.execute("""
        SELECT page_id, url, platform_name, domain, company_id, platform_type
        FROM corpus_view
        WHERE audience = ?
    """, [audience]).fetchall()
    return [dict(r) for r in rows]


def get_pages_for_domain(conn, domain):
    """
    Return all pages for a specific domain.
    Tries exact match, then www. variant, then substring match.
    Logs all available domains if nothing found so the problem is diagnosable.
    """
    # 1. Exact match
    rows = conn.execute("""
        SELECT page_id, url, platform_name, domain, audience
        FROM corpus_view WHERE domain = ? ORDER BY url
    """, [domain]).fetchall()
    if rows:
        return [dict(r) for r in rows]

    # 2. www. variant
    alt = ("www." + domain) if not domain.startswith("www.") else domain[4:]
    rows = conn.execute("""
        SELECT page_id, url, platform_name, domain, audience
        FROM corpus_view WHERE domain = ? ORDER BY url
    """, [alt]).fetchall()
    if rows:
        log.info(f"  '{domain}' matched via variant '{alt}'")
        return [dict(r) for r in rows]

    # 3. Substring match — catches e.g. crowdgen stored as crowdgen.com.au
    stem = domain.split(".")[0]
    rows = conn.execute("""
        SELECT page_id, url, platform_name, domain, audience
        FROM corpus_view WHERE domain LIKE ? ORDER BY url
    """, [f"%{stem}%"]).fetchall()
    if rows:
        log.info(f"  '{domain}' matched via substring '{stem}' to '{rows[0]['domain']}'")
        return [dict(r) for r in rows]

    # Nothing found — log available domains to diagnose
    all_domains = conn.execute(
        "SELECT DISTINCT domain, audience FROM corpus_view ORDER BY domain"
    ).fetchall()
    log.warning(f"  No pages found for '{domain}' (also tried '{alt}', stem '{stem}')")
    log.warning("  Available domains in corpus_view:")
    for d in all_domains:
        log.warning(f"    {d['domain']:<40} audience={d['audience']}")
    return []


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

def split_sentences(text):
    """
    Simple sentence splitter. Good enough for concordance work —
    we are not doing NLP, just finding context windows.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    # Split on .!? followed by whitespace + capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def find_kwic(text, term, context_n):
    """
    Return list of (before_sentences, hit_sentence, after_sentences)
    for every sentence in text containing the term (case-insensitive).
    """
    sentences = split_sentences(text)
    pattern   = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    results   = []
    for i, sent in enumerate(sentences):
        if pattern.search(sent):
            before = sentences[max(0, i - context_n): i]
            after  = sentences[i + 1: i + 1 + context_n]
            results.append((before, sent, after))
    return results


def highlight_term(sentence, term):
    """Wrap term occurrences in >>> <<< markers for readability."""
    pattern = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)
    return pattern.sub(r'>>>\1<<<', sentence)


def truncate_text(text, max_chars=8000):
    """Cap page text to avoid enormous files while keeping full paragraphs."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to end on a paragraph boundary
    last_para = truncated.rfind('\n\n')
    if last_para > max_chars * 0.7:
        truncated = truncated[:last_para]
    return truncated + f"\n\n[... truncated at {max_chars} chars ...]"


# ---------------------------------------------------------------------------
# File writing helpers
# ---------------------------------------------------------------------------

def write_header(f, title, generated_at):
    sep = "=" * 72
    f.write(f"{sep}\n{title}\nGenerated: {generated_at}\n{sep}\n\n")


def write_page_block(f, meta, text, term=None):
    """Write a single page's metadata + text to file."""
    f.write(f"{'—' * 60}\n")
    f.write(f"URL:       {meta.get('url', 'n/a')}\n")
    f.write(f"Platform:  {meta.get('platform_name', meta.get('domain', 'n/a'))}\n")
    f.write(f"Type:      {meta.get('platform_type', 'n/a')}\n")
    f.write(f"Audience:  {meta.get('audience', 'n/a')}\n")
    if "rel_freq" in meta:
        f.write(f"Term freq: {meta['rel_freq']:.2f}‰  (abs: {meta.get('freq','?')})\n")
    f.write(f"{'—' * 60}\n\n")
    body = truncate_text(text)
    if term:
        body = highlight_term(body, term)
    f.write(body)
    f.write("\n\n")


def write_kwic_block(f, meta, kwic_hits, term):
    """Write KWIC concordance entries for one page."""
    if not kwic_hits:
        return
    f.write(f"{'—' * 60}\n")
    f.write(f"Platform: {meta.get('platform_name', meta.get('domain', 'n/a'))}"
            f"  |  {meta.get('url', '')}\n")
    f.write(f"{'—' * 60}\n")
    for i, (before, hit, after) in enumerate(kwic_hits, 1):
        f.write(f"\n  [{i}]\n")
        for s in before:
            f.write(f"      {s}\n")
        f.write(f"  >>> {highlight_term(hit, term)} <<<\n")
        for s in after:
            f.write(f"      {s}\n")
    f.write("\n")


# ---------------------------------------------------------------------------
# Export A: Top pages per key term
# ---------------------------------------------------------------------------

def export_a_top_pages(conn, out_dir, generated_at):
    log.info("Export A — Top pages per key term")
    base = out_dir / "A_top_pages"

    for audience, term_dict in [("client", CLIENT_TERMS),
                                 ("worker", WORKER_TERMS)]:
        aud_dir = base / audience
        aud_dir.mkdir(parents=True, exist_ok=True)

        for term, hypothesis in term_dict.items():
            pages = get_top_pages_for_term(conn, term, audience, TOP_PAGES_N)
            if not pages:
                log.warning(f"  No pages found for '{term}' / {audience}")
                continue

            out_path = aud_dir / f"{term}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                write_header(f,
                    f"EXPORT A — Top pages for term: '{term}'\n"
                    f"Audience: {audience.upper()}  |  Hypothesis: {hypothesis}\n"
                    f"Showing top {len(pages)} pages by relative frequency",
                    generated_at)

                f.write("READING NOTES:\n")
                f.write(f"  Term '{term}' is {audience}-distinctive (Step 1 keyness).\n")
                f.write(f"  Question for Step 2: What is '{term}' doing rhetorically "
                        f"in {audience.upper()} texts?\n")
                f.write(f"  Is its usage consistent with {hypothesis}?\n")
                f.write(f"  Are there unexpected usages that complicate the hypothesis?\n\n")

                for meta in pages:
                    text, url = get_page_text(conn, meta["page_id"])
                    meta["url"] = url
                    write_page_block(f, meta, text, term=term)

            log.info(f"    {audience}/{term}.txt  ({len(pages)} pages)")


# ---------------------------------------------------------------------------
# Export B: KWIC concordance lines
# ---------------------------------------------------------------------------

def export_b_kwic(conn, out_dir, generated_at):
    log.info("Export B — KWIC concordance lines")
    base = out_dir / "B_kwic"
    base.mkdir(parents=True, exist_ok=True)

    # For KWIC we query all pages in the audience, not just top-N per term
    # (we want the full distributional picture of how the term is used)
    for audience, term_dict in [("client", CLIENT_TERMS),
                                 ("worker", WORKER_TERMS)]:
        all_pages = get_all_pages_for_audience(conn, audience)
        log.info(f"  {audience}: {len(all_pages)} pages to scan")

        for term, hypothesis in term_dict.items():
            out_path = base / f"{term}_{audience}.txt"
            total_hits = 0

            with open(out_path, "w", encoding="utf-8") as f:
                write_header(f,
                    f"EXPORT B — KWIC concordance: '{term}' in {audience.upper()} texts\n"
                    f"Hypothesis: {hypothesis}  |  Context: ±{CONTEXT_SENTENCES} sentences",
                    generated_at)

                f.write("READING NOTES:\n")
                f.write(f"  Scan these concordance lines for recurring patterns.\n")
                f.write(f"  What subjects and objects appear with '{term}'?\n")
                f.write(f"  What verbs frame it? What is the implied agent?\n")
                f.write(f"  Note any lines that surprise you or complicate {hypothesis}.\n\n")
                f.write(f"{'=' * 72}\n\n")

                for page_meta in all_pages:
                    text, url = get_page_text(conn, page_meta["page_id"])
                    if not text:
                        continue
                    hits = find_kwic(text, term, CONTEXT_SENTENCES)
                    if hits:
                        page_meta["url"] = url
                        write_kwic_block(f, page_meta, hits, term)
                        total_hits += len(hits)

            log.info(f"    {term}_{audience}.txt  ({total_hits} concordance hits)")


# ---------------------------------------------------------------------------
# Export C: Within-pair page samples
# ---------------------------------------------------------------------------

def export_c_within_pair(conn, out_dir, generated_at):
    log.info("Export C — Within-pair page samples")
    base = out_dir / "C_within_pair"
    base.mkdir(parents=True, exist_ok=True)

    for company_id, b2b_domain, b2w_domain in PAIRS:
        for domain, label in [(b2b_domain, "b2b"), (b2w_domain, "b2w")]:
            pages = get_pages_for_domain(conn, domain)
            if not pages:
                log.warning(f"  No pages found for domain: {domain}")
                continue

            # Sample evenly — take every Nth page to cover different sections
            if len(pages) > PAIR_SAMPLE_N:
                step   = len(pages) // PAIR_SAMPLE_N
                sample = pages[::step][:PAIR_SAMPLE_N]
            else:
                sample = pages

            out_path = base / f"{company_id}_{label}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                audience_str = "CLIENT (B2B)" if label == "b2b" else "WORKER (B2W)"
                write_header(f,
                    f"EXPORT C — Within-pair sample: {domain}\n"
                    f"Company: {company_id.upper()}  |  Audience: {audience_str}\n"
                    f"Pages: {len(sample)} of {len(pages)} total",
                    generated_at)

                f.write("READING NOTES:\n")
                f.write(f"  Read this alongside {company_id}_"
                        f"{'b2w' if label == 'b2b' else 'b2b'}.txt\n")
                f.write(f"  This is the SAME COMPANY communicating to a DIFFERENT audience.\n")
                f.write(f"  Question: What changes? What stays the same?\n")
                f.write(f"  Pay attention to: labour vocabulary, compensation framing,\n")
                f.write(f"  worker agency, automation claims, quality rhetoric.\n\n")
                f.write(f"  Terms to watch: "
                        + ", ".join(list(CLIENT_TERMS.keys())[:6]
                                    + list(WORKER_TERMS.keys())[:6])
                        + "\n\n")

                for meta in sample:
                    text, url = get_page_text(conn, meta["page_id"])
                    meta["url"] = url
                    write_page_block(f, meta, text)

            log.info(f"    {company_id}_{label}.txt  ({len(sample)} pages)")


# ---------------------------------------------------------------------------
# Index file
# ---------------------------------------------------------------------------

def write_index(out_dir, generated_at):
    """Write a plain-text index explaining the export structure."""
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("STEP 2 — GUIDED CLOSE READING CORPUS\n")
        f.write("Nelson (2020) Computational Grounded Theory\n")
        f.write(f"Generated: {generated_at}\n")
        f.write("=" * 72 + "\n\n")

        f.write("PURPOSE\n")
        f.write("-------\n")
        f.write("Step 1 identified statistically distinctive terms.\n")
        f.write("Step 2 asks: what are those terms actually doing?\n")
        f.write("This corpus directs your reading toward the most theoretically\n")
        f.write("productive areas, so you are not browsing blindly.\n\n")

        f.write("STRUCTURE\n")
        f.write("---------\n")
        f.write("A_top_pages/\n")
        f.write("  client/<term>.txt  — pages most responsible for client-side keyness\n")
        f.write("  worker/<term>.txt  — pages most responsible for worker-side keyness\n")
        f.write("  Read these to understand the CONTEXT of each distinctive term.\n\n")

        f.write("B_kwic/\n")
        f.write("  <term>_client.txt  — every use of the term in client texts\n")
        f.write("  <term>_worker.txt  — every use of the term in worker texts\n")
        f.write("  Read these to identify RECURRING PATTERNS across the corpus.\n\n")

        f.write("C_within_pair/\n")
        f.write("  appen_b2b.txt / appen_b2w.txt\n")
        f.write("  toloka_b2b.txt / toloka_b2w.txt\n")
        f.write("  Same company, different audience. Read side by side.\n\n")

        f.write("WHAT TO DO AS YOU READ\n")
        f.write("----------------------\n")
        f.write("1. For each term, note the TYPICAL DISCURSIVE CONTEXT:\n")
        f.write("   - What subjects and objects appear with the term?\n")
        f.write("   - What verbs frame it?\n")
        f.write("   - Who is the implied agent?\n\n")
        f.write("2. Build a CODEBOOK of named patterns as you read:\n")
        f.write("   - Give each pattern a short name (e.g. 'HITL quality frame')\n")
        f.write("   - Note which terms instantiate it\n")
        f.write("   - Collect 2-3 representative example sentences\n")
        f.write("   - Map it to a hypothesis (H1a / H1b / H1c)\n\n")
        f.write("3. Flag SURPRISES — usages that complicate the hypothesis.\n")
        f.write("   These are theoretically important. Note them explicitly.\n\n")
        f.write("4. The codebook you build here becomes the dictionary for Step 3.\n\n")

        f.write("HYPOTHESES FOR REFERENCE\n")
        f.write("------------------------\n")
        f.write("H1a Labour visibility:          Labour terms lower in B2B than B2W\n")
        f.write("H1b Automation myth:            B2B frames outputs as autonomous\n")
        f.write("H1c Strategic hypervisibility:  B2B foregrounds human as quality signal\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("04_step2_export.py — Step 2 Reading Corpus")
    log.info("=" * 60)

    conn         = get_conn()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for table in ["keyness_results", "platform_term_counts", "corpus_view"]:
        if not conn.execute(
            f"SELECT name FROM sqlite_master WHERE type IN ('table','view') "
            f"AND name='{table}'"
        ).fetchone():
            raise RuntimeError(
                f"'{table}' not found — run 01_prepare.py and "
                f"02_step1_frequency.py first.")

    export_a_top_pages(conn, OUTPUT_DIR, generated_at)
    export_b_kwic(conn, OUTPUT_DIR, generated_at)
    export_c_within_pair(conn, OUTPUT_DIR, generated_at)
    write_index(OUTPUT_DIR, generated_at)

    conn.close()

    log.info("=" * 60)
    log.info(f"Step 2 corpus written to: {OUTPUT_DIR.resolve()}")
    log.info("  A_top_pages/  — top pages per key term")
    log.info("  B_kwic/       — KWIC concordance lines")
    log.info("  C_within_pair/ — within-pair samples")
    log.info("  README.txt    — reading guide")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
