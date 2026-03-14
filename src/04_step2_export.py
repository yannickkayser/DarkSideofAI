"""
04_step2_export.py
==================
Produces the Step 2 guided close reading corpus for Nelson (2020)
Computational Grounded Theory.

Pipeline position:
  Stage 4 — Step 2 Export (Close Reading Corpus)
  Reads from: corpus_view, pages, platform_term_counts, keyness_results
  Produces:   output/step_2/  (plain-text .txt files + README)
  Follows:    02_step1_frequency.py (keyness and platform_term_counts)
              01_prepare.py (corpus_view)

What this script does:
  Step 1 (02_step1_frequency.py, 02b, 02c) identified WHAT is statistically
  distinctive about B2B vs B2W language — which terms are disproportionately
  associated with each audience, and how the topic space divides.

  Step 2 asks WHY — what are those distinctive terms actually doing
  rhetorically?  This script organises the raw page text into three curated
  export formats that make close reading tractable:

  Export A — Top pages per key term
    For each term in CLIENT_TERMS and WORKER_TERMS, the TOP_PAGES_N=8 pages
    with the highest relative frequency for that term in the corresponding
    audience register.  These are the pages most responsible for the term's
    keyness signal.  Reading them reveals the discursive context the term
    habitually appears in and whether that context is consistent with the
    hypothesis it has been assigned to.

  Export B — KWIC concordance lines
    Keyword-in-context: every sentence containing the focus term across the
    ENTIRE audience sub-corpus, with ±CONTEXT_SENTENCES=2 surrounding
    sentences for each hit.  Standard close reading input for corpus
    linguistics.  Reading all hits at once reveals recurring patterns,
    preferred collocates, and construction types that individual page reads
    might miss.

  Export C — Within-pair page samples
    For each platform pair (e.g. appen.com / crowdgen.com), a balanced
    sample of pages from both the B2B domain and the B2W domain.  Because
    both sides are operated by the same company, reading them side-by-side
    controls for company-level style variation.  Register differences that
    persist across this control are especially strong evidence of
    audience-driven rhetorical adaptation.

Output structure:
    output/step_2/
      README.txt                    — reading guide for the whole corpus
      A_top_pages/
        client/<term>.txt
        worker/<term>.txt
      B_kwic/
        <term>_client.txt
        <term>_worker.txt
      C_within_pair/
        appen_b2b.txt / appen_b2w.txt
        toloka_b2b.txt / toloka_b2w.txt
        scale_b2b.txt  / scale_b2w.txt
        centific_b2b.txt / centific_b2w.txt
        labelbox_b2b.txt / labelbox_b2w.txt

Key constants (edit to update the reading corpus as the theory evolves):
  CLIENT_TERMS : dict of term → hypothesis for B2B focus terms.
  WORKER_TERMS : dict of term → hypothesis for B2W focus terms.
  PAIRS        : list of (company_id, b2b_domain, b2w_domain) triples.
  TOP_PAGES_N  : number of pages per term in Export A (default 8).
  CONTEXT_SENTENCES : ±window for KWIC (default 2).
  PAIR_SAMPLE_N : pages per platform side in Export C (default 20).

Text handling:
  - truncate_text() caps each page at 8000 chars to prevent huge export
    files while keeping full paragraphs (splits on '\n\n' boundary).
  - highlight_term() wraps hits in >>> <<< markers for visual scanning.
  - find_kwic() uses a simple regex sentence splitter (split on .!? +
    capital) — adequate for close reading; not a full NLP pipeline.

Domain lookup resilience:
  get_pages_for_domain() tries three strategies before giving up:
    1. Exact domain match
    2. www. prefix variant
    3. Domain stem substring match (catches e.g. crowdgen.com.au)
  If nothing is found it logs all available domains to aid diagnosis.

Prerequisites:
  Run in order before this script:
    01_prepare.py          — populates corpus_view
    02_step1_frequency.py  — populates platform_term_counts, keyness_results

  The script raises RuntimeError immediately if corpus_view,
  platform_term_counts, or keyness_results is missing.

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
OUTPUT_DIR  = Path("output/step_2")

# Export A: how many pages per term
TOP_PAGES_N = 8

# Export B: sentence window around each keyword hit
CONTEXT_SENTENCES = 2

# Export C: how many pages per platform side in within-pair sample
# Pages are evenly sampled with stride = len(pages) // PAIR_SAMPLE_N to
# cover all sections of the site rather than just the first 20 URLs
PAIR_SAMPLE_N = 20

# Terms to export — organised by hypothesis and audience register.
# These are the terms Step 1 identified as theoretically meaningful
# after artifact filtering. Edit this list as your close reading evolves.

CLIENT_TERMS = {
    # H1b — Automation myth: B2B texts frame AI outputs as autonomous
    "autonomous":   "H1b",
    "automate":     "H1b",
    "automation":   "H1b",
    "machine":      "H1b",
    # H1c — Strategic hypervisibility: B2B foregrounds human oversight
    #        as a quality and trust signal for enterprise clients
    "human":        "H1c",
    "quality":      "H1c",
    "oversight":    "H1c",
    "annotation":   "H1c",
    "label":        "H1c",
    "datum":        "H1c",   # lemma of 'data' — high B2B keyness (corpus signal)
}

WORKER_TERMS = {
    # H1a — Labour visibility gap: labour terms are suppressed in B2B
    #        (low in client texts) and foregrounded in B2W (high in worker texts)
    "worker":   "H1a",
    "work":     "H1a",
    "job":      "H1a",
    "earn":     "H1a",
    "pay":      "H1a",
    "payment":  "H1a",
    "apply":    "H1a",
    "remote":   "H1a",
    "project":  "H1a",
    "code":     "H1a",   # coding tasks prominent in B2W worker-facing pages
}

# Platform pairs for Export C.
# Each triple: (company_id, b2b_domain, b2w_domain)
# company_id is used as the filename stem (e.g. "appen_b2b.txt")
# Must match domains as stored in corpus_view.domain (get_pages_for_domain
# handles www. / suffix variants automatically)
PAIRS = [
    ("appen",    "appen.com",    "crowdgen.com"),
    ("toloka",   "toloka.ai",    "mindrift.ai"),
    ("scale",    "scale.com",   "remotasks.com"),
    ("centific", "centific.com",  "oneforma.com"),
    ("labelbox", "labelbox.com", "alignerr.com"),
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
    """
    Open and return a SQLite connection to DB_PATH.

    Uses sqlite3.Row so all result rows support key-based column access
    (e.g. row['domain'] or row['rel_freq']).

    Returns:
        sqlite3.Connection with row_factory = sqlite3.Row.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_page_text(conn, page_id):
    """
    Retrieve raw text_content and URL for a page from the pages table.

    The pages table stores the full cleaned text extracted by scraper.py.
    This is the text that was originally tokenised and TF-IDF weighted for
    Step 1 analysis; returning to it allows Step 2 to read in full context.

    Args:
        conn    : Open sqlite3.Connection.
        page_id : Integer primary key of the pages row.

    Returns:
        Tuple (text_content: str, url: str).  Returns ("", "") if the
        page_id does not exist or text_content is NULL.
    """
    row = conn.execute(
        "SELECT text_content, url FROM pages WHERE id = ?", [page_id]
    ).fetchone()
    return (row["text_content"] or "", row["url"]) if row else ("", "")


def get_top_pages_for_term(conn, term, audience, n):
    """
    Return the N pages with the highest relative frequency for a given term
    in the specified audience register.

    Queries platform_term_counts (written by 02_step1_frequency.py) joined
    to corpus_view for metadata.  Relative frequency (rel_freq) is the
    term's count per 1000 tokens for that domain, so it normalises for
    domain size — a term appearing 10 times on a 50-page site has different
    significance than on a 500-page site.

    These are the pages most responsible for the term's keyness signal.
    Reading the top-8 pages per term usually reveals the canonical rhetorical
    context quickly.

    Args:
        conn     : Open sqlite3.Connection.
        term     : Lemmatised unigram string (as stored in pages_tfidf).
        audience : "client" or "worker".
        n        : Number of pages to return.

    Returns:
        List of dicts with keys: domain, rel_freq, freq, page_id, url,
        platform_name, platform_type, company_id.  Ordered by rel_freq DESC.
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
    """
    Return all page_ids and metadata for an audience register.

    Used by Export B (KWIC) which needs to scan every page in the audience
    sub-corpus rather than just the top-N per domain, to give a complete
    distributional picture of how the term is used across the full register.

    Args:
        conn     : Open sqlite3.Connection.
        audience : "client" or "worker".

    Returns:
        List of dicts with keys: page_id, url, platform_name, domain,
        company_id, platform_type.
    """
    rows = conn.execute("""
        SELECT page_id, url, platform_name, domain, company_id, platform_type
        FROM corpus_view
        WHERE audience = ?
    """, [audience]).fetchall()
    return [dict(r) for r in rows]


def get_pages_for_domain(conn, domain):
    """
    Return all pages for a specific domain, with fallback lookup strategies.

    Tries three strategies in order:
      1. Exact domain match — fastest, covers the majority of cases.
      2. www. prefix variant — handles cases where the domain was stored
         with or without the www. prefix.
      3. Domain stem substring — catches storage variants like
         "crowdgen.com.au" when querying "crowdgen.com".

    If nothing is found, logs all available (domain, audience) pairs from
    corpus_view so the mismatch is diagnosable without opening the database
    manually.

    Args:
        conn   : Open sqlite3.Connection.
        domain : Domain string as listed in PAIRS (e.g. "appen.com").

    Returns:
        List of dicts with keys: page_id, url, platform_name, domain, audience.
        Empty list if domain not found after all strategies.
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

    # Nothing found — log all available domains to aid diagnosis
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
    Simple regex sentence splitter for KWIC concordance generation.

    Splits on .!? followed by whitespace and a capital letter.  This is
    intentionally simple (not a full NLP tokeniser) because the output is
    read by a human — a missed split produces slightly longer context
    windows but does not invalidate the concordance.

    Args:
        text : Raw page text string.

    Returns:
        List of sentence strings with leading/trailing whitespace stripped.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def find_kwic(text, term, context_n):
    """
    Find all keyword-in-context occurrences of term in text.

    Uses whole-word boundary matching (\\b) to avoid false positives from
    partial matches (e.g. "automate" should not match "automation" unless
    both are in the term list).

    Args:
        text      : Full page text.
        term      : Lemma string to search for (case-insensitive).
        context_n : Number of surrounding sentences to include before and after.

    Returns:
        List of (before: list[str], hit_sentence: str, after: list[str])
        tuples — one per occurrence of term in text.
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
    """
    Wrap occurrences of term in >>> <<< markers for visual scanning.

    These ASCII markers are more legible than bold/italic in plain-text
    files and survive copy-paste into coding tools without formatting loss.

    Args:
        sentence : Text string to annotate.
        term     : Lemma string to highlight (case-insensitive).

    Returns:
        Annotated string with term wrapped in >>>...<< on every occurrence.
    """
    pattern = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)
    return pattern.sub(r'>>>\1<<<', sentence)


def truncate_text(text, max_chars=8000):
    """
    Cap page text at max_chars while preserving paragraph boundaries.

    A typical scraped page is 2000–6000 chars; some are much longer
    (e.g. long job-listing pages, full technical documentation).  Truncating
    keeps individual export files at a readable size.  Splitting on the last
    '\n\n' within the capped range ensures the text ends at a paragraph
    break rather than mid-sentence.

    Args:
        text      : Full page text.
        max_chars : Hard cap in characters (default 8000).

    Returns:
        Text up to max_chars (or last paragraph boundary above 70% of
        max_chars), followed by a truncation marker if cut.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to end on a paragraph boundary (at least 70% of max_chars in)
    last_para = truncated.rfind('\n\n')
    if last_para > max_chars * 0.7:
        truncated = truncated[:last_para]
    return truncated + f"\n\n[... truncated at {max_chars} chars ...]"


# ---------------------------------------------------------------------------
# File writing helpers
# ---------------------------------------------------------------------------

def write_header(f, title, generated_at):
    """
    Write a standard export file header with title and timestamp.

    Args:
        f            : Open file object (write mode).
        title        : Multi-line title string for this export file.
        generated_at : ISO-formatted timestamp string.
    """
    sep = "=" * 72
    f.write(f"{sep}\n{title}\nGenerated: {generated_at}\n{sep}\n\n")


def write_page_block(f, meta, text, term=None):
    """
    Write a single page's metadata header and text body to an export file.

    Metadata block includes: URL, platform name, platform type, audience,
    and (for Export A) the term's relative and absolute frequency on this
    domain.  If term is provided, occurrences in the text are highlighted
    with >>> <<< markers.

    Args:
        f    : Open file object (write mode).
        meta : Dict with page metadata (url, platform_name, platform_type,
               audience, and optionally rel_freq, freq).
        text : Raw page text to include.
        term : Optional lemma string to highlight in the body text.
    """
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
    """
    Write KWIC concordance entries for one page to an export file.

    Each hit is formatted as a numbered entry: context sentences before,
    the hit sentence wrapped in >>> <<<, context sentences after.

    Skips pages with no hits (kwic_hits is empty list).

    Args:
        f         : Open file object (write mode).
        meta      : Dict with page metadata (platform_name, url).
        kwic_hits : List of (before, hit, after) tuples from find_kwic().
        term      : Lemma string (used to highlight within the hit sentence).
    """
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
    """
    Export A: the TOP_PAGES_N pages with highest term frequency per focus term.

    Iterates over CLIENT_TERMS (B2B) and WORKER_TERMS (B2W).  For each term,
    queries platform_term_counts (via get_top_pages_for_term) to find the
    domains where the term has the highest relative frequency, then writes
    the full text of those pages to a .txt file with reading-guide prompts.

    Reading guide prompts are embedded in each file header to encourage
    structured note-taking:
      - What is the term doing rhetorically?
      - Is usage consistent with the assigned hypothesis?
      - Are there unexpected usages that complicate the hypothesis?

    Output: output/step_2/A_top_pages/{audience}/{term}.txt

    Args:
        conn         : Open sqlite3.Connection.
        out_dir      : Path to output/step_2/ (parent of A_top_pages/).
        generated_at : Timestamp string for file headers.
    """
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
    """
    Export B: KWIC concordance — every occurrence of each term across the audience.

    Unlike Export A (which shows full-page context for the most term-dense
    pages), Export B sweeps the ENTIRE audience sub-corpus and extracts
    every sentence containing the term, with ±CONTEXT_SENTENCES surrounding
    sentences.  This gives a complete distributional picture of how the term
    is used — essential for identifying recurring patterns and construction
    types that might not be apparent from the top-N pages alone.

    Embedded reading guide prompts encourage attention to:
      - Subjects and objects appearing with the term.
      - Verb frames (how is agency attributed?).
      - Lines that surprise or complicate the hypothesis.

    Output: output/step_2/B_kwic/{term}_{audience}.txt
    Each file reports the total number of concordance hits in the log.

    Args:
        conn         : Open sqlite3.Connection.
        out_dir      : Path to output/step_2/ (parent of B_kwic/).
        generated_at : Timestamp string for file headers.
    """
    log.info("Export B — KWIC concordance lines")
    base = out_dir / "B_kwic"
    base.mkdir(parents=True, exist_ok=True)

    # For KWIC we query all pages in the audience — not just top-N per term —
    # to capture the full distributional range of how the term is used
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
    """
    Export C: balanced page samples from both sides of each platform pair.

    For each company in PAIRS, writes two files: one from the B2B domain
    and one from the B2W domain.  The same company communicates to different
    audiences on different sites; this within-pair design controls for
    company-level style variation.  Register differences that persist
    within pairs are the cleanest evidence of audience-driven adaptation.

    Sampling strategy: evenly-spaced stride sampling (every Nth page)
    rather than taking the first PAIR_SAMPLE_N pages.  This ensures coverage
    across different sections of the site (homepage, product pages, blog,
    careers, etc.) rather than just the first N URLs alphabetically.

    Reading guide prompts in each file header direct attention to the terms
    from CLIENT_TERMS and WORKER_TERMS that Step 1 identified as distinctive.

    Output: output/step_2/C_within_pair/{company_id}_{b2b|b2w}.txt

    Args:
        conn         : Open sqlite3.Connection.
        out_dir      : Path to output/step_2/ (parent of C_within_pair/).
        generated_at : Timestamp string for file headers.
    """
    log.info("Export C — Within-pair page samples")
    base = out_dir / "C_within_pair"
    base.mkdir(parents=True, exist_ok=True)

    for company_id, b2b_domain, b2w_domain in PAIRS:
        for domain, label in [(b2b_domain, "b2b"), (b2w_domain, "b2w")]:
            pages = get_pages_for_domain(conn, domain)
            if not pages:
                log.warning(f"  No pages found for domain: {domain}")
                continue

            # Stride sampling: covers site sections rather than just first N pages
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
    """
    Write README.txt to out_dir — a plain-text reading guide for the corpus.

    Explains the three export types, their intended use, and a structured
    note-taking protocol for Step 2 (pattern identification → codebook →
    dictionary for Step 3 theory building).

    Args:
        out_dir      : Path to output/step_2/.
        generated_at : Timestamp string.
    """
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
    """
    Entry point — runs all three exports and writes README.txt.

    Checks for database existence and three prerequisite tables before
    proceeding.  Raises RuntimeError immediately if any prerequisite is
    missing (unlike the visualisation scripts, which skip individual
    figures gracefully — here a missing table means all exports would fail).

    Output summary:
      A_top_pages/  — {len(CLIENT_TERMS) + len(WORKER_TERMS)} × TOP_PAGES_N pages
      B_kwic/       — {same} × (all audience pages scanned for KWIC)
      C_within_pair/ — {len(PAIRS)} × 2 per-side files
      README.txt

    Prerequisites:
      01_prepare.py          (corpus_view)
      02_step1_frequency.py  (platform_term_counts, keyness_results)
    """
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("04_step2_export.py — Step 2 Reading Corpus")
    log.info("=" * 60)

    conn         = get_conn()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Hard prerequisite check — abort if tables are missing
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
