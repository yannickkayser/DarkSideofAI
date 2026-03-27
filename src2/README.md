# src2 — Refined Analysis Pipeline

This folder contains the corrected and streamlined version of the data processing
and analysis pipeline.  The original `src/` folder is preserved unchanged as a
backup and reference.

## Why src2 exists

Several logical errors were identified in the `src/` pipeline after running the
full analysis.  The most significant produced spurious co-occurrence pairs (e.g.
"work–baby") that do not reflect genuine linguistic proximity in the corpus.
`src2/` fixes these errors without altering any raw scraped data.

---

## Bugs fixed relative to src/

### Bug 1 — False co-occurrences from flat-page token lists (CRITICAL)

**Root cause:** `scraper.py` stores page content as a single flat string extracted
with BeautifulSoup's `get_text(separator=' ')`.  On JS-heavy pages, `<main>` or
`<body>` contains hero text, navigation, feature blocks, testimonials, and footer
elements concatenated in document order.  `preprocess.py` then tokenised this as
one flat token list, so the ±5 window in `build_cooccurrence_index` could pair
tokens from completely unrelated page sections — producing co-occurrences like
"work–baby" that have no basis in the actual text.

**Fix (src2/preprocess.py):** spaCy's `sentencizer` component is added to the
pipeline.  Each page is tokenised sentence-by-sentence, and the resulting
`segments` (a list of per-sentence token lists) are stored in a new
`pages_tfidf.segments` column.  Bigrams are also formed only within sentences,
preventing cross-boundary bigrams.

**Fix (src2/02_step1_frequency.py):** `build_cooccurrence_index` now receives
the per-sentence segments and applies the window strictly within each sentence.
Tokens from different sentences are never paired.

### Bug 2 — Window size mismatch

**Root cause:** The methodology specifies a ±15 token window for co-occurrence
(see analysis strategy document).  The code used `WINDOW_SIZE = 5`.

**Fix:** `WINDOW_SIZE = 15` in `src2/02_step1_frequency.py`.

### Bug 3 — `clean_for_embedding` identical to `clean_raw`

**Root cause:** Both functions had identical implementations despite different
documented purposes.  `clean_for_embedding` was supposed to preserve sentence
structure for transformer models, but also stripped numbers.

**Fix (src2/preprocess.py):** `clean_for_embedding` now preserves punctuation and
numbers.  `clean_raw` continues to strip them for keyness/frequency analysis.

---

## Pipeline run order

```
python3 src2/preprocess.py              # adds segments column, re-derives tokens
python3 src2/01_prepare.py              # recreates corpus_view (now includes segments)
python3 src2/01_prepare_additions.py    # rebuilds exclusion tables
python3 src2/02_step1_frequency.py      # keyness + segment-aware co-occurrence
# ... (02b, 02c, 03, 04, 05 to follow)
```

**Note:** `src2/preprocess.py` uses `INSERT OR REPLACE` on `pages_tfidf`, so it
safely re-derives all token data from the raw `pages.text_content` without
deleting any scraped data.  Running it updates unigrams, bigrams, and segments
for all pages.

---

## What is NOT changed in src2

- Scraping scripts (`crawler.py`, `scraper.py`, `sitemap_robots.py`, `main.py`)
  — the raw data is not modified.
- Database schema for `pages`, `websites`, `links` tables — untouched.
- `config/config.py` — audience assignments and WEBSITES map are unchanged.
- All outputs and visualisations in `outputs/` and `STMAnalysis/` — these will
  be regenerated once the fixed pipeline has been run.
