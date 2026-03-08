# DarkSideofAI

A research scraper and NLP analysis pipeline for studying the language of AI data-labeling and crowdwork platforms. The project collects, stores, and analyzes the public web presence of companies that supply training data for AI models — comparing how they speak to **enterprise clients (B2B)** versus how they speak to **workers (B2W)**.

The analytical framework follows Nelson (2020)'s corpus-assisted discourse analysis methodology: keyness scoring (log-likelihood G²), co-occurrence profiling (PMI), and cross-platform comparison.

> **Current corpus:** 33 platforms · ~9,200 pages · ~11.5M words (raw) · ~5.5M tokens (preprocessed)

---

## Table of Contents

1. [Research Context](#research-context)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Pipeline Overview](#pipeline-overview)
7. [Database Schema](#database-schema)
8. [Module Reference](#module-reference)
9. [Analysis Scripts](#analysis-scripts)
10. [Corpus Statistics](#corpus-statistics)
11. [Platform Coverage](#platform-coverage)
12. [Logs & Outputs](#logs--outputs)
13. [Design Notes & Known Issues](#design-notes--known-issues)
14. [Dependencies](#dependencies)

---

## Research Context

This tool was built to support discourse analysis of the AI data-labeling industry — the largely invisible workforce that annotates, labels, and evaluates the training data behind large language models and AI systems.

The platforms in scope fall into three categories:

| Category | Description |
|---|---|
| **Managed Enterprise BPO** | Companies like Scale AI, Appen, iMerit — sell AI training data as a managed B2B service |
| **Algorithmic Crowd Markets** | Platforms like Toloka, Clickworker, Prolific, MTurk — operate marketplace-style gig economies |
| **Impact-Sourcing Firms** | Entities like Sama, CloudFactory, Digital Divide Data — emphasize social-impact or Global South workforce models |

Each platform typically maintains two distinct web presences: one aimed at enterprise clients ("use our data to train your models") and one aimed at workers/contributors ("join our platform and earn money"). This project scrapes both sides and applies computational methods to surface how the language differs.

---

## Project Structure

```
DarkSideofAI/
├── src/
│   ├── main.py                  # Entry point — interactive menu or CLI
│   ├── scraper.py               # Playwright scraper for JS-heavy sites (stealth mode)
│   ├── crawler.py               # BFS link crawler for sites without sitemaps
│   ├── database.py              # SQLite manager — all read/write to scraping.db
│   ├── sitemap_robots.py        # sitemap.xml + robots.txt parser
│   ├── preprocess.py            # NLP preprocessing → pages_tfidf & pages_embedding
│   ├── tfidf_analysis.py        # TF-IDF frequency analysis over corpus
│   ├── 01_prepare.py            # Creates platforms table + corpus_view (run once)
│   ├── 02_step1_frequency.py    # Nelson Step 1: keyness + co-occurrence analysis
│   ├── 03_visualise_step1.py    # Visualisation of keyness/co-occurrence output
│   ├── validator.py             # URL and page-content validation helpers
│   ├── logger.py                # Logging setup (per-domain log files)
│   ├── db_stats.py              # Database statistics report generator
│   ├── analyze_db.py            # Ad-hoc database inspection utilities
│   ├── find_duplicates.py       # Near-duplicate page detection
│   └── test.py                  # Environment / setup verification
├── config/
│   └── config.py                # Website configs, DB path, scraper constants
├── data/
│   └── scraping.db              # SQLite database (generated — not in repo)
├── logs/
│   ├── scraper_<domain>.log     # Per-domain scrape logs (generated)
│   ├── db_stats/
│   │   └── db_stats_report.txt  # Latest stats report
│   └── duplicate/
│       ├── duplicate_report.json
│       └── duplicate_report.txt
├── corpus_dashboard.html        # Auto-generated corpus overview dashboard
├── duplicate_report.json        # Root-level duplicate report (legacy location)
├── duplicate_report.txt
├── db_stats_report.txt
└── requirements.txt
```

---

## Setup & Installation

**Prerequisites:** Python 3.10+, pip

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Playwright browser

```bash
python -m playwright install chromium
```

### 3. Install spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 4. Verify everything works

```bash
cd src
python test.py
```

This runs environment checks: database connectivity, Playwright launch, spaCy model loading, and URL validation.

---

## Configuration

All site configurations live in `config/config.py`. There is no separate config file — the `WEBSITES` dictionary is the source of truth for which domains are scraped, how they are classified, and what metadata they carry.

### Global Constants

| Constant | Default | Description |
|---|---|---|
| `DB_PATH` | `data/scraping.db` | Path to SQLite database |
| `RAW_DATA_DIR` | `data/raw/` | JSON backup directory for raw scraped pages |
| `HEADLESS` | `True` | Run Playwright in headless mode |
| `PAGE_WAIT_TIMEOUT` | `30000` | Page load timeout in ms |
| `NETWORK_IDLE_TIMEOUT` | `10000` | networkidle wait timeout in ms |
| `RATE_LIMIT_DELAY` | `1.5` | Default seconds between requests |
| `RETRY_ATTEMPTS` | `3` | How many times to retry a failed page |
| `RETRY_DELAY` | `2` | Seconds between retry attempts |

### Website Config Schema

Each entry in the `WEBSITES` dict uses the domain as key and this structure as value:

```python
WEBSITES = {
    "example.com": {
        "name":       "Example Platform",       # Human-readable name
        "base_url":   "https://example.com",    # Base URL for scraping
        "type":       "Managed Enterprise BPO", # Free-text category (parsed by 01_prepare.py)
        "audience":   "client",                 # "client", "worker", or "both"
        "rate_limit": 2.0,                      # Seconds between requests (overrides global)
        "max_depth":  3,                        # Max link-follow depth (crawler mode only)
    }
}
```

**`audience` values:**

| Value | Meaning |
|---|---|
| `"client"` | B2B-facing — selling data services to enterprise AI teams |
| `"worker"` | B2W-facing — recruiting annotators, labelers, contributors |
| `"both"` | Single domain serving both audiences |

### Adding a New Website

1. Add an entry to `WEBSITES` in `config/config.py`
2. Choose the appropriate `audience` label
3. Set a conservative `rate_limit` (2.0+ for production sites)
4. Run `python main.py <domain>` to test-scrape
5. After scraping, re-run `01_prepare.py` to refresh the `platforms` table and `corpus_view`

---

## Usage

All commands are run from the `src/` directory.

### Interactive Menu

```bash
python main.py
```

Presents a numbered menu:

```
1. Scrape a single website (sitemap)
2. Scrape a single website (crawler — no sitemap)
3. Scrape all websites
4. Show database statistics
5. List configured websites
6. Exit
```

### CLI Mode

```bash
# Scrape a single domain (sitemap-based discovery)
python main.py mindrift.ai

# Scrape with a page limit
python main.py mindrift.ai 200

# Scrape all configured websites (100 pages each)
python main.py --all 100

# Scrape using BFS crawler (for sites without a sitemap)
python main.py --crawl crowdgen.com 300

# Print database stats
python main.py --stats
```

### NLP Preprocessing

After scraping, run preprocessing to populate the `pages_tfidf` and `pages_embedding` tables:

```bash
python preprocess.py
```

This uses spaCy (`en_core_web_sm`) to lemmatize, remove stopwords (with a custom whitelist for domain-specific terms), and extract filtered bigrams. The output is two NLP-ready tables used by all downstream analysis scripts.

### Database Statistics Report

```bash
python db_stats.py data/scraping.db report.txt
```

Produces a detailed text report covering: pages per website, word counts, TF-IDF token counts, bigram counts, embedding table coverage, and audience breakdowns.

### Duplicate Detection

```bash
python find_duplicates.py
```

Identifies near-duplicate pages within the corpus using content hashing or similarity. Output is written to `logs/duplicate/duplicate_report.json`. `preprocess.py` reads this file to exclude duplicate pages (keeping the longest version per cluster).

---

## Pipeline Overview

The full pipeline runs in the following order:

```
config.py
    │
    ▼
sitemap_robots.py  ──or──  crawler.py
    │                           │
    └───────────┬───────────────┘
                ▼
           scraper.py
          (Playwright)
                │
                ▼
          database.py
         (scraping.db)
                │
           ┌────┴─────┐
           ▼           ▼
     preprocess.py   db_stats.py
           │
      ┌────┴─────────────────┐
      ▼                       ▼
 pages_tfidf           pages_embedding
      │
      ▼
 01_prepare.py
 (platforms + corpus_view)
      │
      ▼
 02_step1_frequency.py
 (keyness + co-occurrence)
      │
      ▼
 03_visualise_step1.py
```

### Stage Descriptions

**Stage 1 — URL Discovery**
`sitemap_robots.py` fetches `robots.txt`, extracts the crawl delay, then recursively parses `sitemap.xml` (including sitemap index files). For sites without a sitemap, `crawler.py` performs a BFS traversal following internal links, respecting `max_depth` and `max_pages` limits.

**Stage 2 — Scraping**
`scraper.py` uses Playwright (Chromium) in stealth mode to render JS-heavy pages. It respects rate limits, handles 429 responses with exponential backoff, and relaunches the browser if it crashes (common on sites like Scale AI). Each page's HTML is rendered, parsed with BeautifulSoup, and CSS color palettes are extracted via `page.evaluate()`. Raw page data is backed up as JSON to `data/raw/<domain>/`.

**Stage 3 — Storage**
`database.py` writes to SQLite. URL deduplication uses SHA-256 hashing. If a page already exists, its content is updated via `ON CONFLICT DO UPDATE`.

**Stage 4 — Preprocessing**
`preprocess.py` reads raw `text_content` from the `pages` table in batches of 200. spaCy lemmatizes text, removes stopwords (with a whitelist preserving domain-relevant terms like "worker", "annotator", "label"), and extracts bigrams that appear at least 3 times in the full corpus. Two output tables are written: `pages_tfidf` (unigrams + bigrams) and `pages_embedding` (clean prose for sentence-transformers, tokenized text for Word2Vec/fastText).

**Stage 5 — Analysis Preparation**
`01_prepare.py` creates a `platforms` table from the `WEBSITES` config (adding `platform_type`, `company_id`, and `hq_region` fields), then creates `corpus_view` — a pre-joined view that all analysis scripts use as their single entry point.

**Stage 6 — Corpus Analysis**
`02_step1_frequency.py` implements Nelson (2020) Step 1: it computes log-likelihood G² scores (keyness) for every term comparing client vs worker pages, both cross-platform and within platform pairs (e.g., appen.com vs crowdgen.com). It also computes PMI-based co-occurrence profiles for the top-N key terms.

---

## Database Schema

The database is a single SQLite file at `data/scraping.db`. Tables are created on first run by `database.py`, and additional tables/views are added by preprocessing and analysis scripts.

### Core Tables (created by `database.py`)

#### `websites`

Stores one record per scraped domain.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `domain` | TEXT UNIQUE | e.g. `scale.com` |
| `name` | TEXT | Human-readable name from config |
| `base_url` | TEXT | Root URL |
| `website_type` | TEXT | Free-text type from config (e.g. `Managed Enterprise BPO`) |
| `created_at` | TIMESTAMP | When the website was first added |
| `last_scraped` | TIMESTAMP | Timestamp of most recent completed scrape |

#### `pages`

One record per scraped URL. The main content table.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `website_id` | INTEGER FK → `websites.id` | Parent website |
| `url` | TEXT UNIQUE | Full page URL |
| `url_hash` | TEXT UNIQUE | SHA-256 of URL (used for deduplication) |
| `title` | TEXT | Page `<title>` or first `<h1>` |
| `text_content` | TEXT | Extracted body text (scripts/styles stripped) |
| `directory` | TEXT | URL path segment (e.g. `/blog`) |
| `html_element` | TEXT | Tag name of the main content container (`main`, `article`, `body`, etc.) |
| `page_depth` | INTEGER | Depth from base URL (sitemap = 0, crawler = actual depth) |
| `status_code` | INTEGER | HTTP response status |
| `content_length` | INTEGER | Raw HTML byte count |
| `css_colors` | TEXT | JSON blob of extracted CSS colors (see below) |
| `scraped_at` | TIMESTAMP | When this page was scraped |

**`css_colors` JSON structure:**

```json
{
  "background_colors": ["rgb(255,255,255)", "rgb(18,18,18)"],
  "text_colors": ["rgb(33,33,33)", "rgb(100,100,100)"],
  "link_colors": ["rgb(0,102,204)"],
  "button_colors": ["rgb(255,87,34)", "rgb(255,255,255)"]
}
```

Colors are extracted via `window.getComputedStyle()` on common elements (`body`, `header`, `main`, `section`, `nav`, `footer`, `p`, `h1–h6`, `a`, `button`).

#### `links`

One record per outbound hyperlink found on a page.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `source_page_id` | INTEGER FK → `pages.id` | Page where the link was found |
| `target_url` | TEXT | Absolute URL of the link target |
| `anchor_text` | TEXT | Visible link text |
| `link_type` | TEXT | `internal` (same domain) or `external` |

> **Note:** Links for a page are fully replaced on re-scrape (`DELETE` then re-insert).

### NLP Tables (created by `preprocess.py`)

#### `pages_tfidf`

One record per page, holding NLP-ready token data for frequency and TF-IDF analysis.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `page_id` | INTEGER FK → `pages.id` | Source page |
| `domain` | TEXT | Convenience copy of domain |
| `audience` | TEXT | Deprecated — use platforms table via corpus_view |
| `tokens` | TEXT | JSON array of lemmatized unigrams (stopwords removed) |
| `bigrams` | TEXT | JSON array of filtered bigrams (min freq 3 in corpus) |

#### `pages_embedding`

One record per page, holding text prepared for embedding models.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `page_id` | INTEGER FK → `pages.id` | Source page |
| `domain` | TEXT | Convenience copy of domain |
| `clean_text` | TEXT | Lightly cleaned prose — punctuation intact, for sentence-transformers |
| `tokenized_text` | TEXT | Space-separated lemmas — for Word2Vec / fastText |

### Analysis Tables (created by `01_prepare.py` and `02_step1_frequency.py`)

#### `platforms`

Structured metadata derived from `config.py`'s `WEBSITES` dictionary. Created by `01_prepare.py`.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `domain` | TEXT UNIQUE | Matches `websites.domain` |
| `platform_type` | TEXT | Canonical type: `crowd_market`, `enterprise_bpo`, `impact_sourcing`, `unknown` |
| `company_id` | TEXT | Groups paired domains (e.g. `appen` for both `appen.com` and `crowdgen.com`) |
| `audience` | TEXT | `client`, `worker`, or `both` (from config) |
| `hq_region` | TEXT | `north` or `south` (derived from `HQ_REGION_RULES` in `01_prepare.py`) |

#### `corpus_view` (SQL VIEW)

A pre-joined view that is the single entry point for all analysis scripts. Joins `pages_tfidf → pages → websites → platforms`. Created by `01_prepare.py`.

Exposes: `page_id`, `url`, `domain`, `platform_type`, `company_id`, `audience`, `hq_region`, `tokens`, `bigrams`.

#### `keyness_results`

Output of `02_step1_frequency.py`. Log-likelihood G² scores for every term.

| Column | Type | Description |
|---|---|---|
| `comparison` | TEXT | `cross_platform` or a `company_id` for within-pair analysis |
| `term` | TEXT | The unigram or bigram |
| `term_type` | TEXT | `unigram` or `bigram` |
| `ll_score` | REAL | Log-likelihood G² — positive = client-distinctive, negative = worker-distinctive |
| `freq_client` | INTEGER | Raw count in client pages |
| `freq_worker` | INTEGER | Raw count in worker pages |
| `rel_freq_client` | REAL | Frequency per 1,000 tokens (client) |
| `rel_freq_worker` | REAL | Frequency per 1,000 tokens (worker) |

#### `cooccurrence_results`

PMI-based co-occurrence profiles for top-N key terms.

| Column | Type | Description |
|---|---|---|
| `comparison` | TEXT | Same as `keyness_results.comparison` |
| `audience` | TEXT | `client` or `worker` |
| `focus_term` | TEXT | The key term being profiled |
| `collocate` | TEXT | The co-occurring term |
| (additional PMI/freq columns) | — | PMI score, co-occurrence count, etc. |

#### `platform_term_counts`

Per-domain term frequency table, used for within-pair comparison.

### Indexes

```sql
CREATE INDEX idx_pages_website     ON pages(website_id);
CREATE INDEX idx_pages_url_hash    ON pages(url_hash);
CREATE INDEX idx_links_source      ON links(source_page_id);
```

---

## Module Reference

### `scraper.py` — `WebScraper`

Playwright-based scraper with stealth mode, rate limiting, retry logic, and CSS color extraction.

**Key methods:**

| Method | Description |
|---|---|
| `scrape_website(max_pages)` | Main entry point. Fetches URLs from sitemap, iterates, saves to DB. |
| `scrape_page(page, url, depth)` | Scrapes a single URL. Returns a `page_data` dict or `None` on failure. |
| `_extract_text_from_page(page)` | Waits for JS hydration, extracts text + links + CSS colors via BeautifulSoup + Playwright evaluate. |
| `_extract_css_colors(page, soup)` | Returns a dict of background/text/link/button colors. |
| `_wait_for_rate_limit()` | Enforces minimum delay between requests. |
| `_save_raw_data(page_data)` | Backs up page data as JSON to `data/raw/<domain>/`. |

**Stealth measures:**
- Launches with `--disable-blink-features=AutomationControlled`
- Sets realistic `User-Agent`, viewport, locale, timezone
- Injects JS to mask `navigator.webdriver`, spoof `navigator.plugins` and `window.chrome`
- Adds random 1.5–3.5s human-like pause per page
- Uses a fresh browser page per URL to avoid stale state

**Handling 429 (rate limited):** Reads `Retry-After` header; defaults to `30 × 2^attempt` seconds. Entire browser is relaunched if it crashes mid-scrape.

---

### `crawler.py` — `LinkCrawler`

BFS crawler for sites without a usable sitemap (e.g. crowdgen.com, telusinternational.ai).

**Constructor params:**

| Param | Default | Description |
|---|---|---|
| `base_url` | — | Starting URL |
| `max_pages` | 1000 | Maximum URLs to discover |
| `max_depth` | 8 | Maximum BFS depth |
| `rate_limit` | 1.5 | Seconds between requests |
| `exclude_extensions` | (see code) | File types to skip (.pdf, .jpg, .css, etc.) |

**Key methods:**

| Method | Description |
|---|---|
| `discover_urls()` | Runs BFS, returns ordered list of internal URLs |
| `_fetch_links(url)` | Fetches page with `requests`, extracts internal hrefs via BeautifulSoup |
| `_normalize(url)` | Strips fragments, trailing slashes, skips excluded extensions |

The crawler is injected into `WebScraper` by monkey-patching `SitemapRobotsParser.get_all_urls` so the scraper's sitemap step returns crawled URLs instead.

---

### `database.py` — `Database`

Thin SQLite wrapper. All scraper writes go through this class.

**Key methods:**

| Method | Description |
|---|---|
| `add_website(domain, name, base_url, website_type)` | Upserts website record, returns `website_id` |
| `add_page(website_id, url, title, ...)` | Inserts page or updates on URL hash conflict |
| `add_link(source_page_id, target_url, anchor_text, link_type)` | Clears old links for page, inserts new ones |
| `page_exists(url)` | SHA-256 lookup — returns `True` if URL already scraped |
| `get_stats(website_id)` | Returns `{total_pages, total_bytes, total_mb}` |
| `update_website_last_scraped(website_id)` | Stamps `last_scraped` on website record |

---

### `sitemap_robots.py` — `SitemapRobotsParser`

**Key methods:**

| Method | Description |
|---|---|
| `get_robots_txt()` | Fetches `/robots.txt`, parses crawl delay |
| `find_sitemap_url(robots_content)` | Extracts sitemap URL from robots.txt or probes common paths |
| `parse_sitemap(sitemap_url)` | Recursively parses sitemap index + leaf sitemaps, returns flat URL list |
| `get_all_urls()` | Top-level: returns `(url_list, crawl_delay)` |

---

### `preprocess.py`

Reads `pages` in batches of 200, applies spaCy NLP, writes `pages_tfidf` and `pages_embedding`.

**Key settings (hardcoded at top of file):**

| Constant | Default | Description |
|---|---|---|
| `DB_PATH` | `data/scraping.db` | Database path |
| `BATCH_SIZE` | 200 | Pages processed per batch |
| `MIN_BIGRAM_FREQ` | 3 | Min corpus frequency for a bigram to be kept |
| `DUPLICATES_FILE` | `logs/duplicate/duplicate_report.json` | If set, duplicate pages are excluded |

**`STOPWORD_WHITELIST`** — a set of words that spaCy would normally remove as stopwords but are retained because they carry meaning in AI-labor discourse (e.g. `worker`, `annotator`, `labeler`, `remote`, `task`, `quality`). Edit this list in `preprocess.py` to adjust for your research focus.

---

### `validator.py`

Stateless validation helpers. All return `(bool, str)` tuples.

| Function | Description |
|---|---|
| `validate_url(url)` | Checks scheme and netloc |
| `validate_text_content(text, min_length=50)` | Rejects empty or very short extractions |
| `validate_page_data(page_data)` | Checks all required fields + URL + content |
| `validate_sitemap(sitemap_urls)` | Checks sitemap returned at least one URL |

---

### `logger.py`

Sets up per-domain rotating log files under `logs/scraper_<domain>.log`.

Helper functions: `log_scrape_start`, `log_scrape_success`, `log_scrape_error`, `log_validation_result`.

---

### `db_stats.py`

Standalone script that queries the database and writes a formatted report to a text file.

```bash
python db_stats.py data/scraping.db logs/db_stats/report.txt
```

Report includes: websites per category, pages per website, word counts, TF-IDF token/bigram counts, audience breakdowns, embedding table coverage.

---

### `find_duplicates.py`

Detects near-duplicate pages within the corpus. Writes results to `logs/duplicate/duplicate_report.json`. The preprocessor reads this file and, per duplicate cluster, retains only the page with the longest `text_content`.

---

## Analysis Scripts

### `01_prepare.py` — Platform Metadata & Corpus View

Run **once** after preprocessing, before any analysis:

```bash
python 01_prepare.py
```

**What it does:**
1. Parses `WEBSITES` config to extract `platform_type`, `company_id`, and `hq_region` for each domain
2. Creates and populates the `platforms` table
3. Creates `corpus_view` — a SQL VIEW that pre-joins `pages_tfidf → pages → websites → platforms`
4. Runs diagnostics and logs any pages with missing platform matches

**`PAIR_RULES`** in `01_prepare.py` maps domain fragments to a shared `company_id`, grouping paired platforms under the same company:

```python
PAIR_RULES = {
    "appen":    "appen",
    "crowdgen": "appen",    # crowdgen.com is Appen's worker portal
    "toloka":   "toloka",
    "mindrift": "toloka",   # mindrift.ai is Toloka's worker portal
}
```

Add new pairs here when you add platforms that have both a B2B and a B2W domain.

**`HQ_REGION_RULES`** maps domain fragments to `"north"` or `"south"` (for Global North/South classification):

```python
HQ_REGION_RULES = {
    "sama":         "south",
    "imerit":       "south",
    "cloudfactory": "south",
    "defined":      "south",
    # Everything else defaults to "north"
}
```

---

### `02_step1_frequency.py` — Keyness & Co-occurrence (Nelson 2020 Step 1)

```bash
python 02_step1_frequency.py
```

Reads from `corpus_view`. Computes:

**Keyness (log-likelihood G²):**
- Positive score → term is over-represented in **client** pages
- Negative score → term is over-represented in **worker** pages
- Runs at two levels: cross-platform (all client vs all worker) and within-pair (per `company_id`)
- Output: `keyness_results` table

**Co-occurrence (PMI):**
- Computes collocate profiles for the top `TOP_N_COOC` (default 150) key terms
- Window size: ±5 tokens
- Minimum co-occurrence count: 10
- Output: `cooccurrence_results` table

**Key settings at top of file:**

| Constant | Default | Description |
|---|---|---|
| `TOP_N_COOC` | 150 | Terms to compute co-occurrence for |
| `MIN_TERM_FREQ` | 5 | Terms below this frequency are skipped |
| `WINDOW_SIZE` | 5 | ±N token co-occurrence window |
| `MIN_PMI_COFREQ` | 10 | Min co-occurrence count for PMI calculation |

---

### `03_visualise_step1.py` — Visualisation

```bash
python 03_visualise_step1.py
```

Reads from `keyness_results` and `cooccurrence_results`. Generates charts/tables for the keyness analysis output. See the script header for specific output formats.

---

### `tfidf_analysis.py`

Runs TF-IDF analysis over the preprocessed corpus. Useful for exploratory inspection of term importance per domain.

---

## Corpus Statistics

As of the latest scrape and preprocessing run (February 2026):

| Metric | Value |
|---|---|
| Websites scraped | 33 |
| Total pages | ~9,200 |
| Total words (raw) | ~11.5M |
| Total tokens (preprocessed) | ~5.5M |
| Total bigrams (preprocessed) | ~4.2M |
| Pages with embeddings | 7,529 (100% coverage) |

**Audience breakdown (preprocessed corpus):**

| Audience | Pages | Tokens | Avg tokens/page |
|---|---|---|---|
| client | 5,299 | 3,579,578 | 675 |
| both | 1,292 | 1,157,788 | 896 |
| worker | 929 | 777,625 | 837 |

---

## Platform Coverage

### Managed Enterprise BPO

| Domain | Pages | Words | Notes |
|---|---|---|---|
| defined.ai | 940 | 605,648 | Data marketplace |
| www.lxt.ai | 762 | 1,224,706 | Global AI training data |
| mindy-support.com | 672 | 676,765 | |
| scale.com | 618 | 615,294 | + Remotasks worker portal |
| www.appen.com | 551 | 490,809 | B2B side |
| www.opentrain.ai | 517 | 782,332 | |
| www.superannotate.com | 325 | 503,692 | High-skill labeling |
| outlier.ai | 227 | 243,785 | RLHF / expert freelancers |
| www.abaka.ai | 227 | 244,329 | |
| www.crowdworks.ai | 53 | 66,477 | Korean platform |
| flipside.ai | 17 | 2,125 | |
| surgehq.ai | 1 | 119 | Partial |
| www.telusinternational.com | 1 | 35 | B2B side; .ai is worker portal |

### Algorithmic Crowd Markets

| Domain | Pages | Words | Notes |
|---|---|---|---|
| crowdgen.com | 490 | 450,422 | Appen worker portal |
| www.clickworker.com | 410 | 736,496 | |
| www.prolific.com | 377 | 383,600 | Research participant recruitment |
| www.dataannotation.tech | 171 | 334,681 | RLHF gigs |
| www.oneforma.com | 143 | 127,168 | Centific worker portal |
| mindrift.ai | 113 | 234,767 | Toloka worker portal |
| www.microworkers.com | 109 | 116,710 | |
| www.alignerr.com | 63 | 852,887 | |
| toloka.ai | 50 | 31,122 | Platform side |
| joinstellar.ai | 20 | 9,528 | |
| www.remotasks.com | 10 | 22,690 | Scale AI worker portal |
| remoter.me | 8 | 13,250 | |
| www.mturk.com | 1 | 866 | Partial (bot-blocked) |
| www.telusinternational.ai | 1 | 4,217 | Worker portal |

### Impact-Sourcing Firms

| Domain | Pages | Words | Notes |
|---|---|---|---|
| imerit.net | 726 | 616,938 | India-HQ, social impact |
| www.cloudfactory.com | 440 | 467,594 | Nepal/Kenya operations |
| www.digitaldividedata.com | 415 | 495,790 | Global South focus |
| www.sama.com | 392 | 1,007,067 | Kenya-focused |
| humansintheloop.org | 346 | 147,759 | Conflict-affected workers |

---

## Logs & Outputs

| Path | Description |
|---|---|
| `logs/scraper_<domain>.log` | Per-domain scrape log with timestamps, status codes, validation results |
| `logs/db_stats/db_stats_report.txt` | Latest database statistics report |
| `logs/duplicate/duplicate_report.json` | Duplicate cluster data used by preprocessor |
| `logs/duplicate/duplicate_report.txt` | Human-readable duplicate report |
| `data/raw/<domain>/` | JSON backups of every scraped page (one file per page) |
| `corpus_dashboard.html` | Auto-generated HTML corpus overview |

---

## Design Notes & Known Issues

**URL deduplication uses SHA-256 hashing** rather than string comparison, which is faster at scale but means URLs that differ only in case or trailing slash are treated as different. Normalize URLs in `config.py` `base_url` entries to avoid collecting near-duplicate pages.

**`audience` in `pages_tfidf` is unreliable** — it was derived from URL pattern matching in early versions of `preprocess.py` and produces `unknown` for many pages. The `platforms` table (created by `01_prepare.py`) is the authoritative source of audience classification. All analysis scripts should join via `corpus_view`, not use the `pages_tfidf.audience` column directly.

**The `links` table is destructively updated** — every re-scrape of a page deletes and re-inserts all its links. This means the link graph reflects the most recent scrape, not historical state.

**Playwright browser crashes** are handled by a full browser relaunch inside `scraper.py`. Scale.com in particular has been observed to kill the Chromium process mid-scrape.

**MTurk, SurgeHQ, and Telus International** have very low page counts — they either block scrapers aggressively or have sparse sitemaps. These domains are included in the corpus but have limited analytical weight.

**`pages_tfidf.bigrams`** only contains bigrams that appear ≥3 times across the full corpus. This threshold is set by `MIN_BIGRAM_FREQ` in `preprocess.py`. If you extend the corpus significantly, you may want to raise this threshold to reduce noise.

**Stealth measures may need updating** as bot-detection services evolve. If a site consistently returns 403s or empty pages, check the scrape log and consider adjusting the user-agent string or adding additional `context.add_init_script` patches.

---

## Dependencies

```
# Core scraping
playwright==1.40.0
beautifulsoup4==4.12.3
requests==2.31.0

# HTML parsing
lxml==5.1.0

# NLP
spacy>=3.0           # + en_core_web_sm model

# Utilities
python-dateutil==2.8.2
```

**Runtime requirements:**

- Python 3.10+
- Chromium (installed via `playwright install chromium`)
- spaCy `en_core_web_sm` (installed via `python -m spacy download en_core_web_sm`)
- SQLite3 (bundled with Python)

---

*Last corpus update: February 2026*
