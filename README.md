# Web Scraper

My web scraper that collects, stores, and analyzes text content from AI data-labeling and crowdwork platforms (e.g. Appen, Scale AI, Toloka, Outlier, Prolific). 

---

## Project Structure

```
├── src/
│   ├── main.py               # Entry point — interactive menu or CLI
│   ├── scraper.py            # Playwright-based scraper (JS-heavy sites)
│   ├── database.py           # SQLite manager (websites, pages, links)
│   ├── sitemap_robots.py     # Sitemap + robots.txt parser
│   ├── preprocess.py         # NLP preprocessing → TF-IDF & embedding tables
│   ├── tfidf_analysis.py     # TF-IDF analysis over scraped corpus
│   ├── validator.py          # URL and content validation helpers
│   ├── logger.py             # Logging setup
│   ├── db_stats.py           # Database statistics report generator
│   └── test.py               # Setup verification tests
├── config/
│   └── config.py             # Website configs, DB path, scraper settings
├── data/
│   └── scraping.db           # SQLite database (generated)
└── logs/                     # Per-domain log files (generated)
```

---

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
python -m playwright install chromium
python -m spacy download en_core_web_sm
```

**Verify setup:**
```bash
python test.py
```

---

## Usage

**Interactive mode:**
```bash
python main.py
```

**CLI mode:**
```bash
# Scrape a single website
python main.py mindrift.ai

# Scrape all configured websites (limit pages per site)
python main.py --all 100

# Show database stats
python main.py --stats
```

**Preprocess scraped text for NLP:**
```bash
python preprocess.py
```

**Generate a database statistics report:**
```bash
python db_stats.py data/scraping.db report.txt
```

---

## Pipeline

```
config.py  →  sitemap_robots.py  →  scraper.py  →  database.py
                                                         ↓
                                                   preprocess.py
                                                         ↓
                                                  tfidf_analysis.py
```

1. **Scraper** reads site configs, fetches URLs from sitemaps, and uses Playwright to render JS-heavy pages.
2. **Database** stores pages with full text, metadata, extracted links, and CSS color palettes.
3. **Preprocessor** reads raw pages and writes two NLP-ready tables: `pages_tfidf` (lemmatized unigrams + bigrams) and `pages_embedding` (clean prose for sentence-transformers, tokenized text for Word2Vec/fastText).
4. **TF-IDF analysis** runs over the processed corpus.

---

## Database Schema

| Table | Description |
|---|---|
| `websites` | Domain, name, type/category, last scraped |
| `pages` | URL, title, text content, depth, status, CSS colors |
| `links` | Source page → target URL with anchor text |
| `pages_tfidf` | Lemmatized tokens + filtered bigrams per page |
| `pages_embedding` | Clean text and tokenized text for embeddings |

---

## Current Data

As of the latest scrape: **27 websites · 7802 pages · ~9.1M words**

Categories covered: Managed Enterprise BPO, Algorithmic Crowd Markets, Impact-Sourcing Firms.

---

## Requirements

- Python 3.10+
- Playwright (Chromium)
- spaCy `en_core_web_sm`
- beautifulsoup4, requests, lxml