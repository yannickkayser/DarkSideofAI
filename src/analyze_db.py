"""
Descriptive statistics + topic modeling for scraped website database.
Outputs a self-contained HTML dashboard.

Improvements over previous version:
  - Structural page filter: token_count, page_depth, content_length thresholds
    remove cookie banners, nav stubs, auth pages before LDA sees any documents.
  - Boilerplate token blocklist: removes cookie/GDPR/nav/auth tokens that survive
    lemmatization but signal page chrome rather than content.
  - Stratified LDA: when pages_tfidf is available, runs a separate model per
    audience ('worker' / 'client') so topics reflect content themes within each
    population, not the population split itself.
  - Coherence-guided topic count: tries N=4..10, picks the elbow on CV coherence
    (falls back to N_TOPICS if gensim is unavailable).


Usage: python analyze_db.py <path_to_database.db> [output.html]
"""
import sqlite3
import sys
import re
import json
import math
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# ── Optional coherence (gensim) ─────────────────────────────────────────────
try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False


# ── Config ───────────────────────────────────────────────────────────────────
N_TOPICS          = 6     # fallback if coherence search is skipped
N_TOPICS_MIN      = 4     # coherence search range low
N_TOPICS_MAX      = 10    # coherence search range high
N_TOP_WORDS       = 12
N_TOP_PAGES       = 5

# Structural filters — pages failing any of these are excluded from LDA
MIN_TOKEN_COUNT   = 80    # removes stubs, cookie banners, nav pages (~15 tokens)
MIN_CONTENT_BYTES = 500   # removes near-empty structural pages
MIN_PAGE_DEPTH    = 0     # removes homepages (often pure navigation hubs)
MAX_PAGE_DEPTH    = 5     # removes deep pagination / index cruft

# Raw-text fallback threshold (used when pages_tfidf does not exist)
MIN_RAW_LENGTH    = 300

# ── Boilerplate token blocklist ───────────────────────────────────────────────
# Tokens that survive lemmatization but signal page chrome, not content.
# Applied as a post-filter on unigrams/bigrams from pages_tfidf before LDA.
BOILERPLATE = {
    # Cookie / GDPR chrome
    'cookie', 'cookies', 'consent', 'gdpr', 'accept', 'preference', 'preferences',
    'tracking', 'functional', 'necessary', 'opt', 'optout', 'optin',
    'analytics', 'statistic', 'statistics', 'personalise', 'personalize',
    # Navigation chrome
    'menu', 'navigation', 'nav', 'header', 'footer', 'sidebar', 'breadcrumb',
    'click', 'button', 'link', 'home', 'back', 'next', 'previous', 'prev',
    'toggle', 'dropdown', 'hamburger', 'icon', 'logo',
    # Auth boilerplate
    'login', 'logout', 'signin', 'signup', 'sign_in', 'sign_up',
    'register', 'password', 'username', 'forgot', 'reset',
    'verify', 'verification', 'authenticate', 'authentication',
    # Generic web chrome
    'search', 'filter', 'sort', 'page', 'result', 'results',
    'load', 'loading', 'error', 'submit', 'continue', 'cancel',
    'close', 'open', 'show', 'hide', 'expand', 'collapse',
    # Legal boilerplate
    'privacy', 'term', 'terms', 'policy', 'policies', 'copyright',
    'reserved', 'condition', 'conditions', 'disclaimer', 'notice',
    'gdpr', 'ccpa', 'regulation',
    # Generic site scaffolding
    'read_more', 'learn_more', 'click_here', 'find_out',
    'subscribe', 'newsletter', 'follow', 'share', 'tweet',
    'comment', 'reply', 'like', 'dislike',
    # months of the year
    'january','february','march','april','may','june','july','august',
    'september','october','november','december',
}


# ── DB helpers ───────────────────────────────────────────────────────────────
def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def clean(text):
    text = re.sub(r'\s+', ' ', text or '')
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower().strip()


def table_exists(cur, name):
    return cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


# ── Token helpers ─────────────────────────────────────────────────────────────
def apply_blocklist(tokens):
    """Remove boilerplate tokens and blocklisted bigrams from a token list."""
    out = []
    for t in tokens:
        parts = t.split('_')
        # drop if any part of a unigram or bigram is boilerplate
        if any(p in BOILERPLATE for p in parts):
            continue
        out.append(t)
    return out


def build_doc(row, title_boost=2):
    """Reconstruct a document string from pages_tfidf row, blocklist applied."""
    uni   = json.loads(row['unigrams'] or '[]')
    big   = json.loads(row['bigrams']  or '[]')
    title = clean(row['title'] or '').split()
    tokens = title * title_boost + apply_blocklist(uni + big)
    return ' '.join(tokens)


# ── Coherence-guided topic count ──────────────────────────────────────────────
def best_n_topics(docs_tokenized, n_min, n_max):
    """
    Compute CV coherence for n_min..n_max topics.
    Returns the n with the highest coherence score.
    Falls back to N_TOPICS if gensim unavailable or corpus too small.
    """
    if not HAS_GENSIM or len(docs_tokenized) < 20:
        return N_TOPICS

    dictionary = Dictionary(docs_tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(d) for d in docs_tokenized]

    scores = {}
    for n in range(n_min, n_max + 1):
        from gensim.models import LdaModel
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n,
                       passes=10, random_state=42)
        cm  = CoherenceModel(model=lda, texts=docs_tokenized,
                             dictionary=dictionary, coherence='c_v')
        scores[n] = cm.get_coherence()
        print(f"    coherence n={n}: {scores[n]:.4f}")

    best = max(scores, key=scores.get)
    print(f"  → Best topic count: {best} (coherence {scores[best]:.4f})")
    return best


# ── LDA runner ────────────────────────────────────────────────────────────────
def run_lda(docs, meta, label=''):
    """
    Fit TF-IDF + LDA on docs. Returns list of topic dicts.
    meta is a parallel list of row dicts for example attribution.
    """
    if len(docs) < 10:
        return []

    vec = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.90,
        stop_words=None,           # already cleaned by preprocess.py
        ngram_range=(1, 1),        # bigrams arrive as token_token strings
        token_pattern=r'[a-z][a-z_]+',
    )
    X = vec.fit_transform(docs)
    if X.shape[1] == 0:
        return []

    # coherence search on tokenized form
    docs_tok = [d.split() for d in docs]
    n = best_n_topics(docs_tok, N_TOPICS_MIN, min(N_TOPICS_MAX, len(docs) // 3))
    n = max(2, min(n, len(docs) // 3))

    print(f"  [{label}] Fitting LDA with {n} topics on {len(docs)} docs…")
    lda = LatentDirichletAllocation(
        n_components=n,
        max_iter=30,
        learning_method='batch',
        random_state=42,
    )
    lda.fit(X)

    terms      = vec.get_feature_names_out()
    doc_topics = lda.transform(X)
    topics     = []

    for t_idx in range(n):
        top_idx   = lda.components_[t_idx].argsort()[-N_TOP_WORDS:][::-1]
        top_words = [terms[i] for i in top_idx]
        dominant  = [i for i, _ in enumerate(doc_topics) if doc_topics[i].argmax() == t_idx]
        examples  = sorted(dominant,
                           key=lambda i: doc_topics[i][t_idx],
                           reverse=True)[:N_TOP_PAGES]

        topics.append({
            'label':    label,
            'index':    t_idx + 1,
            'keywords': top_words,
            'purpose':  infer_purpose(top_words),
            'count':    len(dominant),
            'pct':      round(100 * len(dominant) / len(docs), 1),
            'examples': [
                {
                    'title':    (meta[i].get('title') or 'untitled')[:70],
                    'url':      meta[i]['url'],
                    'domain':   meta[i].get('domain', ''),
                    'audience': meta[i].get('audience', ''),
                }
                for i in examples
            ],
        })

    return topics


# ── Data collection ───────────────────────────────────────────────────────────
def collect_all(conn):
    cur  = conn.cursor()
    data = {}

    # ── Overview ──────────────────────────────────────────────────────────────
    data['n_sites']  = cur.execute("SELECT COUNT(*) FROM websites").fetchone()[0]
    data['n_pages']  = cur.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    data['n_links']  = cur.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    data['total_mb'] = (
        cur.execute("SELECT COALESCE(SUM(content_length),0) FROM pages").fetchone()[0] or 0
    ) / 1e6

    # ── Per-site breakdown ────────────────────────────────────────────────────
    data['sites'] = [dict(r) for r in cur.execute("""
        SELECT w.domain, w.website_type,
               COUNT(p.id)                        AS n_pages,
               COALESCE(AVG(p.page_depth),0)      AS avg_depth,
               COALESCE(AVG(p.content_length),0)  AS avg_len,
               COALESCE(SUM(p.content_length),0)  AS total_len
        FROM websites w LEFT JOIN pages p ON p.website_id = w.id
        GROUP BY w.id ORDER BY n_pages DESC
    """).fetchall()]

    # ── URL structure ─────────────────────────────────────────────────────────
    urls = [r[0] for r in cur.execute("SELECT url FROM pages").fetchall()]
    seg_counter, depth_counter = Counter(), Counter()
    for url in urls:
        path  = re.sub(r'^https?://[^/]+', '', url).strip('/')
        parts = [p for p in path.split('/') if p]
        depth_counter[len(parts)] += 1
        for seg in parts:
            seg_counter[seg] += 1
    data['top_segments'] = seg_counter.most_common(15)
    data['depth_dist']   = sorted(depth_counter.items())

    # ── Content length distribution ───────────────────────────────────────────
    lengths = [r[0] for r in cur.execute(
        "SELECT content_length FROM pages WHERE content_length IS NOT NULL"
    ).fetchall()]
    if lengths:
        a = np.array(lengths, dtype=float)
        data['len_stats'] = {
            'min': int(a.min()), 'median': int(np.median(a)),
            'mean': int(a.mean()), 'max': int(a.max()),
        }
        buckets = [
            (0,       500,    '<500 B'),
            (500,     5000,   '0.5–5 KB'),
            (5000,    50000,  '5–50 KB'),
            (50000,   500000, '50–500 KB'),
            (500000,  1e12,   '>500 KB'),
        ]
        data['len_buckets'] = [
            {'label': lbl, 'count': int(((a >= lo) & (a < hi)).sum())}
            for lo, hi, lbl in buckets
        ]
    else:
        data['len_stats']   = {}
        data['len_buckets'] = []

    # ── HTTP status codes ─────────────────────────────────────────────────────
    data['status_codes'] = [
        {'code': str(r['code'] or 'unknown'), 'count': r['n']}
        for r in cur.execute("""
            SELECT COALESCE(status_code, 0) AS code, COUNT(*) AS n
            FROM pages GROUP BY code ORDER BY n DESC
        """).fetchall()
    ]

    # ── Topic modeling ────────────────────────────────────────────────────────
    use_preprocessed = table_exists(cur, 'pages_tfidf')
    data['topic_source'] = 'pages_tfidf' if use_preprocessed else 'pages (raw)'
    data['topics']              = []
    data['domain_topic_matrix'] = []
    data['filter_stats']        = {}

    if use_preprocessed:
        # Structural filter applied in SQL — cheap and fast
        all_rows = cur.execute(f"""
            SELECT t.page_id  AS id,
                   t.url,
                   t.unigrams,
                   t.bigrams,
                   t.audience,
                   t.token_count,
                   p.title,
                   p.page_depth,
                   p.content_length,
                   COALESCE(w.domain,
                       replace(replace(t.url,'https://',''),'http://',''))
                       AS domain
            FROM   pages_tfidf  t
            JOIN   pages        p ON p.id       = t.page_id
            LEFT   JOIN websites w ON w.id = p.website_id
            WHERE  t.token_count    >= {MIN_TOKEN_COUNT}
              AND  p.content_length >= {MIN_CONTENT_BYTES}
              AND  p.page_depth     >= {MIN_PAGE_DEPTH}
              AND  p.page_depth     <= {MAX_PAGE_DEPTH}
        """).fetchall()

        total_before = cur.execute("SELECT COUNT(*) FROM pages_tfidf").fetchone()[0]
        total_after  = len(all_rows)
        data['filter_stats'] = {
            'before': total_before,
            'after':  total_after,
            'removed': total_before - total_after,
            'thresholds': {
                'min_tokens':       MIN_TOKEN_COUNT,
                'min_bytes':        MIN_CONTENT_BYTES,
                'depth_range':      f"{MIN_PAGE_DEPTH}–{MAX_PAGE_DEPTH}",
            }
        }
        print(f"  Structural filter: {total_before} → {total_after} pages "
              f"({total_before - total_after} removed as boilerplate/stubs)")

        # Split by audience — run separate LDA models per stratum
        audiences = sorted({r['audience'] for r in all_rows if r['audience']})
        if len(audiences) > 1:
            print(f"  Audiences found: {audiences} — running stratified LDA")
            for aud in audiences:
                aud_rows = [r for r in all_rows if r['audience'] == aud]
                docs     = [build_doc(r) for r in aud_rows]
                meta     = [dict(r) for r in aud_rows]
                print(f"  [{aud}] {len(docs)} pages after filter")
                data['topics'] += run_lda(docs, meta, label=aud)
        else:
            # Single audience or no audience labels — run combined
            docs = [build_doc(r) for r in all_rows]
            meta = [dict(r) for r in all_rows]
            data['topics'] += run_lda(docs, meta, label='all')

        # Domain × topic matrix (across all topics, audience-aware)
        all_meta = [dict(r) for r in all_rows]
        domains  = sorted({m['domain'] for m in all_meta})

        # Re-run a single combined pass just for the matrix (lightweight)
        if all_meta:
            all_docs = [build_doc(r) for r in all_rows]
            vec_m    = TfidfVectorizer(
                max_features=3000, min_df=2, max_df=0.90,
                token_pattern=r'[a-z][a-z_]+',
            )
            Xm = vec_m.fit_transform(all_docs)
            n_combined = max(2, min(N_TOPICS, len(all_docs) // 3))
            lda_m = LatentDirichletAllocation(
                n_components=n_combined, max_iter=20,
                learning_method='batch', random_state=42,
            )
            lda_m.fit(Xm)
            dt_m = lda_m.transform(Xm)
            for domain in domains:
                idxs = [i for i, m in enumerate(all_meta) if m['domain'] == domain]
                if not idxs:
                    continue
                dist = dt_m[idxs].mean(axis=0).tolist()
                data['domain_topic_matrix'].append({
                    'domain': domain,
                    'dist':   dist,
                    'n':      n_combined,
                })

    else:
        # ── Raw fallback ─────────────────────────────────────────────────────
        rows = cur.execute("""
            SELECT p.id, p.url, p.title, p.text_content, w.domain
            FROM   pages p JOIN websites w ON w.id = p.website_id
            WHERE  LENGTH(COALESCE(p.text_content,'')) > ?
        """, (MIN_RAW_LENGTH,)).fetchall()

        docs = [clean(f"{r['title'] or ''} {r['title'] or ''} {r['text_content'] or ''}")
                for r in rows]
        meta = [dict(r) for r in rows]
        data['topics'] += run_lda(docs, meta, label='all')

        domains = sorted({m['domain'] for m in meta})
        if docs:
            vec_m = TfidfVectorizer(
                max_features=3000, min_df=2, max_df=0.90, stop_words='english',
            )
            Xm   = vec_m.fit_transform(docs)
            n_combined = max(2, min(N_TOPICS, len(docs) // 3))
            lda_m = LatentDirichletAllocation(
                n_components=n_combined, max_iter=20,
                learning_method='batch', random_state=42,
            )
            lda_m.fit(Xm)
            dt_m = lda_m.transform(Xm)
            for domain in domains:
                idxs = [i for i, m in enumerate(meta) if m['domain'] == domain]
                if not idxs:
                    continue
                dist = dt_m[idxs].mean(axis=0).tolist()
                data['domain_topic_matrix'].append({
                    'domain': domain, 'dist': dist, 'n': n_combined,
                })

    return data


# ── Purpose heuristics ────────────────────────────────────────────────────────
_PURPOSE_RULES = [
    (['blog', 'post', 'article', 'author', 'publish', 'story', 'write', 'read'],
     'Editorial / Blog', 'BLOG'),
    (['product', 'price', 'buy', 'cart', 'shop', 'order', 'checkout', 'purchase'],
     'E-commerce', 'SHOP'),
    (['doc', 'api', 'reference', 'parameter', 'function', 'method', 'code', 'endpoint'],
     'Technical Docs', 'DOCS'),
    (['task', 'annotate', 'annotation', 'label', 'label_task', 'review', 'submit_task'],
     'Annotation / Tasks', 'TASK'),
    (['worker', 'freelancer', 'earn', 'earning', 'payment', 'pay', 'rate', 'income'],
     'Worker / Earnings', 'WORK'),
    (['client', 'enterprise', 'solution', 'platform', 'service', 'pricing', 'plan'],
     'Client / Platform', 'PLAT'),
    (['automate', 'automation', 'ai', 'model', 'algorithm', 'machine', 'intelligence'],
     'AI / Automation', 'AI'),
    (['quality', 'accuracy', 'guideline', 'instruction', 'criteria', 'standard'],
     'Quality / Guidelines', 'QA'),
    (['contact', 'support', 'help', 'faq', 'question', 'answer', 'ticket'],
     'Support / Help', 'HELP'),
    (['about', 'team', 'mission', 'company', 'history', 'career', 'job'],
     'Company / About', 'CORP'),
    (['news', 'press', 'release', 'announcement', 'update', 'launch'],
     'News / Press', 'NEWS'),
    (['tutorial', 'guide', 'how', 'step', 'learn', 'course', 'lesson', 'training'],
     'Tutorials / Learning', 'LEARN'),
    (['privacy', 'term', 'legal', 'policy', 'gdpr', 'condition'],
     'Legal / Policy', 'LEGAL'),
]


def infer_purpose(words):
    word_set = set(words)
    best_label, best_tag, best_score = 'General Content', 'GEN', 0
    for keywords, label, tag in _PURPOSE_RULES:
        score = sum(1 for k in keywords
                    if k in word_set or any(k in w for w in word_set))
        if score > best_score:
            best_label, best_tag, best_score = label, tag, score
    return {'label': best_label, 'tag': best_tag}


# ── HTML template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Web Corpus · Analysis Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap');

:root {
  --bg:    #0d0f14;
  --surf:  #13161e;
  --surf2: #1a1e2a;
  --bdr:   #252a38;
  --acc:   #c8f060;
  --acc2:  #60c8f0;
  --acc3:  #f060c8;
  --acc4:  #f0c860;
  --text:  #e8eaf2;
  --muted: #6b7280;
  --serif: 'Instrument Serif', Georgia, serif;
  --mono:  'DM Mono', monospace;
  --sans:  'Outfit', sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-weight:300;line-height:1.6;min-height:100vh}
body::before{content:'';position:fixed;inset:0;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");pointer-events:none;z-index:1000;opacity:.35}

.shell{max-width:1340px;margin:0 auto;padding:0 2rem 6rem}

header{padding:4rem 0 3rem;border-bottom:1px solid var(--bdr);margin-bottom:3rem;display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:1rem}
.h-title{font-family:var(--serif);font-size:clamp(2.2rem,5vw,3.8rem);font-weight:400;line-height:1.1}
.h-title em{font-style:italic;color:var(--acc)}
.h-meta{font-family:var(--mono);font-size:.72rem;color:var(--muted);text-align:right;line-height:2}

.sl{font-family:var(--mono);font-size:.65rem;letter-spacing:.18em;text-transform:uppercase;color:var(--acc);margin-bottom:1.4rem;display:flex;align-items:center;gap:.8rem}
.sl::after{content:'';flex:1;height:1px;background:var(--bdr)}
.sl sub{color:var(--muted);font-size:.6rem;letter-spacing:0;text-transform:none}

.stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1px;background:var(--bdr);border:1px solid var(--bdr);border-radius:12px;overflow:hidden;margin-bottom:3.5rem}
.stat-cell{background:var(--surf);padding:1.6rem 1.8rem;transition:background .2s}
.stat-cell:hover{background:var(--surf2)}
.stat-val{font-family:var(--serif);font-size:2.6rem;line-height:1;color:var(--acc);margin-bottom:.3rem}
.stat-lbl{font-family:var(--mono);font-size:.68rem;color:var(--muted);letter-spacing:.08em}

.g2{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:3.5rem}
@media(max-width:900px){.g2{grid-template-columns:1fr}}

.card{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;padding:1.8rem;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--acc) 0%,transparent 100%);opacity:.6}
.card-title{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:1.4rem}
.chart-wrap{position:relative}

/* filter banner */
.filter-banner{background:var(--surf2);border:1px solid var(--bdr);border-left:3px solid var(--acc);border-radius:8px;padding:1rem 1.4rem;margin-bottom:2rem;font-family:var(--mono);font-size:.75rem;color:var(--muted);display:flex;gap:2rem;flex-wrap:wrap;align-items:center}
.filter-banner strong{color:var(--acc);font-weight:500}
.filter-stat{display:flex;flex-direction:column;gap:.1rem}
.filter-stat span:first-child{color:var(--text);font-size:.85rem}

/* site table */
.st{width:100%;border-collapse:collapse;font-size:.85rem}
.st th{font-family:var(--mono);font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);text-align:left;padding:.5rem .8rem;border-bottom:1px solid var(--bdr)}
.st td{padding:.7rem .8rem;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle}
.st tr:hover td{background:var(--surf2)}
.dp{display:inline-block;background:var(--surf2);border:1px solid var(--bdr);border-radius:4px;padding:.1rem .5rem;font-family:var(--mono);font-size:.75rem}
.tb{font-family:var(--mono);font-size:.65rem;padding:.15rem .5rem;border-radius:20px;background:rgba(200,240,96,.1);color:var(--acc);border:1px solid rgba(200,240,96,.25)}
.mb{height:4px;background:var(--bdr);border-radius:2px;overflow:hidden;width:120px}
.mf{height:100%;background:var(--acc);border-radius:2px}
.nm{font-family:var(--mono);font-size:.8rem}

/* audience tabs */
.aud-tabs{display:flex;gap:.5rem;margin-bottom:1.5rem;flex-wrap:wrap}
.aud-tab{font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;padding:.4rem 1rem;border-radius:6px;border:1px solid var(--bdr);background:var(--surf2);color:var(--muted);cursor:pointer;transition:all .15s}
.aud-tab:hover{border-color:var(--acc2);color:var(--acc2)}
.aud-tab.active{background:rgba(200,240,96,.1);border-color:var(--acc);color:var(--acc)}

/* topic grid */
.tg{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1.2rem;margin-bottom:1rem}
.tg.hidden{display:none}
.tc{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;padding:1.5rem;position:relative;overflow:hidden;transition:transform .2s,border-color .2s}
.tc:hover{transform:translateY(-2px);border-color:rgba(200,240,96,.3)}
.tc::after{content:attr(data-num);position:absolute;top:-.5rem;right:1rem;font-family:var(--serif);font-style:italic;font-size:5rem;color:rgba(255,255,255,.03);line-height:1;pointer-events:none}
.tc-head{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1rem;gap:.5rem}
.tc-tag{font-family:var(--mono);font-size:.6rem;letter-spacing:.12em;padding:.25rem .6rem;border-radius:4px;background:rgba(96,200,240,.12);color:var(--acc2);border:1px solid rgba(96,200,240,.25);white-space:nowrap}
.tc-purpose{font-size:.9rem;font-weight:500;color:var(--text);flex:1}
.tc-pct{font-family:var(--mono);font-size:.78rem;color:var(--acc);white-space:nowrap}
.kws{display:flex;flex-wrap:wrap;gap:.35rem;margin-bottom:1.1rem}
.kw{font-family:var(--mono);font-size:.65rem;padding:.2rem .5rem;border-radius:4px;background:var(--surf2);border:1px solid var(--bdr);color:var(--muted)}
.kw.k0{color:var(--acc2);border-color:rgba(96,200,240,.3)}
.kw.k1{color:var(--acc);border-color:rgba(200,240,96,.25)}
.ex{border-top:1px solid var(--bdr);padding-top:.9rem}
.ex-lbl{font-family:var(--mono);font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem}
.ex-item{font-size:.78rem;color:var(--muted);padding:.2rem 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ex-item a{color:inherit;text-decoration:none}
.ex-item:hover{color:var(--text)}
.aud-badge{font-family:var(--mono);font-size:.58rem;padding:.1rem .35rem;border-radius:3px;background:rgba(96,200,240,.12);color:var(--acc2);border:1px solid rgba(96,200,240,.2);margin-right:.35rem}
.aud-badge.worker{background:rgba(200,240,96,.12);color:var(--acc);border-color:rgba(200,240,96,.2)}
.aud-badge.client{background:rgba(240,96,200,.12);color:var(--acc3);border-color:rgba(240,96,200,.2)}

/* heatmap */
.hm-wrap{overflow-x:auto}
.hm{border-collapse:collapse;font-size:.78rem;width:100%}
.hm th{font-family:var(--mono);font-size:.62rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding:.4rem .6rem;text-align:center}
.hm th:first-child{text-align:left}
.hm td{padding:.5rem .6rem;text-align:center;font-family:var(--mono);font-size:.72rem;border:1px solid rgba(255,255,255,.03)}
.hm td:first-child{text-align:left;font-size:.75rem;white-space:nowrap;color:var(--muted)}

footer{border-top:1px solid var(--bdr);padding-top:2rem;margin-top:2rem;font-family:var(--mono);font-size:.65rem;color:var(--muted);display:flex;justify-content:space-between;flex-wrap:wrap;gap:.5rem}

@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.a{opacity:0;animation:fadeUp .5s ease forwards}

/* ── word cloud section ────────────────────────────────────────────── */
.wc-section{margin-bottom:3.5rem}
.wc-picker{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:1.5rem}
.wc-btn{font-family:var(--mono);font-size:.65rem;letter-spacing:.08em;padding:.35rem .9rem;border-radius:20px;border:1px solid var(--bdr);background:transparent;color:var(--muted);cursor:pointer;transition:all .18s;white-space:nowrap}
.wc-btn:hover{border-color:var(--acc2);color:var(--acc2)}
.wc-btn.active{background:var(--surf2);border-color:var(--acc);color:var(--acc)}
.wc-stage{position:relative;background:var(--surf);border:1px solid var(--bdr);border-radius:16px;overflow:hidden}
.wc-stage::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--acc),var(--acc2),var(--acc3));opacity:.7;z-index:2}
.wc-canvas-wrap{position:relative;width:100%;height:420px}
.wc-canvas-wrap canvas{position:absolute;inset:0;width:100%;height:100%}
.wc-legend{display:flex;flex-wrap:wrap;gap:.5rem 1.2rem;padding:1rem 1.4rem 1.4rem;border-top:1px solid var(--bdr)}
.wc-legend-item{display:flex;align-items:center;gap:.4rem;font-family:var(--mono);font-size:.65rem;color:var(--muted);cursor:pointer;transition:color .15s}
.wc-legend-item:hover{color:var(--text)}
.wc-legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.wc-info{position:absolute;bottom:1rem;right:1.2rem;font-family:var(--mono);font-size:.6rem;color:rgba(107,114,128,.4);pointer-events:none;z-index:3}
.wc-tooltip{position:absolute;background:var(--surf2);border:1px solid var(--bdr);border-radius:6px;padding:.4rem .8rem;font-family:var(--mono);font-size:.72rem;color:var(--text);pointer-events:none;opacity:0;transition:opacity .15s;z-index:10;white-space:nowrap}
</style>
</head>
<body>
<div class="shell">

<header>
  <div><div class="h-title">Web Corpus<br><em>Analysis Dashboard</em></div></div>
  <div class="h-meta" id="meta-stamp"></div>
</header>

<div class="sl">Overview</div>
<div class="stat-row a" style="animation-delay:.1s" id="stat-row"></div>

<div class="sl">Website Breakdown</div>
<div class="card a" style="animation-delay:.2s;margin-bottom:3.5rem">
  <div class="card-title">Pages per domain</div>
  <table class="st" id="site-table"></table>
</div>

<div class="g2 a" style="animation-delay:.25s">
  <div class="card">
    <div class="card-title">URL depth distribution</div>
    <div class="chart-wrap" style="height:220px"><canvas id="depthChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Content length buckets</div>
    <div class="chart-wrap" style="height:220px"><canvas id="lenChart"></canvas></div>
  </div>
</div>

<div class="g2 a" style="animation-delay:.3s">
  <div class="card">
    <div class="card-title">Top URL path segments</div>
    <div class="chart-wrap" style="height:280px"><canvas id="segChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">HTTP status codes</div>
    <div class="chart-wrap" style="height:280px"><canvas id="statusChart"></canvas></div>
  </div>
</div>

<div class="sl a" style="animation-delay:.35s">
  Topic Modeling
  <sub id="topic-src">LDA — latent dirichlet allocation</sub>
</div>
<div id="filter-banner-wrap" class="a" style="animation-delay:.37s"></div>
<div class="aud-tabs a" style="animation-delay:.38s" id="aud-tabs"></div>
<div id="topic-sections"></div>

<div class="sl a" style="animation-delay:.45s">Domain &times; Topic Matrix</div>
<div class="card a" style="animation-delay:.47s;margin-bottom:3.5rem">
  <div class="card-title">Average topic probability per domain (combined model)</div>
  <div class="hm-wrap"><table class="hm" id="heatmap"></table></div>
</div>

<div class="sl a" style="animation-delay:.5s">Word Landscape <sub>topic keyword visualisation — click a topic to explore</sub></div>
<div class="wc-section a" style="animation-delay:.52s">
  <div class="wc-picker" id="wc-picker"></div>
  <div class="wc-stage">
    <div class="wc-canvas-wrap">
      <canvas id="wc-bg"></canvas>
      <canvas id="wc-main"></canvas>
      <div class="wc-tooltip" id="wc-tooltip"></div>
      <div class="wc-info">hover a word for details</div>
    </div>
    <div class="wc-legend" id="wc-legend"></div>
  </div>
</div>

<footer>
  <span>Generated by analyze_db.py</span>
  <span id="footer-stamp"></span>
</footer>
</div>

<script>
const D = __DATA_PLACEHOLDER__;

const ACC='#c8f060',ACC2='#60c8f0',ACC3='#f060c8',ACC4='#f0c860';
const MUTED='#6b7280',BDR='#252a38';
const PAL=['#c8f060','#60c8f0','#f060c8','#f0c860','#60f0c8','#c860f0','#f08060','#60f080'];

Chart.defaults.color=MUTED;
Chart.defaults.borderColor=BDR;
Chart.defaults.font.family="'DM Mono', monospace";
Chart.defaults.font.size=11;

// ── stamps ────────────────────────────────────────────────────────────────
const now=new Date();
document.getElementById('meta-stamp').innerHTML=
  `Generated<br>${now.toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'})}<br>`+
  `${now.toLocaleTimeString('en-GB',{hour:'2-digit',minute:'2-digit'})}`;
document.getElementById('footer-stamp').textContent=
  `${D.n_pages.toLocaleString()} pages · ${D.n_sites} sites · ${D.total_mb.toFixed(2)} MB`;

const srcEl=document.getElementById('topic-src');
if(D.topic_source&&D.topic_source.includes('tfidf')){
  srcEl.textContent='pages_tfidf — lemmatized · boilerplate filtered · stratified by audience';
  srcEl.style.color='var(--acc)';
}else{
  srcEl.textContent='raw text — run preprocess.py for cleaner topics';
  srcEl.style.color='var(--acc3)';
}

// ── overview stats ────────────────────────────────────────────────────────
const topicCount=D.topics.length;
[
  [D.n_sites,                        'websites'     ],
  [D.n_pages.toLocaleString(),        'pages scraped'],
  [D.n_links.toLocaleString(),        'links found'  ],
  [D.total_mb.toFixed(1)+' MB',       'total content'],
  [topicCount,                        'topics found' ],
].forEach(([v,l])=>{
  document.getElementById('stat-row').innerHTML+=
    `<div class="stat-cell"><div class="stat-val">${v}</div><div class="stat-lbl">${l}</div></div>`;
});

// ── filter banner ─────────────────────────────────────────────────────────
if(D.filter_stats&&D.filter_stats.before){
  const fs=D.filter_stats;
  const pct=((fs.removed/fs.before)*100).toFixed(1);
  document.getElementById('filter-banner-wrap').innerHTML=`
  <div class="filter-banner">
    <strong>Structural filter applied</strong>
    <div class="filter-stat"><span>${fs.before.toLocaleString()}</span><span>pages before filter</span></div>
    <div class="filter-stat"><span style="color:var(--acc)">${fs.after.toLocaleString()}</span><span>pages modeled</span></div>
    <div class="filter-stat"><span style="color:var(--acc3)">${fs.removed.toLocaleString()} (${pct}%)</span><span>removed as stubs / boilerplate</span></div>
    <div class="filter-stat"><span>${fs.thresholds.min_tokens} tokens</span><span>min token count</span></div>
    <div class="filter-stat"><span>${fs.thresholds.min_bytes} B</span><span>min content size</span></div>
    <div class="filter-stat"><span>depth ${fs.thresholds.depth_range}</span><span>URL depth range</span></div>
  </div>`;
}

// ── site table ────────────────────────────────────────────────────────────
const maxP=Math.max(...D.sites.map(s=>s.n_pages),1);
const tbl=document.getElementById('site-table');
tbl.innerHTML=`<thead><tr>
  <th>Domain</th><th>Type</th><th>Pages</th><th style="width:130px"></th>
  <th>Avg depth</th><th>Avg KB</th><th>Total MB</th>
</tr></thead>`;
const tb=document.createElement('tbody');
D.sites.forEach(s=>{
  tb.innerHTML+=`<tr>
    <td><span class="dp">${s.domain}</span></td>
    <td><span class="tb">${s.website_type||'n/a'}</span></td>
    <td class="nm">${s.n_pages.toLocaleString()}</td>
    <td><div class="mb"><div class="mf" style="width:${s.n_pages/maxP*100}%"></div></div></td>
    <td class="nm">${s.avg_depth.toFixed(1)}</td>
    <td class="nm">${(s.avg_len/1024).toFixed(1)}</td>
    <td class="nm">${(s.total_len/1e6).toFixed(2)}</td>
  </tr>`;
});
tbl.appendChild(tb);

// ── chart helpers ─────────────────────────────────────────────────────────
function bar(id,labels,values,color,horiz=false){
  new Chart(document.getElementById(id),{
    type:'bar',
    data:{labels,datasets:[{data:values,backgroundColor:color+'33',borderColor:color,borderWidth:1,borderRadius:3}]},
    options:{indexAxis:horiz?'y':'x',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{x:{grid:{color:BDR},ticks:{color:MUTED}},y:{grid:{color:BDR},ticks:{color:MUTED}}}}
  });
}
function donut(id,labels,values){
  new Chart(document.getElementById(id),{
    type:'doughnut',
    data:{labels,datasets:[{data:values,backgroundColor:PAL.map(c=>c+'55'),borderColor:PAL,borderWidth:1,hoverOffset:6}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'62%',
      plugins:{legend:{position:'right',labels:{boxWidth:10,padding:12,color:MUTED,font:{size:10}}}}}
  });
}

if(D.depth_dist.length)  bar('depthChart',D.depth_dist.map(d=>'depth '+d[0]),D.depth_dist.map(d=>d[1]),ACC2);
if(D.len_buckets.length) bar('lenChart',D.len_buckets.map(b=>b.label),D.len_buckets.map(b=>b.count),ACC);
if(D.top_segments.length)bar('segChart',D.top_segments.map(s=>s[0]),D.top_segments.map(s=>s[1]),ACC3,true);
if(D.status_codes.length)donut('statusChart',D.status_codes.map(s=>s.code),D.status_codes.map(s=>s.count));

// ── audience-stratified topic sections ────────────────────────────────────
const audiences=[...new Set(D.topics.map(t=>t.label))].sort();
const tabsEl=document.getElementById('aud-tabs');
const sectEl=document.getElementById('topic-sections');

const audColors={'worker':'var(--acc)','client':'var(--acc3)','all':'var(--acc2)'};

audiences.forEach((aud,ai)=>{
  // tab
  const tab=document.createElement('button');
  tab.className='aud-tab'+(ai===0?' active':'');
  tab.dataset.aud=aud;
  const audTopics=D.topics.filter(t=>t.label===aud);
  tab.innerHTML=`${aud} <span style="opacity:.5">${audTopics.length} topics</span>`;
  tab.style.setProperty('--tab-color', audColors[aud]||ACC2);
  tabsEl.appendChild(tab);

  // grid section
  const section=document.createElement('div');
  section.className='tg a'+(ai===0?'':' hidden');
  section.id='tg-'+aud;
  section.style.animationDelay='.4s';

  audTopics.forEach((t,i)=>{
    const kws=t.keywords.map((w,j)=>`<span class="kw ${j===0?'k0':j===1?'k1':''}">${w}</span>`).join('');
    const exs=t.examples.map(e=>{
      const aC=e.audience==='worker'?'worker':e.audience==='client'?'client':'';
      const aBadge=e.audience?`<span class="aud-badge ${aC}">${e.audience}</span>`:'';
      return `<div class="ex-item" title="${e.url}">${aBadge}<a href="${e.url}" target="_blank">${e.title||e.url}</a></div>`;
    }).join('');
    section.innerHTML+=`
    <div class="tc" data-num="${t.index}">
      <div class="tc-head">
        <span class="tc-tag">${t.purpose.tag}</span>
        <span class="tc-purpose">${t.purpose.label}</span>
        <span class="tc-pct">${t.pct}%</span>
      </div>
      <div class="kws">${kws}</div>
      <div class="ex">
        <div class="ex-lbl">Top pages</div>
        ${exs}
      </div>
    </div>`;
  });
  sectEl.appendChild(section);
});

// tab switching
tabsEl.addEventListener('click',e=>{
  const btn=e.target.closest('.aud-tab');
  if(!btn) return;
  tabsEl.querySelectorAll('.aud-tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
  audiences.forEach(a=>{
    const el=document.getElementById('tg-'+a);
    if(el) el.classList.toggle('hidden', a!==btn.dataset.aud);
  });
});

// ── heatmap ───────────────────────────────────────────────────────────────
if(D.domain_topic_matrix.length){
  const nT=D.domain_topic_matrix[0].dist.length;
  const ht=document.getElementById('heatmap');
  ht.innerHTML=`<thead><tr><th>Domain</th>${Array.from({length:nT},(_,i)=>`<th>T${i+1}</th>`).join('')}</tr></thead>`;
  const colMax=Array(nT).fill(0);
  D.domain_topic_matrix.forEach(r=>r.dist.forEach((v,j)=>{if(v>colMax[j])colMax[j]=v;}));
  const htb=document.createElement('tbody');
  D.domain_topic_matrix.forEach(row=>{
    const cells=row.dist.map((v,j)=>{
      const norm=colMax[j]?v/colMax[j]:0;
      const alpha=Math.round(norm*180).toString(16).padStart(2,'0');
      const fg=norm>.55?'#0d0f14':MUTED;
      return `<td style="background:${PAL[j%PAL.length]+alpha};color:${fg}">${v.toFixed(2)}</td>`;
    }).join('');
    htb.innerHTML+=`<tr><td>${row.domain}</td>${cells}</tr>`;
  });
  ht.appendChild(htb);
}
// ══════════════════════════════════════════════════════════════════
// WORD LANDSCAPE — animated canvas word cloud with physics
// ══════════════════════════════════════════════════════════════════
(function(){
  if(!D.topics.length) return;

  const bgCvs   = document.getElementById('wc-bg');
  const cvs     = document.getElementById('wc-main');
  const picker  = document.getElementById('wc-picker');
  const legend  = document.getElementById('wc-legend');
  const tooltip = document.getElementById('wc-tooltip');

  // Per-topic palette — cycling through accent family
  const T_COLORS = [
    ['#c8f060','#a8d040','#88b020'],  // lime
    ['#60c8f0','#40a8d0','#2088b0'],  // cyan
    ['#f060c8','#d040a8','#b02088'],  // pink
    ['#f0c860','#d0a840','#b08820'],  // gold
    ['#60f0c8','#40d0a8','#20b088'],  // teal
    ['#c860f0','#a840d0','#8820b0'],  // purple
    ['#f08060','#d06040','#b04020'],  // orange
    ['#60f080','#40d060','#20b040'],  // green
  ];

  // Assign a stable color palette per topic across all audiences
  const topicColorMap = {};
  D.topics.forEach((t,i)=>{
    topicColorMap[`${t.label}_${t.index}`] = T_COLORS[i % T_COLORS.length];
  });

  // ── Word node class ────────────────────────────────────────────
  class WordNode {
    constructor(word, rank, topicIdx, colors, cx, cy, radius, angle){
      this.word     = word;
      this.rank     = rank;          // 0 = highest weight
      this.colors   = colors;
      this.topicIdx = topicIdx;
      // font size: rank 0 → 32px, rank N → 11px
      this.fontSize = Math.max(11, 32 - rank * 1.8);
      // orbit radius with jitter
      this.orbitR   = radius + (Math.random()-0.5)*28;
      this.baseAngle= angle;
      this.angle    = angle;
      // slow individual drift speed
      this.speed    = (0.00015 + Math.random()*0.00025) * (Math.random()<0.5?1:-1);
      // gentle float offset
      this.floatPhase = Math.random()*Math.PI*2;
      this.floatAmp   = 2 + Math.random()*4;
      // target & current position
      this.tx = cx + Math.cos(angle)*this.orbitR;
      this.ty = cy + Math.sin(angle)*this.orbitR;
      this.x  = this.tx; this.y = this.ty;
      this.cx = cx; this.cy = cy;
      // interaction
      this.alpha    = 0;
      this.targeted = false;
    }
    update(t, cx, cy){
      this.cx = cx; this.cy = cy;
      this.angle += this.speed;
      const float = Math.sin(t*0.0008 + this.floatPhase) * this.floatAmp;
      this.tx = cx + Math.cos(this.angle)*this.orbitR;
      this.ty = cy + Math.sin(this.angle)*this.orbitR + float;
      // smooth lerp toward target
      this.x += (this.tx - this.x) * 0.04;
      this.y += (this.ty - this.y) * 0.04;
      // fade in
      this.alpha = Math.min(1, this.alpha + 0.015);
    }
    draw(ctx, hovered){
      const a    = hovered===this ? 1 : (hovered ? 0.25 : this.alpha);
      const col  = this.colors[Math.min(this.rank<3?0:this.rank<7?1:2, 2)];
      ctx.save();
      ctx.globalAlpha = a;
      ctx.font        = `${this.rank<2?'500':'400'} ${this.fontSize}px 'DM Mono', monospace`;
      ctx.fillStyle   = col;
      // subtle glow for top words
      if(this.rank < 3 && hovered !== this){
        ctx.shadowColor = col;
        ctx.shadowBlur  = 8;
      }
      if(hovered===this){
        ctx.shadowColor = col;
        ctx.shadowBlur  = 20;
      }
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(this.word, this.x, this.y);
      ctx.restore();
    }
    // axis-aligned bounding box for hit testing
    bbox(ctx){
      ctx.font = `${this.fontSize}px 'DM Mono', monospace`;
      const w  = ctx.measureText(this.word).width;
      const h  = this.fontSize;
      return {x:this.x-w/2, y:this.y-h/2, w, h};
    }
    contains(mx, my, ctx){
      const b = this.bbox(ctx);
      return mx>=b.x && mx<=b.x+b.w && my>=b.y && my<=b.y+b.h;
    }
  }

  // ── Background renderer — constellation lines ──────────────────
  function drawBackground(ctx, W, H, nodes, time){
    ctx.clearRect(0,0,W,H);
    // draw faint connector lines between nearby words of same topic
    nodes.forEach((a,i)=>{
      nodes.slice(i+1).forEach(b=>{
        if(a.topicIdx!==b.topicIdx) return;
        const dx=a.x-b.x, dy=a.y-b.y;
        const d =Math.sqrt(dx*dx+dy*dy);
        if(d>160) return;
        const alpha=(1-d/160)*0.06*Math.min(a.alpha,b.alpha);
        ctx.beginPath();
        ctx.moveTo(a.x,a.y);
        ctx.lineTo(b.x,b.y);
        ctx.strokeStyle=a.colors[0];
        ctx.globalAlpha=alpha;
        ctx.lineWidth=1;
        ctx.stroke();
        ctx.globalAlpha=1;
      });
    });
  }

  // ── Layout builder ─────────────────────────────────────────────
  function buildNodes(topics, W, H){
    const nodes  = [];
    const nT     = topics.length;
    // place topic centres in a circle around the canvas centre
    const cx0=W/2, cy0=H/2;
    const layoutR = Math.min(W,H)*0.28;

    topics.forEach((t,ti)=>{
      const key    = `${t.label}_${t.index}`;
      const colors = topicColorMap[key] || T_COLORS[ti%T_COLORS.length];
      // centre of this topic cluster
      const tAngle = (ti/nT)*Math.PI*2 - Math.PI/2;
      const tcx    = nT===1 ? cx0 : cx0+Math.cos(tAngle)*layoutR;
      const tcy    = nT===1 ? cy0 : cy0+Math.sin(tAngle)*layoutR;
      const wordR  = Math.min(W,H)*(nT===1?0.36:0.22);

      t.keywords.forEach((word,rank)=>{
        // distribute words in a spiral around cluster centre
        const wAngle = (rank/t.keywords.length)*Math.PI*2 + ti*(Math.PI*0.37);
        const r      = wordR*(0.3 + (rank/t.keywords.length)*0.7);
        nodes.push(new WordNode(word,rank,ti,colors,tcx,tcy,r,wAngle));
      });
    });
    return nodes;
  }

  // ── State ──────────────────────────────────────────────────────
  let activeTopics = [];
  let nodes        = [];
  let hoveredNode  = null;
  let animId       = null;
  let startTime    = performance.now();

  function resize(){
    const rect = cvs.parentElement.getBoundingClientRect();
    const W=rect.width, H=rect.height;
    [cvs, bgCvs].forEach(c=>{ c.width=W; c.height=H; });
    nodes = buildNodes(activeTopics, W, H);
  }

  function loop(ts){
    animId = requestAnimationFrame(loop);
    const W=cvs.width, H=cvs.height;
    if(!W||!H) return;
    const t=ts-startTime;
    const cx=W/2, cy=H/2;

    // update
    nodes.forEach(n=>n.update(t,
      activeTopics.length===1 ? cx : n.cx,
      activeTopics.length===1 ? cy : n.cy
    ));

    // draw bg (connection lines) — throttled
    if(Math.round(t/16)%3===0){
      const bgCtx=bgCvs.getContext('2d');
      drawBackground(bgCtx,W,H,nodes,t);
    }

    // draw words
    const ctx=cvs.getContext('2d');
    ctx.clearRect(0,0,W,H);
    // draw non-hovered first, hovered on top
    nodes.filter(n=>n!==hoveredNode).forEach(n=>n.draw(ctx,hoveredNode));
    if(hoveredNode) hoveredNode.draw(ctx,hoveredNode);
  }

  // ── Mouse interaction ──────────────────────────────────────────
  cvs.addEventListener('mousemove',e=>{
    const r=cvs.getBoundingClientRect();
    const mx=(e.clientX-r.left)*(cvs.width/r.width);
    const my=(e.clientY-r.top)*(cvs.height/r.height);
    const ctx=cvs.getContext('2d');
    const hit=nodes.find(n=>n.contains(mx,my,ctx))||null;
    if(hit!==hoveredNode){
      hoveredNode=hit;
      cvs.style.cursor=hit?'pointer':'default';
    }
    if(hit){
      const rank=hit.rank+1;
      tooltip.style.opacity='1';
      tooltip.style.left=(e.clientX-cvs.getBoundingClientRect().left+14)+'px';
      tooltip.style.top =(e.clientY-cvs.getBoundingClientRect().top -10)+'px';
      tooltip.textContent=`${hit.word}  ·  rank #${rank} in topic`;
    } else {
      tooltip.style.opacity='0';
    }
  });
  cvs.addEventListener('mouseleave',()=>{
    hoveredNode=null;
    tooltip.style.opacity='0';
    cvs.style.cursor='default';
  });

  // ── Topic picker ───────────────────────────────────────────────
  function setActive(selectedTopics){
    activeTopics=selectedTopics;
    resize();
    buildLegend(selectedTopics);
  }

  function buildLegend(topics){
    legend.innerHTML='';
    topics.forEach((t,i)=>{
      const key=`${t.label}_${t.index}`;
      const col=(topicColorMap[key]||T_COLORS[i%T_COLORS.length])[0];
      const item=document.createElement('div');
      item.className='wc-legend-item';
      item.innerHTML=`<div class="wc-legend-dot" style="background:${col}"></div>
        <span>${t.label} · T${t.index} · ${t.purpose.label}</span>`;
      legend.appendChild(item);
    });
  }

  // Build "All" + individual topic buttons
  const allBtn=document.createElement('button');
  allBtn.className='wc-btn active';
  allBtn.textContent='All topics';
  allBtn.onclick=()=>{
    picker.querySelectorAll('.wc-btn').forEach(b=>b.classList.remove('active'));
    allBtn.classList.add('active');
    setActive(D.topics);
  };
  picker.appendChild(allBtn);

  // Group by audience for labelling
  const audGroups={};
  D.topics.forEach(t=>{
    if(!audGroups[t.label]) audGroups[t.label]=[];
    audGroups[t.label].push(t);
  });

  // Audience group buttons
  Object.entries(audGroups).forEach(([aud,tList])=>{
    if(Object.keys(audGroups).length>1){
      const gb=document.createElement('button');
      gb.className='wc-btn';
      gb.textContent=aud;
      gb.onclick=()=>{
        picker.querySelectorAll('.wc-btn').forEach(b=>b.classList.remove('active'));
        gb.classList.add('active');
        setActive(tList);
      };
      picker.appendChild(gb);
    }
    // Individual topic buttons
    tList.forEach(t=>{
      const key=`${t.label}_${t.index}`;
      const col=(topicColorMap[key]||T_COLORS[0])[0];
      const btn=document.createElement('button');
      btn.className='wc-btn';
      btn.style.borderColor=col+'55';
      btn.innerHTML=`<span style="color:${col}">${t.purpose.tag}</span> ${t.label}·T${t.index}`;
      btn.onclick=()=>{
        picker.querySelectorAll('.wc-btn').forEach(b=>b.classList.remove('active'));
        btn.classList.add('active');
        setActive([t]);
      };
      picker.appendChild(btn);
    });
  });

  // ── Init ───────────────────────────────────────────────────────
  window.addEventListener('resize',()=>{ if(activeTopics.length) resize(); });
  // Defer until element in viewport for perf
  const obs=new IntersectionObserver(entries=>{
    if(entries[0].isIntersecting){
      obs.disconnect();
      setActive(D.topics);
      animId=requestAnimationFrame(loop);
    }
  },{threshold:0.1});
  obs.observe(cvs.parentElement);
})();
</script>
</body>
</html>
"""


# ── Render ────────────────────────────────────────────────────────────────────
def render_html(data, output_path):
    html = HTML_TEMPLATE.replace('__DATA_PLACEHOLDER__', json.dumps(data))
    Path(output_path).write_text(html, encoding='utf-8')
    print(f"  Dashboard written -> {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_db.py <database.db> [output.html]")
        sys.exit(1)

    db_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "corpus_dashboard.html"

    if not Path(db_path).exists():
        print(f"File not found: {db_path}")
        sys.exit(1)

    print("Connecting...")
    conn = connect(db_path)
    print("Collecting statistics...")
    data = collect_all(conn)
    conn.close()
    print("Rendering dashboard...")
    render_html(data, out_path)
    print("Done.")


if __name__ == "__main__":
    main()
