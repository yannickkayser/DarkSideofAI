"""
TF-IDF Analysis — Proof of Concept
===================================
Reads preprocessed data from pages_tfidf and produces:
  1. Top TF-IDF terms per audience (worker vs client) — bar charts
  2. Term frequency heatmap across individual websites
  3. Word clouds per audience type

Output: figures saved to ./tfidf_output/ directory
"""

import sqlite3
import json
import sys
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Add project root so config is importable (src/ → project root)
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent.parent))
from config.config import WEBSITES

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH    = str(Path(__file__).parent.parent / "data" / "scraping.db")
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "test"
OUTPUT_DIR.mkdir(exist_ok=True)

TOP_N_TERMS   = 20      # top terms to show in bar charts and heatmap
MAX_PAGES     = 2000    # cap pages per audience to keep memory manageable
EXCLUDE_AUDIENCES = {"both"}   # exclude mixed-audience sites

# Visual style
WORKER_COLOR = "#2E86AB"   # blue
CLIENT_COLOR = "#E84855"   # red
BOTH_COLOR   = "#F4A261"   # orange (unused but kept for reference)

FONT_TITLE  = {"fontsize": 14, "fontweight": "bold"}
FONT_AXIS   = {"fontsize": 11}
FONT_TICK   = {"fontsize": 9}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — build domain → website name map from config
# ---------------------------------------------------------------------------

DOMAIN_NAMES: dict[str, str] = {
    domain: site["name"] for domain, site in WEBSITES.items()
}

def domain_from_url(url: str) -> str:
    for domain in WEBSITES:
        if domain in url:
            return domain
    return "unknown"


# ---------------------------------------------------------------------------
# 1. Load data from database
# ---------------------------------------------------------------------------

def load_tfidf_data(db_path: str) -> pd.DataFrame:
    """Load pages_tfidf rows, excluding 'both' and 'unknown' audience pages."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("""
        SELECT page_id, url, audience, unigrams, bigrams, token_count
        FROM   pages_tfidf
        WHERE  audience NOT IN ('both', 'unknown')
          AND  token_count > 10
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(
            "No data found in pages_tfidf. "
            "Run preprocess.py first to populate the table."
        )

    records = []
    for row in rows:
        unigrams = json.loads(row["unigrams"] or "[]")
        bigrams  = json.loads(row["bigrams"]  or "[]")
        # Combine unigrams and bigrams into one token string for TF-IDF
        tokens   = unigrams + bigrams
        records.append({
            "page_id":  row["page_id"],
            "url":      row["url"],
            "audience": row["audience"],
            "domain":   domain_from_url(row["url"]),
            "tokens":   " ".join(tokens),
        })

    df = pd.DataFrame(records)
    log.info(
        f"Loaded {len(df)} pages  "
        f"({(df.audience=='worker').sum()} worker, "
        f"{(df.audience=='client').sum()} client)"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Compute TF-IDF matrices
# ---------------------------------------------------------------------------

def compute_tfidf(
    df: pd.DataFrame,
    max_features: int = 5000,
) -> tuple[TfidfVectorizer, np.ndarray, list[str]]:
    """Fit TF-IDF on the full corpus and return (vectorizer, matrix, feature_names)."""
    log.info("Fitting TF-IDF vectorizer on full corpus...")
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),   # tokens already include pre-computed bigrams
        min_df=3,             # ignore terms in fewer than 3 pages
        sublinear_tf=True,    # log normalization — reduces impact of very frequent terms
    )
    matrix = vec.fit_transform(df["tokens"])
    names  = vec.get_feature_names_out().tolist()
    log.info(f"  Vocabulary size: {len(names)} terms")
    return vec, matrix, names


def top_terms_by_audience(
    df: pd.DataFrame,
    matrix: np.ndarray,
    feature_names: list[str],
    top_n: int = TOP_N_TERMS,
) -> dict[str, pd.Series]:
    """
    For each audience, compute mean TF-IDF score per term across all its pages.
    Returns {audience: Series(term → mean_score)} sorted descending.
    """
    results = {}
    for audience in ["worker", "client"]:
        mask        = (df["audience"] == audience).to_numpy()
        sub_matrix  = matrix[mask]
        mean_scores = np.asarray(sub_matrix.mean(axis=0)).flatten()
        series = pd.Series(mean_scores, index=feature_names).sort_values(ascending=False)
        results[audience] = series.head(top_n)
        log.info(f"  Top term ({audience}): '{series.index[0]}' (score={series.iloc[0]:.4f})")
    return results


def top_terms_by_domain(
    df: pd.DataFrame,
    matrix: np.ndarray,
    feature_names: list[str],
    top_n: int = TOP_N_TERMS,
) -> pd.DataFrame:
    """
    Mean TF-IDF score per term per domain.
    Returns a DataFrame (domains × top_n_terms).
    """
    domains = [d for d in df["domain"].unique() if d != "unknown"]
    records = {}

    for domain in sorted(domains):
        mask        = (df["domain"] == domain).to_numpy()
        if mask.sum() == 0:
            continue
        sub_matrix  = matrix[mask]
        mean_scores = np.asarray(sub_matrix.mean(axis=0)).flatten()
        series      = pd.Series(mean_scores, index=feature_names)
        records[DOMAIN_NAMES.get(domain, domain)] = series

    domain_df = pd.DataFrame(records).T   # domains × all terms

    # Select the top_n terms with highest variance across domains
    # (these are the most discriminating terms for the heatmap)
    top_terms = domain_df.var(axis=0).sort_values(ascending=False).head(top_n).index
    return domain_df[top_terms]


# ---------------------------------------------------------------------------
# 3. Visualizations
# ---------------------------------------------------------------------------

def plot_top_terms(
    top_terms: dict[str, pd.Series],
    output_dir: Path,
):
    """Bar charts: top TF-IDF terms for worker vs client, side by side."""
    log.info("Plotting top TF-IDF terms per audience...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Top TF-IDF Terms by Audience Type",
        fontsize=16, fontweight="bold", y=1.01
    )

    configs = [
        ("worker", WORKER_COLOR, axes[0]),
        ("client", CLIENT_COLOR, axes[1]),
    ]

    for audience, color, ax in configs:
        series = top_terms[audience]
        terms  = series.index.tolist()[::-1]   # reverse for horizontal bar
        scores = series.values[::-1]

        bars = ax.barh(terms, scores, color=color, alpha=0.85, edgecolor="white")

        # Value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", ha="left", fontsize=8, color="#444"
            )

        ax.set_title(
            f"{'Worker' if audience == 'worker' else 'Client'}-Facing Sites",
            **FONT_TITLE, color=color
        )
        ax.set_xlabel("Mean TF-IDF Score", **FONT_AXIS)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, scores.max() * 1.18)

    plt.tight_layout()
    out = output_dir / "01_top_tfidf_terms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


def plot_heatmap(
    domain_df: pd.DataFrame,
    df_pages: pd.DataFrame,
    output_dir: Path,
):
    """Heatmap: top discriminating terms across individual websites."""
    log.info("Plotting TF-IDF heatmap across websites...")

    # Sort rows: workers first, then clients
    audience_lookup = {
        DOMAIN_NAMES.get(d, d): WEBSITES[d]["audience"]
        for d in WEBSITES
        if WEBSITES[d].get("audience") not in EXCLUDE_AUDIENCES
    }
    def sort_key(name):
        aud = audience_lookup.get(name, "z")
        return (0 if aud == "worker" else 1, name)

    ordered = sorted(domain_df.index.tolist(), key=sort_key)
    domain_df = domain_df.loc[[r for r in ordered if r in domain_df.index]]

    # Row colors to indicate audience
    row_colors = [
        WORKER_COLOR if audience_lookup.get(name) == "worker" else CLIENT_COLOR
        for name in domain_df.index
    ]

    n_sites, n_terms = domain_df.shape
    fig_h = max(8, n_sites * 0.45 + 2)
    fig_w = max(14, n_terms * 0.7 + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = domain_df.values
    im   = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # Axes labels
    ax.set_xticks(range(n_terms))
    ax.set_xticklabels(domain_df.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(n_sites))
    ax.set_yticklabels(domain_df.index, fontsize=9)

    # Colour the y-tick labels by audience
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)

    # Audience label strip on the left
    for i, name in enumerate(domain_df.index):
        aud   = audience_lookup.get(name, "")
        color = WORKER_COLOR if aud == "worker" else CLIENT_COLOR
        ax.annotate(
            aud, xy=(-0.01, i), xycoords=("axes fraction", "data"),
            ha="right", va="center", fontsize=7.5,
            color=color, fontweight="bold"
        )

    plt.colorbar(im, ax=ax, shrink=0.6, label="Mean TF-IDF Score")
    ax.set_title(
        f"Top {n_terms} Discriminating Terms Across Websites\n"
        "(sorted: worker top, client bottom — colour = audience)",
        **FONT_TITLE
    )
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    out = output_dir / "02_tfidf_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


def plot_wordclouds(
    df: pd.DataFrame,
    matrix,
    feature_names: list[str],
    output_dir: Path,
):
    """Word clouds: one per audience, sized by mean TF-IDF score."""
    log.info("Generating word clouds...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("TF-IDF Word Clouds by Audience Type", fontsize=16, fontweight="bold")

    configs = [
        ("worker", WORKER_COLOR, axes[0]),
        ("client", CLIENT_COLOR, axes[1]),
    ]

    for audience, color, ax in configs:
        mask        = (df["audience"] == audience).to_numpy()
        sub_matrix  = matrix[mask]
        mean_scores = np.asarray(sub_matrix.mean(axis=0)).flatten()
        freq_dict   = {
            term: float(score)
            for term, score in zip(feature_names, mean_scores)
            if score > 0
        }

        wc = WordCloud(
            width=900, height=600,
            background_color="white",
            colormap="Blues" if audience == "worker" else "Reds",
            max_words=120,
            prefer_horizontal=0.85,
            collocations=False,
        ).generate_from_frequencies(freq_dict)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"{'Worker' if audience == 'worker' else 'Client'}-Facing Sites",
            **FONT_TITLE, color=color, pad=12
        )

    plt.tight_layout()
    out = output_dir / "03_wordclouds.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


def plot_term_overlap(
    top_terms: dict[str, pd.Series],
    output_dir: Path,
):
    """
    Bonus: horizontal diverging chart showing terms that are
    distinctively worker vs client (i.e. appear in one but not both top lists).
    Useful for seeing what language is exclusive to each audience.
    """
    log.info("Plotting distinctive term divergence chart...")

    worker_terms = set(top_terms["worker"].index)
    client_terms = set(top_terms["client"].index)

    # Terms unique to each audience
    only_worker = {t: top_terms["worker"][t] for t in worker_terms - client_terms}
    only_client = {t: top_terms["client"][t] for t in client_terms - worker_terms}

    if not only_worker and not only_client:
        log.info("  No distinctive terms found — skipping divergence chart.")
        return

    # Build diverging data: worker scores positive, client scores negative
    all_terms  = sorted(
        list(only_worker.keys()) + list(only_client.keys()),
        key=lambda t: only_worker.get(t, 0) - only_client.get(t, 0),
    )
    scores     = [only_worker.get(t, 0) - only_client.get(t, 0) for t in all_terms]
    colors     = [WORKER_COLOR if s > 0 else CLIENT_COLOR for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(6, len(all_terms) * 0.38)))
    ax.barh(all_terms, scores, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#333", linewidth=0.8)

    ax.set_xlabel("← Distinctive to Client          Distinctive to Worker →", **FONT_AXIS)
    ax.set_title("Terms Exclusive to One Audience Type\n(not present in the other's top terms)", **FONT_TITLE)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=WORKER_COLOR, label="Worker-facing"),
            Patch(color=CLIENT_COLOR, label="Client-facing"),
        ],
        fontsize=9, loc="lower right"
    )

    plt.tight_layout()
    out = output_dir / "04_distinctive_terms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# H1 Term Dictionaries
# ---------------------------------------------------------------------------
# H1a — Labour-denoting terms (visibility of human work)
LABOUR_TERMS = [
    "worker", "workers", "annotator", "annotators", "labeler", "labelers",
    "labeller", "labellers", "moderator", "moderators", "contributor",
    "contributors", "human", "humans", "person", "people", "freelancer",
    "freelancers", "tasker", "taskers", "crowd", "crowdworker",
]

# H1b — Automation-myth terms (AI/tech framing without labour)
AUTOMATION_TERMS = [
    "ai", "algorithm", "algorithms", "automated", "automation", "autonomous",
    "autonomously", "intelligent", "intelligence", "machine", "model", "models",
    "pipeline", "scalable", "solution", "solutions", "technology",
]

# H1b — Task/work terms (labour-grounding language)
TASK_TERMS = [
    "task", "tasks", "job", "jobs", "project", "projects", "work",
    "assignment", "assignments", "gig", "earn", "earning", "income", "pay",
]

# H1c — Quality/ethics framing (strategic hypervisibility)
QUALITY_TERMS = [
    "quality", "accuracy", "trust", "ethics", "ethical", "excellence",
    "responsible", "responsibility", "expert", "expertise", "reliable",
    "reliability", "precision", "verified", "transparent", "transparency",
]


def count_term_frequencies(
    df: pd.DataFrame,
    term_list: list[str],
) -> pd.DataFrame:
    """
    Count how often each term appears per page (raw token match),
    then return mean frequency per page, grouped by audience.
    Returns a DataFrame: rows = terms, columns = ['worker', 'client'].
    """
    records = []
    for _, row in df.iterrows():
        token_list = row["tokens"].split()
        token_set  = set(token_list)          # for presence check
        audience   = row["audience"]
        for term in term_list:
            count = token_list.count(term)    # raw occurrences per page
            records.append({"term": term, "audience": audience, "count": count})

    freq_df = pd.DataFrame(records)
    # Mean occurrences per page, by term and audience
    pivot = (
        freq_df.groupby(["term", "audience"])["count"]
        .mean()
        .unstack(fill_value=0)
    )
    # Only keep terms that appear at least somewhere
    pivot = pivot[(pivot > 0).any(axis=1)]
    return pivot


# ---------------------------------------------------------------------------
# H1a — Visibility Gap
# ---------------------------------------------------------------------------

def plot_h1a_visibility_gap(df: pd.DataFrame, output_dir: Path):
    """
    H1a: Compare mean frequency of labour-denoting terms per page
    between worker-facing and client-facing sites.
    Produces a grouped bar chart and a ratio chart.
    """
    log.info("H1a — Plotting visibility gap...")

    pivot = count_term_frequencies(df, LABOUR_TERMS)

    # Ensure both columns exist
    for col in ["worker", "client"]:
        if col not in pivot.columns:
            pivot[col] = 0.0

    # Sort by combined frequency descending
    pivot["total"] = pivot["worker"] + pivot["client"]
    pivot = pivot.sort_values("total", ascending=False).drop(columns="total")
    pivot = pivot.head(15)   # top 15 most-used labour terms

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "H1a — Visibility Gap: Labour-Denoting Term Frequency\n"
        "Mean occurrences per page by audience type",
        fontsize=14, fontweight="bold"
    )

    # --- Left: grouped bar chart ---
    ax = axes[0]
    terms  = pivot.index.tolist()
    x      = np.arange(len(terms))
    width  = 0.38

    bars_w = ax.bar(x - width/2, pivot["worker"], width,
                    color=WORKER_COLOR, alpha=0.85, label="Worker-facing")
    bars_c = ax.bar(x + width/2, pivot["client"],  width,
                    color=CLIENT_COLOR,  alpha=0.85, label="Client-facing")

    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Mean occurrences per page", **FONT_AXIS)
    ax.set_title("Raw frequency comparison", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bars
    for bar in list(bars_w) + list(bars_c):
        h = bar.get_height()
        if h > 0.005:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    # --- Right: visibility ratio chart (worker / client) ---
    ax2 = axes[1]
    # Avoid division by zero — add small epsilon
    eps   = 1e-6
    ratio = (pivot["worker"] + eps) / (pivot["client"] + eps)
    ratio = ratio.sort_values(ascending=True)

    colors = [WORKER_COLOR if r >= 1 else CLIENT_COLOR for r in ratio]
    ax2.barh(ratio.index, ratio.values, color=colors, alpha=0.85, edgecolor="white")
    ax2.axvline(1.0, color="#333", linewidth=1.2, linestyle="--",
                label="Equal frequency (ratio = 1)")

    ax2.set_xlabel("Worker / Client frequency ratio  (log scale)", **FONT_AXIS)
    ax2.set_title(
        "Visibility ratio\n"
        "> 1 = more visible in worker-facing  |  < 1 = more visible in client-facing",
        fontsize=11
    )
    ax2.set_xscale("log")
    ax2.legend(fontsize=9)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = output_dir / "H1a_visibility_gap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# H1b — Automation Myth
# ---------------------------------------------------------------------------

def plot_h1b_automation_myth(df: pd.DataFrame, output_dir: Path):
    """
    H1b: Compare automation terms vs task/work terms across audiences.
    Shows:
      - Left:  mean frequency of automation vs task terms per page per audience
      - Right: stacked proportion chart — share of automation vs task vocabulary
    """
    log.info("H1b — Plotting automation myth...")

    auto_pivot = count_term_frequencies(df, AUTOMATION_TERMS)
    task_pivot = count_term_frequencies(df, TASK_TERMS)

    for p in [auto_pivot, task_pivot]:
        for col in ["worker", "client"]:
            if col not in p.columns:
                p[col] = 0.0

    # Aggregate: total mean frequency per category per audience
    auto_sum = auto_pivot[["worker", "client"]].sum()
    task_sum = task_pivot[["worker", "client"]].sum()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "H1b — Automation Myth: Automation vs Task/Work Language\n"
        "Total mean term frequency per page by audience type",
        fontsize=14, fontweight="bold"
    )

    # --- Left: grouped bar — automation vs task vocabulary totals ---
    ax = axes[0]
    categories = ["Automation terms", "Task/Work terms"]
    worker_vals = [auto_sum["worker"], task_sum["worker"]]
    client_vals = [auto_sum["client"], task_sum["client"]]

    x     = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, worker_vals, width,
           color=WORKER_COLOR, alpha=0.85, label="Worker-facing")
    ax.bar(x + width/2, client_vals,  width,
           color=CLIENT_COLOR,  alpha=0.85, label="Client-facing")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Summed mean occurrences per page", **FONT_AXIS)
    ax.set_title("Vocabulary category totals", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    for i, (wv, cv) in enumerate(zip(worker_vals, client_vals)):
        ax.text(i - width/2, wv + 0.01, f"{wv:.3f}",
                ha="center", va="bottom", fontsize=9, color=WORKER_COLOR, fontweight="bold")
        ax.text(i + width/2, cv + 0.01, f"{cv:.3f}",
                ha="center", va="bottom", fontsize=9, color=CLIENT_COLOR, fontweight="bold")

    # --- Right: stacked proportion (automation vs task share per audience) ---
    ax2 = axes[1]
    audiences = ["Worker-facing", "Client-facing"]
    auto_vals = [auto_sum["worker"], auto_sum["client"]]
    task_vals = [task_sum["worker"], task_sum["client"]]
    totals    = [a + t for a, t in zip(auto_vals, task_vals)]

    auto_pct = [a / tot * 100 if tot > 0 else 0 for a, tot in zip(auto_vals, totals)]
    task_pct = [t / tot * 100 if tot > 0 else 0 for t, tot in zip(task_vals, totals)]

    x2 = np.arange(len(audiences))
    b1 = ax2.bar(x2, auto_pct, color="#E76F51", alpha=0.9, label="Automation terms")
    b2 = ax2.bar(x2, task_pct, bottom=auto_pct, color="#457B9D", alpha=0.9, label="Task/Work terms")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(audiences, fontsize=11)
    ax2.set_ylabel("Share of combined vocabulary (%)", **FONT_AXIS)
    ax2.set_title(
        "Vocabulary composition\n"
        "Automation vs Task/Work share per audience",
        fontsize=11
    )
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_ylim(0, 115)
    ax2.spines[["top", "right"]].set_visible(False)

    # Percentage labels inside bars
    for i, (ap, tp) in enumerate(zip(auto_pct, task_pct)):
        if ap > 3:
            ax2.text(i, ap/2, f"{ap:.1f}%", ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold")
        if tp > 3:
            ax2.text(i, ap + tp/2, f"{tp:.1f}%", ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold")

    plt.tight_layout()
    out = output_dir / "H1b_automation_myth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# H1c — Strategic Hypervisibility
# ---------------------------------------------------------------------------

def plot_h1c_hypervisibility(df: pd.DataFrame, output_dir: Path):
    """
    H1c: In client-facing texts, is labour language paired with quality/ethics terms?
    Shows co-occurrence: for each page, does it contain BOTH labour AND quality terms?
    Produces:
      - Left:  % of pages containing labour + quality terms together, by audience
      - Right: mean frequency of quality terms per page, by audience
    """
    log.info("H1c — Plotting strategic hypervisibility...")

    records = []
    for _, row in df.iterrows():
        token_set = set(row["tokens"].split())
        has_labour  = any(t in token_set for t in LABOUR_TERMS)
        has_quality = any(t in token_set for t in QUALITY_TERMS)
        has_both    = has_labour and has_quality

        quality_count = sum(row["tokens"].split().count(t) for t in QUALITY_TERMS)
        labour_count  = sum(row["tokens"].split().count(t) for t in LABOUR_TERMS)

        records.append({
            "audience":      row["audience"],
            "has_labour":    has_labour,
            "has_quality":   has_quality,
            "has_both":      has_both,
            "quality_count": quality_count,
            "labour_count":  labour_count,
        })

    page_df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "H1c — Strategic Hypervisibility: Labour + Quality Language Co-occurrence\n"
        "Does client-facing discourse pair human labour with quality/ethics framing?",
        fontsize=14, fontweight="bold"
    )

    audiences  = ["worker", "client"]
    aud_labels = ["Worker-facing", "Client-facing"]
    colors     = [WORKER_COLOR, CLIENT_COLOR]

    # --- Left: % of pages with labour-only, quality-only, both, neither ---
    ax = axes[0]
    categories = ["Labour only", "Quality only", "Both", "Neither"]
    cat_data   = {aud: [] for aud in audiences}

    for aud in audiences:
        sub   = page_df[page_df.audience == aud]
        total = len(sub)
        labour_only  = ((sub.has_labour) & (~sub.has_quality)).sum() / total * 100
        quality_only = ((~sub.has_labour) & (sub.has_quality)).sum() / total * 100
        both         = sub.has_both.sum() / total * 100
        neither      = ((~sub.has_labour) & (~sub.has_quality)).sum() / total * 100
        cat_data[aud] = [labour_only, quality_only, both, neither]

    x     = np.arange(len(categories))
    width = 0.35

    for i, (aud, label, color) in enumerate(zip(audiences, aud_labels, colors)):
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, cat_data[aud], width,
                        color=color, alpha=0.85, label=label)
        for bar, val in zip(bars, cat_data[aud]):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("% of pages", **FONT_AXIS)
    ax.set_title("Page-level term co-occurrence breakdown", fontsize=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Right: mean quality term frequency per page ---
    ax2 = axes[1]
    means = [page_df[page_df.audience == aud]["quality_count"].mean()
             for aud in audiences]

    bars = ax2.bar(aud_labels, means, color=colors, alpha=0.85, width=0.45,
                   edgecolor="white")
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    ax2.set_ylabel("Mean quality/ethics term occurrences per page", **FONT_AXIS)
    ax2.set_title(
        "Quality/ethics vocabulary intensity\n"
        "Higher in client-facing = strategic hypervisibility",
        fontsize=11
    )
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = output_dir / "H1c_hypervisibility.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")

def print_summary(df: pd.DataFrame, top_terms: dict[str, pd.Series]):
    log.info("=" * 60)
    log.info("CORPUS SUMMARY")
    log.info(f"  Total pages analysed : {len(df)}")
    for aud in ["worker", "client"]:
        sub = df[df.audience == aud]
        domains = sub["domain"].nunique()
        log.info(f"  {aud.capitalize():8s}: {len(sub):5d} pages across {domains} websites")

    log.info("\nTOP 10 TERMS PER AUDIENCE")
    for aud in ["worker", "client"]:
        terms = top_terms[aud].head(10)
        log.info(f"  {aud.upper()}:")
        for term, score in terms.items():
            log.info(f"    {term:<30s} {score:.5f}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(db_path: str = DB_PATH):
    log.info("=" * 60)
    log.info("TF-IDF ANALYSIS — PROOF OF CONCEPT")
    log.info(f"  Database  : {db_path}")
    log.info(f"  Output dir: {OUTPUT_DIR}")
    log.info("=" * 60)

    # 1. Load
    df = load_tfidf_data(db_path)

    # 2. Fit TF-IDF
    vec, matrix, feature_names = compute_tfidf(df)

    # 3. Compute aggregated scores
    top_terms = top_terms_by_audience(df, matrix, feature_names)
    domain_df = top_terms_by_domain(df, matrix, feature_names)

    # 4. Print summary
    print_summary(df, top_terms)

    # 5. Visualizations — general TF-IDF
    plot_top_terms(top_terms, OUTPUT_DIR)
    plot_heatmap(domain_df, df, OUTPUT_DIR)
    plot_wordclouds(df, matrix, feature_names, OUTPUT_DIR)
    plot_term_overlap(top_terms, OUTPUT_DIR)

    # 6. H1 hypothesis visualizations
    log.info("-" * 60)
    log.info("Running H1 hypothesis visualisations...")
    plot_h1a_visibility_gap(df, OUTPUT_DIR)
    plot_h1b_automation_myth(df, OUTPUT_DIR)
    plot_h1c_hypervisibility(df, OUTPUT_DIR)

    log.info("=" * 60)
    log.info(f"All figures saved to: {OUTPUT_DIR}/")
    log.info("  --- General TF-IDF ---")
    log.info("  01_top_tfidf_terms.png   — bar charts worker vs client")
    log.info("  02_tfidf_heatmap.png     — term heatmap across websites")
    log.info("  03_wordclouds.png        — word clouds per audience")
    log.info("  04_distinctive_terms.png — terms exclusive to each audience")
    log.info("  --- H1 Hypotheses ---")
    log.info("  H1a_visibility_gap.png   — labour term frequency + ratio by audience")
    log.info("  H1b_automation_myth.png  — automation vs task vocabulary by audience")
    log.info("  H1c_hypervisibility.png  — labour+quality co-occurrence by audience")
    log.info("=" * 60)


if __name__ == "__main__":
    run()