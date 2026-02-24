"""
TF-IDF Analysis — Proof of Concept
===================================
Reads preprocessed data from pages_tfidf and produces:
1. Top TF-IDF terms per audience (worker vs client) — bar charts
2. Term frequency heatmap across individual websites
3. Word clouds per audience type

Output: figures saved to ./output/test directory
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
        mask        = df["audience"] == audience
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
        mask        = df["domain"] == domain
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
        mask        = df["audience"] == audience
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
# 4. Summary stats table
# ---------------------------------------------------------------------------

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

    # 5. Visualizations
    plot_top_terms(top_terms, OUTPUT_DIR)
    plot_heatmap(domain_df, df, OUTPUT_DIR)
    plot_wordclouds(df, matrix, feature_names, OUTPUT_DIR)
    plot_term_overlap(top_terms, OUTPUT_DIR)

    log.info("=" * 60)
    log.info(f"All figures saved to: {OUTPUT_DIR}/")
    log.info("  01_top_tfidf_terms.png  — bar charts worker vs client")
    log.info("  02_tfidf_heatmap.png    — term heatmap across websites")
    log.info("  03_wordclouds.png       — word clouds per audience")
    log.info("  04_distinctive_terms.png — terms exclusive to each audience")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
