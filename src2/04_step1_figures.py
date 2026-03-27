"""
04_step1_figures.py
===================
All Step 1 visualisations for the DarkSideofAI thesis.

Consolidates:
  src/03_visualise_step1.py               (Figures 1–6, S1)
  src/03b_visualise_distinctiveness_topics.py  (Figures 7–12)
  src/04_step1_narrative_figures.py       (Narrative Figures N1–N7)
  src/05_pca_domain_figures.py            (PCA Figures P-A through P-F)

Pipeline position:
  Stage 4 — Visualisation (run after 02_step1_analysis.py and 03_step1_topics.py)
  Prerequisites:
    02_step1_analysis.py  (keyness_results, cooccurrence_results,
                           platform_term_counts, distinctiveness_matrix,
                           term_exclusivity, domain_quality)
    03_step1_topics.py    (topic_terms, document_topics, topic_audience_profile,
                           step2_sample)

Figures produced:
  ── Core Step 1 (src2/ output/step_1/) ──────────────────────────────────────
  fig1_keyness_bar            Top B2B/B2W distinctive terms by LL G²
  fig2_cooccurrence_network   PMI collocate profiles for selected focus terms
  fig3_frequency_comparison   Hypothesis vocabulary frequencies grouped by H1a/b/c
  fig4_within_pair            Within-pair diverging bar (appen, toloka)
  fig5_platform_heatmap       Term frequency heatmap across all platforms
  fig6_theory_cooccurrence    Theory-driven co-occurrence for all H1a-c terms
  figS1_register_scatter      Full vocabulary register gap scatter (log-log)
  fig7_distinctiveness_heatmap  Domain JSD matrix (02b data)
  fig8_exclusivity_volcano    Term exclusivity vs frequency volcano plot
  fig9_pca_scatter            Document-topic PCA scatter by audience
  fig10_topic_audience_profile  Per-topic B2B/B2W diverging bar chart
  fig11_collocate_divergence  PMI profile divergence ranking
  fig12_step2_sample_map      PCA with Step 2 sample highlighted

  ── Narrative sequence (output/step_1/narrative/) ───────────────────────────
  N1_vocab_terrain            Register gap scatter (argumentative version)
  N2_exclusive_vocab          Top exclusive vocabulary per side
  N3_shared_divergent         Shared terms, divergent collocate contexts
  N4_topic_profiles           Topic audience balance (unified chart)
  N5_topic_hyp_alignment      Topic-hypothesis alignment map
  N6_document_space           PCA document space
  N7_step2_sample_map         Step 2 sample overlay on PCA

  ── PCA domain analysis (output/step_1/pca/) ─────────────────────────────────
  P-A_domain_landscape        Domain strip plot: mean PC1 per platform
  P-B_pca_scatter             PCA scatter coloured by domain
  P-C_within_domain_var       Within-domain PC1 variation strip plot
  P-D_step2_sample_map        Step 2 sample coloured by hypothesis
  P-E_topic_pc_loadings       Topic loadings on PC1 and PC2
  P-F_pca_domain_shapes       PCA scatter with per-domain shapes

Dual style:
  pub — clean, thesis-ready (white background, no value annotations)
  exp — exploratory/annotated (tinted background, value labels, centroids)

Missing-table resilience:
  Each figure checks whether its prerequisite tables exist before querying.
  If a table is missing the figure is skipped with a WARNING log message.
  This allows partial runs (e.g. if 03_step1_topics.py has not yet been run).

Usage:
    python3 src2/04_step1_figures.py
    # Toggle figure groups in FIGURE_GROUPS dict at the top of main().
"""

import json
import math
import logging
import sqlite3
import statistics
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH        = "data/scraping_2.db"
OUTPUT_DIR     = Path("output/step_1/")
OUTPUT_NARR    = Path("output/step_1/narrative/")
OUTPUT_PCA     = Path("output/step_1/pca/")
DPI            = 150
EXT            = "jpg"      # "png" for lossless; "pdf" for vector

# ── Figure parameters ────────────────────────────────────────────────────
TOP_N          = 20     # terms per direction in keyness bar charts
TOP_PAIR_N     = 12     # terms per direction in within-pair charts
MIN_COFREQ     = 10     # minimum co-occurrence frequency (= DB minimum)
COOC_TOP_N     = 8      # collocates shown per side in network/theory figs
NETWORK_N      = 18     # max collocates shown per side in network fig
F2_TOP_N       = 25     # terms per direction in narrative F2
F2_LL_MIN      = 10.83  # minimum |LL| (p < 0.001)
HEATMAP_MAX_DOMAINS = 40
VOLCANO_LABEL_N     = 20
VOLCANO_FREQ_FLOOR  = 20
PCA_ALPHA           = 0.55
PCA_SIZE            = 18
TOPIC_BAR_TOP_TERMS = 5
DIVERGENCE_TOP_N    = 30
SAMPLE_MARKER_SIZE  = 70
FE_TOP_N       = 12   # topics to show at each end of each PC
FE_TOP_TERMS   = 4    # topic terms to show per bar label
FE_MIN_DOCS    = 5    # minimum dominant documents to include topic

# Focuses terms for Fig2 PMI network
COOC_FOCUS_TERMS = ["human", "annotation", "autonomous", "earn"]

# Theory terms for Fig6
FIG6_GROUPS = {
    "H1a — Labour visibility":        ["worker", "labour", "task", "earn", "pay"],
    "H1b — Automation myth":           ["autonomous", "machine", "automation", "intelligent"],
    "H1c — Strategic hypervisibility": ["human", "quality", "oversight", "annotation", "label"],
}
FIG6_TOP_N = 8

# Domains excluded from PCA (same as 03_step1_topics.py)
PCA_EXCLUDE_DOMAINS = {"www.sama.com", "mindy-support.com", "scale.com"}

# ---------------------------------------------------------------------------
# Colour palette — consistent across all figures
# ---------------------------------------------------------------------------

C_CLIENT  = "#1B4F8A"   # deep blue  — client (B2B) register
C_WORKER  = "#C0392B"   # deep red   — worker (B2W) register
C_SHARED  = "#6C757D"   # grey       — shared / neutral
C_H1C     = "#E67E22"   # orange     — H1c (strategic hypervisibility)
C_BG_PUB  = "#FFFFFF"   # pure white — pub style background
C_BG_EXP  = "#F7F9FC"   # off-white  — exp style background
C_GRID    = "#DEE2E6"   # light grey — axis grid lines
C_TEXT    = "#1A1A2E"   # near-black — primary text
C_SUBTEXT = "#6C757D"   # medium grey — secondary / annotation text
C_ACCENT  = "#E67E22"   # orange     — Step 2 sample highlights

PAL = dict(
    b2b=C_CLIENT, b2w=C_WORKER, h1a=C_WORKER, h1b=C_CLIENT, h1c=C_H1C,
    neutral=C_SHARED, grid=C_GRID, text=C_TEXT, sub=C_SUBTEXT, bg=C_BG_PUB,
    highlight="#F4D03F",
)

FONT_TITLE = {"fontsize": 13, "fontweight": "bold",   "color": C_TEXT}
FONT_LABEL = {"fontsize": 10, "fontweight": "normal", "color": C_SUBTEXT}
FONT_ANNOT = {"fontsize":  8, "color": C_SUBTEXT}

HYPOTHESIS_VOCAB = {
    "H1a — Labour visibility": {
        "terms":  {"worker", "labour", "task", "job", "earn", "pay", "payment",
                   "annotator", "gig", "contractor", "wage", "labeller",
                   "freelance", "income"},
        "color":  C_WORKER,
        "marker": "o",
    },
    "H1b — Automation myth": {
        "terms":  {"autonomous", "machine", "automate", "automation", "algorithm",
                   "pipeline", "deploy", "inference", "neural", "llm",
                   "intelligent", "scalable"},
        "color":  C_CLIENT,
        "marker": "s",
    },
    "H1c — Strategic hypervisibility": {
        "terms":  {"human", "quality", "oversight", "annotation", "label",
                   "expert", "accuracy", "datum", "review", "verification"},
        "color":  C_H1C,
        "marker": "^",
    },
}

HYP_COLORS = {
    "H1a": C_WORKER,
    "H1b": C_CLIENT,
    "H1c": C_H1C,
    "H2":  "#8E44AD",
    "H3":  "#27AE60",
    "H4":  "#2980B9",
    "H5":  "#D35400",
}

# Artifact terms — display-time filter (secondary to DB-level exclusion)
ARTIFACT_TERMS = {
    "cookie", "set_cookie", "cooky", "/hr", "/hr_remote", "remote_apply",
    "feb", "opportunity_feb", "faq", "faq_help", "help_desk", "desk",
    "subscribe", "website", "account", "access", "enable", "microworker",
    "shall", "youtube", "zeynep", "koouchnir", "gavrilov", "unga", "gary",
    "yalda", "monarch", "warhol", "fremont", "pittsburgh", "mpii",
    "experience.with", "rhml", "ead", "cc0", "ft",
    "hole", "overfit", "surprised", "christmas", "morale", "high-quality",
    "slash", "500", "pickup", "loophole", "conceptually", "housing",
    "firefighting", "sidestep", "wary", "downward", "jira", "voluman",
    "squeeze", "retrofit", "yt", "ml",
    "deciphering", "trafficking", "recap", "ueberwinden", "bildbearbeitung",
    "sicherstellung", "kunst", "human-le", "pto", "generous",
    "dhanesh", "ramachandram", "outlet", "daniela", "braga", "forbe",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def has_table(conn, name: str) -> bool:
    return bool(conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone())


def apply_base_style(ax, bg):
    ax.set_facecolor(bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_axisbelow(True)


def save(fig, name: str, style: str, out_dir: Path = None):
    """Save figure to <out_dir>/<name>_<style>.<EXT>."""
    d = (out_dir or OUTPUT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}_{style}.{EXT}"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(),
                format=EXT if EXT in ("jpg", "jpeg") else None)
    log.info(f"  Saved: {path}")
    plt.close(fig)


def shorten_domain(d: str) -> str:
    for ext in [".com", ".ai", ".net", ".org", ".tech", ".me"]:
        d = d.replace(ext, "")
    return d.replace("www.", "")


def apply_seaborn_theme():
    if _HAS_SNS:
        sns.set_theme(style="whitegrid", palette="muted",
                      rc={"figure.facecolor": C_BG_PUB,
                          "axes.facecolor":   C_BG_PUB,
                          "axes.edgecolor":   C_GRID,
                          "grid.color":       C_GRID,
                          "text.color":       C_TEXT})


def _artifact_ph():
    """Parameterised NOT IN placeholder for ARTIFACT_TERMS."""
    return ",".join("?" * len(ARTIFACT_TERMS))


# ---------------------------------------------------------------------------
# DB query helpers
# ---------------------------------------------------------------------------

def fetch_top_client(conn, comparison, n=TOP_N):
    rows = conn.execute(f"""
        SELECT term, ll_score, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = ?
          AND ll_score > 0
          AND term_type = 'unigram'
          AND term NOT IN ({_artifact_ph()})
        ORDER BY ll_score DESC LIMIT ?
    """, [comparison] + list(ARTIFACT_TERMS) + [n]).fetchall()
    return [dict(r) for r in rows]


def fetch_top_worker(conn, comparison, n=TOP_N):
    rows = conn.execute(f"""
        SELECT term, ll_score, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = ?
          AND ll_score < 0
          AND term_type = 'unigram'
          AND term NOT IN ({_artifact_ph()})
        ORDER BY ll_score ASC LIMIT ?
    """, [comparison] + list(ARTIFACT_TERMS) + [n]).fetchall()
    return [dict(r) for r in rows]


def fetch_cooccurrence(conn, comparison, audience, focus, min_freq=MIN_COFREQ,
                       top_n=NETWORK_N):
    rows = conn.execute(f"""
        SELECT collocate, pmi, cofreq
        FROM cooccurrence_results
        WHERE comparison = ? AND audience = ? AND focus_term = ?
          AND cofreq >= ?
          AND collocate NOT IN ({_artifact_ph()})
        ORDER BY pmi DESC LIMIT ?
    """, [comparison, audience, focus, min_freq]
         + list(ARTIFACT_TERMS) + [top_n]).fetchall()
    return [dict(r) for r in rows]


def fetch_term_freqs(conn, terms, comparison="cross_platform"):
    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(f"""
        SELECT term, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = ? AND term IN ({placeholders}) AND term_type = 'unigram'
    """, [comparison] + terms).fetchall()
    return {r["term"]: dict(r) for r in rows}


def fetch_platform_terms(conn, terms):
    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(f"""
        SELECT domain, audience, term, rel_freq
        FROM platform_term_counts
        WHERE term IN ({placeholders}) AND term_type = 'unigram'
    """, terms).fetchall()
    result, audiences = defaultdict(dict), {}
    for r in rows:
        result[r["domain"]][r["term"]] = r["rel_freq"]
        audiences[r["domain"]] = r["audience"]
    return dict(result), audiences


# ===========================================================================
# ── SECTION A: Core Step 1 Figures (from 03_visualise_step1.py) ────────────
# ===========================================================================

def fig_keyness_bar(conn, style):
    """Fig 1 — Cross-platform keyness: top B2B and B2W distinctive terms."""
    log.info(f"Fig 1 — Keyness bar ({style})")
    client_top = fetch_top_client(conn, "cross_platform", TOP_N)
    worker_top = list(reversed(fetch_top_worker(conn, "cross_platform", TOP_N)))
    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor=bg)
    fig.subplots_adjust(wspace=0.5)

    for ax, data, colour, label in [
        (axes[0], client_top, C_CLIENT, "Client-distinctive (B2B)"),
        (axes[1], worker_top, C_WORKER, "Worker-distinctive (B2W)"),
    ]:
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center",
                    transform=ax.transAxes, **FONT_LABEL)
            ax.set_title(label, **FONT_TITLE)
            continue
        terms  = [r["term"] for r in data]
        scores = [abs(r["ll_score"]) for r in data]
        y_pos  = np.arange(len(terms))
        bars   = ax.barh(y_pos, scores, color=colour, alpha=0.85,
                         edgecolor="white", linewidth=0.4, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9, color=C_TEXT)
        ax.set_xlabel("Log-likelihood (G²)", **FONT_LABEL)
        ax.set_title(label, **FONT_TITLE, pad=12)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
        apply_base_style(ax, bg)
        if style == "exp":
            for i, (bar, row) in enumerate(zip(bars, data)):
                ax.text(bar.get_width() + max(scores) * 0.01, i,
                        f"B2B {row['rel_freq_client']:.2f}‰ / B2W {row['rel_freq_worker']:.2f}‰",
                        va="center", **FONT_ANNOT)

    fig.suptitle("Cross-Platform Keyness Analysis: B2B vs B2W Distinctive Terms",
                 **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01,
             "Log-likelihood G²  •  Artifact terms excluded  •  Unigrams only",
             ha="center", **FONT_ANNOT)
    save(fig, "fig1_keyness_bar", style)


def fig_cooccurrence_network(conn, style):
    """Fig 2 — PMI collocate profiles for theory-selected key terms."""
    log.info(f"Fig 2 — Co-occurrence network ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    n_terms = len(COOC_FOCUS_TERMS)
    fig, axes = plt.subplots(n_terms, 2,
                             figsize=(16, n_terms * 3.2), facecolor=bg)
    fig.subplots_adjust(hspace=0.55, wspace=0.55)

    for row_idx, focus in enumerate(COOC_FOCUS_TERMS):
        for col_idx, (audience, colour, reg_label) in enumerate([
            ("client", C_CLIENT, "B2B"),
            ("worker", C_WORKER, "B2W"),
        ]):
            ax = axes[row_idx, col_idx]
            cooc = fetch_cooccurrence(conn, "cross_platform", audience, focus,
                                      min_freq=MIN_COFREQ, top_n=COOC_TOP_N)
            ax.set_title(f'"{focus}"  —  {reg_label}',
                         fontsize=10, fontweight="bold", color=colour, pad=8)
            ax.set_facecolor(bg)
            if not cooc:
                ax.text(0.5, 0.5, "No collocates above threshold",
                        ha="center", va="center",
                        transform=ax.transAxes, **FONT_LABEL)
                ax.axis("off")
                continue
            collocates = [r["collocate"] for r in reversed(cooc)]
            pmi_vals   = [r["pmi"]       for r in reversed(cooc)]
            y_pos      = np.arange(len(collocates))
            bars = ax.barh(y_pos, pmi_vals, color=colour, alpha=0.80,
                           edgecolor="white", linewidth=0.4, height=0.65)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(collocates, fontsize=8.5, color=C_TEXT)
            ax.set_xlabel("PMI score", **FONT_LABEL)
            ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
            apply_base_style(ax, bg)
            if style == "exp":
                for bar, row in zip(bars, reversed(cooc)):
                    ax.text(bar.get_width() + max(pmi_vals) * 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            f"f={row['cofreq']}",
                            va="center", fontsize=6.5, color=C_SUBTEXT)

    fig.suptitle(
        "Co-occurrence Profiles: Top PMI Collocates for Key Terms by Audience Register",
        **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01,
             f"PMI = Pointwise Mutual Information  •  Min co-freq={MIN_COFREQ}  "
             "•  Directed to Step 2 close reading",
             ha="center", **FONT_ANNOT)
    save(fig, "fig2_cooccurrence_network", style)


def fig_frequency_comparison(conn, style):
    """Fig 3 — Relative frequencies of hypothesis terms by group."""
    log.info(f"Fig 3 — Frequency comparison ({style})")
    groups = {
        "H1a — Labour visibility":        ["worker", "work", "job", "earn", "pay", "payment"],
        "H1b — Automation myth":           ["autonomous", "automate", "automation", "machine", "model"],
        "H1c — Strategic hypervisibility": ["human", "quality", "oversight", "annotation", "label"],
    }
    all_terms = [t for v in groups.values() for t in v]
    freq_data = fetch_term_freqs(conn, all_terms)
    terms  = [t for t in all_terms if t in freq_data]
    c_freq = [freq_data[t]["rel_freq_client"] for t in terms]
    w_freq = [freq_data[t]["rel_freq_worker"] for t in terms]
    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=bg)
    ax.set_facecolor(bg)
    x      = np.arange(len(terms))
    width  = 0.38
    bars_c = ax.bar(x - width/2, c_freq, width, label="Client (B2B)",
                    color=C_CLIENT, alpha=0.88, edgecolor="white", linewidth=0.5)
    bars_w = ax.bar(x + width/2, w_freq, width, label="Worker (B2W)",
                    color=C_WORKER, alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=35, ha="right", fontsize=9, color=C_TEXT)
    ax.set_ylabel("Relative frequency (per 1,000 tokens)", **FONT_LABEL)
    ax.set_title("Relative Frequency of Theoretically Key Terms by Audience Register",
                 **FONT_TITLE, pad=12)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
    apply_base_style(ax, bg)

    group_sizes = [len(v) for v in groups.values()]
    boundaries, running = [], 0
    for size in group_sizes[:-1]:
        running += size
        boundaries.append(running - 0.5)
    for xb in boundaries:
        ax.axvline(xb, color=C_GRID, linewidth=1.2, linestyle="--")

    centres, running = [], 0
    for size in group_sizes:
        centres.append(running + size / 2 - 0.5)
        running += size
    ymax = ax.get_ylim()[1]
    for xc, gl in zip(centres, groups.keys()):
        ax.text(xc, ymax * 0.97, gl, ha="center", va="top",
                fontsize=8, color=C_SUBTEXT, style="italic")

    if style == "exp":
        for bar in list(bars_c) + list(bars_w):
            h = bar.get_height()
            if h > 0.08:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        f"{h:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=C_SUBTEXT)

    ax.legend(frameon=False, fontsize=9)
    fig.text(0.5, -0.03,
             "Terms grouped by hypothesis  •  Frequency per 1,000 lemmatized tokens  •  Cross-platform corpus",
             ha="center", **FONT_ANNOT)
    save(fig, "fig3_frequency_comparison", style)


def fig_within_pair(conn, style):
    """Fig 4 — Within-pair keyness (appen, toloka)."""
    log.info(f"Fig 4 — Within-pair ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=bg)
    fig.subplots_adjust(wspace=0.55)

    for ax, company_id, pair_label in [
        (axes[0], "appen",  "Appen (B2B) vs CrowdGen (B2W)"),
        (axes[1], "toloka", "Toloka (B2B) vs Mindrift (B2W)"),
    ]:
        client_top = fetch_top_client(conn, company_id, TOP_PAIR_N)
        worker_top = fetch_top_worker(conn, company_id, TOP_PAIR_N)
        if not client_top and not worker_top:
            ax.text(0.5, 0.5, "No data", ha="center",
                    transform=ax.transAxes, **FONT_LABEL)
            ax.set_title(pair_label, **FONT_TITLE)
            continue

        worker_rev = list(reversed(worker_top))
        all_terms  = ([r["term"] for r in worker_rev] + ["— — —"] +
                      [r["term"] for r in client_top])
        all_scores = ([-abs(r["ll_score"]) for r in worker_rev] + [0] +
                      [abs(r["ll_score"]) for r in client_top])
        colours    = ([C_WORKER] * len(worker_rev) + ["none"] +
                      [C_CLIENT] * len(client_top))

        y_pos = np.arange(len(all_terms))
        bars  = ax.barh(y_pos, all_scores, color=colours, alpha=0.85,
                        edgecolor="white", linewidth=0.4, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_terms, fontsize=8.5, color=C_TEXT)
        ax.axvline(0, color=C_TEXT, linewidth=0.8)
        ax.set_xlabel("Log-likelihood G²  (← Worker | Client →)", **FONT_LABEL)
        ax.set_title(pair_label, **FONT_TITLE, pad=10)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
        ax.yaxis.grid(False)
        apply_base_style(ax, bg)

        if style == "exp":
            nonzero = [abs(s) for s in all_scores if s != 0]
            max_abs = max(nonzero) if nonzero else 1
            for bar, score in zip(bars, all_scores):
                if score != 0:
                    offset = max_abs * 0.02
                    ax.text(score + (offset if score > 0 else -offset),
                            bar.get_y() + bar.get_height() / 2,
                            f"{abs(score):.0f}", va="center",
                            ha="left" if score > 0 else "right",
                            fontsize=6.5, color=C_SUBTEXT)

    fig.legend(handles=[
        mpatches.Patch(color=C_CLIENT, label="Client-distinctive (B2B)"),
        mpatches.Patch(color=C_WORKER, label="Worker-distinctive (B2W)"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Within-Pair Keyness: Same Company, Different Audience Register",
                 **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01,
             "Paired comparison controls for company-level variation  •  "
             "Differences attributable to audience only",
             ha="center", **FONT_ANNOT)
    save(fig, "fig4_within_pair", style)


def fig_platform_heatmap(conn, style):
    """Fig 5 — Term frequency heatmap across all platforms."""
    log.info(f"Fig 5 — Platform heatmap ({style})")
    heatmap_terms = [
        "worker", "job", "earn", "pay",
        "autonomous", "automate", "machine", "model",
        "human", "quality", "oversight", "annotation", "label", "datum",
    ]
    col_boundaries = [3.5, 7.5]
    col_centres    = [1.5, 5.5, 10.5]
    col_labels     = ["H1a", "H1b", "H1c"]

    term_data, audiences = fetch_platform_terms(conn, heatmap_terms)
    domains_client = sorted([d for d, a in audiences.items() if a == "client"])
    domains_both   = sorted([d for d, a in audiences.items() if a == "both"])
    domains_worker = sorted([d for d, a in audiences.items() if a == "worker"])
    all_domains    = domains_client + domains_both + domains_worker
    n_client_rows  = len(domains_client) + len(domains_both)

    matrix = np.zeros((len(all_domains), len(heatmap_terms)))
    for i, domain in enumerate(all_domains):
        for j, term in enumerate(heatmap_terms):
            matrix[i, j] = term_data.get(domain, {}).get(term, 0)

    vmax = np.percentile(matrix[matrix > 0], 95) if (matrix > 0).any() else 1
    if style == "pub":
        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        plot_matrix = matrix / row_max
        vmax_plot   = 1.0
        cbar_label  = "Normalised relative frequency (row max = 1)"
        cmap        = "Blues"
    else:
        plot_matrix = np.clip(matrix, 0, vmax)
        vmax_plot   = vmax
        cbar_label  = f"Relative frequency per 1,000 tokens (capped at p95={vmax:.1f})"
        cmap        = "YlOrRd"

    short_labels = [shorten_domain(d) for d in all_domains]
    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(14, max(8, len(all_domains) * 0.5)),
                           facecolor=bg)
    ax.set_facecolor(bg)
    im = ax.imshow(plot_matrix, aspect="auto", cmap=cmap,
                   vmin=0, vmax=vmax_plot, interpolation="nearest")
    ax.set_xticks(np.arange(len(heatmap_terms)))
    ax.set_xticklabels(heatmap_terms, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(all_domains)))
    ax.set_yticklabels(short_labels, fontsize=9)
    for i, domain in enumerate(all_domains):
        aud = audiences.get(domain, "both")
        col = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SUBTEXT
        ax.get_yticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_fontweight("bold" if aud != "both" else "normal")

    ax.axhline(n_client_rows - 0.5, color=C_TEXT, linewidth=1.5, linestyle="--")
    ax.text(len(heatmap_terms) - 0.4, n_client_rows - 0.7,
            "▲ Client  |  Worker ▼", ha="right", va="bottom",
            fontsize=8, color=C_TEXT)
    for xb in col_boundaries:
        ax.axvline(xb, color=C_TEXT, linewidth=1.0, linestyle=":")
    for xc, gl in zip(col_centres, col_labels):
        ax.text(xc, -1.1, gl, ha="center", va="top",
                fontsize=8, color=C_SUBTEXT, style="italic",
                transform=ax.get_xaxis_transform())
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8, color=C_SUBTEXT)
    cbar.ax.tick_params(labelsize=7)
    if style == "exp":
        for i in range(len(all_domains)):
            for j in range(len(heatmap_terms)):
                val = plot_matrix[i, j]
                if val > vmax_plot * 0.05:
                    ax.text(j, i, f"{matrix[i,j]:.1f}",
                            ha="center", va="center", fontsize=5.5,
                            color="white" if val > vmax_plot * 0.6 else C_TEXT)
    ax.set_title("Term Frequency Heatmap Across Platforms", **FONT_TITLE, pad=12)
    fig.text(0.5, -0.03,
             "Blue labels = client  •  Red labels = worker  •  Columns grouped by hypothesis",
             ha="center", **FONT_ANNOT)
    save(fig, "fig5_platform_heatmap", style)


def fig_theory_cooccurrence(conn, style):
    """Fig 6 — Theory-driven co-occurrence profiles for all H1a-c terms."""
    log.info(f"Fig 6 — Theory co-occurrence ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    ordered_terms = [t for terms in FIG6_GROUPS.values() for t in terms]
    available = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT focus_term FROM cooccurrence_results "
            "WHERE comparison = 'cross_platform'"
        ).fetchall()
    }
    focus_terms = [t for t in ordered_terms if t in available]
    if not focus_terms:
        log.warning("  No theory-driven focus terms in DB — re-run 02_step1_analysis.py.")
        return

    n_terms = len(focus_terms)
    fig, axes = plt.subplots(n_terms, 2, figsize=(16, n_terms * 3.2), facecolor=bg)
    if n_terms == 1:
        axes = [axes]
    fig.subplots_adjust(hspace=0.55, wspace=0.55)

    for row_idx, focus in enumerate(focus_terms):
        for col_idx, (audience, colour, reg_label) in enumerate([
            ("client", C_CLIENT, "B2B"),
            ("worker", C_WORKER, "B2W"),
        ]):
            ax = axes[row_idx][col_idx] if n_terms > 1 else axes[col_idx]
            ax.set_facecolor(bg)
            cooc = fetch_cooccurrence(conn, "cross_platform", audience, focus,
                                      min_freq=MIN_COFREQ, top_n=FIG6_TOP_N)
            group_short = next(
                (k.split("—")[0].strip() for k, v in FIG6_GROUPS.items() if focus in v), "")
            ax.set_title(f'"{focus}"  —  {reg_label}  [{group_short}]',
                         fontsize=10, fontweight="bold", color=colour, pad=8)
            if not cooc:
                ax.text(0.5, 0.5, "No collocates above threshold",
                        ha="center", va="center",
                        transform=ax.transAxes, **FONT_LABEL)
                ax.axis("off")
                continue
            collocates = [r["collocate"] for r in reversed(cooc)]
            pmi_vals   = [r["pmi"]       for r in reversed(cooc)]
            y_pos      = np.arange(len(collocates))
            bars = ax.barh(y_pos, pmi_vals, color=colour, alpha=0.80,
                           edgecolor="white", linewidth=0.4, height=0.65)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(collocates, fontsize=8.5, color=C_TEXT)
            ax.set_xlabel("PMI score", **FONT_LABEL)
            ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
            apply_base_style(ax, bg)
            if style == "exp":
                for bar, row in zip(bars, reversed(cooc)):
                    ax.text(bar.get_width() + max(pmi_vals) * 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            f"f={row['cofreq']}", va="center",
                            fontsize=6.5, color=C_SUBTEXT)

    fig.suptitle("Theory-Driven Co-occurrence Profiles: Terms Central to H1a–H1c",
                 **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01,
             f"Terms selected for theoretical relevance  •  PMI, min co-freq={MIN_COFREQ}",
             ha="center", **FONT_ANNOT)
    save(fig, "fig6_theory_cooccurrence", style)


def fig_register_scatter(conn, style):
    """Fig S1 — Full vocabulary register gap scatter (log-log)."""
    log.info(f"Fig S1 — Register scatter ({style})")
    all_rows = conn.execute(f"""
        SELECT term, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = 'cross_platform' AND term_type = 'unigram'
          AND term NOT IN ({_artifact_ph()})
    """, list(ARTIFACT_TERMS)).fetchall()
    if not all_rows:
        log.warning("  No keyness data — skipping.")
        return

    term_to_hyp = {}
    for hyp_key, cfg in HYPOTHESIS_VOCAB.items():
        for t in cfg["terms"]:
            if t not in term_to_hyp:
                term_to_hyp[t] = (hyp_key, cfg)

    EPS = 0.001
    bg_x, bg_y = [], []
    hyp_data = {k: {"x": [], "y": [], "labels": []} for k in HYPOTHESIS_VOCAB}
    for row in all_rows:
        xv = math.log10(max(row["rel_freq_worker"], EPS))
        yv = math.log10(max(row["rel_freq_client"],  EPS))
        if row["term"] in term_to_hyp:
            hyp_key, _ = term_to_hyp[row["term"]]
            hyp_data[hyp_key]["x"].append(xv)
            hyp_data[hyp_key]["y"].append(yv)
            hyp_data[hyp_key]["labels"].append(row["term"])
        else:
            bg_x.append(xv)
            bg_y.append(yv)

    bg_colour = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=bg_colour)
    ax.set_facecolor(bg_colour)
    ax.scatter(bg_x, bg_y, s=7, color=C_GRID, alpha=0.30,
               edgecolors="none", zorder=1)
    all_x = bg_x + [v for d in hyp_data.values() for v in d["x"]]
    all_y = bg_y + [v for d in hyp_data.values() for v in d["y"]]
    if all_x and all_y:
        lo = min(min(all_x), min(all_y)) - 0.15
        hi = max(max(all_x), max(all_y)) + 0.15
        ax.plot([lo, hi], [lo, hi], color=C_SUBTEXT, linewidth=1.0,
                linestyle="--", alpha=0.55, zorder=2)
    for hyp_key, cfg in HYPOTHESIS_VOCAB.items():
        layer = hyp_data[hyp_key]
        if not layer["x"]:
            continue
        ax.scatter(layer["x"], layer["y"], s=70, color=cfg["color"], alpha=0.92,
                   edgecolors="white", linewidths=0.6, marker=cfg["marker"], zorder=4)
        for xv, yv, lbl in zip(layer["x"], layer["y"], layer["labels"]):
            ax.annotate(lbl, (xv, yv), fontsize=7.5, color=cfg["color"],
                        fontweight="bold", textcoords="offset points",
                        xytext=(5, 4), zorder=5)

    ax.text(0.97, 0.03, "← more B2W-distinctive →", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=C_WORKER, style="italic", alpha=0.75)
    ax.text(0.03, 0.97, "← more B2B-distinctive →", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color=C_CLIENT, style="italic",
            alpha=0.75, rotation=90)
    ax.set_xlabel("log₁₀(relative frequency in B2W — worker register)", **FONT_LABEL)
    ax.set_ylabel("log₁₀(relative frequency in B2B — client register)", **FONT_LABEL)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg_colour)

    legend_entries = [
        plt.Line2D([0], [0], color=C_SUBTEXT, linestyle="--", alpha=0.6,
                   label="Equal frequency (y = x)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GRID,
                   markersize=7, alpha=0.6,
                   label=f"All terms (n={len(bg_x)+sum(len(d['x']) for d in hyp_data.values()):,})"),
    ]
    for hyp_key, cfg in HYPOTHESIS_VOCAB.items():
        short = hyp_key.split("—")[0].strip()
        legend_entries.append(
            plt.Line2D([0], [0], marker=cfg["marker"], color="w",
                       markerfacecolor=cfg["color"], markersize=9, label=short)
        )
    ax.legend(handles=legend_entries, loc="upper left", frameon=True,
              fontsize=8.5, facecolor=bg_colour, edgecolor=C_GRID)
    ax.set_title("Register Gap: Full Vocabulary Distribution by Audience Frequency",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "Each dot = one unigram  •  Above diagonal = B2B-distinctive  •  "
             "Below = B2W-distinctive  •  Hypothesis vocabulary should cluster in predicted regions",
             ha="center", **FONT_ANNOT)
    save(fig, "figS1_register_scatter", style)


# ===========================================================================
# ── SECTION B: Distinctiveness & Topic Figures (from 03b_) ─────────────────
# ===========================================================================

def fig_distinctiveness_heatmap(conn, style):
    """Fig 7 — Cross-domain linguistic distinctiveness heatmap (JSD)."""
    if not has_table(conn, "distinctiveness_matrix"):
        log.warning("  Fig 7 skipped — distinctiveness_matrix not found.")
        return
    log.info(f"Fig 7 — Distinctiveness heatmap ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    rows = conn.execute("""
        SELECT domain_a, domain_b, jsd
        FROM distinctiveness_matrix ORDER BY domain_a, domain_b
    """).fetchall()
    if not rows:
        log.warning("  Fig 7 skipped — no JSD data.")
        return

    domains_ordered = []
    seen = set()
    # Get audience info
    aud_map = {}
    aud_rows = conn.execute(
        "SELECT DISTINCT domain, audience FROM domain_quality"
    ).fetchall() if has_table(conn, "domain_quality") else []
    for r in aud_rows:
        aud_map[r["domain"]] = r["audience"]

    jsd_map = {}
    for r in rows:
        jsd_map[(r["domain_a"], r["domain_b"])] = r["jsd"]
        jsd_map[(r["domain_b"], r["domain_a"])] = r["jsd"]
        for d in (r["domain_a"], r["domain_b"]):
            if d not in seen:
                domains_ordered.append(d)
                seen.add(d)

    if len(domains_ordered) > HEATMAP_MAX_DOMAINS:
        log.warning(f"  Capping heatmap at {HEATMAP_MAX_DOMAINS} domains.")
        domains_ordered = domains_ordered[:HEATMAP_MAX_DOMAINS]

    n = len(domains_ordered)
    matrix = np.zeros((n, n))
    for i, da in enumerate(domains_ordered):
        for j, db in enumerate(domains_ordered):
            if i != j:
                matrix[i, j] = jsd_map.get((da, db), 0)

    vmax = np.percentile(matrix[matrix > 0], 95) if (matrix > 0).any() else 1
    short_labels = [shorten_domain(d) for d in domains_ordered]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), max(9, n * 0.4)), facecolor=bg)
    ax.set_facecolor(bg)
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)
    for i, domain in enumerate(domains_ordered):
        aud = aud_map.get(domain, "both")
        col = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SUBTEXT
        ax.get_xticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_color(col)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Jensen-Shannon Divergence", fontsize=8, color=C_SUBTEXT)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Cross-Domain Linguistic Distinctiveness (JSD)", **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "Blue = client-facing  •  Red = worker-facing  •  "
             "Higher JSD = more linguistically distinct",
             ha="center", **FONT_ANNOT)
    save(fig, "fig7_distinctiveness_heatmap", style)


def fig_exclusivity_volcano(conn, style):
    """Fig 8 — Term exclusivity vs. log10(total frequency) volcano plot."""
    if not has_table(conn, "term_exclusivity"):
        log.warning("  Fig 8 skipped — term_exclusivity not found.")
        return
    log.info(f"Fig 8 — Exclusivity volcano ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    rows = conn.execute("""
        SELECT term, exclusivity_index, total_freq, category
        FROM term_exclusivity
        WHERE total_freq >= ?
        ORDER BY exclusivity_index DESC
    """, (VOLCANO_FREQ_FLOOR,)).fetchall()
    if not rows:
        log.warning("  Fig 8 skipped — no exclusivity data.")
        return

    x_all = [math.log10(max(r["total_freq"], 1)) for r in rows]
    y_all = [r["exclusivity_index"] for r in rows]
    c_all = [C_CLIENT if r["category"] == "client_exclusive"
             else C_WORKER if r["category"] == "worker_exclusive"
             else C_SHARED for r in rows]

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg)
    ax.set_facecolor(bg)
    ax.scatter(x_all, y_all, c=c_all, alpha=0.55, s=18, edgecolors="none", zorder=2)

    # Label top N by exclusivity
    top_n = sorted(enumerate(rows), key=lambda iv: iv[1]["exclusivity_index"],
                   reverse=True)[:VOLCANO_LABEL_N]
    for idx, row in top_n:
        colour = C_CLIENT if row["category"] == "client_exclusive" else \
                 C_WORKER if row["category"] == "worker_exclusive" else C_SHARED
        ax.annotate(row["term"], (x_all[idx], y_all[idx]),
                    fontsize=7.5, color=colour, fontweight="bold",
                    textcoords="offset points", xytext=(4, 3))

    ax.set_xlabel("log₁₀(total frequency)", **FONT_LABEL)
    ax.set_ylabel("Exclusivity index (B2B–B2W)", **FONT_LABEL)
    ax.axhline(0, color=C_SUBTEXT, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)
    ax.legend(handles=[
        mpatches.Patch(color=C_CLIENT, label="Client-leaning"),
        mpatches.Patch(color=C_WORKER, label="Worker-leaning"),
        mpatches.Patch(color=C_SHARED, label="Shared"),
    ], loc="upper right", frameon=True, fontsize=9,
               facecolor=bg, edgecolor=C_GRID)
    ax.set_title("Term Exclusivity vs Frequency", **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "Above zero = more exclusive to B2B  •  Below zero = more exclusive to B2W  •  "
             f"Floor: min freq={VOLCANO_FREQ_FLOOR}",
             ha="center", **FONT_ANNOT)
    save(fig, "fig8_exclusivity_volcano", style)


def fig_pca_scatter(conn, style):
    """Fig 9 — Document-topic PCA scatter coloured by audience."""
    if not has_table(conn, "document_topics"):
        log.warning("  Fig 9 skipped — document_topics not found.")
        return
    log.info(f"Fig 9 — PCA scatter ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    rows = conn.execute("""
        SELECT pca_1, pca_2, audience, domain
        FROM document_topics
        WHERE pca_1 IS NOT NULL AND pca_2 IS NOT NULL
    """).fetchall()
    if not rows:
        log.warning("  Fig 9 skipped — no PCA data.")
        return

    # Two-panel: full corpus left, with PCA_EXCLUDE_DOMAINS filtered right
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=bg)
    fig.subplots_adjust(wspace=0.35)

    for ax, filtered, title_suffix in [
        (axes[0], False, "(all domains)"),
        (axes[1], True,  f"(excluding outliers)"),
    ]:
        ax.set_facecolor(bg)
        subset = [r for r in rows
                  if (not filtered) or (r["domain"] not in PCA_EXCLUDE_DOMAINS)]
        for audience, colour in [("client", C_CLIENT), ("worker", C_WORKER)]:
            pts = [(r["pca_1"], r["pca_2"]) for r in subset
                   if r["audience"] == audience]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, c=colour, alpha=PCA_ALPHA, s=PCA_SIZE,
                           edgecolors="none", zorder=2,
                           label=f"{'Client (B2B)' if audience=='client' else 'Worker (B2W)'}")
                if style == "exp" and len(xs) > 2:
                    from matplotlib.patches import Ellipse as _Ell
                    cx, cy = np.mean(xs), np.mean(ys)
                    sx, sy = np.std(xs), np.std(ys)
                    ell = _Ell((cx, cy), 2 * sx, 2 * sy,
                               angle=0, linewidth=1.2, linestyle="--",
                               edgecolor=colour, facecolor="none", alpha=0.4, zorder=1)
                    ax.add_patch(ell)

        if filtered:
            # Annotate excluded domains
            excl_domains = set(r["domain"] for r in rows
                               if r["domain"] in PCA_EXCLUDE_DOMAINS)
            if excl_domains:
                ax.text(0.02, 0.02,
                        f"Excluded: {', '.join(shorten_domain(d) for d in excl_domains)}",
                        transform=ax.transAxes, fontsize=7, color=C_SUBTEXT,
                        va="bottom", style="italic")

        ax.set_xlabel("PC1", **FONT_LABEL)
        ax.set_ylabel("PC2", **FONT_LABEL)
        ax.set_title(f"Document-Topic PCA {title_suffix}", **FONT_TITLE, pad=10)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
        apply_base_style(ax, bg)

    handles = [
        mpatches.Patch(color=C_CLIENT, label="Client (B2B)"),
        mpatches.Patch(color=C_WORKER, label="Worker (B2W)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("PCA of Document-Topic Space: Audience Register Separation",
                 **FONT_TITLE)
    save(fig, "fig9_pca_scatter", style)


def fig_topic_audience_profile(conn, style):
    """Fig 10 — Per-topic B2B/B2W balance diverging bar chart."""
    if not has_table(conn, "topic_audience_profile"):
        log.warning("  Fig 10 skipped — topic_audience_profile not found.")
        return
    log.info(f"Fig 10 — Topic audience profile ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    rows = conn.execute("""
        SELECT topic_id, topic_label, avg_weight_client, avg_weight_worker,
               client_share, category
        FROM topic_audience_profile
        ORDER BY client_share DESC
    """).fetchall()
    if not rows:
        log.warning("  Fig 10 skipped — no data.")
        return

    # Diverging bar: client share centred at 0.5
    topic_ids    = [r["topic_id"] for r in rows]
    labels       = [f"T{r['topic_id']}" + (f" {r['topic_label']}" if r["topic_label"] else "")
                    for r in rows]
    client_share = [r["client_share"] for r in rows]
    cat_colours  = {
        "client_leaning": C_CLIENT,
        "worker_leaning": C_WORKER,
        "shared":         C_SHARED,
    }
    bar_colours = [cat_colours.get(r["category"], C_SHARED) for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(8, len(rows) * 0.35)), facecolor=bg)
    ax.set_facecolor(bg)
    y_pos = np.arange(len(rows))
    # Plot as (client_share - 0.5) centred at zero
    centred = [cs - 0.5 for cs in client_share]
    bars = ax.barh(y_pos, centred, color=bar_colours, alpha=0.85,
                   edgecolor="white", linewidth=0.4, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color=C_TEXT, linewidth=0.8)
    ax.set_xlabel("← Worker-leaning  |  Client-leaning →", **FONT_LABEL)
    ax.set_title("Topic Audience Profile: B2B/B2W Balance per Topic",
                 **FONT_TITLE, pad=12)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
    ax.set_xlim(-0.5, 0.5)
    ax.xaxis.set_ticklabels([f"{x+0.5:.1f}" for x in ax.get_xticks()])
    apply_base_style(ax, bg)

    if style == "exp":
        for bar, row in zip(bars, rows):
            val = bar.get_width()
            ax.text(val + (0.01 if val >= 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2,
                    f"c={row['client_share']:.2f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=6.5, color=C_SUBTEXT)

    ax.legend(handles=[
        mpatches.Patch(color=C_CLIENT, label="Client-leaning (>65% B2B)"),
        mpatches.Patch(color=C_WORKER, label="Worker-leaning (>65% B2W)"),
        mpatches.Patch(color=C_SHARED, label="Shared"),
    ], loc="lower right", frameon=True, fontsize=9,
               facecolor=bg, edgecolor=C_GRID)
    save(fig, "fig10_topic_audience_profile", style)


def fig_collocate_divergence(conn, style):
    """Fig 11 — PMI profile divergence ranking for focus terms."""
    if not has_table(conn, "cooccurrence_results"):
        log.warning("  Fig 11 skipped — cooccurrence_results not found.")
        return
    log.info(f"Fig 11 — Collocate divergence ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    rows = conn.execute("""
        SELECT focus_term, audience, collocate, pmi
        FROM cooccurrence_results
        WHERE comparison = 'cross_platform'
    """).fetchall()
    if not rows:
        log.warning("  Fig 11 skipped — no data.")
        return

    profiles = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        profiles[r["focus_term"]][r["audience"]][r["collocate"]] = r["pmi"]

    divergences = {}
    for term, aud_data in profiles.items():
        if "client" not in aud_data or "worker" not in aud_data:
            continue
        c_vec = aud_data["client"]
        w_vec = aud_data["worker"]
        all_c = set(c_vec) | set(w_vec)
        if len(all_c) < 3:
            continue
        dot   = sum(c_vec.get(c, 0) * w_vec.get(c, 0) for c in all_c)
        mag_c = math.sqrt(sum(v ** 2 for v in c_vec.values()))
        mag_w = math.sqrt(sum(v ** 2 for v in w_vec.values()))
        cos_sim = (dot / (mag_c * mag_w)) if (mag_c and mag_w) else 0.0
        divergences[term] = round(1.0 - cos_sim, 6)

    if not divergences:
        log.warning("  Fig 11 skipped — no divergences computed.")
        return

    sorted_div = sorted(divergences.items(), key=lambda x: x[1],
                        reverse=True)[:DIVERGENCE_TOP_N]
    terms  = [t for t, _ in sorted_div]
    scores = [d for _, d in sorted_div]
    y_pos  = np.arange(len(terms))

    # Colour gradient from low to high divergence
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    bar_colours = [cmap(norm(s)) for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(8, len(terms) * 0.4)), facecolor=bg)
    ax.set_facecolor(bg)
    bars = ax.barh(y_pos, scores, color=bar_colours, edgecolor="white",
                   linewidth=0.4, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=9, color=C_TEXT)
    ax.set_xlabel("Collocate profile divergence (1 − cosine)", **FONT_LABEL)
    ax.set_title("PMI Profile Divergence: Same Word, Different Framing",
                 **FONT_TITLE, pad=12)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
    apply_base_style(ax, bg)
    if style == "exp":
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{score:.3f}", va="center", fontsize=7, color=C_SUBTEXT)
    fig.text(0.5, -0.02,
             "0 = identical collocate profile in B2B & B2W  •  "
             "1 = completely different framing  •  "
             "High divergence = strong Step 2 candidate",
             ha="center", **FONT_ANNOT)
    save(fig, "fig11_collocate_divergence", style)


def fig_step2_sample_map(conn, style):
    """Fig 12 — PCA with Step 2 sample highlighted."""
    if not has_table(conn, "document_topics") or not has_table(conn, "step2_sample"):
        log.warning("  Fig 12 skipped — document_topics or step2_sample not found.")
        return
    log.info(f"Fig 12 — Step 2 sample map ({style})")
    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    all_docs = conn.execute("""
        SELECT pca_1, pca_2, audience, domain
        FROM document_topics
        WHERE pca_1 IS NOT NULL AND pca_2 IS NOT NULL
          AND domain NOT IN ('www.sama.com', 'mindy-support.com', 'scale.com')
    """).fetchall()
    sample_ids = {
        r["page_id"]: r["sampling_reason"]
        for r in conn.execute("SELECT page_id, sampling_reason FROM step2_sample").fetchall()
    }
    sample_docs = conn.execute("""
        SELECT dt.pca_1, dt.pca_2, dt.audience, dt.domain, dt.page_id, s.sampling_reason
        FROM document_topics dt
        JOIN step2_sample s ON s.page_id = dt.page_id
        WHERE dt.pca_1 IS NOT NULL AND dt.pca_2 IS NOT NULL
    """).fetchall() if sample_ids else []

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=bg)
    ax.set_facecolor(bg)

    # Background scatter
    for audience, colour in [("client", C_CLIENT), ("worker", C_WORKER)]:
        pts = [(r["pca_1"], r["pca_2"]) for r in all_docs if r["audience"] == audience]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=colour, alpha=0.12, s=8, edgecolors="none", zorder=1)

    # Sample overlay coloured by hypothesis
    hyp_plotted = set()
    for row in sample_docs:
        reason = row["sampling_reason"] or ""
        hyp = next((k for k in HYP_COLORS if reason.startswith(k)), "H1a")
        colour = HYP_COLORS.get(hyp, C_ACCENT)
        label  = hyp if hyp not in hyp_plotted else ""
        ax.scatter([row["pca_1"]], [row["pca_2"]], c=colour,
                   s=SAMPLE_MARKER_SIZE, edgecolors="white", linewidths=0.8,
                   zorder=5, alpha=0.92,
                   label=label)
        hyp_plotted.add(hyp)

    ax.set_xlabel("PC1", **FONT_LABEL)
    ax.set_ylabel("PC2", **FONT_LABEL)
    ax.set_title("Step 2 Sampling Map: Selected Pages in PCA Document Space",
                 **FONT_TITLE, pad=12)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)
    # Legend: background audiences + sample hypotheses
    handles = [
        mpatches.Patch(color=C_CLIENT, alpha=0.4, label="Background: Client (B2B)"),
        mpatches.Patch(color=C_WORKER, alpha=0.4, label="Background: Worker (B2W)"),
    ]
    for hyp in sorted(hyp_plotted):
        handles.append(mpatches.Patch(color=HYP_COLORS[hyp],
                                      label=f"Sample: {hyp}"))
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=8.5,
              facecolor=bg, edgecolor=C_GRID)
    fig.text(0.5, -0.02,
             "Large markers = Step 2 sample  •  Background = full corpus  •  "
             "Colour = hypothesis group",
             ha="center", **FONT_ANNOT)
    save(fig, "fig12_step2_sample_map", style)


# ===========================================================================
# ── SECTION C: Narrative Figures (from 04_step1_narrative_figures.py) ───────
# ===========================================================================

def _narrative_save(fig, name, style):
    save(fig, name, style, out_dir=OUTPUT_NARR)


def fig_narrative_vocab_terrain(conn, style):
    """N1 — Vocabulary terrain: register gap scatter (narrative version)."""
    log.info(f"  N1 — Vocab terrain ({style})")
    if not has_table(conn, "keyness_results"):
        log.warning("    Skipped — keyness_results not found.")
        return
    apply_seaborn_theme()
    fig_register_scatter(conn, style)   # reuse Section A figure; also save to narrative dir
    # (The register scatter is already saved to OUTPUT_DIR by fig_register_scatter)


def fig_narrative_exclusive_vocab(conn, style):
    """N2 — Exclusive vocabulary per side."""
    log.info(f"  N2 — Exclusive vocab ({style})")
    if not has_table(conn, "keyness_results"):
        log.warning("    Skipped — keyness_results not found.")
        return
    client_top = fetch_top_client(conn, "cross_platform", F2_TOP_N)
    worker_top = list(reversed(fetch_top_worker(conn, "cross_platform", F2_TOP_N)))
    # Filter by LL threshold
    client_top = [r for r in client_top if abs(r["ll_score"]) >= F2_LL_MIN]
    worker_top = [r for r in worker_top if abs(r["ll_score"]) >= F2_LL_MIN]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    apply_seaborn_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor=bg)
    fig.subplots_adjust(wspace=0.5)
    for ax, data, colour, label in [
        (axes[0], client_top, C_CLIENT, "Exclusive to B2B  (client-facing)"),
        (axes[1], worker_top, C_WORKER, "Exclusive to B2W  (worker-facing)"),
    ]:
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            continue
        terms  = [r["term"] for r in data]
        scores = [abs(r["ll_score"]) for r in data]
        y_pos  = np.arange(len(terms))
        bars   = ax.barh(y_pos, scores, color=colour, alpha=0.85,
                         edgecolor="white", linewidth=0.4, height=0.65)
        ax.set_yticks(y_pos); ax.set_yticklabels(terms, fontsize=9, color=C_TEXT)
        ax.set_xlabel("Log-likelihood G²", **FONT_LABEL)
        ax.set_title(label, **FONT_TITLE, pad=12)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
        apply_base_style(ax, bg)
        if style == "exp":
            for bar, row in zip(bars, data):
                ax.text(bar.get_width() + max(scores) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{row['rel_freq_client']:.2f} / {row['rel_freq_worker']:.2f}",
                        va="center", **FONT_ANNOT)
    fig.suptitle("Exclusive Vocabulary: What Each Register Talks About",
                 **FONT_TITLE, y=1.01)
    _narrative_save(fig, "N2_exclusive_vocab", style)


def fig_narrative_shared_divergent(conn, style):
    """N3 — Shared terms, divergent collocate contexts."""
    log.info(f"  N3 — Shared divergent ({style})")
    if not has_table(conn, "cooccurrence_results"):
        log.warning("    Skipped — cooccurrence_results not found.")
        return
    # Show PMI profiles for 4 shared terms
    shared_focus = ["human", "quality", "work", "annotation"]
    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    apply_seaborn_theme()
    fig, axes = plt.subplots(len(shared_focus), 2,
                             figsize=(16, len(shared_focus) * 3.2), facecolor=bg)
    fig.subplots_adjust(hspace=0.55, wspace=0.55)
    for row_idx, focus in enumerate(shared_focus):
        for col_idx, (audience, colour, reg_label) in enumerate([
            ("client", C_CLIENT, "B2B"), ("worker", C_WORKER, "B2W")
        ]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(bg)
            cooc = fetch_cooccurrence(conn, "cross_platform", audience, focus,
                                      min_freq=MIN_COFREQ, top_n=COOC_TOP_N)
            ax.set_title(f'"{focus}"  —  {reg_label}',
                         fontsize=10, fontweight="bold", color=colour)
            if not cooc:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, **FONT_LABEL)
                ax.axis("off")
                continue
            collocates = [r["collocate"] for r in reversed(cooc)]
            pmi_vals   = [r["pmi"]       for r in reversed(cooc)]
            y_pos      = np.arange(len(collocates))
            ax.barh(y_pos, pmi_vals, color=colour, alpha=0.80,
                    edgecolor="white", linewidth=0.4, height=0.65)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(collocates, fontsize=8.5, color=C_TEXT)
            ax.set_xlabel("PMI", **FONT_LABEL)
            ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
            apply_base_style(ax, bg)
    fig.suptitle("Same Word, Different World: Shared Terms in Divergent Contexts",
                 **FONT_TITLE, y=1.01)
    _narrative_save(fig, "N3_shared_divergent", style)


def fig_narrative_topic_profiles(conn, style):
    """N4 — Unified topic audience balance chart."""
    if not has_table(conn, "topic_audience_profile"):
        log.warning("  N4 skipped — topic_audience_profile not found.")
        return
    log.info(f"  N4 — Topic profiles ({style})")
    apply_seaborn_theme()
    # Call the same logic as fig10 but save to narrative dir
    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    rows = conn.execute("""
        SELECT topic_id, topic_label, client_share, category
        FROM topic_audience_profile ORDER BY client_share DESC
    """).fetchall()
    if not rows:
        return
    labels = [f"T{r['topic_id']}" for r in rows]
    centred = [r["client_share"] - 0.5 for r in rows]
    cat_colours = {"client_leaning": C_CLIENT, "worker_leaning": C_WORKER, "shared": C_SHARED}
    bar_colours = [cat_colours.get(r["category"], C_SHARED) for r in rows]
    fig, ax = plt.subplots(figsize=(10, max(8, len(rows) * 0.35)), facecolor=bg)
    ax.set_facecolor(bg)
    ax.barh(np.arange(len(rows)), centred, color=bar_colours, alpha=0.85,
            edgecolor="white", linewidth=0.4, height=0.65)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color=C_TEXT, linewidth=0.8)
    ax.set_xlabel("← Worker-leaning  |  Client-leaning →", **FONT_LABEL)
    ax.set_title("Topic Audience Balance: Themes by Register Alignment", **FONT_TITLE, pad=12)
    apply_base_style(ax, bg)
    _narrative_save(fig, "N4_topic_profiles", style)


def fig_narrative_topic_hyp_alignment(conn, style):
    """N5 — Topic-hypothesis alignment map."""
    if not has_table(conn, "topic_terms") or not has_table(conn, "topic_audience_profile"):
        log.warning("  N5 skipped — topic_terms or topic_audience_profile not found.")
        return
    log.info(f"  N5 — Topic-hypothesis alignment ({style})")
    apply_seaborn_theme()
    from collections import defaultdict

    hyp_vocab = {
        "H1a": {"worker", "labour", "task", "job", "earn", "pay"},
        "H1b": {"autonomous", "machine", "automate", "automation", "algorithm", "pipeline"},
        "H1c": {"human", "quality", "oversight", "annotation", "label", "datum"},
    }
    topic_rows = conn.execute("""
        SELECT topic_id, term, rank FROM topic_terms WHERE rank <= 15
    """).fetchall()
    profile_rows = conn.execute("""
        SELECT topic_id, client_share, category FROM topic_audience_profile
    """).fetchall()
    profile_map = {r["topic_id"]: dict(r) for r in profile_rows}

    overlap = defaultdict(dict)
    for hyp_key, terms in hyp_vocab.items():
        topic_terms_map = defaultdict(set)
        for r in topic_rows:
            topic_terms_map[r["topic_id"]].add(r["term"])
        for tid, top_terms in topic_terms_map.items():
            overlap[hyp_key][tid] = len(top_terms & terms)

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    hyp_keys = list(hyp_vocab.keys())
    topic_ids = sorted({r["topic_id"] for r in profile_rows})
    matrix = np.zeros((len(hyp_keys), len(topic_ids)))
    for i, hk in enumerate(hyp_keys):
        for j, tid in enumerate(topic_ids):
            matrix[i, j] = overlap[hk].get(tid, 0)

    fig, ax = plt.subplots(figsize=(max(14, len(topic_ids) * 0.45), 5), facecolor=bg)
    ax.set_facecolor(bg)
    im = ax.imshow(matrix, aspect="auto", cmap="Oranges",
                   vmin=0, interpolation="nearest")
    ax.set_yticks(range(len(hyp_keys)))
    ax.set_yticklabels(hyp_keys, fontsize=10)
    ax.set_xticks(range(len(topic_ids)))
    ax.set_xticklabels([f"T{t}" for t in topic_ids], fontsize=7, rotation=45, ha="right")
    # Colour x-labels by audience
    for i, tid in enumerate(topic_ids):
        prof = profile_map.get(tid, {})
        cat  = prof.get("category", "shared") if prof else "shared"
        col  = C_CLIENT if cat == "client_leaning" else \
               C_WORKER if cat == "worker_leaning" else C_SHARED
        ax.get_xticklabels()[i].set_color(col)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Term overlap with hypothesis vocabulary", fontsize=8)
    ax.set_title("Topic-Hypothesis Alignment: Which Topics Speak to Each Hypothesis",
                 **FONT_TITLE, pad=12)
    _narrative_save(fig, "N5_topic_hyp_alignment", style)


def fig_narrative_document_space(conn, style):
    """N6 — PCA document space (narrative version)."""
    log.info(f"  N6 — Document space ({style})")
    if not has_table(conn, "document_topics"):
        log.warning("    Skipped.")
        return
    apply_seaborn_theme()
    fig_pca_scatter(conn, style)   # reuse Section B figure


def fig_narrative_step2_map(conn, style):
    """N7 — Step 2 sample map (narrative version)."""
    log.info(f"  N7 — Step 2 map ({style})")
    if not has_table(conn, "document_topics") or not has_table(conn, "step2_sample"):
        log.warning("    Skipped.")
        return
    apply_seaborn_theme()
    fig_step2_sample_map(conn, style)  # reuse Section B figure


# ===========================================================================
# ── SECTION D: PCA Domain Figures (from 05_pca_domain_figures.py) ──────────
# ===========================================================================

def _pca_save(fig, name, style):
    save(fig, name, style, out_dir=OUTPUT_PCA)


def fig_pca_domain_landscape(conn, style):
    """P-A — Domain landscape strip plot: mean PC1 per platform."""
    if not has_table(conn, "document_topics"):
        log.warning("  P-A skipped — document_topics not found.")
        return
    log.info(f"  P-A — Domain landscape ({style})")

    rows = conn.execute("""
        SELECT domain, audience, pca_1
        FROM document_topics WHERE pca_1 IS NOT NULL
    """).fetchall()
    if not rows:
        return

    domain_data = defaultdict(list)
    domain_audience = {}
    for r in rows:
        domain_data[r["domain"]].append(r["pca_1"])
        domain_audience[r["domain"]] = r["audience"]

    domain_stats = {
        d: (statistics.mean(vals), len(vals),
            statistics.stdev(vals) if len(vals) > 1 else 0)
        for d, vals in domain_data.items()
    }
    sorted_domains = sorted(domain_stats.keys(),
                            key=lambda d: domain_stats[d][0])

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(8, max(8, len(sorted_domains) * 0.4)), facecolor=bg)
    ax.set_facecolor(bg)

    y_pos = np.arange(len(sorted_domains))
    for i, domain in enumerate(sorted_domains):
        mean_pc1, n, std = domain_stats[domain]
        aud  = domain_audience.get(domain, "both")
        col  = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SHARED
        size = max(20, min(200, n * 2))
        ax.scatter([mean_pc1], [i], s=size, c=col, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3)
        if style == "exp":
            ax.errorbar([mean_pc1], [i], xerr=[std], fmt="none",
                        ecolor=col, alpha=0.4, linewidth=1.2, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten_domain(d) for d in sorted_domains], fontsize=7)
    for i, domain in enumerate(sorted_domains):
        aud = domain_audience.get(domain, "both")
        col = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SHARED
        ax.get_yticklabels()[i].set_color(col)

    ax.set_xlabel("PC1 (mean per domain)", **FONT_LABEL)
    ax.axvline(0, color=C_SUBTEXT, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Domain Landscape: Platform Position on Register Axis",
                 **FONT_TITLE, pad=12)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)
    fig.text(0.5, -0.02,
             "Dot size ∝ number of pages  •  Blue = client  •  Red = worker  •  "
             "Sorted by mean PC1",
             ha="center", **FONT_ANNOT)
    _pca_save(fig, "PA_domain_landscape", style)


def fig_pca_scatter_domain(conn, style):
    """P-B — PCA scatter coloured by domain."""
    if not has_table(conn, "document_topics"):
        log.warning("  P-B skipped — document_topics not found.")
        return
    log.info(f"  P-B — PCA scatter domain ({style})")

    rows = conn.execute("""
        SELECT pca_1, pca_2, audience, domain
        FROM document_topics WHERE pca_1 IS NOT NULL AND pca_2 IS NOT NULL
          AND domain NOT IN ('www.sama.com', 'mindy-support.com', 'scale.com')
    """).fetchall()
    if not rows:
        return

    all_domains = sorted(set(r["domain"] for r in rows))
    cmap_fn = plt.cm.get_cmap("tab20", min(len(all_domains), 20))
    domain_colour = {d: cmap_fn(i % 20) for i, d in enumerate(all_domains)}

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=bg)
    ax.set_facecolor(bg)

    for domain in all_domains:
        pts = [(r["pca_1"], r["pca_2"]) for r in rows if r["domain"] == domain]
        if pts:
            xs, ys = zip(*pts)
            marker = "o" if "client" in (r["audience"] for r in rows if r["domain"] == domain) else "^"
            ax.scatter(xs, ys, c=[domain_colour[domain]], alpha=PCA_ALPHA, s=PCA_SIZE,
                       edgecolors="none", zorder=2,
                       label=shorten_domain(domain) if len(pts) >= 5 else None)

    ax.set_xlabel("PC1", **FONT_LABEL)
    ax.set_ylabel("PC2", **FONT_LABEL)
    ax.set_title("PCA Document Space: Individual Platforms", **FONT_TITLE, pad=12)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)
    if len(all_domains) <= 20:
        ax.legend(loc="upper right", frameon=True, fontsize=7,
                  facecolor=bg, edgecolor=C_GRID, ncol=2)
    _pca_save(fig, "PB_pca_scatter_domain", style)


def fig_pca_within_domain_var(conn, style):
    """P-C — Within-domain PC1 variation strip plot."""
    if not has_table(conn, "document_topics"):
        log.warning("  P-C skipped.")
        return
    log.info(f"  P-C — Within-domain variation ({style})")

    rows = conn.execute("""
        SELECT domain, audience, pca_1
        FROM document_topics WHERE pca_1 IS NOT NULL
          AND domain NOT IN ('www.sama.com', 'mindy-support.com', 'scale.com')
    """).fetchall()
    if not rows:
        return

    domain_data    = defaultdict(list)
    domain_audience = {}
    for r in rows:
        domain_data[r["domain"]].append(r["pca_1"])
        domain_audience[r["domain"]] = r["audience"]

    # Only domains with enough pages
    domains_valid = {d for d, vals in domain_data.items() if len(vals) >= 5}
    sorted_domains = sorted(domains_valid,
                            key=lambda d: statistics.mean(domain_data[d]))

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(10, max(8, len(sorted_domains) * 0.45)), facecolor=bg)
    ax.set_facecolor(bg)

    for i, domain in enumerate(sorted_domains):
        vals = domain_data[domain]
        aud  = domain_audience.get(domain, "both")
        col  = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SHARED
        y    = np.full(len(vals), i) + np.random.normal(0, 0.12, len(vals))
        ax.scatter(vals, y, c=col, alpha=0.35, s=8, edgecolors="none", zorder=2)
        mean_val = statistics.mean(vals)
        ax.scatter([mean_val], [i], c=col, s=80, edgecolors="white",
                   linewidths=0.8, zorder=4)

    ax.set_yticks(range(len(sorted_domains)))
    ax.set_yticklabels([shorten_domain(d) for d in sorted_domains], fontsize=7)
    for i, domain in enumerate(sorted_domains):
        aud = domain_audience.get(domain, "both")
        col = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SHARED
        ax.get_yticklabels()[i].set_color(col)

    ax.set_xlabel("PC1", **FONT_LABEL)
    ax.axvline(0, color=C_SUBTEXT, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Within-Domain PC1 Variation", **FONT_TITLE, pad=12)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)
    fig.text(0.5, -0.02,
             "Jittered dots = individual pages  •  Solid circle = domain mean  •  "
             "Variation shows register is a choice, not a fixed platform property",
             ha="center", **FONT_ANNOT)
    _pca_save(fig, "PC_within_domain_var", style)


def fig_pca_topic_loadings(conn, style):
    """P-E — Topic loadings on PC1 and PC2."""
    if not has_table(conn, "document_topics") or not has_table(conn, "topic_terms"):
        log.warning("  P-E skipped.")
        return
    log.info(f"  P-E — Topic PC loadings ({style})")

    # We need the PCA model loadings stored in topic_terms
    # They are not stored directly; instead we compute correlation of topic vectors with PCA coords
    rows = conn.execute("""
        SELECT dt.page_id, dt.pca_1, dt.pca_2, dt.topic_vector, dt.audience
        FROM document_topics dt
        WHERE dt.pca_1 IS NOT NULL AND dt.pca_2 IS NOT NULL
          AND dt.topic_vector IS NOT NULL
          AND dt.domain NOT IN ('www.sama.com', 'mindy-support.com', 'scale.com')
    """).fetchall()
    if not rows:
        return

    pca1_vals = np.array([r["pca_1"] for r in rows])
    pca2_vals = np.array([r["pca_2"] for r in rows])
    try:
        topic_vecs = np.array([json.loads(r["topic_vector"]) for r in rows])
    except (json.JSONDecodeError, TypeError):
        log.warning("  P-E skipped — could not parse topic_vector.")
        return

    n_topics = topic_vecs.shape[1]
    corr_pc1 = np.array([np.corrcoef(topic_vecs[:, t], pca1_vals)[0, 1]
                         for t in range(n_topics)])
    corr_pc2 = np.array([np.corrcoef(topic_vecs[:, t], pca2_vals)[0, 1]
                         for t in range(n_topics)])

    # Get top terms for labels
    topic_top_terms = defaultdict(list)
    for r in conn.execute(
        "SELECT topic_id, term FROM topic_terms WHERE rank <= ? ORDER BY rank",
        (FE_TOP_TERMS,)
    ).fetchall():
        topic_top_terms[r["topic_id"]].append(r["term"])

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=bg)
    fig.subplots_adjust(wspace=0.4)

    for ax, corr, pc_label in [(axes[0], corr_pc1, "PC1"), (axes[1], corr_pc2, "PC2")]:
        top_idx = np.argsort(np.abs(corr))[::-1][:FE_TOP_N]
        sorted_by_corr = sorted(top_idx, key=lambda i: corr[i])
        labels  = [f"T{t}: {', '.join(topic_top_terms[t][:FE_TOP_TERMS])}"
                   for t in sorted_by_corr]
        vals    = [corr[t] for t in sorted_by_corr]
        colours = [C_CLIENT if v > 0 else C_WORKER for v in vals]
        y_pos   = np.arange(len(labels))
        ax.barh(y_pos, vals, color=colours, alpha=0.82,
                edgecolor="white", linewidth=0.4, height=0.65)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7.5, color=C_TEXT)
        ax.axvline(0, color=C_TEXT, linewidth=0.8)
        ax.set_xlabel(f"Correlation with {pc_label}", **FONT_LABEL)
        ax.set_title(f"Topic Loadings on {pc_label}", **FONT_TITLE, pad=10)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
        apply_base_style(ax, bg)

    fig.suptitle("Topic Loadings on Principal Components",
                 **FONT_TITLE)
    fig.text(0.5, -0.02,
             "Correlation between per-topic document weight and PCA coordinate  •  "
             "Blue = loads with B2B side; Red = loads with B2W side",
             ha="center", **FONT_ANNOT)
    _pca_save(fig, "PE_topic_pc_loadings", style)


def fig_pca_domain_shapes(conn, style):
    """P-F — PCA scatter with per-domain shapes (up to 20 domains)."""
    if not has_table(conn, "document_topics"):
        log.warning("  P-F skipped.")
        return
    log.info(f"  P-F — PCA domain shapes ({style})")

    rows = conn.execute("""
        SELECT pca_1, pca_2, audience, domain
        FROM document_topics WHERE pca_1 IS NOT NULL AND pca_2 IS NOT NULL
          AND domain NOT IN ('www.sama.com', 'mindy-support.com', 'scale.com')
    """).fetchall()
    if not rows:
        return

    all_domains = sorted(set(r["domain"] for r in rows),
                         key=lambda d: len([r for r in rows if r["domain"] == d]),
                         reverse=True)
    top_domains = all_domains[:20]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "8",
               "p", "+", "x", "1", "2", "3", "4", "|", "_", "H"]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=bg)
    ax.set_facecolor(bg)

    domain_audience = {r["domain"]: r["audience"] for r in rows}
    for i, domain in enumerate(top_domains):
        pts = [(r["pca_1"], r["pca_2"]) for r in rows if r["domain"] == domain]
        if pts:
            xs, ys = zip(*pts)
            aud = domain_audience.get(domain, "both")
            col = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SHARED
            ax.scatter(xs, ys, c=col, alpha=0.55, s=18,
                       marker=markers[i % len(markers)],
                       edgecolors="none", zorder=2,
                       label=shorten_domain(domain))

    # Other domains as grey dots
    other_rows = [r for r in rows if r["domain"] not in top_domains]
    if other_rows:
        ax.scatter([r["pca_1"] for r in other_rows],
                   [r["pca_2"] for r in other_rows],
                   c=C_GRID, alpha=0.25, s=8, edgecolors="none", zorder=1,
                   label="Other")

    ax.set_xlabel("PC1", **FONT_LABEL)
    ax.set_ylabel("PC2", **FONT_LABEL)
    ax.set_title("PCA Document Space: Per-Domain Shapes",
                 **FONT_TITLE, pad=12)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)
    ax.legend(loc="upper right", frameon=True, fontsize=7,
              facecolor=bg, edgecolor=C_GRID, ncol=2)
    _pca_save(fig, "PF_pca_domain_shapes", style)


# ===========================================================================
# Main
# ===========================================================================

def main():
    """
    Generate all Step 1 figures.

    Figures are grouped into four sections, each controllable via the
    FIGURE_GROUPS dict below.  Missing prerequisite tables cause the
    relevant figure to be skipped with a WARNING (not an abort).

    Run order:
        python3 src2/02_step1_analysis.py   (prerequisite for Figs 1–8)
        python3 src2/03_step1_topics.py     (prerequisite for Figs 9–12 + N/P)
        python3 src2/04_step1_figures.py    (this script)
    """
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("04_step1_figures.py — All Step 1 Visualisations")
    log.info("=" * 60)

    # Toggle figure sections here
    FIGURE_GROUPS = {
        "core":      True,   # Figs 1–6, S1 (keyness, co-occurrence, freq)
        "dist":      True,   # Figs 7–12  (distinctiveness, PCA, topics)
        "narrative": True,   # N1–N7      (argumentative narrative sequence)
        "pca":       True,   # P-A to P-F (PCA domain analysis)
    }

    conn = get_conn()

    for style in ("pub", "exp"):
        log.info(f"\n{'=' * 28}  {style.upper()}  {'=' * 28}")

        # ── Section A: Core Step 1 ───────────────────────────────────────
        if FIGURE_GROUPS["core"]:
            log.info("── Core Step 1 Figures ──")
            fig_keyness_bar(conn, style)
            fig_cooccurrence_network(conn, style)
            fig_frequency_comparison(conn, style)
            fig_within_pair(conn, style)
            fig_platform_heatmap(conn, style)
            fig_theory_cooccurrence(conn, style)
            fig_register_scatter(conn, style)

        # ── Section B: Distinctiveness & Topics ─────────────────────────
        if FIGURE_GROUPS["dist"]:
            log.info("── Distinctiveness & Topic Figures ──")
            fig_distinctiveness_heatmap(conn, style)
            fig_exclusivity_volcano(conn, style)
            fig_pca_scatter(conn, style)
            fig_topic_audience_profile(conn, style)
            fig_collocate_divergence(conn, style)
            fig_step2_sample_map(conn, style)

        # ── Section C: Narrative sequence ───────────────────────────────
        if FIGURE_GROUPS["narrative"]:
            log.info("── Narrative Figures ──")
            fig_narrative_exclusive_vocab(conn, style)
            fig_narrative_shared_divergent(conn, style)
            fig_narrative_topic_profiles(conn, style)
            fig_narrative_topic_hyp_alignment(conn, style)

        # ── Section D: PCA domain analysis ──────────────────────────────
        if FIGURE_GROUPS["pca"]:
            log.info("── PCA Domain Figures ──")
            fig_pca_domain_landscape(conn, style)
            fig_pca_scatter_domain(conn, style)
            fig_pca_within_domain_var(conn, style)
            fig_pca_topic_loadings(conn, style)
            fig_pca_domain_shapes(conn, style)

    conn.close()
    log.info("=" * 60)
    log.info("All figures generated.")
    log.info(f"  Core + distinctiveness  → {OUTPUT_DIR.resolve()}")
    log.info(f"  Narrative               → {OUTPUT_NARR.resolve()}")
    log.info(f"  PCA domain              → {OUTPUT_PCA.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
