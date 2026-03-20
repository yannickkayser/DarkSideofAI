"""
04_step1_narrative_figures.py
==============================
Narrative-driven Step 1 figures for the DarkSideofAI thesis.

Argumentative arc
-----------------
  Stage 1 — Establish the register gap         → F1  vocabulary terrain
  Stage 2A — What each side talks about         → F2  exclusive vocabulary
  Stage 2B — Same word, different world         → F3  shared terms, divergent contexts
  Stage 3  — Patterns aggregate to themes       → F4  topic profiles (unified chart)
  Stage 4  — Themes map onto hypotheses         → F5  topic-hypothesis alignment
  Stage 5  — Document-level structure validated → F6  PCA document space
  Stage 6  — Justify the Step 2 selection       → F7  Step 2 sample map

Visual framework
----------------
  Seaborn is used as the theme engine:  one call to sns.set_theme() applies
  consistent typography, white background, cleaned spines, and grid styling
  to every matplotlib figure in the session.  No style arguments needed.

  Alternative (IEEE/Nature paper aesthetic):
      pip install SciencePlots
      # then in apply_theme(), replace sns.set_theme() with:
      import scienceplots; plt.style.use(["science", "no-latex"])

Usage
-----
  python3 src/04_step1_narrative_figures.py
  Toggle figures in FIGURES dict; all visual params in the CONFIG section.
"""

import sqlite3
import math
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False


# =============================================================================
# ── CONFIG ────────────────────────────────────────────────────────────────────
# =============================================================================

DB_PATH    = "data/scraping.db"
OUTPUT_DIR = Path("output/step_1/narrative/")
DPI        = 150
EXT        = "jpg"    # "png" for lossless, "pdf" for vector

# ── toggle figures ─────────────────────────────────────────────────────────────
FIGURES = {
    "f1_vocab_terrain":       False, #True,
    "f2_exclusive_vocab":     False, #True,
    "f3_shared_divergent":    True,
    "f4_topic_profiles":      False, #True,
    "f5_topic_hyp_alignment": False, #True,
    "f6_document_space":      False, #True,
    "f7_step2_sample_map":    False, #True,
}

# ── palette ────────────────────────────────────────────────────────────────────
PAL = dict(
    b2b     = "#1B4F8A",
    b2w     = "#C0392B",
    h1c     = "#E67E22",
    neutral = "#BBBBBB",
    grid    = "#E0E0E0",
    text    = "#1A1A2E",
    sub     = "#6C757D",
    bg      = "#FFFFFF",
    accent  = "#E67E22",
)

# ── per-figure parameters ──────────────────────────────────────────────────────

# F1 — vocabulary terrain
F1_MIN_FREQ    = 3      # minimum rel_freq on both sides (per 10 000) to include
F1_LABEL_N     = 22     # number of terms to annotate (ranked by asymmetry × freq)
F1_BG_ALPHA    = 0.40   # background dot opacity
F1_NEAR_DIAG   = 0.10   # |log_fc − log_fw| threshold for "shared zone" (grey)

# F2 — exclusive vocabulary
F2_TOP_N       = 25     # terms per direction
F2_LL_MIN      = 10.83  # minimum |LL| (p < 0.001)

# F3 — shared terms, divergent contexts
F3_TERMS       = ["human", "work", "quality", "earn"]  # ← edit freely
F3_N_COLLOC    = 8      # collocates per side per term
F3_MIN_COFREQ  = 5      # minimum co-occurrence frequency

# F4 — topic profiles (single diverging bar)
F4_TOP_TERMS   = 4      # terms shown per topic bar

# F5 — topic-hypothesis alignment
F5_TOP_K       = 15     # top-K topic terms used for hypothesis overlap

# F6 — PCA document space
F6_DOT_ALPHA   = 0.40
F6_DOT_SIZE    = 14
F6_ELLIPSE_STD = 2.0    # n_std for the 95% confidence ellipse
F6_N_EXTREME   = 2      # number of extreme topics to annotate per axis direction

# F7 — step 2 sample map
F7_BG_ALPHA    = 0.10
F7_FG_SIZE     = 100
F7_INSET       = True   # draw a zoomed inset panel of the sample region

# ── hypothesis vocabulary ──────────────────────────────────────────────────────
# Used by F2 (★ markers), F5 (alignment matrix), F7 (sample colouring).
HYP = {
    "H1a": dict(
        label  = "H1a — Labour visibility",
        terms  = {"worker", "labour", "task", "job", "earn", "pay", "payment",
                  "annotator", "gig", "contractor", "wage", "labeller",
                  "freelance", "income", "work"},
        color  = PAL["b2w"],
        marker = "o",
    ),
    "H1b": dict(
        label  = "H1b — Automation myth",
        terms  = {"autonomous", "machine", "automate", "automation", "algorithm",
                  "pipeline", "deploy", "inference", "neural", "llm",
                  "intelligent", "scalable"},
        color  = PAL["b2b"],
        marker = "s",
    ),
    "H1c": dict(
        label  = "H1c — Strategic hypervisibility",
        terms  = {"human", "quality", "oversight", "annotation", "label",
                  "expert", "accuracy", "review", "verification",
                  "curate"},
        color  = PAL["h1c"],
        marker = "^",
    ),
}

# ── artifact / noise terms ─────────────────────────────────────────────────────
NOISE = {
    "cookie", "set_cookie", "cooky", "/hr", "/hr_remote", "remote_apply",
    "feb", "opportunity_feb", "faq", "faq_help", "help_desk", "subscribe",
    "website", "account", "access", "enable", "microworker", "shall", "youtube",
    "zeynep", "koouchnir", "gavrilov", "unga", "gary", "yalda",
    "monarch", "warhol", "fremont", "pittsburgh", "mpii",
    "experience.with", "rhml", "ead", "cc0", "ft",
    "hole", "overfit", "surprised", "christmas", "morale", "high-quality",
    "slash", "500", "pickup", "loophole", "conceptually", "housing",
    "firefighting", "sidestep", "wary", "downward", "jira", "voluman",
    "squeeze", "retrofit", "yt", "outli",
    "deciphering", "trafficking", "recap", "ueberwinden", "bildbearbeitung",
    "sicherstellung", "kunst", "human-le", "pto", "generous",
    "dhanesh", "ramachandram", "outlet", "daniela", "braga", "forbe", "january", 
    "february", "march", "abril", "may", "june", "july", "september", "october",
    "november", "december"
}


# =============================================================================
# ── THEME ─────────────────────────────────────────────────────────────────────
# =============================================================================

def apply_theme():
    rc = {
        "figure.facecolor":   PAL["bg"],
        "axes.facecolor":     PAL["bg"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     PAL["grid"],
        "axes.labelcolor":    PAL["text"],
        "axes.labelsize":     10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.titlecolor":    PAL["text"],
        "axes.grid":          True,
        "grid.color":         PAL["grid"],
        "grid.linewidth":     0.6,
        "grid.linestyle":     ":",
        "axes.axisbelow":     True,
        "xtick.color":        PAL["sub"],
        "ytick.color":        PAL["sub"],
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   PAL["grid"],
        "legend.fontsize":    9,
        "font.family":        "sans-serif",
        "text.color":         PAL["text"],
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
    }
    if _HAS_SNS:
        sns.set_theme(style="ticks", rc=rc)
        log.info("Theme: seaborn 'ticks'")
    else:
        plt.rcParams.update(rc)
        log.info("Theme: manual rcParams")


# =============================================================================
# ── DATABASE / FIGURE HELPERS ─────────────────────────────────────────────────
# =============================================================================

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _ph(items):
    return ",".join("?" * len(items))

def table_exists(conn, name):
    return bool(conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone())

def save_fig(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.{EXT}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    log.info(f"  → {path}")
    plt.close(fig)

def _clean_ax(ax, grid_axis="y"):
    ax.spines["left"].set_color(PAL["grid"])
    ax.spines["bottom"].set_color(PAL["grid"])
    if grid_axis == "x":
        ax.yaxis.grid(False)
    elif grid_axis == "y":
        ax.xaxis.grid(False)
    elif grid_axis == "none":
        ax.grid(False)

def _caption(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha="center", va="top",
             fontsize=8, color=PAL["sub"], style="italic")

def _confidence_ellipse(ax, x, y, n_std=2.0, color=None, **kwargs):
    """Draw an n_std-sigma confidence ellipse for the data (x, y)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(max(vals[0], 0))
    height = 2 * n_std * np.sqrt(max(vals[1], 0))
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=width, height=height, angle=angle,
                  facecolor=color, alpha=0.08,
                  edgecolor=color, linewidth=1.5, linestyle="--",
                  zorder=1, **kwargs)
    ax.add_patch(ell)


# =============================================================================
# F1 — VOCABULARY TERRAIN
# =============================================================================

def fig_f1_vocab_terrain(conn):
    """
    F1 — Log-log scatter of all unigrams by B2B vs B2W frequency.

    Dots are coloured by which side of the diagonal they fall on
    (blue = B2B-dominant, red = B2W-dominant, grey = shared zone).
    The top F1_LABEL_N most asymmetric high-frequency terms are annotated —
    these are the empirically prominent words that define the register gap,
    selected without any theoretical pre-seeding.

    Labelling strategy: score = |log_fc − log_fw| × log10(total_freq + 1)
    This prioritises terms that are both highly asymmetric AND frequent
    enough to be analytically meaningful.
    """
    log.info("F1 — vocabulary terrain")
    if not table_exists(conn, "keyness_results"):
        log.warning("  keyness_results not found — skipping F1.")
        return

    noise_ph = _ph(NOISE)
    rows = conn.execute(f"""
        SELECT term, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = 'cross_platform'
          AND term_type  = 'unigram'
          AND term NOT IN ({noise_ph})
    """, list(NOISE)).fetchall()

    if not rows:
        log.warning("  No data — skipping F1.")
        return

    EPS = F1_MIN_FREQ / 20
    xs, ys, terms, totals = [], [], [], []
    for r in rows:
        fc = max(r["rel_freq_client"] or 0, EPS)
        fw = max(r["rel_freq_worker"] or 0, EPS)
        if fc < F1_MIN_FREQ / 3 and fw < F1_MIN_FREQ / 3:
            continue
        xs.append(math.log10(fw))
        ys.append(math.log10(fc))
        terms.append(r["term"])
        totals.append((r["rel_freq_client"] or 0) + (r["rel_freq_worker"] or 0))

    # ── per-point colour: above diagonal = B2B, below = B2W, near = grey ──
    diff = [ys[i] - xs[i] for i in range(len(terms))]
    colors = []
    for d in diff:
        if d > F1_NEAR_DIAG:
            colors.append(PAL["b2b"])
        elif d < -F1_NEAR_DIAG:
            colors.append(PAL["b2w"])
        else:
            colors.append(PAL["neutral"])

    # ── size proportional to total frequency (capped) ─────────────────────
    max_t  = max(totals) or 1
    sizes  = [max(4, min(60, (t / max_t) * 80)) for t in totals]

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.scatter(xs, ys, c=colors, s=sizes, alpha=F1_BG_ALPHA,
               linewidths=0, zorder=2)

    # ── diagonal ───────────────────────────────────────────────────────────
    lo = min(min(xs), min(ys)) - 0.15
    hi = max(max(xs), max(ys)) + 0.15
    ax.plot([lo, hi], [lo, hi], "--", color=PAL["sub"], linewidth=1.0,
            zorder=1, alpha=0.7, label="Equal usage  (y = x)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # ── label top terms by asymmetry × frequency score ─────────────────────
    score = [abs(diff[i]) * math.log10(max(totals[i], 1) + 1)
             for i in range(len(terms))]
    ranked = sorted(range(len(terms)), key=lambda i: score[i], reverse=True)

    labelled = 0
    for i in ranked:
        if labelled >= F1_LABEL_N:
            break
        col = (PAL["b2b"] if diff[i] > F1_NEAR_DIAG
               else PAL["b2w"] if diff[i] < -F1_NEAR_DIAG
               else PAL["sub"])
        ax.annotate(terms[i], (xs[i], ys[i]),
                    fontsize=8, color=col, fontweight="bold",
                    xytext=(4, 3), textcoords="offset points")
        labelled += 1

    # ── zone labels ────────────────────────────────────────────────────────
    ax.text(hi - 0.05, lo + 0.10, "B2B dominant\n(client register)",
            ha="right", va="bottom", fontsize=9, color=PAL["b2b"],
            style="italic", alpha=0.80)
    ax.text(lo + 0.05, hi - 0.10, "B2W dominant\n(worker register)",
            ha="left", va="top", fontsize=9, color=PAL["b2w"],
            style="italic", alpha=0.80)

    # Legend: audience colour patches + diagonal
    legend_h = [
        mpatches.Patch(color=PAL["b2b"], alpha=0.7, label="B2B-dominant term"),
        mpatches.Patch(color=PAL["b2w"], alpha=0.7, label="B2W-dominant term"),
        mpatches.Patch(color=PAL["neutral"], alpha=0.7, label="Shared zone (±0.10)"),
        plt.Line2D([0], [0], linestyle="--", color=PAL["sub"],
                   linewidth=1, label="Equal usage (y = x)"),
    ]
    ax.legend(handles=legend_h, loc="lower right",
              handlelength=1.2, handletextpad=0.5)

    ax.set_xlabel("log₁₀ relative frequency — Worker corpus (B2W)", fontsize=10)
    ax.set_ylabel("log₁₀ relative frequency — Client corpus (B2B)", fontsize=10)
    ax.set_title("F1 — Vocabulary Terrain: Register Distribution of the Full Lexicon",
                 pad=14)
    _clean_ax(ax, grid_axis="both")
    _caption(fig, "Each point = one unigram; colour = register dominance; "
                  "size ∝ total corpus frequency.  "
                  "Labelled terms = most asymmetric high-frequency words "
                  f"(top {F1_LABEL_N} by asymmetry × frequency score, no theory pre-seeding).")
    save_fig(fig, "f1_vocab_terrain")


# =============================================================================
# F2 — EXCLUSIVE VOCABULARY
# =============================================================================

def fig_f2_exclusive_vocab(conn):
    """
    F2 — Side-by-side horizontal bar charts of the most audience-distinctive terms.

    Both panels read left-to-right with term labels on the y-axis (left side
    of each panel).  Terms in the hypothesis vocabulary are marked ★ to
    distinguish theory-seeded from empirically discovered patterns.

    Reading the figure:
      - Terms at the top = highest log-likelihood (most diagnostically exclusive)
      - ★ terms confirm the hypothesised vocabulary IS selectively used
      - Unmarked terms at high LL are empirical discoveries
    """
    log.info("F2 — exclusive vocabulary")
    if not table_exists(conn, "keyness_results"):
        log.warning("  keyness_results not found — skipping F2.")
        return

    noise_ph = _ph(NOISE)
    all_hyp  = {t for cfg in HYP.values() for t in cfg["terms"]}

    b2b_rows = conn.execute(f"""
        SELECT term, ll_score FROM keyness_results
        WHERE comparison = 'cross_platform' AND term_type = 'unigram'
          AND ll_score >= {F2_LL_MIN}
          AND term NOT IN ({noise_ph})
        ORDER BY ll_score DESC LIMIT {F2_TOP_N}
    """, list(NOISE)).fetchall()

    b2w_rows = conn.execute(f"""
        SELECT term, ll_score FROM keyness_results
        WHERE comparison = 'cross_platform' AND term_type = 'unigram'
          AND ll_score <= -{F2_LL_MIN}
          AND term NOT IN ({noise_ph})
        ORDER BY ll_score ASC LIMIT {F2_TOP_N}
    """, list(NOISE)).fetchall()

    if not b2b_rows and not b2w_rows:
        log.warning("  No data — skipping F2.")
        return

    all_hyp  = {t for cfg in HYP.values() for t in cfg["terms"]}

    def _prep(rows):
        # ★ prefix + store whether it is a hypothesis term for bar styling
        labels = [("★ " if r["term"] in all_hyp else "") + r["term"]
                  for r in rows]
        scores = [abs(r["ll_score"]) for r in rows]
        is_hyp = [r["term"] in all_hyp for r in rows]
        return labels, scores, is_hyp

    b2b_labels, b2b_scores, b2b_hyp = _prep(b2b_rows)
    b2w_labels, b2w_scores, b2w_hyp = _prep(b2w_rows)

    n_b2b = len(b2b_labels)
    n_b2w = len(b2w_labels)
    height = max(9, max(n_b2b, n_b2w) * 0.38)

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(15, height),
        gridspec_kw={"wspace": 0.55}
    )

    def _panel(ax, labels, scores, is_hyp_list, col, title):
        ypos = np.arange(len(labels))
        bars = ax.barh(ypos, scores, color=col, alpha=0.78,
                       edgecolor="white", linewidth=0.5, height=0.72)
        for bar, hyp in zip(bars, is_hyp_list):
            if hyp:
                bar.set_edgecolor(PAL["accent"])
                bar.set_linewidth(2.0)
                bar.set_alpha(0.92)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_title(title, color=col, fontsize=11, pad=9)
        ax.set_xlabel("|Log-likelihood|", fontsize=9)
        # value labels at bar ends
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                    f"{score:,.0f}", va="center", fontsize=7.5, color=PAL["sub"])
        _clean_ax(ax, grid_axis="x")

    _panel(ax_l, b2b_labels, b2b_scores, b2b_hyp, PAL["b2b"],
           "Client-distinctive  (B2B ↑)")
    _panel(ax_r, b2w_labels, b2w_scores, b2w_hyp, PAL["b2w"],
           "Worker-distinctive  (B2W ↑)")

    fig.suptitle("F2 — Exclusive Vocabulary: Terms Most Distinctive to Each Audience",
                 fontsize=12, fontweight="bold", y=1.01)
    _caption(fig, "★ = term in hypothesis vocabulary (H1a / H1b / H1c)  •  "
                  "Orange border = hypothesis term  •  Ranked by log-likelihood (descending)")
    save_fig(fig, "f2_exclusive_vocab")


# =============================================================================
# F3 — SHARED TERMS, DIVERGENT CONTEXTS
# =============================================================================

def fig_f3_shared_divergent(conn):
    """
    F3 — PMI collocate panels: same term, different discourse environment.

    Layout: 2-column grid.  Each row = one focus term from F3_TERMS.
    The focus term is used as the row header (bold suptitle-style text
    spanning both panels), so it is clearly legible without cluttering
    the chart area.

    Left panel = B2B collocates (blue); Right = B2W collocates (red).
    Edit F3_TERMS at the top to change which terms are shown.
    """
    log.info("F3 — shared terms, divergent contexts")
    if not table_exists(conn, "cooccurrence_results"):
        log.warning("  cooccurrence_results not found — skipping F3.")
        return

    noise_ph = _ph(NOISE)
    n_terms  = len(F3_TERMS)

    fig = plt.figure(figsize=(14, 3.6 * n_terms))
    # Outer grid: one row per term, two columns (left / right)
    outer = gridspec.GridSpec(
        n_terms, 2,
        hspace=0.80, wspace=0.40,
        left=0.08, right=0.96,
        top=0.93, bottom=0.04
    )

    any_drawn = False
    for row_i, term in enumerate(F3_TERMS):
        b2b_rows = conn.execute(f"""
            SELECT collocate, pmi FROM cooccurrence_results
            WHERE comparison = 'cross_platform' AND audience = 'client'
              AND focus_term = ? AND cofreq >= ?
              AND collocate NOT IN ({noise_ph})
            ORDER BY pmi DESC LIMIT {F3_N_COLLOC}
        """, [term, F3_MIN_COFREQ] + list(NOISE)).fetchall()

        b2w_rows = conn.execute(f"""
            SELECT collocate, pmi FROM cooccurrence_results
            WHERE comparison = 'cross_platform' AND audience = 'worker'
              AND focus_term = ? AND cofreq >= ?
              AND collocate NOT IN ({noise_ph})
            ORDER BY pmi DESC LIMIT {F3_N_COLLOC}
        """, [term, F3_MIN_COFREQ] + list(NOISE)).fetchall()

        if not b2b_rows and not b2w_rows:
            log.warning(f"  No collocate data for '{term}' — skipping.")
            continue

        any_drawn = True
        ax_l = fig.add_subplot(outer[row_i, 0])
        ax_r = fig.add_subplot(outer[row_i, 1])

        def _bars(ax, crows, col, title):
            if not crows:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color=PAL["sub"])
                ax.axis("off")
                return
            clabels = [r["collocate"] for r in crows][::-1]
            pmis    = [r["pmi"]       for r in crows][::-1]
            ypos    = np.arange(len(clabels))
            ax.barh(ypos, pmis, color=col, alpha=0.82,
                    edgecolor="white", linewidth=0.4, height=0.68)
            ax.set_yticks(ypos)
            ax.set_yticklabels(clabels, fontsize=9)
            ax.set_xlabel("PMI", fontsize=8.5, labelpad=2)
            ax.set_title(title, fontsize=9.5, color=col, pad=5)
            _clean_ax(ax, grid_axis="x")

        # Focus term as row label — placed as a shared super-title for the row
        # by positioning text in figure coords at the vertical centre of this row.
        row_top    = 1.0 - row_i / n_terms
        row_bottom = 1.0 - (row_i + 1) / n_terms
        row_mid_y  = (row_top + row_bottom) / 2

        fig.text(0.5, row_mid_y + 0.02,
                 f'Focus term: "{term}"',
                 ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=PAL["text"],
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="#F5F5F5", edgecolor=PAL["grid"]))

        _bars(ax_l, b2b_rows, PAL["b2b"], "B2B collocate context")
        _bars(ax_r, b2w_rows, PAL["b2w"], "B2W collocate context")

    if not any_drawn:
        plt.close(fig)
        return

    fig.suptitle("F3 — Shared Terms, Divergent Contexts: "
                 "Same Word, Different Discourse Environment",
                 fontsize=12, fontweight="bold")
    _caption(fig, "Each row = one focus term present in both registers.  "
                  "Left = B2B top collocates; Right = B2W top collocates, "
                  "both ranked by PMI.  "
                  "Large mismatch → same surface form, different framing → priority for Step 2.",
             y=-0.015)
    save_fig(fig, "f3_shared_divergent")


# =============================================================================
# F4 — TOPIC PROFILES (unified diverging bar chart)
# =============================================================================

def fig_f4_topic_profiles(conn):
    """
    F4 — Single unified horizontal diverging bar chart: all LDA topics.

    x-axis: B2B share − 0.5  (positive = B2B-dominant, negative = B2W-dominant)
    Each topic = one row, sorted from most B2B (top) to most B2W (bottom).
    Bar colour: blue (B2B), red (B2W), or grey (shared zone ±0.05).
    Right-side annotation: top F4_TOP_TERMS terms per topic.

    This single-axis layout makes all topics directly comparable and
    reveals the full spectrum from B2B pole to B2W pole at a glance.
    """
    log.info("F4 — topic profiles (unified chart)")
    if not (table_exists(conn, "topic_audience_profile") and
            table_exists(conn, "topic_terms")):
        log.warning("  topic tables not found — skipping F4.")
        return

    profiles = conn.execute("""
        SELECT topic_id, client_share, category,
               n_dominant_client, n_dominant_worker
        FROM topic_audience_profile
        ORDER BY client_share DESC
    """).fetchall()

    if not profiles:
        log.warning("  No topic_audience_profile data — skipping F4.")
        return

    topic_top = defaultdict(list)
    for r in conn.execute(f"""
        SELECT topic_id, term FROM topic_terms
        WHERE rank <= {F4_TOP_TERMS}
        ORDER BY topic_id, rank
    """).fetchall():
        topic_top[r["topic_id"]].append(r["term"])

    n = len(profiles)
    # Divergence value: B2B_share − 0.5
    devs    = [(p["client_share"] or 0.5) - 0.5 for p in profiles]
    n_doms  = [(p["n_dominant_client"] or 0) + (p["n_dominant_worker"] or 0)
               for p in profiles]
    y_labels = [f"T{p['topic_id']}  (n={nd})"
                for p, nd in zip(profiles, n_doms)]
    bar_cols = []
    for d in devs:
        if d > 0.05:
            bar_cols.append(PAL["b2b"])
        elif d < -0.05:
            bar_cols.append(PAL["b2w"])
        else:
            bar_cols.append(PAL["neutral"])

    # Figure height scales with number of topics
    fig_h = max(8, n * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    ypos = np.arange(n)
    bars = ax.barh(ypos, devs, color=bar_cols, alpha=0.80,
                   edgecolor="white", linewidth=0.5, height=0.72)

    # Shared-zone band
    ax.axvspan(-0.05, 0.05, color=PAL["neutral"], alpha=0.10, zorder=0,
               label="Shared zone (±5 pp)")
    ax.axvline(0, color=PAL["text"], linewidth=0.9, zorder=3)

    # Term annotations to the right of each bar
    max_abs = max(abs(d) for d in devs) if devs else 0.1
    for i, (prof, dev) in enumerate(zip(profiles, devs)):
        words = "  ·  ".join(topic_top.get(prof["topic_id"], []))
        x_text = max_abs * 1.04  # always right edge — fixed position
        ax.text(x_text, i, words,
                va="center", ha="left", fontsize=7.5,
                color=PAL["sub"], style="italic")

    ax.set_yticks(ypos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("B2B share − 0.5   (positive = B2B-dominant; negative = B2W-dominant)",
                  fontsize=9.5)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:+.2f}")
    )

    # Pole annotations at the x-axis ends
    xlims = ax.get_xlim()
    ax.text(xlims[1] * 0.95, n - 0.5, "← B2B pole",
            ha="right", va="top", fontsize=9, color=PAL["b2b"], style="italic")
    ax.text(xlims[0] * 0.95, n - 0.5, "B2W pole →",
            ha="left", va="top", fontsize=9, color=PAL["b2w"], style="italic")

    # Legend
    legend_h = [
        mpatches.Patch(color=PAL["b2b"], alpha=0.80, label="B2B-dominant topic"),
        mpatches.Patch(color=PAL["b2w"], alpha=0.80, label="B2W-dominant topic"),
        mpatches.Patch(color=PAL["neutral"], alpha=0.40, label="Shared zone (±5 pp)"),
    ]
    ax.legend(handles=legend_h, loc="lower right", fontsize=9)

    ax.set_title("F4 — Topic Profiles: Audience Balance per LDA Topic", pad=12)
    ax.invert_yaxis()   # most B2B at top
    _clean_ax(ax, grid_axis="x")
    _caption(fig, "Sorted B2B-dominant (top) → B2W-dominant (bottom).  "
                  "Bar length = deviation from 50/50 balance.  "
                  f"Right-side labels = top {F4_TOP_TERMS} topic terms (italic).")
    save_fig(fig, "f4_topic_profiles")


# =============================================================================
# F5 — TOPIC-HYPOTHESIS ALIGNMENT
# =============================================================================

def fig_f5_topic_hyp_alignment(conn):
    """
    F5 — Heatmap matrix: LDA topics (rows) × hypotheses (columns).

    Overlap score = |topic top-K terms ∩ hypothesis vocabulary| / K.
    Cell annotations list the matching terms so alignment is verifiable.
    Column backgrounds are shaded in each hypothesis's colour for visual
    grouping; row labels include a B2B/B2W tag for audience context.
    """
    log.info("F5 — topic-hypothesis alignment")
    if not table_exists(conn, "topic_terms"):
        log.warning("  topic_terms not found — skipping F5.")
        return

    topic_rows = conn.execute(f"""
        SELECT topic_id, term FROM topic_terms
        WHERE rank <= {F5_TOP_K}
        ORDER BY topic_id, rank
    """).fetchall()

    if not topic_rows:
        log.warning("  No data in topic_terms — skipping F5.")
        return

    topic_terms_map = defaultdict(list)
    for r in topic_rows:
        topic_terms_map[r["topic_id"]].append(r["term"])

    topic_ids = sorted(topic_terms_map.keys())
    hyp_keys  = list(HYP.keys())

    overlap = np.zeros((len(topic_ids), len(hyp_keys)))
    matches = {}
    for ri, tid in enumerate(topic_ids):
        top_k = set(topic_terms_map[tid][:F5_TOP_K])
        for ci, hk in enumerate(hyp_keys):
            shared = top_k & HYP[hk]["terms"]
            overlap[ri, ci] = len(shared) / max(F5_TOP_K, 1)
            if shared:
                matches[(ri, ci)] = sorted(shared)

    cs_map = {}
    if table_exists(conn, "topic_audience_profile"):
        for r in conn.execute(
            "SELECT topic_id, client_share FROM topic_audience_profile"
        ).fetchall():
            cs_map[r["topic_id"]] = r["client_share"] or 0.5

    fig_h = max(5, len(topic_ids) * 0.58 + 2.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "overlap", ["#FFFFFF", "#FDE8D8", "#F0A06A", "#C0392B"], N=256
    )
    vmax = max(0.001, overlap.max())
    im = ax.imshow(overlap, cmap=cmap, vmin=0, vmax=vmax,
                   aspect="auto", zorder=2)

    # Cell annotations
    for (ri, ci), mterms in matches.items():
        label = f"n={len(mterms)}\n" + ", ".join(mterms[:3])
        ax.text(ci, ri, label, ha="center", va="center",
                fontsize=6.5, color=PAL["text"], zorder=4)

    # Column dividers
    for ci in range(len(hyp_keys)):
        ax.axvline(ci - 0.5, color="#CCCCCC", linewidth=0.6, zorder=3)

    # Hypothesis column background tint (subtle) — using axvspan
    for ci, hk in enumerate(hyp_keys):
        ax.axvspan(ci - 0.5, ci + 0.5,
                   color=HYP[hk]["color"], alpha=0.05, zorder=0)

    # X ticks — colored per hypothesis
    ax.set_xticks(range(len(hyp_keys)))
    ax.set_xticklabels([HYP[k]["label"] for k in hyp_keys],
                       fontsize=9.5, rotation=15, ha="right")
    for tick, hk in zip(ax.get_xticklabels(), hyp_keys):
        tick.set_color(HYP[hk]["color"])
        tick.set_fontweight("bold")

    # Y ticks
    ax.set_yticks(range(len(topic_ids)))
    row_labels = []
    for tid in topic_ids:
        cs  = cs_map.get(tid, 0.5)
        tag = " ▲" if cs > 0.60 else (" ▼" if cs < 0.40 else " ≈")
        row_labels.append(f"Topic {tid}{tag}")
    ax.set_yticklabels(row_labels, fontsize=8.5)

    plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02,
                 label=f"Overlap score  (shared terms ÷ {F5_TOP_K})")

    ax.set_title("F5 — Topic–Hypothesis Alignment: Where LDA Meets Theory",
                 pad=14)
    ax.set_xlabel("Hypothesis  (coloured per H1a / H1b / H1c)", fontsize=9)
    ax.set_ylabel("LDA topic  (▲ = B2B-leaning, ▼ = B2W-leaning, ≈ = shared)",
                  fontsize=8.5)
    _clean_ax(ax, grid_axis="none")
    _caption(fig, "High overlap = topic directly engages hypothesis vocabulary.  "
                  "Zero-overlap rows = empirical discoveries not pre-specified by theory.",
             y=-0.025)
    save_fig(fig, "f5_topic_hyp_alignment")


# =============================================================================
# F6 — DOCUMENT SPACE (PCA)
# =============================================================================

def fig_f6_document_space(conn):
    """
    F6 — PCA document space: audience separation with confidence ellipses.

    What PCA on topic models means
    --------------------------------
    The LDA model assigns each document a probability distribution over
    K topics.  PCA then reduces this K-dimensional document-topic matrix
    to 2 principal components that capture maximum variance.

    PC1 (x-axis) = the dominant axis of variation across the corpus.
      If the register gap is real and pervasive, B2B and B2W documents
      should separate along PC1 — this is the key quality check.

    PC2 (y-axis) = secondary variance, orthogonal to PC1.
      Typically captures thematic variation WITHIN one audience
      (e.g. domain-specific B2B topics: medical vs automotive vs legal).

    Quality checks shown on the figure:
      - Audience centroids (×): if far apart relative to spread → good separation
      - 95% confidence ellipses: overlap = registers share topic space;
        separation = structurally distinct topic usage
      - 4 extreme topic annotations: which topics anchor each pole
    """
    log.info("F6 — document space PCA")
    if not table_exists(conn, "document_topics"):
        log.warning("  document_topics not found — skipping F6.")
        return

    rows = conn.execute("""
        SELECT audience, dominant_topic, pca_1, pca_2
        FROM document_topics
        WHERE pca_1 IS NOT NULL AND pca_2 IS NOT NULL
    """).fetchall()

    n_excl = conn.execute(
        "SELECT COUNT(*) FROM document_topics WHERE pca_1 IS NULL"
    ).fetchone()[0]

    if not rows:
        log.warning("  No valid PCA coordinates — skipping F6.")
        return

    # Top words per topic (for extreme centroid labels only)
    topic_words = defaultdict(list)
    if table_exists(conn, "topic_terms"):
        for r in conn.execute(f"""
            SELECT topic_id, term FROM topic_terms
            WHERE rank <= 3
            ORDER BY topic_id, rank
        """).fetchall():
            topic_words[r["topic_id"]].append(r["term"])

    # Split by audience
    aud_pts = {
        "client": {"x": [], "y": [], "col": PAL["b2b"], "lbl": "Client (B2B)"},
        "worker": {"x": [], "y": [], "col": PAL["b2w"], "lbl": "Worker (B2W)"},
    }
    for r in rows:
        if r["audience"] in aud_pts:
            aud_pts[r["audience"]]["x"].append(r["pca_1"])
            aud_pts[r["audience"]]["y"].append(r["pca_2"])

    # Topic centroids
    topic_pts = defaultdict(lambda: {"x": [], "y": []})
    for r in rows:
        topic_pts[r["dominant_topic"]]["x"].append(r["pca_1"])
        topic_pts[r["dominant_topic"]]["y"].append(r["pca_2"])

    topic_centroids = {
        tid: (np.mean(v["x"]), np.mean(v["y"]))
        for tid, v in topic_pts.items()
    }

    # Select only the most extreme topics to annotate (avoid clutter)
    if topic_centroids:
        sorted_by_pc1 = sorted(topic_centroids.items(), key=lambda x: x[1][0])
        sorted_by_pc2 = sorted(topic_centroids.items(), key=lambda x: x[1][1])
        extreme_tids = set()
        for lst in [sorted_by_pc1, sorted_by_pc2]:
            for tid, _ in lst[:F6_N_EXTREME]:
                extreme_tids.add(tid)
            for tid, _ in lst[-F6_N_EXTREME:]:
                extreme_tids.add(tid)
    else:
        extreme_tids = set()

    fig, ax = plt.subplots(figsize=(9, 8))

    # ── scatter + ellipses ─────────────────────────────────────────────────
    for aud, d in aud_pts.items():
        if not d["x"]:
            continue
        ax.scatter(d["x"], d["y"], c=d["col"], s=F6_DOT_SIZE,
                   alpha=F6_DOT_ALPHA, linewidths=0, zorder=2,
                   label=d["lbl"])
        _confidence_ellipse(ax, np.array(d["x"]), np.array(d["y"]),
                            n_std=F6_ELLIPSE_STD, color=d["col"])
        ax.plot(np.mean(d["x"]), np.mean(d["y"]), "X",
                color=d["col"], markersize=13, zorder=6,
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"{d['lbl']} centroid")

    # ── extreme topic annotations only ────────────────────────────────────
    for tid in extreme_tids:
        cx, cy = topic_centroids[tid]
        wlabel = ", ".join(topic_words.get(tid, [])[:2])
        ax.text(cx, cy, f"T{tid}: {wlabel}",
                ha="center", va="center", fontsize=7, color=PAL["sub"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=PAL["grid"], alpha=0.85), zorder=5)

    # ── PC interpretation labels ───────────────────────────────────────────
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.text(xlims[0] + (xlims[1] - xlims[0]) * 0.02, ylims[0] + 0.02,
            "← B2B / client-register topics",
            fontsize=8, color=PAL["b2b"], style="italic", va="bottom")
    ax.text(xlims[1] - (xlims[1] - xlims[0]) * 0.02, ylims[0] + 0.02,
            "B2W / worker-register topics →",
            fontsize=8, color=PAL["b2w"], style="italic", va="bottom", ha="right")

    if n_excl > 0:
        ax.text(0.98, 0.98,
                f"n = {n_excl} pages excluded\nbefore PCA fitting\n(template-heavy domain)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=PAL["sub"], style="italic",
                bbox=dict(boxstyle="round,pad=0.30",
                          facecolor=PAL["bg"], edgecolor=PAL["grid"], alpha=0.85))

    ax.set_xlabel("PC 1  (primary variation axis — separates audience registers)",
                  fontsize=9.5)
    ax.set_ylabel("PC 2  (secondary variation — thematic diversity within registers)",
                  fontsize=9.5)
    ax.set_title("F6 — Document Space: Audience Separation in Topic-PCA Space")
    ax.legend(loc="upper right", fontsize=8.5, markerscale=1.3)
    _clean_ax(ax, grid_axis="both")
    _caption(fig,
             "× = audience centroid  •  Dashed ellipse = 95% confidence region  •  "
             f"Only the {len(extreme_tids)} most extreme topic centroids labelled.  "
             "Non-overlapping ellipses = systematic audience-level register separation.")
    save_fig(fig, "f6_document_space")


# =============================================================================
# F7 — STEP 2 SAMPLE MAP
# =============================================================================

def fig_f7_step2_sample_map(conn):
    """
    F7 — PCA scatter with Step 2 sample highlighted + zoomed inset.

    Background: all documents, very faded (corpus shape visible).
    Foreground: Step 2 sample pages, coloured by hypothesis alignment
                (inferred from dominant_topic → best-matching hypothesis).

    A zoomed inset panel (bottom-right) magnifies the main sample cluster
    so individual sample points are legible even when compressed in the
    full-corpus view.
    """
    log.info("F7 — step 2 sample map")
    if not (table_exists(conn, "document_topics") and
            table_exists(conn, "step2_sample")):
        log.warning("  Required tables not found — skipping F7.")
        return

    all_docs = conn.execute("""
        SELECT page_id, audience, pca_1, pca_2, dominant_topic
        FROM document_topics
        WHERE pca_1 IS NOT NULL AND pca_2 IS NOT NULL
    """).fetchall()

    sample_ids = {r["page_id"] for r in conn.execute(
        "SELECT page_id FROM step2_sample"
    ).fetchall()}

    if not all_docs or not sample_ids:
        log.warning("  No data — skipping F7.")
        return

    # Build topic → best hypothesis (same as F5)
    topic_best_hyp = {}
    if table_exists(conn, "topic_terms"):
        ttmap = defaultdict(list)
        for r in conn.execute(f"""
            SELECT topic_id, term FROM topic_terms
            WHERE rank <= {F5_TOP_K}
            ORDER BY topic_id, rank
        """).fetchall():
            ttmap[r["topic_id"]].append(r["term"])
        for tid, tterms in ttmap.items():
            best_hyp, best_score = None, 0
            for hk, cfg in HYP.items():
                sc = len(set(tterms) & cfg["terms"])
                if sc > best_score:
                    best_score, best_hyp = sc, hk
            if best_hyp and best_score > 0:
                topic_best_hyp[tid] = best_hyp

    sample_docs = [r for r in all_docs if r["page_id"] in sample_ids]
    bg_docs     = [r for r in all_docs if r["page_id"] not in sample_ids]

    fig, ax = plt.subplots(figsize=(9, 8))

    # ── faded background ───────────────────────────────────────────────────
    for aud, col in [("client", PAL["b2b"]), ("worker", PAL["b2w"])]:
        bx = [r["pca_1"] for r in bg_docs if r["audience"] == aud]
        by = [r["pca_2"] for r in bg_docs if r["audience"] == aud]
        ax.scatter(bx, by, c=col, s=8, alpha=F7_BG_ALPHA,
                   linewidths=0, zorder=2)

    # ── sampled pages ──────────────────────────────────────────────────────
    drawn_hyps = set()
    sx_all, sy_all = [], []

    for r in sample_docs:
        hk = topic_best_hyp.get(r["dominant_topic"])
        if hk:
            col    = HYP[hk]["color"]
            marker = HYP[hk]["marker"]
            label  = HYP[hk]["label"] if hk not in drawn_hyps else "_"
            drawn_hyps.add(hk)
        else:
            col    = PAL["b2b"] if r["audience"] == "client" else PAL["b2w"]
            marker = "D"
            label  = "_"
        ax.scatter(r["pca_1"], r["pca_2"], c=col, s=F7_FG_SIZE,
                   marker=marker, alpha=0.92, linewidths=1.3,
                   edgecolors="white", zorder=5, label=label)
        sx_all.append(r["pca_1"])
        sy_all.append(r["pca_2"])

    # ── zoomed inset ───────────────────────────────────────────────────────
    if F7_INSET and sx_all:
        pad = 0.06
        x0, x1 = min(sx_all) - pad, max(sx_all) + pad
        y0, y1 = min(sy_all) - pad, max(sy_all) + pad

        ax_ins = ax.inset_axes([0.60, 0.02, 0.38, 0.36])
        # background in inset
        for aud, col in [("client", PAL["b2b"]), ("worker", PAL["b2w"])]:
            bx = [r["pca_1"] for r in all_docs
                  if r["audience"] == aud and x0 <= r["pca_1"] <= x1
                  and y0 <= r["pca_2"] <= y1]
            by = [r["pca_2"] for r in all_docs
                  if r["audience"] == aud and x0 <= r["pca_1"] <= x1
                  and y0 <= r["pca_2"] <= y1]
            ax_ins.scatter(bx, by, c=col, s=5, alpha=F7_BG_ALPHA * 1.5,
                           linewidths=0)
        for r in sample_docs:
            hk  = topic_best_hyp.get(r["dominant_topic"])
            col = HYP[hk]["color"] if hk else (
                PAL["b2b"] if r["audience"] == "client" else PAL["b2w"])
            mk  = HYP[hk]["marker"] if hk else "D"
            ax_ins.scatter(r["pca_1"], r["pca_2"], c=col, s=40,
                           marker=mk, alpha=0.92,
                           linewidths=0.8, edgecolors="white", zorder=5)

        ax_ins.set_xlim(x0, x1)
        ax_ins.set_ylim(y0, y1)
        ax_ins.set_title("Sample region (zoomed)", fontsize=7.5,
                         color=PAL["sub"], pad=3)
        ax_ins.tick_params(labelsize=6.5)
        ax_ins.spines[:].set_color(PAL["grid"])
        ax_ins.set_facecolor(PAL["bg"])
        # Indicate inset region on main axes
        ax.indicate_inset_zoom(ax_ins, edgecolor=PAL["sub"])

    # ── legend ─────────────────────────────────────────────────────────────
    bg_h = [
        plt.Line2D([0], [0], linestyle="none", marker="s", markersize=8,
                   color=PAL["b2b"], alpha=0.20, label="B2B (background)"),
        plt.Line2D([0], [0], linestyle="none", marker="s", markersize=8,
                   color=PAL["b2w"], alpha=0.20, label="B2W (background)"),
    ]
    hyp_h = [
        plt.Line2D([0], [0], linestyle="none",
                   marker=HYP[k]["marker"], markersize=9,
                   color=HYP[k]["color"], label=HYP[k]["label"])
        for k in HYP if k in drawn_hyps
    ]
    ax.legend(handles=bg_h + hyp_h, loc="upper left",
              fontsize=8.5, frameon=True)

    ax.set_xlabel("PC 1", fontsize=10)
    ax.set_ylabel("PC 2", fontsize=10)
    ax.set_title(f"F7 — Step 2 Sample Map: {len(sample_docs)} Pages Selected for Close Reading")
    _clean_ax(ax, grid_axis="both")
    _caption(fig,
             "Sample coloured by hypothesis alignment (dominant topic ↔ hypothesis vocabulary).  "
             "◆ = no hypothesis overlap (empirical discovery selection).  "
             "Inset = zoomed view of the sample cluster.",
             y=-0.015)
    save_fig(fig, "f7_step2_sample_map")


# =============================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────────
# =============================================================================

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def main():
    apply_theme()
    conn = get_conn()

    dispatch = {
        "f1_vocab_terrain":       fig_f1_vocab_terrain,
        "f2_exclusive_vocab":     fig_f2_exclusive_vocab,
        "f3_shared_divergent":    fig_f3_shared_divergent,
        "f4_topic_profiles":      fig_f4_topic_profiles,
        "f5_topic_hyp_alignment": fig_f5_topic_hyp_alignment,
        "f6_document_space":      fig_f6_document_space,
        "f7_step2_sample_map":    fig_f7_step2_sample_map,
    }

    for name, fn in dispatch.items():
        if FIGURES.get(name, True):
            try:
                fn(conn)
            except Exception as exc:
                log.error(f"  {name} failed: {exc}", exc_info=True)
        else:
            log.info(f"  {name} skipped (FIGURES['{name}'] = False)")

    conn.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
