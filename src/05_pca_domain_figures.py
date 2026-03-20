"""
05_pca_domain_figures.py
=========================
Four PCA-focused figures for the DarkSideofAI thesis argumentative structure.

Argumentative purpose
---------------------
  Figure A — Domain landscape strip plot
      One dot per domain positioned on PC1, size = n_pages, colour = audience,
      error bars = within-domain std.  Reads as a "map of platforms" showing
      the register gradient from technical/client-like to labour/worker-like.

  Figure B — Full PCA scatter with domain colouring
      Individual page scatter coloured by domain.  Audience confidence ellipses
      retained but framed as showing register range, not hard clusters.  Makes
      the methodological rigour of the sampling visible.

  Figure C — Within-domain PC1 variation (strip plot)
      PC1 value distribution per domain, sorted by mean, coloured by audience.
      Shows that register consistency varies across platforms — key evidence that
      register is a strategic choice, not a fixed platform property.

  Figure D — Step 2 sample map
      Background scatter (grey) overlaid with sample pages (large markers)
      coloured by hypothesis (H1a/H1b/H1c).  Makes the sampling logic
      transparent and connects PCA position to theoretical motivation.

Usage
-----
  python3 src/05_pca_domain_figures.py
  Toggle figures with FIGURES dict; all visual params in the CONFIG section.
"""

import sqlite3
import logging
import statistics
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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
OUTPUT_DIR = Path("output/step_1/pca/")
DPI        = 150
EXT        = "jpg"

# ── toggle individual figures ─────────────────────────────────────────────────
FIGURES = {
    "fa_domain_landscape":    True,
    "fb_pca_scatter":         True,
    "fc_within_domain_var":   True,
    "fd_step2_sample_map":    True,
    "fe_topic_pc_loadings":   True,
    "ff_pca_domain_shapes":   True,
}

# ── Figure F parameters ───────────────────────────────────────────────────────
FF_DOT_ALPHA   = 0.55
FF_DOT_SIZE    = 18     # marker size
FF_MAX_DOMAINS = 20     # domains with distinct colours; rest shown in grey

# ── Figure E parameters ───────────────────────────────────────────────────────
FE_TOP_N      = 12    # number of topics to show at each end of each PC
FE_TOP_TERMS  = 4     # number of topic terms to show per bar label
FE_MIN_DOCS   = 5     # minimum documents with topic as dominant to include

# ── colour palette (consistent with 04_step1_narrative_figures.py) ────────────
PAL = dict(
    b2b     = "#1B4F8A",   # client / B2B
    b2w     = "#C0392B",   # worker / B2W
    h1a     = "#C0392B",   # H1a — Labour Visibility (red)
    h1b     = "#1B4F8A",   # H1b — Automation Myth  (blue)
    h1c     = "#E67E22",   # H1c — Strategic Hypervisibility (orange)
    neutral = "#BBBBBB",
    grid    = "#E0E0E0",
    text    = "#1A1A2E",
    sub     = "#6C757D",
    bg      = "#FFFFFF",
    highlight = "#F4D03F", # crowdgen.com anomaly highlight
)

# ── hypothesis metadata ───────────────────────────────────────────────────────
HYP = {
    "H1a": dict(label="H1a — Labour Visibility",          color=PAL["h1a"], marker="o"),
    "H1b": dict(label="H1b — Automation Myth",            color=PAL["h1b"], marker="s"),
    "H1c": dict(label="H1c — Strategic Hypervisibility",  color=PAL["h1c"], marker="^"),
}

# ── Figure A parameters ───────────────────────────────────────────────────────
FA_MIN_PAGES   = 3      # minimum pages for a domain to appear
FA_DOT_SCALE   = 8      # sqrt(n_pages) * FA_DOT_SCALE = marker size
FA_ERR_ALPHA   = 0.55   # opacity of error bars

# ── Figure B parameters ───────────────────────────────────────────────────────
FB_DOT_ALPHA   = 0.35
FB_DOT_SIZE    = 12
FB_ELLIPSE_STD = 2.0    # n_std for audience confidence ellipses
FB_MAX_DOMAINS = 20     # cap on legend entries (busiest domains)

# ── Figure C parameters ───────────────────────────────────────────────────────
FC_MIN_PAGES   = 5      # minimum pages for a domain to appear in strip plot
FC_DOT_ALPHA   = 0.45
FC_DOT_SIZE    = 14
FC_JITTER      = 0.18   # vertical jitter to separate overlapping dots

# =============================================================================
# ── LOGGING ───────────────────────────────────────────────────────────────────
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# ── THEME ─────────────────────────────────────────────────────────────────────
# =============================================================================

def apply_theme():
    rc = {
        "figure.facecolor":  PAL["bg"],
        "axes.facecolor":    PAL["bg"],
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    PAL["grid"],
        "axes.labelcolor":   PAL["text"],
        "axes.labelsize":    10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.titlecolor":   PAL["text"],
        "axes.grid":         True,
        "grid.color":        PAL["grid"],
        "grid.linewidth":    0.6,
        "grid.linestyle":    ":",
        "axes.axisbelow":    True,
        "xtick.color":       PAL["sub"],
        "ytick.color":       PAL["sub"],
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.framealpha": 0.92,
        "legend.edgecolor":  PAL["grid"],
        "legend.fontsize":   9,
        "font.family":       "sans-serif",
        "text.color":        PAL["text"],
        "figure.dpi":        DPI,
        "savefig.dpi":       DPI,
        "savefig.bbox":      "tight",
    }
    if _HAS_SNS:
        sns.set_theme(style="ticks", rc=rc)
        log.info("Theme: seaborn 'ticks'")
    else:
        plt.rcParams.update(rc)
        log.info("Theme: manual rcParams")

# =============================================================================
# ── HELPERS ───────────────────────────────────────────────────────────────────
# =============================================================================

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def save_fig(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.{EXT}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    log.info(f"  → {path}")
    plt.close(fig)

def _caption(fig, text, y=-0.02):
    fig.text(0.5, y, text, ha="center", va="top",
             fontsize=8, color=PAL["sub"], style="italic")

def _confidence_ellipse(ax, x, y, n_std=2.0, color=None, **kwargs):
    """Draw an n_std-sigma confidence ellipse for the data (x, y)."""
    from matplotlib.patches import Ellipse
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width  = 2 * n_std * np.sqrt(max(vals[0], 0))
    height = 2 * n_std * np.sqrt(max(vals[1], 0))
    cx, cy = np.mean(x), np.mean(y)
    el = Ellipse((cx, cy), width, height, angle=angle,
                 facecolor=color, alpha=0.08, edgecolor=color,
                 linewidth=1.5, linestyle="--", **kwargs)
    ax.add_patch(el)

# =============================================================================
# ── DATA LOADING ──────────────────────────────────────────────────────────────
# =============================================================================

def load_document_topics(conn):
    """Return list of dicts: domain, audience, pca_1, pca_2."""
    rows = conn.execute("""
        SELECT domain, audience, pca_1, pca_2, dominant_topic
        FROM   document_topics
        WHERE  pca_1 IS NOT NULL
        ORDER  BY domain
    """).fetchall()
    return [dict(r) for r in rows]

def compute_domain_stats(docs):
    """
    Aggregate page-level PCA data by domain.

    Returns dict keyed by domain:
      n_pages, audience, pca1_mean, pca1_std, pca1_vals, pca2_vals
    """
    by_domain = defaultdict(lambda: {"pca1": [], "pca2": [], "audience": None})
    for d in docs:
        dom = d["domain"]
        by_domain[dom]["pca1"].append(d["pca_1"])
        by_domain[dom]["pca2"].append(d["pca_2"] or 0.0)
        by_domain[dom]["audience"] = d["audience"]

    stats = {}
    for dom, data in by_domain.items():
        vals = data["pca1"]
        if len(vals) < 1:
            continue
        stats[dom] = {
            "n_pages":    len(vals),
            "audience":   data["audience"],
            "pca1_mean":  statistics.mean(vals),
            "pca1_std":   statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "pca1_vals":  vals,
            "pca2_vals":  data["pca2"],
        }
    return stats

def load_step2_sample(conn):
    """
    Return list of dicts for Step 2 sample with PCA coords and hypothesis label.
    hypothesis extracted from sampling_reason prefix (e.g. 'H1a_topic_11_...').
    """
    rows = conn.execute("""
        SELECT s.domain, s.audience, s.sampling_reason, s.priority_rank,
               dt.pca_1, dt.pca_2
        FROM   step2_sample s
        LEFT JOIN document_topics dt ON dt.page_id = s.page_id
        WHERE  dt.pca_1 IS NOT NULL
    """).fetchall()
    result = []
    for r in rows:
        reason = r["sampling_reason"] or ""
        hyp = reason.split("_topic_")[0] if "_topic_" in reason else reason.split("_")[0]
        # Normalise to short key: H1a_visibility → H1a
        for key in ("H1a", "H1b", "H1c"):
            if hyp.startswith(key):
                hyp = key
                break
        result.append({
            "domain":   r["domain"],
            "audience": r["audience"],
            "hyp":      hyp,
            "pca_1":    r["pca_1"],
            "pca_2":    r["pca_2"] or 0.0,
            "rank":     r["priority_rank"],
        })
    return result

# =============================================================================
# ── FIGURE A — Domain Landscape Strip Plot ────────────────────────────────────
# =============================================================================

def fig_a_domain_landscape(domain_stats):
    """
    One dot per domain on the PC1 axis.
    Dot size encodes page count; colour encodes audience; error bars show std.
    Sorted by PC1 mean to read as a register gradient.
    """
    log.info("Figure A: domain landscape strip plot")

    # Filter and sort
    entries = [
        (dom, s) for dom, s in domain_stats.items()
        if s["n_pages"] >= FA_MIN_PAGES
    ]
    entries.sort(key=lambda x: x[1]["pca1_mean"])

    n = len(entries)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.38)))

    y_positions = list(range(n))

    for yi, (dom, s) in enumerate(entries):
        aud     = s["audience"] or "client"
        color   = PAL["b2b"] if aud == "client" else PAL["b2w"]
        dot_sz  = (np.sqrt(s["n_pages"]) * FA_DOT_SCALE) ** 1.1

        # Error bar (std)
        ax.errorbar(
            s["pca1_mean"], yi,
            xerr=s["pca1_std"],
            fmt="none",
            ecolor=color, elinewidth=1.2, capsize=3, alpha=FA_ERR_ALPHA,
        )
        # Domain dot
        ax.scatter(
            s["pca1_mean"], yi,
            s=dot_sz, color=color,
            zorder=3, edgecolors="white", linewidths=0.6,
            alpha=0.9,
        )

    # y-axis: domain labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [e[0] for e in entries],
        fontsize=8.5
    )
    # Colour y-tick labels by audience
    for tick, (dom, s) in zip(ax.get_yticklabels(), entries):
        aud = s["audience"] or "client"
        tick.set_color(PAL["b2b"] if aud == "client" else PAL["b2w"])

    ax.set_xlabel("PC1", fontsize=10)

    # Size legend
    for n_ex in [50, 200, 500]:
        sz = (np.sqrt(n_ex) * FA_DOT_SCALE) ** 1.1
        ax.scatter([], [], s=sz, color=PAL["neutral"],
                   label=f"{n_ex} pages", edgecolors="white")

    # Audience legend patches
    client_patch = mpatches.Patch(color=PAL["b2b"], label="Client-facing (B2B)")
    worker_patch = mpatches.Patch(color=PAL["b2w"], label="Worker-facing (B2W)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [client_patch, worker_patch],
        labels  + ["Client-facing (B2B)", "Worker-facing (B2W)"],
        loc="lower right", fontsize=8, title="Page count / Audience",
        title_fontsize=8,
    )

    ax.set_title("PC1 Position by Domain", pad=10)
    ax.axvline(0, color=PAL["grid"], linewidth=1.0, linestyle="--", zorder=0)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)

    _caption(fig,
        "Each dot = one domain. Position = mean PC1. "
        "Error bars = within-domain std. Size = number of pages. "
        "Colour = audience type."
    )

    save_fig(fig, "fa_domain_landscape")

# =============================================================================
# ── FIGURE B — Full PCA Scatter with Domain Colouring ─────────────────────────
# =============================================================================

def fig_b_pca_scatter(docs, domain_stats):
    """
    Page-level PCA scatter coloured by domain.
    Audience confidence ellipses framed as register ranges.
    Axes annotated with PC1/PC2 interpretations.
    """
    log.info("Figure B: full PCA scatter with domain colouring")

    # Build domain → colour map (busiest domains get distinct colours, rest grey)
    all_domains = sorted(domain_stats, key=lambda d: -domain_stats[d]["n_pages"])
    top_domains = all_domains[:FB_MAX_DOMAINS]

    # Use a qualitative palette that still respects client/worker distinction
    client_domains = [d for d in top_domains
                      if (domain_stats[d]["audience"] or "client") == "client"]
    worker_domains = [d for d in top_domains
                      if (domain_stats[d]["audience"] or "client") != "client"]

    # Generate shades: clients = blues, workers = reds/oranges
    client_colors = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(client_domains), 1)))
    worker_colors = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(worker_domains), 1)))

    domain_color = {}
    for i, d in enumerate(client_domains):
        domain_color[d] = client_colors[i]
    for i, d in enumerate(worker_domains):
        domain_color[d] = worker_colors[i]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Scatter all pages
    plotted_domains = set()
    for doc in docs:
        dom  = doc["domain"]
        col  = domain_color.get(dom, PAL["neutral"])
        alpha = FB_DOT_ALPHA if dom in top_domains else 0.15
        ax.scatter(
            doc["pca_1"], doc.get("pca_2") or 0.0,
            s=FB_DOT_SIZE, color=col, alpha=alpha,
            edgecolors="none", zorder=2,
        )
        plotted_domains.add(dom)

    # Domain legend (top 10 by page count for readability)
    legend_domains = all_domains[:10]
    dom_handles = [
        mlines.Line2D([], [], marker="o", linestyle="none",
                      color=domain_color.get(d, PAL["neutral"]),
                      markersize=6, label=d)
        for d in legend_domains
    ]

    ax.legend(handles=dom_handles,
              loc="upper left", fontsize=7.5,
              title="Top domains", title_fontsize=8)

    ax.set_xlabel("PC1", fontsize=9.5)
    ax.set_ylabel("PC2", fontsize=9.5)

    ax.set_title("PCA Document Space — Page-Level Distribution by Domain", pad=10)

    _caption(fig,
        f"Each dot = one page (n={len(docs):,}). "
        "Top 10 domains shown in legend; remaining pages shown in grey."
    )

    save_fig(fig, "fb_pca_scatter")

# =============================================================================
# ── FIGURE C — Within-Domain PC1 Variation (Strip Plot) ──────────────────────
# =============================================================================

def fig_c_within_domain_variation(domain_stats):
    """
    Horizontal strip plot: one row per domain, dots = individual pages on PC1.
    Sorted by mean PC1. Highlights register-consistent and anomalous domains.
    """
    log.info("Figure C: within-domain variation strip plot")

    entries = [
        (dom, s) for dom, s in domain_stats.items()
        if s["n_pages"] >= FC_MIN_PAGES
    ]
    entries.sort(key=lambda x: x[1]["pca1_mean"])

    n = len(entries)
    rng = np.random.default_rng(42)   # fixed seed for reproducible jitter

    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.42)))

    for yi, (dom, s) in enumerate(entries):
        aud   = s["audience"] or "client"
        color = PAL["b2b"] if aud == "client" else PAL["b2w"]
        vals  = s["pca1_vals"]

        # Jitter on y
        jitter = rng.uniform(-FC_JITTER, FC_JITTER, len(vals))
        ax.scatter(
            vals, [yi + j for j in jitter],
            s=FC_DOT_SIZE, color=color, alpha=FC_DOT_ALPHA,
            edgecolors="none", zorder=2,
        )

        # Mean marker
        ax.scatter(
            s["pca1_mean"], yi,
            s=80, color=color, zorder=4,
            edgecolors="white", linewidths=1.0, marker="D",
        )

        # Std range line
        ax.plot(
            [s["pca1_mean"] - s["pca1_std"], s["pca1_mean"] + s["pca1_std"]],
            [yi, yi],
            color=color, linewidth=1.8, alpha=0.5, zorder=3,
            solid_capstyle="round",
        )

    # y-axis
    ax.set_yticks(range(n))
    ax.set_yticklabels([e[0] for e in entries], fontsize=8.5)
    for tick, (dom, s) in zip(ax.get_yticklabels(), entries):
        aud = s["audience"] or "client"
        tick.set_color(PAL["b2b"] if aud == "client" else PAL["b2w"])

    ax.set_xlabel("PC1", fontsize=10)

    ax.set_title("Within-Domain PC1 Variation by Platform", pad=10)
    ax.axvline(0, color=PAL["grid"], linewidth=1.0, linestyle="--", zorder=0)
    ax.yaxis.grid(False)

    client_patch = mpatches.Patch(color=PAL["b2b"], label="Client-facing (B2B)")
    worker_patch = mpatches.Patch(color=PAL["b2w"], label="Worker-facing (B2W)")
    mean_marker  = mlines.Line2D([], [], marker="D", linestyle="none",
                                 color=PAL["neutral"], markersize=6,
                                 label="Domain mean (◆)")
    ax.legend(handles=[client_patch, worker_patch, mean_marker],
              loc="lower right", fontsize=8)

    _caption(fig,
        "Each dot = one page. ◆ = domain mean PC1. Horizontal line = ±1 std. "
        "Sorted by mean PC1. Colour = audience label."
    )

    save_fig(fig, "fc_within_domain_variation")

# =============================================================================
# ── FIGURE D — Step 2 Sample Map ──────────────────────────────────────────────
# =============================================================================

def fig_d_step2_sample_map(docs, sample):
    """
    Background page scatter (grey) overlaid with Step 2 sample pages
    coloured and shaped by hypothesis (H1a / H1b / H1c).
    Domain labels for each sampled page.
    """
    log.info("Figure D: Step 2 sample map")

    fig, ax = plt.subplots(figsize=(11, 7))

    # Background: all pages
    bg_x = [d["pca_1"] for d in docs]
    bg_y = [d["pca_2"] or 0.0 for d in docs]
    ax.scatter(bg_x, bg_y, s=8, color=PAL["neutral"], alpha=0.25,
               edgecolors="none", zorder=1, label="_nolegend_")

    # Sample pages per hypothesis
    for hyp_key, hdata in HYP.items():
        pts = [s for s in sample if s["hyp"] == hyp_key]
        if not pts:
            continue
        sx = [p["pca_1"] for p in pts]
        sy = [p["pca_2"] for p in pts]
        ax.scatter(sx, sy,
                   s=120, color=hdata["color"], marker=hdata["marker"],
                   edgecolors="white", linewidths=0.8,
                   zorder=4, label=f'{hdata["label"]}  (n={len(pts)})',
                   alpha=0.92)

        # Domain labels for sample pages (de-duplicate overlapping positions)
        seen_positions = []
        for p in pts:
            pos = (round(p["pca_1"], 2), round(p["pca_2"], 2))
            if pos in seen_positions:
                continue
            seen_positions.append(pos)
            ax.annotate(
                p["domain"],
                xy=(p["pca_1"], p["pca_2"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=6.5, color=hdata["color"], alpha=0.85,
                zorder=5,
            )

    # Confidence ellipses for context
    b2b_x = [d["pca_1"] for d in docs if (d["audience"] or "client") == "client"]
    b2b_y = [d["pca_2"] or 0.0 for d in docs if (d["audience"] or "client") == "client"]
    b2w_x = [d["pca_1"] for d in docs if (d["audience"] or "client") != "client"]
    b2w_y = [d["pca_2"] or 0.0 for d in docs if (d["audience"] or "client") != "client"]
    _confidence_ellipse(ax, np.array(b2b_x), np.array(b2b_y), n_std=2.0,
                        color=PAL["b2b"])
    _confidence_ellipse(ax, np.array(b2w_x), np.array(b2w_y), n_std=2.0,
                        color=PAL["b2w"])

    # Ellipse labels
    ax.text(np.mean(b2b_x), np.percentile(b2b_y, 95) + 0.005,
            "Client cluster", color=PAL["b2b"], fontsize=8.5,
            ha="center", fontstyle="italic")
    ax.text(np.mean(b2w_x), np.percentile(b2w_y, 95) + 0.005,
            "Worker cluster", color=PAL["b2w"], fontsize=8.5,
            ha="center", fontstyle="italic")

    ax.set_xlabel("PC1", fontsize=9.5)
    ax.set_ylabel("PC2", fontsize=9.5)

    ax.set_title("Step 2 Sample Distribution in PCA Document Space", pad=10)
    ax.legend(loc="upper left", fontsize=8.5,
              title="Hypothesis", title_fontsize=9)

    _caption(fig,
        f"Grey dots = all corpus pages (n={len(docs):,}). "
        f"Coloured markers = Step 2 sample (n={len(sample)}), stratified by hypothesis. "
        "Dashed ellipses = 95% confidence regions per audience cluster."
    )

    save_fig(fig, "fd_step2_sample_map")

# =============================================================================
# ── FIGURE E — Topic Contributions to PC1 and PC2 ────────────────────────────
# =============================================================================

def load_topic_pc_positions(conn):
    """
    For each topic, compute its mean PC1 and PC2 across all documents
    where it is the dominant topic.  Also fetch the top terms per topic
    from topic_terms for bar labels.

    Returns list of dicts:
      topic_id, mean_pc1, mean_pc2, n_docs, terms, client_share
    """
    # Mean PC position per dominant topic
    rows = conn.execute("""
        SELECT  dt.dominant_topic                  AS topic_id,
                AVG(dt.pca_1)                      AS mean_pc1,
                AVG(dt.pca_2)                      AS mean_pc2,
                COUNT(*)                           AS n_docs,
                AVG(CASE WHEN dt.audience = 'client' THEN 1.0 ELSE 0.0 END)
                                                   AS client_share
        FROM    document_topics dt
        WHERE   dt.pca_1 IS NOT NULL
        GROUP   BY dt.dominant_topic
        HAVING  COUNT(*) >= ?
        ORDER   BY mean_pc1
    """, (FE_MIN_DOCS,)).fetchall()

    # Top terms per topic
    term_rows = conn.execute("""
        SELECT  topic_id, term, rank
        FROM    topic_terms
        WHERE   rank <= ?
        ORDER   BY topic_id, rank
    """, (FE_TOP_TERMS,)).fetchall()

    terms_by_topic = defaultdict(list)
    for r in term_rows:
        terms_by_topic[r["topic_id"]].append(r["term"])

    result = []
    for r in rows:
        tid = r["topic_id"]
        result.append({
            "topic_id":    tid,
            "mean_pc1":    r["mean_pc1"],
            "mean_pc2":    r["mean_pc2"],
            "n_docs":      r["n_docs"],
            "client_share": r["client_share"],
            "terms":       terms_by_topic.get(tid, [f"T{tid}"]),
        })
    return result


def fig_e_topic_pc_loadings(topic_positions):
    """
    Two-panel figure showing topic contributions to PC1 (left) and PC2 (right).
    Each panel shows FE_TOP_N topics at each extreme of the PC axis as
    horizontal bars, labelled with their top terms.
    Bar colour encodes client_share (blue = client-dominant, red = worker-dominant).
    """
    log.info("Figure E: topic PC loadings")

    if not topic_positions:
        log.warning("  No topic positions — skipping Figure E")
        return

    def _make_panel(ax, positions, pc_key, title):
        """Draw one ranked bar panel for a given PC axis."""
        # Sort by PC value, take extremes
        sorted_pos = sorted(positions, key=lambda x: x[pc_key])
        bottom_n   = sorted_pos[:FE_TOP_N]           # most negative
        top_n      = sorted_pos[-FE_TOP_N:][::-1]    # most positive, reversed
        entries    = bottom_n + top_n

        # Deduplicate if corpus has fewer topics than 2*FE_TOP_N
        seen = set()
        deduped = []
        for e in entries:
            if e["topic_id"] not in seen:
                deduped.append(e)
                seen.add(e["topic_id"])
        entries = deduped

        values  = [e[pc_key] for e in entries]
        labels  = [", ".join(e["terms"][:FE_TOP_TERMS]) for e in entries]
        shares  = [e["client_share"] for e in entries]

        # Colour: interpolate between b2w (0.0) and b2b (1.0) via client_share
        def _bar_color(share):
            if share >= 0.65:
                return PAL["b2b"]
            elif share <= 0.35:
                return PAL["b2w"]
            else:
                return PAL["h1c"]   # mixed / neutral → orange

        colors  = [_bar_color(s) for s in shares]
        y_pos   = list(range(len(entries)))

        bars = ax.barh(y_pos, values, color=colors, alpha=0.80,
                       edgecolor="white", linewidth=0.4, height=0.7)

        # Term labels inside/outside bars
        for yi, (val, label) in enumerate(zip(values, labels)):
            ha    = "left"  if val >= 0 else "right"
            xoff  = 0.002   if val >= 0 else -0.002
            ax.text(xoff, yi, label,
                    va="center", ha=ha, fontsize=7.5,
                    color=PAL["text"])

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"T{e['topic_id']}  ({e['n_docs']})" for e in entries],
            fontsize=7.5
        )
        ax.axvline(0, color=PAL["grid"], linewidth=1.0, zorder=0)
        ax.set_xlabel(pc_key.upper(), fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.yaxis.grid(False)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, max(7, FE_TOP_N * 0.55)),
        sharey=False
    )

    _make_panel(ax1, topic_positions, "mean_pc1",
                f"Topics by PC1 position  (top/bottom {FE_TOP_N})")
    _make_panel(ax2, topic_positions, "mean_pc2",
                f"Topics by PC2 position  (top/bottom {FE_TOP_N})")

    # Shared colour legend
    b2b_patch = mpatches.Patch(color=PAL["b2b"], alpha=0.8,
                                label="Client-dominant topic")
    b2w_patch = mpatches.Patch(color=PAL["b2w"], alpha=0.8,
                                label="Worker-dominant topic")
    mix_patch = mpatches.Patch(color=PAL["h1c"], alpha=0.8,
                                label="Mixed-audience topic")
    fig.legend(
        handles=[b2b_patch, b2w_patch, mix_patch],
        loc="lower center", ncol=3, fontsize=8.5,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle("Topic Positions on PC1 and PC2", fontsize=12,
                 fontweight="bold", y=1.01)

    _caption(fig,
        "Each bar = one topic. Position = mean PC coordinate across documents "
        "where that topic is dominant. "
        f"Labels show top {FE_TOP_TERMS} terms. "
        "Colour = proportion of client-facing documents."
    )

    plt.tight_layout()
    save_fig(fig, "fe_topic_pc_loadings")


# =============================================================================
# ── FIGURE F — PCA Scatter: Domain Colours + Audience Shapes ─────────────────
# =============================================================================

def fig_f_pca_domain_shapes(docs, domain_stats):
    """
    PCA scatter where:
      - colour  encodes domain  (each platform gets a distinct hue)
      - shape   encodes audience type  (■ client-facing, ● worker-facing)

    Top FF_MAX_DOMAINS domains by page count receive distinct colours;
    remaining domains are rendered in grey.
    """
    log.info("Figure F: PCA scatter — domain colours + audience shapes")

    # Rank domains by page count for colour assignment
    ranked = sorted(domain_stats, key=lambda d: -domain_stats[d]["n_pages"])
    top_domains = ranked[:FF_MAX_DOMAINS]

    # Build a qualitative colour map from tab20 (20 colours)
    tab20 = plt.cm.get_cmap("tab20", FF_MAX_DOMAINS)
    domain_color = {dom: tab20(i) for i, dom in enumerate(top_domains)}

    fig, ax = plt.subplots(figsize=(12, 7))

    # Separate client and worker pages for shape-encoding
    client_docs = [d for d in docs if (d["audience"] or "client") == "client"]
    worker_docs = [d for d in docs if (d["audience"] or "client") != "client"]

    # Plot client pages as squares (marker="s")
    for doc in client_docs:
        dom   = doc["domain"]
        color = domain_color.get(dom, PAL["neutral"])
        alpha = FF_DOT_ALPHA if dom in top_domains else 0.12
        ax.scatter(
            doc["pca_1"], doc.get("pca_2") or 0.0,
            s=FF_DOT_SIZE, color=color, alpha=alpha,
            marker="s", edgecolors="none", zorder=2,
        )

    # Plot worker pages as circles (marker="o")
    for doc in worker_docs:
        dom   = doc["domain"]
        color = domain_color.get(dom, PAL["neutral"])
        alpha = FF_DOT_ALPHA if dom in top_domains else 0.12
        ax.scatter(
            doc["pca_1"], doc.get("pca_2") or 0.0,
            s=FF_DOT_SIZE, color=color, alpha=alpha,
            marker="o", edgecolors="none", zorder=2,
        )

    # ── Legend: domain colours ────────────────────────────────────────────
    dom_handles = []
    for dom in top_domains:
        col = domain_color[dom]
        aud = domain_stats[dom]["audience"] or "client"
        mkr = "s" if aud == "client" else "o"
        dom_handles.append(
            mlines.Line2D([], [], marker=mkr, linestyle="none",
                          color=col, markersize=6, label=dom)
        )

    # Shape legend entries
    shape_client = mlines.Line2D([], [], marker="s", linestyle="none",
                                  color=PAL["sub"], markersize=7,
                                  label="Client-facing  (■)")
    shape_worker = mlines.Line2D([], [], marker="o", linestyle="none",
                                  color=PAL["sub"], markersize=7,
                                  label="Worker-facing  (●)")

    # Two-part legend: shapes first, then domains
    ax.legend(
        handles=[shape_client, shape_worker] + dom_handles,
        loc="upper left", fontsize=7, ncol=2,
        title="Audience / Domain", title_fontsize=8,
        framealpha=0.9,
    )

    ax.set_xlabel("PC1", fontsize=9.5)
    ax.set_ylabel("PC2", fontsize=9.5)
    ax.set_title("PCA Document Space — Domain and Audience", pad=10)

    _caption(fig,
        f"Each dot = one page (n={len(docs):,}). "
        "Colour = domain. Shape = audience type  (■ client-facing, ● worker-facing). "
        f"Top {FF_MAX_DOMAINS} domains shown in colour; remaining pages in grey."
    )

    save_fig(fig, "ff_pca_domain_shapes")


# =============================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    apply_theme()
    conn = get_conn()

    log.info("Loading document_topics …")
    docs         = load_document_topics(conn)
    domain_stats = compute_domain_stats(docs)
    log.info(f"  {len(docs):,} pages across {len(domain_stats)} domains")

    if FIGURES["fa_domain_landscape"]:
        fig_a_domain_landscape(domain_stats)

    if FIGURES["fb_pca_scatter"]:
        fig_b_pca_scatter(docs, domain_stats)

    if FIGURES["fc_within_domain_var"]:
        fig_c_within_domain_variation(domain_stats)

    if FIGURES["fd_step2_sample_map"]:
        if conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='step2_sample'"
        ).fetchone():
            sample = load_step2_sample(conn)
            log.info(f"  {len(sample)} pages in step2_sample")
            fig_d_step2_sample_map(docs, sample)
        else:
            log.warning("step2_sample table not found — skipping Figure D")

    if FIGURES["fe_topic_pc_loadings"]:
        topic_positions = load_topic_pc_positions(conn)
        log.info(f"  {len(topic_positions)} topics with PC positions")
        fig_e_topic_pc_loadings(topic_positions)

    if FIGURES["ff_pca_domain_shapes"]:
        fig_f_pca_domain_shapes(docs, domain_stats)

    conn.close()
    log.info("Done.")

if __name__ == "__main__":
    main()
