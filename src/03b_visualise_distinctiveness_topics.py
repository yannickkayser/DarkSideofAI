"""
03b_visualise_distinctiveness_topics.py
========================================
Produces visualisations for the Step 1 extension analyses: cross-domain
distinctiveness (02b) and LDA topic modelling / Step 2 sampling (02c).

Pipeline position:
  Stage 3 — Visualisation (Part B)
  Reads from: distinctiveness_matrix, term_exclusivity,
               document_topics, topic_audience_profile, topic_terms,
               cooccurrence_results, step2_sample
  Produces:   output/step_1/fig7_* … fig12_*  (pub and exp variants)
  Follows:    03_visualise_step1.py  (Figures 1–6)
  Before:     04_step2_export.py  (Step 2 close reading corpus)

What this script does:
  Generates Figures 7–12 of the Step 1 analysis, covering the outputs of
  02b_step1_distinctiveness.py (JSD distinctiveness matrix, term exclusivity)
  and 02c_step1_topics.py (LDA topic model, PCA document space, collocate
  divergence, Step 2 sampling rationale).

  Like 03_visualise_step1.py, every figure is produced in two styles:
    pub  — clean, thesis-ready (white background, minimal annotation)
    exp  — exploratory/analytical (tinted background, value labels, centroids)
  This dual-output strategy means the same underlying data can be used both
  for the thesis figures (pub) and for iterative analytical work (exp).

Figures produced:
  7.  fig7_distinctiveness_heatmap   — n×n domain JSD matrix
  8.  fig8_exclusivity_volcano       — term exclusivity index vs log10(freq)
  9.  fig9_pca_scatter               — document-topic PCA by audience
  10. fig10_topic_audience_profile   — per-topic B2B/B2W diverging bar chart
  11. fig11_collocate_divergence     — PMI profile divergence ranking
  12. fig12_step2_sample_map         — PCA with Step 2 sample highlighted

Prerequisites (database tables):
  - distinctiveness_matrix   (from 02b: JSD per domain pair)
  - term_exclusivity         (from 02b: exclusivity index per term)
  - document_topics          (from 02c: per-page topic weights + PCA coords)
  - topic_audience_profile   (from 02c: per-topic B2B/B2W share)
  - topic_terms              (from 02c: top terms per topic)
  - cooccurrence_results     (from 02_step1_frequency: PMI co-occurrence)
  - step2_sample             (from 02c: hypothesis-stratified sample)

  If a prerequisite table is missing, the corresponding figure is skipped
  with a WARNING log message — it does not abort the run.

Output:
  output/step_1/fig7_*_{pub,exp}.jpg  through  fig12_*_{pub,exp}.jpg
  12 .jpg files total (6 figures × 2 styles).
  JPEG at 150 dpi; filenames follow the fig<N>_<slug>_<style>.jpg pattern
  established in 03_visualise_step1.py for consistent thesis figure numbering.

Key design decisions:
  - Missing-table resilience: each figure function checks whether its tables
    exist before querying; partial runs (e.g. if 02c has not yet been run)
    produce whatever figures are possible.
  - Colour palette: C_CLIENT (#1B4F8A blue) / C_WORKER (#C0392B red) /
    C_SHARED (#6C757D grey) — identical to 03_visualise_step1.py so all
    figures share a consistent visual language.
  - HEATMAP_MAX_DOMAINS=40 cap: prevents the heatmap from becoming
    illegible when the corpus has many domains; logged as WARNING if hit.
  - fig7 95th-percentile colour ceiling: same rationale as fig5 in 03_ —
    prevents outlier pairs from compressing all other variation to near-zero.
  - fig8 VOLCANO_FREQ_FLOOR=20: low-frequency terms are noisy exclusivity
    estimates; the floor removes them from the scatter without removing them
    from other analyses.
  - fig11 recomputes collocate divergence from cooccurrence_results rather
    than reading from a stored column, so the figure is always consistent
    with the underlying PMI data regardless of how 02c was run.
  - fig12 overlays faded background points with vivid sample points using
    zorder=5, making the sampling coverage immediately legible.

Thesis note:
  Figures 7–12 appear in the Step 1 Results section and the appendices.
  fig9 and fig12 together constitute the core PCA narrative: fig9 shows
  that audience separates the document space; fig12 shows how the Step 2
  sample was selected to cover both the shared region (theoretically rich)
  and the audience-polar regions (for contrast).

Usage:
    python3 src/03b_visualise_distinctiveness_topics.py
"""

import sqlite3
import json
import math
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH    = "data/scraping.db"
OUTPUT_DIR = Path("output/step_1/")

# Figure-specific settings
HEATMAP_MAX_DOMAINS = 40        # cap for readability; log WARNING if exceeded
VOLCANO_LABEL_N     = 20        # total terms to label across all three categories
VOLCANO_FREQ_FLOOR  = 20        # minimum total freq to appear in volcano plot
PCA_ALPHA           = 0.55      # dot transparency for PCA scatter (overlap clarity)
PCA_SIZE            = 18        # dot size for PCA scatter
TOPIC_BAR_TOP_TERMS = 5         # terms shown alongside each topic bar (label)
DIVERGENCE_TOP_N    = 30        # number of focus terms to rank in fig11
SAMPLE_MARKER_SIZE  = 70        # size of Step 2 sample dots in fig12

# ---------------------------------------------------------------------------
# Colour palette — matches 03_visualise_step1.py exactly
# ---------------------------------------------------------------------------
C_CLIENT  = "#1B4F8A"   # deep blue — B2B / client-side
C_WORKER  = "#C0392B"   # deep red  — B2W / worker-side
C_SHARED  = "#6C757D"   # grey      — shared / neutral
C_BG_PUB  = "#FFFFFF"   # pure white — pub style background
C_BG_EXP  = "#F7F9FC"   # off-white  — exp style background
C_GRID    = "#DEE2E6"   # light grey — axis and grid lines
C_TEXT    = "#1A1A2E"   # near-black — primary text
C_SUBTEXT = "#6C757D"   # medium grey — secondary text / annotations
C_ACCENT  = "#E67E22"   # orange — used for Step 2 sample highlights (fig12)
                        #          and collocate divergence colour gradient (fig11)

FONT_TITLE = {"fontsize": 13, "fontweight": "bold",   "color": C_TEXT}
FONT_LABEL = {"fontsize": 10, "fontweight": "normal", "color": C_SUBTEXT}
FONT_ANNOT = {"fontsize":  8, "color": C_SUBTEXT}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    """
    Open and return a SQLite connection to DB_PATH.

    Uses sqlite3.Row so all result rows support both positional and
    key-based column access (e.g. row['domain_a'] or row[0]).

    Returns:
        sqlite3.Connection with row_factory = sqlite3.Row.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def require_tables(conn, tables):
    """
    Raise RuntimeError if any of the specified tables are missing.

    Used at the start of figure functions whose queries depend on a
    specific table.  Provides a clear diagnostic message rather than
    an opaque SQL error.

    Args:
        conn   : Open sqlite3.Connection.
        tables : Iterable of table name strings to verify.

    Raises:
        RuntimeError: If any table is absent from sqlite_master.
    """
    for table in tables:
        if not conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
        ).fetchone():
            raise RuntimeError(
                f"Table '{table}' not found — run the prerequisite script first.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def apply_base_style(ax, bg):
    """
    Apply the project's standard minimal axis style.

    Removes top/right spines, colours remaining spines C_GRID, sets
    tick colour and font size, and moves grid lines behind data (axisbelow).

    Args:
        ax : matplotlib Axes to style in-place.
        bg : Background colour string (C_BG_PUB or C_BG_EXP).
    """
    ax.set_facecolor(bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_axisbelow(True)


def save(fig, name, style):
    """
    Save a matplotlib figure to OUTPUT_DIR/{name}_{style}.jpg and close it.

    Creates OUTPUT_DIR (and parents) if it does not exist.
    Uses JPEG at 150 dpi with tight bounding box.  Format matches the
    convention established in 03_visualise_step1.py.

    Args:
        fig   : matplotlib Figure to save.
        name  : Filename stem, e.g. "fig7_distinctiveness_heatmap".
        style : "pub" or "exp" — appended before the extension.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_{style}.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="jpeg")
    log.info(f"  Saved: {path}")
    plt.close(fig)


def shorten_domain(d):
    """
    Strip common TLD suffixes and www. prefix for compact axis labels.

    Examples:
        "appen.com"  → "appen"
        "www.telusinternational.ai" → "telusinternational"
        "crowdgen.com.au" → still shortened to "crowdgen" after .com removal

    Args:
        d : Domain string.

    Returns:
        Shortened string suitable for tick labels on heatmap axes.
    """
    for ext in [".com", ".ai", ".net", ".org", ".tech", ".me"]:
        d = d.replace(ext, "")
    return d.replace("www.", "")


# ---------------------------------------------------------------------------
# Figure 7: Domain distinctiveness heatmap (JSD)
# ---------------------------------------------------------------------------

def fig_distinctiveness_heatmap(conn, style):
    """
    Produce Figure 7: cross-domain linguistic distinctiveness heatmap.

    Data source: distinctiveness_matrix table (written by 02b_step1_distinctiveness.py).
    Each cell shows the Jensen-Shannon Divergence (JSD) between a pair of
    domains, computed on the high-variance vocabulary (above-median cross-domain
    variance filter applied in 02b to remove uninformative common terms).

    Layout:
      - Domains sorted client-first then worker, producing a visual
        block structure where the upper-left quadrant = intra-client
        distances, lower-right = intra-worker, and off-diagonal blocks
        = cross-audience distances.
      - A dashed boundary line separates the two audience blocks.
      - Axis tick labels are coloured C_CLIENT (blue) for client domains,
        C_WORKER (red) for worker domains.
      - Colour scale: "Blues" (pub) / "YlOrRd" (exp).
        Ceiling at 95th percentile of non-zero cells — prevents one
        highly-distinctive outlier pair from compressing all other variation
        to near-zero in the colour map.
      - exp style only, and only when n ≤ 25: cell value annotations
        (white text on dark cells, C_TEXT on light cells).

    Output: fig7_distinctiveness_heatmap_{pub,exp}.jpg

    Args:
        conn  : Open sqlite3.Connection.
        style : "pub" or "exp".
    """
    log.info(f"Figure 7 — Distinctiveness heatmap ({style})")

    rows = conn.execute("""
        SELECT domain_a, audience_a, domain_b, audience_b, jsd
        FROM distinctiveness_matrix
    """).fetchall()

    if not rows:
        log.warning("  No data in distinctiveness_matrix — skipping.")
        return

    # Build domain list and audience map
    domains_set = set()
    audiences   = {}
    for r in rows:
        domains_set.add(r["domain_a"])
        domains_set.add(r["domain_b"])
        audiences[r["domain_a"]] = r["audience_a"]
        audiences[r["domain_b"]] = r["audience_b"]

    # Sort: client domains first, then both (paired), then worker
    clients = sorted([d for d, a in audiences.items() if a == "client"])
    workers = sorted([d for d, a in audiences.items() if a == "worker"])
    both    = sorted([d for d, a in audiences.items() if a not in ("client", "worker")])
    domains = clients + both + workers

    if len(domains) > HEATMAP_MAX_DOMAINS:
        log.warning(f"  {len(domains)} domains — capping at {HEATMAP_MAX_DOMAINS}")
        domains = domains[:HEATMAP_MAX_DOMAINS]

    n = len(domains)
    d2i = {d: i for i, d in enumerate(domains)}

    # Fill symmetric matrix (distinctiveness_matrix stores each pair once)
    matrix = np.zeros((n, n))
    for r in rows:
        i = d2i.get(r["domain_a"])
        j = d2i.get(r["domain_b"])
        if i is not None and j is not None:
            matrix[i, j] = r["jsd"]
            matrix[j, i] = r["jsd"]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(9, n * 0.50)),
                           facecolor=bg)
    ax.set_facecolor(bg)

    cmap = "YlOrRd" if style == "exp" else "Blues"
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=np.percentile(matrix[matrix > 0], 95)
                   if (matrix > 0).any() else 1)

    short_labels = [shorten_domain(d) for d in domains]
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(short_labels, fontsize=8)

    # Colour axis labels by audience type
    n_client = len(clients) + len(both)
    for i, domain in enumerate(domains):
        aud = audiences.get(domain, "both")
        col = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SUBTEXT
        ax.get_xticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_color(col)

    # Dashed line separating client and worker audience blocks
    if clients and workers:
        boundary = len(clients) + len(both) - 0.5
        ax.axhline(boundary, color=C_TEXT, linewidth=1.2, linestyle="--")
        ax.axvline(boundary, color=C_TEXT, linewidth=1.2, linestyle="--")

    # Cell value annotations in exp mode (only when matrix is small enough to read)
    if style == "exp" and n <= 25:
        vmax = im.get_clim()[1]
        for i in range(n):
            for j in range(n):
                if i != j:
                    val = matrix[i, j]
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=5, color="white" if val > vmax * 0.6 else C_TEXT)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Jensen-Shannon Divergence (normalised)", fontsize=8, color=C_SUBTEXT)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Cross-Domain Linguistic Distinctiveness (High-Variance Terms)",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "JSD computed on high-variance terms only (above-median cross-domain variance)  •  "
             "Blue labels = B2B  •  Red labels = B2W",
             ha="center", **FONT_ANNOT)
    save(fig, "fig7_distinctiveness_heatmap", style)


# ---------------------------------------------------------------------------
# Figure 8: Term exclusivity volcano plot
# ---------------------------------------------------------------------------

def fig_exclusivity_volcano(conn, style):
    """
    Produce Figure 8: term exclusivity index vs log10(total frequency).

    Data source: term_exclusivity table (written by 02b_step1_distinctiveness.py).
    Exclusivity index = prevalence_client − prevalence_worker ∈ [−1, +1].
    Only unigrams with total_freq ≥ VOLCANO_FREQ_FLOOR are plotted to avoid
    noisy low-frequency estimates dominating the periphery.

    Layout:
      - X-axis: exclusivity index (left = worker-exclusive, right = client).
      - Y-axis: log10(total corpus frequency) — high-freq terms near the top.
      - Shaded grey band at x ∈ [−0.25, 0.25]: the "shared" zone where
        terms appear in both registers at similar rates.  These terms are
        theoretically important for rhetorical analysis because the SAME word
        carries different connotations in client vs worker contexts.
      - Top-N terms labelled per category: client_exclusive, worker_exclusive,
        shared.  Labels use arrows in exp style for legibility.
      - Five colour levels: client_exclusive → leaning_client → shared →
        leaning_worker → worker_exclusive.

    Interpretation guide:
      Upper-right  = high-frequency client-exclusive terms (core B2B vocabulary)
      Upper-left   = high-frequency worker-exclusive terms (core B2W vocabulary)
      Upper-centre = high-frequency shared terms (ambiguous framing candidates)
      Lower periphery = low-frequency domain-specific jargon

    Output: fig8_exclusivity_volcano_{pub,exp}.jpg

    Args:
        conn  : Open sqlite3.Connection.
        style : "pub" or "exp".
    """
    log.info(f"Figure 8 — Exclusivity volcano plot ({style})")

    rows = conn.execute(f"""
        SELECT term, term_type, exclusivity_index, category, total_freq
        FROM term_exclusivity
        WHERE total_freq >= {VOLCANO_FREQ_FLOOR}
          AND term_type = 'unigram'
    """).fetchall()

    if not rows:
        log.warning("  No data in term_exclusivity — skipping.")
        return

    terms  = [r["term"] for r in rows]
    x_vals = [r["exclusivity_index"] for r in rows]
    y_vals = [math.log10(r["total_freq"]) for r in rows]
    cats   = [r["category"] for r in rows]

    cat_colors = {
        "client_exclusive": C_CLIENT,
        "leaning_client":   "#5B9BD5",
        "shared":           C_SHARED,
        "leaning_worker":   "#E88B8B",
        "worker_exclusive": C_WORKER,
    }
    colors = [cat_colors.get(c, C_SHARED) for c in cats]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=bg)
    ax.set_facecolor(bg)

    ax.scatter(x_vals, y_vals, c=colors, s=14, alpha=0.6,
               edgecolors="white", linewidths=0.3)

    # Grey band for the "shared" zone defined by SHARED_BAND in 02b
    ax.axvspan(-0.25, 0.25, color=C_GRID, alpha=0.25, zorder=0)
    ax.axvline(0, color=C_GRID, linewidth=0.8, linestyle="--")

    ax.set_xlabel("Exclusivity Index  (← Worker-exclusive | Shared | Client-exclusive →)",
                  **FONT_LABEL)
    ax.set_ylabel("log₁₀(Total Corpus Frequency)", **FONT_LABEL)
    ax.set_xlim(-1.1, 1.1)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)

    # Label the most interesting terms: top-N by frequency in each tail + shared centre
    labelled = set()

    def label_top(subset_indices, n, ha_default="center"):
        """Label the highest-frequency terms in a subset."""
        sorted_idx = sorted(subset_indices, key=lambda i: y_vals[i], reverse=True)
        for idx in sorted_idx[:n]:
            if terms[idx] in labelled:
                continue
            labelled.add(terms[idx])
            ha = "right" if x_vals[idx] < -0.3 else "left" if x_vals[idx] > 0.3 else "center"
            ax.annotate(terms[idx],
                        (x_vals[idx], y_vals[idx]),
                        fontsize=6.5 if style == "pub" else 7,
                        color=C_TEXT,
                        textcoords="offset points",
                        xytext=(4, 4), ha=ha,
                        arrowprops=dict(arrowstyle="-", color=C_GRID,
                                        linewidth=0.4)
                        if style == "exp" else None)

    client_excl = [i for i, c in enumerate(cats) if c == "client_exclusive"]
    worker_excl = [i for i, c in enumerate(cats) if c == "worker_exclusive"]
    shared_idx  = [i for i, c in enumerate(cats) if c == "shared"]

    n_per = VOLCANO_LABEL_N // 3
    label_top(client_excl, n_per)
    label_top(worker_excl, n_per)
    label_top(shared_idx, n_per)

    # Legend
    legend_entries = [
        mpatches.Patch(color=C_CLIENT,   label="Client-exclusive"),
        mpatches.Patch(color="#5B9BD5",   label="Leaning client"),
        mpatches.Patch(color=C_SHARED,    label="Shared"),
        mpatches.Patch(color="#E88B8B",   label="Leaning worker"),
        mpatches.Patch(color=C_WORKER,    label="Worker-exclusive"),
    ]
    ax.legend(handles=legend_entries, loc="upper right", frameon=True,
              fontsize=8, facecolor=bg, edgecolor=C_GRID)

    ax.set_title("Term Exclusivity: Platform Prevalence by Audience Register",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "Exclusivity = prevalence_client − prevalence_worker  •  "
             "Grey band = shared terms (|E| ≤ 0.25)  •  "
             "Shared high-frequency terms are prime Step 2 candidates",
             ha="center", **FONT_ANNOT)
    save(fig, "fig8_exclusivity_volcano", style)


# ---------------------------------------------------------------------------
# Figure 9: PCA scatter — document-topic space
# ---------------------------------------------------------------------------

def fig_pca_scatter(conn, style):
    """
    Produce Figure 9: PCA projection of document-topic space, coloured by audience.

    Data source: document_topics table (written by 02c_step1_topics.py).
    Each row is one page; pca_1 and pca_2 are the first two principal
    components of the N_TOPICS-dimensional document-topic weight matrix
    computed by LDA.

    Interpretation:
      If audience is a dominant structural axis in the corpus, client and
      worker pages should cluster separately along PC1 (the maximum-variance
      direction).  The degree of separation validates the Step 1 hypothesis
      that B2B and B2W platforms occupy meaningfully different topic spaces.

    exp style adds:
      - Centroid markers (X) for each audience cluster.
      - Centroid labels with audience name.

    Output: fig9_pca_scatter_{pub,exp}.jpg

    Args:
        conn  : Open sqlite3.Connection.
        style : "pub" or "exp".
    """
    log.info(f"Figure 9 — PCA scatter ({style})")

    rows = conn.execute("""
        SELECT page_id, domain, audience, dominant_topic, pca_1, pca_2
        FROM document_topics
    """).fetchall()

    if not rows:
        log.warning("  No data in document_topics — skipping.")
        return

    x_vals = [r["pca_1"] for r in rows]
    y_vals = [r["pca_2"] for r in rows]
    auds   = [r["audience"] for r in rows]
    colors = [C_CLIENT if a == "client" else C_WORKER for a in auds]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=bg)
    ax.set_facecolor(bg)

    ax.scatter(x_vals, y_vals, c=colors, s=PCA_SIZE, alpha=PCA_ALPHA,
               edgecolors="white", linewidths=0.2)

    ax.set_xlabel("PC 1", **FONT_LABEL)
    ax.set_ylabel("PC 2", **FONT_LABEL)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)

    if style == "exp":
        # Mark and label audience centroids
        for aud, col, label in [("client", C_CLIENT, "B2B centroid"),
                                ("worker", C_WORKER, "B2W centroid")]:
            xs = [x_vals[i] for i, a in enumerate(auds) if a == aud]
            ys = [y_vals[i] for i, a in enumerate(auds) if a == aud]
            if xs:
                cx, cy = np.mean(xs), np.mean(ys)
                ax.plot(cx, cy, "X", color=col, markersize=14,
                        markeredgecolor="white", markeredgewidth=1.5)
                ax.annotate(label, (cx, cy), fontsize=8, color=col,
                            fontweight="bold",
                            textcoords="offset points", xytext=(10, 10))

    # Legend
    client_patch = mpatches.Patch(color=C_CLIENT, alpha=0.7, label="Client (B2B)")
    worker_patch = mpatches.Patch(color=C_WORKER, alpha=0.7, label="Worker (B2W)")
    ax.legend(handles=[client_patch, worker_patch], loc="upper right",
              frameon=True, fontsize=9, facecolor=bg, edgecolor=C_GRID)

    ax.set_title("Document-Topic Space: PCA Projection Coloured by Audience",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "Each dot = one page  •  Colour = audience register  •  "
             "Separation along PC1 indicates audience is a dominant structural axis",
             ha="center", **FONT_ANNOT)
    save(fig, "fig9_pca_scatter", style)


# ---------------------------------------------------------------------------
# Figure 10: Topic audience profile — diverging horizontal bars
# ---------------------------------------------------------------------------

def fig_topic_audience_profile(conn, style):
    """
    Produce Figure 10: per-topic B2B/B2W balance as a diverging bar chart.

    Data sources:
      - topic_audience_profile (from 02c): client_share per topic, category
        (client_leaning / shared / worker_leaning), dominant page counts.
      - topic_terms (from 02c): top terms per topic for labelling.

    Layout:
      - Bar length = client_share − 0.5.  Bars to the right = client-leaning
        topics; bars to the left = worker-leaning; zero = balanced (shared).
      - Grey shaded band at [−0.15, 0.15] marks the shared zone
        (threshold 0.65 used in 02c for client_share to qualify as "leaning").
      - Topics sorted from most client-leaning (top) to most worker-leaning
        (bottom), so reading the chart top-to-bottom traverses the full
        rhetorical spectrum of the corpus.
      - Each bar is labelled with the topic's top-5 terms, providing
        immediate interpretive context without needing to cross-reference
        the LDA output separately.
      - exp style adds client_share value annotations on bar ends.

    Output: fig10_topic_audience_profile_{pub,exp}.jpg

    Args:
        conn  : Open sqlite3.Connection.
        style : "pub" or "exp".
    """
    log.info(f"Figure 10 — Topic audience profile ({style})")

    profiles = conn.execute("""
        SELECT topic_id, category, client_share,
               avg_weight_client, avg_weight_worker,
               n_dominant_client, n_dominant_worker
        FROM topic_audience_profile
        ORDER BY client_share DESC
    """).fetchall()

    if not profiles:
        log.warning("  No data in topic_audience_profile — skipping.")
        return

    # Fetch top terms per topic for axis labels
    topic_terms = {}
    term_rows = conn.execute(f"""
        SELECT topic_id, term
        FROM topic_terms
        WHERE rank <= {TOPIC_BAR_TOP_TERMS}
        ORDER BY topic_id, rank
    """).fetchall()
    for r in term_rows:
        topic_terms.setdefault(r["topic_id"], []).append(r["term"])

    topic_ids     = [r["topic_id"] for r in profiles]
    client_shares = [r["client_share"] for r in profiles]
    categories    = [r["category"] for r in profiles]

    # Bar length = deviation from balanced 0.5 split
    divergences = [cs - 0.5 for cs in client_shares]

    cat_colors = {
        "client_leaning": C_CLIENT,
        "shared":         C_SHARED,
        "worker_leaning": C_WORKER,
    }
    colors = [cat_colors.get(c, C_SHARED) for c in categories]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    n_topics = len(topic_ids)
    fig, ax = plt.subplots(figsize=(14, max(8, n_topics * 0.4)), facecolor=bg)
    ax.set_facecolor(bg)

    y_pos = np.arange(n_topics)
    bars = ax.barh(y_pos, divergences, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.4, height=0.7)

    # Y-axis: topic id + top terms for interpretive context
    y_labels = []
    for tid in topic_ids:
        terms = topic_terms.get(tid, [])
        term_str = ", ".join(terms[:TOPIC_BAR_TOP_TERMS])
        y_labels.append(f"T{tid}: {term_str}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7.5, color=C_TEXT)
    ax.axvline(0, color=C_TEXT, linewidth=0.8)
    ax.set_xlabel("← Worker-leaning  |  Client-leaning →", **FONT_LABEL)
    ax.set_xlim(-0.55, 0.55)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)

    # Shade the shared zone
    ax.axvspan(-0.15, 0.15, color=C_GRID, alpha=0.2, zorder=0)

    if style == "exp":
        # Annotate each bar with its client_share value
        for bar, prof in zip(bars, profiles):
            w = bar.get_width()
            offset = 0.01 if w >= 0 else -0.01
            ha = "left" if w >= 0 else "right"
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2,
                    f"{prof['client_share']:.2f}",
                    va="center", ha=ha, fontsize=6.5, color=C_SUBTEXT)

    # Legend
    legend_entries = [
        mpatches.Patch(color=C_CLIENT, label="Client-leaning"),
        mpatches.Patch(color=C_SHARED, label="Shared"),
        mpatches.Patch(color=C_WORKER, label="Worker-leaning"),
    ]
    ax.legend(handles=legend_entries, loc="lower right", frameon=True,
              fontsize=8, facecolor=bg, edgecolor=C_GRID)

    ax.set_title("Topic Audience Profile: LDA Topics by B2B/B2W Balance",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "Bar = client_share − 0.5  •  Grey band = shared zone  •  "
             "Topics ranked from most client-leaning (top) to most worker-leaning (bottom)",
             ha="center", **FONT_ANNOT)
    save(fig, "fig10_topic_audience_profile", style)


# ---------------------------------------------------------------------------
# Figure 11: Collocate divergence ranking
# ---------------------------------------------------------------------------

def fig_collocate_divergence(conn, style):
    """
    Produce Figure 11: collocate profile divergence ranking for THEORY_FOCUS_TERMS.

    Data source: cooccurrence_results table (written by 02_step1_frequency.py).
    Divergence is recomputed here from raw PMI values rather than read from
    a stored column, ensuring the figure is always consistent with the
    underlying co-occurrence data.

    Divergence formula (same as 02c_step1_topics.py compute_collocate_divergence):
        divergence(term) = 1 − cosine_similarity(PMI_client_vector, PMI_worker_vector)

    where each PMI vector is built over the union of all collocates across
    both audiences.  Cosine similarity approaches 1 when the two audiences
    use the term in similar collocate environments; approaches 0 when they
    do not overlap at all.

    Interpretation:
      High divergence = the term's discursive neighbourhood differs most
      between B2B and B2W.  These are the terms carrying the richest
      analytical signal for Step 2 close reading, because the same word
      is being deployed in fundamentally different ways.

    Layout:
      - Horizontal bars ranked from lowest (bottom) to highest (top) divergence.
      - Gradient colour: grey (low divergence) → orange (mid) → red (high).
      - Shows top DIVERGENCE_TOP_N=30 focus terms.
      - exp style adds numeric labels on bar ends.

    Output: fig11_collocate_divergence_{pub,exp}.jpg

    Args:
        conn  : Open sqlite3.Connection.
        style : "pub" or "exp".
    """
    log.info(f"Figure 11 — Collocate divergence ({style})")

    # Check for prerequisite table
    table_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='cooccurrence_results'"
    ).fetchone()
    if not table_check:
        log.warning("  cooccurrence_results not found — skipping fig11.")
        return

    rows = conn.execute("""
        SELECT focus_term, audience, collocate, pmi
        FROM cooccurrence_results
        WHERE comparison = 'cross_platform'
    """).fetchall()

    if not rows:
        log.warning("  No cross_platform co-occurrence data — skipping fig11.")
        return

    # Build PMI vectors per (focus_term, audience)
    profiles = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        profiles[r["focus_term"]][r["audience"]][r["collocate"]] = r["pmi"]

    # Compute 1 − cosine_similarity for each focus term
    divergences = {}
    for term, aud_data in profiles.items():
        if "client" not in aud_data or "worker" not in aud_data:
            continue
        c_vec = aud_data["client"]
        w_vec = aud_data["worker"]
        all_collocates = set(c_vec) | set(w_vec)
        if len(all_collocates) < 3:
            continue   # too few collocates for meaningful cosine

        dot  = sum(c_vec.get(c, 0) * w_vec.get(c, 0) for c in all_collocates)
        mag_c = math.sqrt(sum(c_vec.get(c, 0) ** 2 for c in all_collocates))
        mag_w = math.sqrt(sum(w_vec.get(c, 0) ** 2 for c in all_collocates))
        cos_sim = dot / (mag_c * mag_w) if mag_c > 0 and mag_w > 0 else 0
        divergences[term] = round(1.0 - cos_sim, 6)

    if not divergences:
        log.warning("  No divergence scores computed — skipping fig11.")
        return

    # Sort and take top N, then reverse so highest is at the top of the chart
    ranked = sorted(divergences.items(), key=lambda x: x[1], reverse=True)
    ranked = ranked[:DIVERGENCE_TOP_N]
    ranked.reverse()   # lowest divergence at bottom, highest at top of bar chart

    terms = [r[0] for r in ranked]
    divs  = [r[1] for r in ranked]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(12, max(6, len(terms) * 0.35)), facecolor=bg)
    ax.set_facecolor(bg)

    y_pos = np.arange(len(terms))

    # Colour gradient: C_GRID (grey) → C_ACCENT (orange) → C_WORKER (red)
    # This emphasises the high-divergence terms that most need Step 2 attention
    norm = plt.Normalize(vmin=min(divs), vmax=max(divs))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "div_cmap", [C_GRID, C_ACCENT, C_WORKER])
    colors = [cmap(norm(d)) for d in divs]

    bars = ax.barh(y_pos, divs, color=colors, alpha=0.88,
                   edgecolor="white", linewidth=0.4, height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms, fontsize=9, color=C_TEXT)
    ax.set_xlabel("Collocate Divergence  (1 − cosine similarity of PMI profiles)",
                  **FONT_LABEL)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)

    if style == "exp":
        # Numeric labels on bar ends
        for bar, d in zip(bars, divs):
            ax.text(bar.get_width() + max(divs) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{d:.3f}", va="center", fontsize=7, color=C_SUBTEXT)

    ax.set_title("Collocate Profile Divergence: Same Term, Different Framing by Audience",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "High divergence = the term's discursive neighbourhood differs most "
             "between B2B and B2W  •  These terms carry the richest analytical signal for Step 2",
             ha="center", **FONT_ANNOT)
    save(fig, "fig11_collocate_divergence", style)


# ---------------------------------------------------------------------------
# Figure 12: Step 2 sample map — PCA with sampled pages highlighted
# ---------------------------------------------------------------------------

def fig_step2_sample_map(conn, style):
    """
    Produce Figure 12: PCA document space with Step 2 sample highlighted.

    Data sources:
      - document_topics (from 02c): pca_1, pca_2 for all pages.
      - step2_sample (from 02c): page_ids selected for Step 2 close reading,
        with priority_rank and dominant_topic.

    This figure complements fig9 (plain PCA scatter) by showing where in the
    document-topic space the Step 2 sample falls.  A well-designed sample
    should:
      1. Include pages from both the shared topic region (centre of the PCA)
         — theoretically rich because the same topics are used by both
         audiences but framed differently.
      2. Include high-contrast pages from each audience-polar region (edges)
         — useful for establishing the outer rhetorical boundaries.
      3. Be balanced across audiences — roughly equal numbers of B2B and B2W.

    Layout:
      - Background: all non-sampled pages, faded (alpha=0.15, small dots).
      - Foreground: sampled pages, vivid (alpha=0.85, larger dots) with
        orange accent borders (C_ACCENT) to make them immediately identifiable.
      - zorder=5 ensures sampled points are always drawn on top.
      - exp style: top 12 sampled pages annotated with
        #priority_rank and dominant topic number.

    Output: fig12_step2_sample_map_{pub,exp}.jpg

    Args:
        conn  : Open sqlite3.Connection.
        style : "pub" or "exp".
    """
    log.info(f"Figure 12 — Step 2 sample map ({style})")

    # All documents for background scatter
    all_docs = conn.execute("""
        SELECT page_id, audience, pca_1, pca_2, dominant_topic
        FROM document_topics
    """).fetchall()

    if not all_docs:
        log.warning("  No data in document_topics — skipping fig12.")
        return

    # Step 2 sampled pages
    sample_rows = conn.execute("""
        SELECT page_id, audience, dominant_topic, priority_rank,
               topic_weight, collocate_divergence
        FROM step2_sample
        ORDER BY priority_rank
    """).fetchall()

    sample_ids  = {r["page_id"] for r in sample_rows}
    sample_meta = {r["page_id"]: dict(r) for r in sample_rows}

    # Separate background (not sampled) and foreground (sampled) points
    bg_x = [r["pca_1"] for r in all_docs if r["page_id"] not in sample_ids]
    bg_y = [r["pca_2"] for r in all_docs if r["page_id"] not in sample_ids]
    bg_c = [C_CLIENT if r["audience"] == "client" else C_WORKER
            for r in all_docs if r["page_id"] not in sample_ids]

    # Sampled docs need PCA coordinates from document_topics (step2_sample lacks them)
    doc_pca = {r["page_id"]: (r["pca_1"], r["pca_2"]) for r in all_docs}
    s_x, s_y, s_c, s_edge = [], [], [], []
    for pid in sample_ids:
        if pid in doc_pca:
            px, py = doc_pca[pid]
            s_x.append(px)
            s_y.append(py)
            meta = sample_meta.get(pid, {})
            s_c.append(C_CLIENT if meta.get("audience") == "client" else C_WORKER)
            s_edge.append(C_ACCENT)

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(12, 9), facecolor=bg)
    ax.set_facecolor(bg)

    # Layer 1: background — faded, small
    ax.scatter(bg_x, bg_y, c=bg_c, s=10, alpha=0.15,
               edgecolors="none")

    # Layer 2: sampled — vivid, larger, orange border (zorder=5 = on top)
    if s_x:
        ax.scatter(s_x, s_y, c=s_c, s=SAMPLE_MARKER_SIZE, alpha=0.85,
                   edgecolors=C_ACCENT, linewidths=1.5, zorder=5)

    ax.set_xlabel("PC 1", **FONT_LABEL)
    ax.set_ylabel("PC 2", **FONT_LABEL)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)

    # exp mode: annotate the top 12 sampled pages with rank and topic
    if style == "exp" and sample_rows:
        top_sample = [r for r in sample_rows
                      if r["page_id"] in doc_pca][:12]
        for r in top_sample:
            px, py = doc_pca[r["page_id"]]
            ax.annotate(f"#{r['priority_rank']} T{r['dominant_topic']}",
                        (px, py), fontsize=6, color=C_TEXT,
                        textcoords="offset points", xytext=(5, 5),
                        arrowprops=dict(arrowstyle="-", color=C_ACCENT,
                                        linewidth=0.5))

    # Legend with all four point types
    legend_entries = [
        mpatches.Patch(color=C_CLIENT, alpha=0.15, label="B2B (background)"),
        mpatches.Patch(color=C_WORKER, alpha=0.15, label="B2W (background)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_CLIENT,
                   markeredgecolor=C_ACCENT, markersize=9,
                   label="B2B (Step 2 sample)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_WORKER,
                   markeredgecolor=C_ACCENT, markersize=9,
                   label="B2W (Step 2 sample)"),
    ]
    ax.legend(handles=legend_entries, loc="upper right", frameon=True,
              fontsize=8, facecolor=bg, edgecolor=C_GRID)

    n_sample = len(sample_ids)
    ax.set_title("Step 2 Sampling Strategy: Selected Pages in Document-Topic Space",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             f"{n_sample} pages selected  •  "
             "Orange border = sampled for close reading  •  "
             "Selection based on shared-topic exemplarity × collocate divergence",
             ha="center", **FONT_ANNOT)
    save(fig, "fig12_step2_sample_map", style)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Entry point — produces all 12 figures (6 figures × 2 styles).

    Checks for database existence first; then warns for missing prerequisite
    tables (but does not abort — each figure function handles absence
    gracefully by skipping and logging a warning).

    Prerequisites:
      Run in order before this script:
        01_prepare.py
        02_step1_frequency.py    (for cooccurrence_results)
        02b_step1_distinctiveness.py   (for distinctiveness_matrix, term_exclusivity)
        02c_step1_topics.py      (for document_topics, topic_audience_profile,
                                  topic_terms, step2_sample)
    """
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("03b_visualise_distinctiveness_topics.py — Figures 7–12")
    log.info("=" * 60)

    conn = get_conn()

    # Warn (don't abort) for missing prerequisite tables
    for table in ["distinctiveness_matrix", "term_exclusivity",
                  "document_topics", "topic_audience_profile", "step2_sample"]:
        if not conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
        ).fetchone():
            log.warning(f"  Table '{table}' not found — "
                        f"figures depending on it will be skipped.")

    for style in ["pub", "exp"]:
        log.info(f"\n{'='*30} Style: {style.upper()} {'='*30}")
        fig_distinctiveness_heatmap(conn, style)
        fig_exclusivity_volcano(conn, style)
        fig_pca_scatter(conn, style)
        fig_topic_audience_profile(conn, style)
        fig_collocate_divergence(conn, style)
        fig_step2_sample_map(conn, style)

    conn.close()
    log.info("=" * 60)
    log.info(f"All figures saved to {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
