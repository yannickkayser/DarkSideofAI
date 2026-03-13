"""
03b_visualise_distinctiveness_topics.py
=======================================
Produces visualisations of the Step 1 extensions: distinctiveness analysis
(02b) and topic modelling / Step 2 sampling (02c).

Six figures, each in two styles (pub = clean thesis, exp = annotated):
  7.  fig7_distinctiveness_heatmap  — domain×domain JSD matrix
  8.  fig8_exclusivity_volcano      — term exclusivity index vs frequency
  9.  fig9_pca_scatter              — document-topic PCA coloured by audience
 10.  fig10_topic_audience_profile  — per-topic B2B/B2W balance
 11.  fig11_collocate_divergence    — PMI profile divergence ranking
 12.  fig12_step2_sample_map        — PCA with Step 2 sample highlighted

Prerequisites:
  - 02b_step1_distinctiveness.py (distinctiveness_matrix, term_exclusivity)
  - 02c_step1_topics.py (document_topics, topic_audience_profile, step2_sample)

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
HEATMAP_MAX_DOMAINS = 40        # cap for readability
VOLCANO_LABEL_N     = 20        # terms to label in volcano plot
VOLCANO_FREQ_FLOOR  = 20        # minimum total freq to appear in volcano
PCA_ALPHA           = 0.55      # dot transparency for scatter
PCA_SIZE            = 18        # dot size
TOPIC_BAR_TOP_TERMS = 5         # terms shown alongside topic bars
DIVERGENCE_TOP_N    = 30        # focus terms to show in divergence chart
SAMPLE_MARKER_SIZE  = 70        # size of highlighted sample dots

# ---------------------------------------------------------------------------
# Colour palette — matches 03_visualise_step1.py
# ---------------------------------------------------------------------------
C_CLIENT  = "#1B4F8A"
C_WORKER  = "#C0392B"
C_SHARED  = "#6C757D"
C_BG_PUB  = "#FFFFFF"
C_BG_EXP  = "#F7F9FC"
C_GRID    = "#DEE2E6"
C_TEXT    = "#1A1A2E"
C_SUBTEXT = "#6C757D"
C_ACCENT  = "#E67E22"       # for highlights (Step 2 sample markers)

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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def require_tables(conn, tables):
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
    ax.set_facecolor(bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_axisbelow(True)


def save(fig, name, style):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_{style}.jpg"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), format="jpeg")
    log.info(f"  Saved: {path}")
    plt.close(fig)


def shorten_domain(d):
    for ext in [".com", ".ai", ".net", ".org", ".tech", ".me"]:
        d = d.replace(ext, "")
    return d.replace("www.", "")


# ---------------------------------------------------------------------------
# Figure 7: Domain distinctiveness heatmap (JSD)
# ---------------------------------------------------------------------------

def fig_distinctiveness_heatmap(conn, style):
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

    # Sort: client domains first, then worker
    clients = sorted([d for d, a in audiences.items() if a == "client"])
    workers = sorted([d for d, a in audiences.items() if a == "worker"])
    both    = sorted([d for d, a in audiences.items() if a not in ("client", "worker")])
    domains = clients + both + workers

    if len(domains) > HEATMAP_MAX_DOMAINS:
        log.warning(f"  {len(domains)} domains — capping at {HEATMAP_MAX_DOMAINS}")
        domains = domains[:HEATMAP_MAX_DOMAINS]

    n = len(domains)
    d2i = {d: i for i, d in enumerate(domains)}

    # Fill symmetric matrix
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

    # Colour axis labels by audience
    n_client = len(clients) + len(both)
    for i, domain in enumerate(domains):
        aud = audiences.get(domain, "both")
        col = C_CLIENT if aud == "client" else C_WORKER if aud == "worker" else C_SUBTEXT
        ax.get_xticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_color(col)

    # Audience boundary line
    if clients and workers:
        boundary = len(clients) + len(both) - 0.5
        ax.axhline(boundary, color=C_TEXT, linewidth=1.2, linestyle="--")
        ax.axvline(boundary, color=C_TEXT, linewidth=1.2, linestyle="--")

    # Value annotations in exp mode
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

    # Shared band shading
    ax.axvspan(-0.25, 0.25, color=C_GRID, alpha=0.25, zorder=0)
    ax.axvline(0, color=C_GRID, linewidth=0.8, linestyle="--")

    ax.set_xlabel("Exclusivity Index  (← Worker-exclusive | Shared | Client-exclusive →)",
                  **FONT_LABEL)
    ax.set_ylabel("log₁₀(Total Corpus Frequency)", **FONT_LABEL)
    ax.set_xlim(-1.1, 1.1)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)

    # Label the most interesting terms
    # Strategy: label top N by frequency in each tail + shared centre
    labelled = set()

    def label_top(subset_indices, n, ha_default="center"):
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
        # Annotate cluster centres per audience
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

    # Fetch top terms per topic for labels
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

    # Divergence from 0.5 centre
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

    # Y-axis: topic id + top terms
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

    # Shared band
    ax.axvspan(-0.15, 0.15, color=C_GRID, alpha=0.2, zorder=0)

    if style == "exp":
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
    log.info(f"Figure 11 — Collocate divergence ({style})")

    # Recompute divergence from cooccurrence_results (same logic as 02c)
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

    divergences = {}
    for term, aud_data in profiles.items():
        if "client" not in aud_data or "worker" not in aud_data:
            continue
        c_vec = aud_data["client"]
        w_vec = aud_data["worker"]
        all_collocates = set(c_vec) | set(w_vec)
        if len(all_collocates) < 3:
            continue

        dot  = sum(c_vec.get(c, 0) * w_vec.get(c, 0) for c in all_collocates)
        mag_c = math.sqrt(sum(c_vec.get(c, 0) ** 2 for c in all_collocates))
        mag_w = math.sqrt(sum(w_vec.get(c, 0) ** 2 for c in all_collocates))
        cos_sim = dot / (mag_c * mag_w) if mag_c > 0 and mag_w > 0 else 0
        divergences[term] = round(1.0 - cos_sim, 6)

    if not divergences:
        log.warning("  No divergence scores computed — skipping fig11.")
        return

    # Sort and take top N
    ranked = sorted(divergences.items(), key=lambda x: x[1], reverse=True)
    ranked = ranked[:DIVERGENCE_TOP_N]
    ranked.reverse()   # so highest is at top of horizontal bar chart

    terms = [r[0] for r in ranked]
    divs  = [r[1] for r in ranked]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(12, max(6, len(terms) * 0.35)), facecolor=bg)
    ax.set_facecolor(bg)

    y_pos = np.arange(len(terms))

    # Colour gradient: low divergence = grey, high = accent
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
    log.info(f"Figure 12 — Step 2 sample map ({style})")

    # All documents
    all_docs = conn.execute("""
        SELECT page_id, audience, pca_1, pca_2, dominant_topic
        FROM document_topics
    """).fetchall()

    if not all_docs:
        log.warning("  No data in document_topics — skipping fig12.")
        return

    # Sampled pages
    sample_rows = conn.execute("""
        SELECT page_id, audience, dominant_topic, priority_rank,
               topic_weight, collocate_divergence
        FROM step2_sample
        ORDER BY priority_rank
    """).fetchall()

    sample_ids = {r["page_id"] for r in sample_rows}
    sample_meta = {r["page_id"]: dict(r) for r in sample_rows}

    bg_x = [r["pca_1"] for r in all_docs if r["page_id"] not in sample_ids]
    bg_y = [r["pca_2"] for r in all_docs if r["page_id"] not in sample_ids]
    bg_c = [C_CLIENT if r["audience"] == "client" else C_WORKER
            for r in all_docs if r["page_id"] not in sample_ids]

    # Sampled docs — need PCA coords from document_topics
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

    # Background: all non-sampled docs, faded
    ax.scatter(bg_x, bg_y, c=bg_c, s=10, alpha=0.15,
               edgecolors="none")

    # Foreground: sampled docs, vivid with accent border
    if s_x:
        ax.scatter(s_x, s_y, c=s_c, s=SAMPLE_MARKER_SIZE, alpha=0.85,
                   edgecolors=C_ACCENT, linewidths=1.5, zorder=5)

    ax.set_xlabel("PC 1", **FONT_LABEL)
    ax.set_ylabel("PC 2", **FONT_LABEL)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg)

    # Label top-ranked samples in exp mode
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

    # Legend
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
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("03b_visualise_distinctiveness_topics.py — Figures 7–12")
    log.info("=" * 60)

    conn = get_conn()

    # Check prerequisites — warn but don't fail (individual figs handle missing data)
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
