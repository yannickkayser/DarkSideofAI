"""
03_visualise_step1.py
=====================
Produces visualisations of Step 1 keyness and co-occurrence results.

Five figures, each in two styles:
  1. Keyness bar chart        — top client vs worker distinctive terms
  2. Co-occurrence network    — 'human' collocates in client vs worker
  3. Frequency comparison     — key terms side by side across audiences
  4. Within-pair comparison   — appen and toloka keyness side by side
  5. Platform heatmap         — term frequency across all domains

Output: PNG files saved to OUTPUT_DIR
Prerequisites: 02_step1_frequency.py must have been run.

Usage:
    python3 src/03_visualise_step1.py
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
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH    = "data/scraping.db"
OUTPUT_DIR = Path("output/step_1/")
TOP_N      = 20       # terms per direction in keyness charts
MIN_COFREQ = 15       # minimum co-occurrence freq for network edges
NETWORK_N  = 18       # max collocates per side in co-occurrence network

# Terms to exclude — scraping artifacts identified in Step 1 interpretation
ARTIFACT_TERMS = {
    "cookie", "set_cookie", "cooky", "/hr", "/hr_remote", "remote_apply",
    "faq", "faq_help", "youtube", "feb", "opportunity_feb", "shall",
    "help_desk", "desk", "subscribe", "microworker", "enable",
    "website", "account", "access", "zeynep", "koouchnir", "gavrilov",
    "unga", "gary", "yalda", "monarch", "warhol", "fremont", "pittsburgh",
    "experience.with", "rhml", "ead", "cc0", "ft", "mpii",
}

# ---------------------------------------------------------------------------
# Colour palette — consistent across all figures
# ---------------------------------------------------------------------------
C_CLIENT   = "#1B4F8A"   # deep blue  — client/B2B
C_WORKER   = "#C0392B"   # deep red   — worker/B2W
C_NEUTRAL  = "#ECF0F1"
C_BG_PUB   = "#FFFFFF"
C_BG_EXP   = "#F7F9FC"
C_GRID     = "#DEE2E6"
C_TEXT     = "#1A1A2E"
C_SUBTEXT  = "#6C757D"

FONT_TITLE  = {"fontsize": 13, "fontweight": "bold",  "color": C_TEXT,    "fontfamily": "DejaVu Sans"}
FONT_LABEL  = {"fontsize": 10, "fontweight": "normal","color": C_SUBTEXT, "fontfamily": "DejaVu Sans"}
FONT_TICK   = {"fontsize":  9, "color": C_TEXT}
FONT_ANNOT  = {"fontsize":  8, "color": C_SUBTEXT}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_keyness(conn, comparison: str, min_cofreq: int = 0) -> list:
    rows = conn.execute("""
        SELECT term, ll_score, rel_freq_client, rel_freq_worker, term_type,
               freq_client, freq_worker
        FROM keyness_results
        WHERE comparison = ?
          AND term NOT IN ({})
        ORDER BY ll_score DESC
    """.format(",".join("?" * len(ARTIFACT_TERMS))),
    [comparison] + list(ARTIFACT_TERMS)).fetchall()
    return [dict(r) for r in rows]


def fetch_cooccurrence(conn, comparison: str, audience: str,
                       focus: str, min_freq: int = MIN_COFREQ) -> list:
    rows = conn.execute("""
        SELECT collocate, pmi, cofreq
        FROM cooccurrence_results
        WHERE comparison = ? AND audience = ? AND focus_term = ?
          AND cofreq >= ?
          AND collocate NOT IN ({})
        ORDER BY pmi DESC
        LIMIT ?
    """.format(",".join("?" * len(ARTIFACT_TERMS))),
    [comparison, audience, focus, min_freq]
    + list(ARTIFACT_TERMS) + [NETWORK_N]).fetchall()
    return [dict(r) for r in rows]


def fetch_platform_terms(conn, terms: list) -> dict:
    """Return {domain: {term: rel_freq}} for a list of terms."""
    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(f"""
        SELECT domain, audience, term, rel_freq
        FROM platform_term_counts
        WHERE term IN ({placeholders})
          AND term_type = 'unigram'
    """, terms).fetchall()

    result = defaultdict(dict)
    audiences = {}
    for r in rows:
        result[r["domain"]][r["term"]] = r["rel_freq"]
        audiences[r["domain"]] = r["audience"]
    return dict(result), audiences


# ---------------------------------------------------------------------------
# Shared styling helpers
# ---------------------------------------------------------------------------

def apply_base_style(ax, bg: str):
    ax.set_facecolor(bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)


def save(fig, name: str, style: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_{style}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Keyness bar chart
# ---------------------------------------------------------------------------

def fig_keyness_bar(conn, style: str):
    log.info(f"Figure 1 — Keyness bar chart ({style})")
    rows = fetch_keyness(conn, "cross_platform")

    # Separate and take top N each direction, unigrams only
    client_top = [r for r in rows if r["ll_score"] > 0
                  and r["term_type"] == "unigram"][:TOP_N]
    worker_top = [r for r in rows if r["ll_score"] < 0
                  and r["term_type"] == "unigram"][:TOP_N]
    worker_top = list(reversed(worker_top))   # most distinctive at top

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=bg)
    fig.subplots_adjust(wspace=0.45)

    for ax, data, colour, label, direction in [
        (axes[0], client_top, C_CLIENT, "Client-distinctive (B2B)", "client"),
        (axes[1], worker_top, C_WORKER, "Worker-distinctive (B2W)", "worker"),
    ]:
        terms  = [r["term"] for r in data]
        scores = [abs(r["ll_score"]) for r in data]
        y_pos  = np.arange(len(terms))

        bars = ax.barh(y_pos, scores, color=colour,
                       alpha=0.85 if style == "pub" else 0.75,
                       edgecolor="white", linewidth=0.4, height=0.65)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms, fontsize=9, color=C_TEXT)
        ax.set_xlabel("Log-likelihood (G²)", **FONT_LABEL)
        ax.set_title(label, **FONT_TITLE, pad=12)
        apply_base_style(ax, bg)

        if style == "exp":
            # Annotate with relative frequencies
            for i, (bar, row) in enumerate(zip(bars, data)):
                ax.text(bar.get_width() + max(scores) * 0.01, i,
                        f"{row['rel_freq_' + direction]:.2f}‰",
                        va="center", **FONT_ANNOT)

    sup = "Cross-Platform Keyness Analysis: B2B vs B2W Distinctive Terms"
    sub = "Log-likelihood G² statistic  •  Positive = overrepresented in that register  •  Artifact terms excluded"
    fig.suptitle(sup, **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01, sub, ha="center", **FONT_ANNOT)

    save(fig, "fig1_keyness_bar", style)


# ---------------------------------------------------------------------------
# Figure 2: Co-occurrence network for 'human'
# ---------------------------------------------------------------------------

def fig_cooccurrence_network(conn, style: str):
    log.info(f"Figure 2 — Co-occurrence network ({style})")

    client_cooc = fetch_cooccurrence(conn, "cross_platform", "client", "human")
    worker_cooc = fetch_cooccurrence(conn, "cross_platform", "worker", "human")

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=bg)

    for ax, cooc, colour, title in [
        (axes[0], client_cooc, C_CLIENT, "\"human\" in Client texts (B2B)"),
        (axes[1], worker_cooc, C_WORKER, "\"human\" in Worker texts (B2W)"),
    ]:
        G = nx.Graph()
        G.add_node("human", node_type="focus")

        for row in cooc:
            col  = row["collocate"]
            pmi  = row["pmi"]
            freq = row["cofreq"]
            G.add_node(col, node_type="collocate")
            G.add_edge("human", col, weight=pmi, freq=freq)

        if len(G.nodes) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center",
                    transform=ax.transAxes, **FONT_LABEL)
            ax.set_title(title, **FONT_TITLE)
            ax.axis("off")
            continue

        pos = nx.spring_layout(G, seed=42, k=2.2)

        # Edge width scaled by PMI
        edges     = G.edges(data=True)
        edge_w    = [d["weight"] * 0.4 for _, _, d in edges]
        edge_col  = [colour + "99" for _ in edges]

        nx.draw_networkx_edges(G, pos, ax=ax,
                               width=edge_w, edge_color=edge_col, alpha=0.7)

        # Node sizes
        node_sizes  = [1800 if n == "human" else
                       300 + G[n]["human"]["freq"] * 0.3
                       if "human" in G[n] else 300
                       for n in G.nodes]
        node_colors = [colour if n == "human" else colour + "55"
                       for n in G.nodes]

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_size=node_sizes,
                               node_color=node_colors,
                               edgecolors=colour, linewidths=1.2)

        # Labels
        label_sizes = {n: (11 if n == "human" else 8) for n in G.nodes}
        for node, (x, y) in pos.items():
            ax.text(x, y, node,
                    fontsize=label_sizes[node],
                    fontweight="bold" if node == "human" else "normal",
                    ha="center", va="center",
                    color="white" if node == "human" else C_TEXT,
                    zorder=5)

        if style == "exp":
            # Add PMI scores as edge labels
            edge_labels = {(u, v): f"{d['weight']:.1f}"
                           for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                         font_size=6,
                                         font_color=C_SUBTEXT, ax=ax)

        ax.set_title(title, **FONT_TITLE, pad=12)
        ax.set_facecolor(bg)
        ax.axis("off")

    fig.suptitle("Co-occurrence Network: 'human' Collocates by Audience Register",
                 **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01,
             f"Edge weight = PMI score  •  Node size ∝ co-occurrence frequency  •  Min freq={MIN_COFREQ}",
             ha="center", **FONT_ANNOT)
    save(fig, "fig2_cooccurrence_network", style)


# ---------------------------------------------------------------------------
# Figure 3: Frequency comparison — theoretically key terms
# ---------------------------------------------------------------------------

def fig_frequency_comparison(conn, style: str):
    log.info(f"Figure 3 — Frequency comparison ({style})")

    # Theoretically motivated term set from H1a, H1b, H1c
    focus_terms = [
        # H1a — labour visibility
        "worker", "work", "job", "earn", "pay", "payment",
        # H1b — automation myth
        "autonomous", "automate", "automation", "machine", "model",
        # H1c — strategic hypervisibility
        "human", "quality", "oversight", "annotation", "label",
    ]

    rows = conn.execute("""
        SELECT term, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = 'cross_platform'
          AND term IN ({})
          AND term_type = 'unigram'
    """.format(",".join("?" * len(focus_terms))),
    focus_terms).fetchall()

    # Build lookup and preserve order
    data = {r["term"]: r for r in rows}
    terms   = [t for t in focus_terms if t in data]
    c_freq  = [data[t]["rel_freq_client"] for t in terms]
    w_freq  = [data[t]["rel_freq_worker"] for t in terms]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=bg)
    ax.set_facecolor(bg)

    x     = np.arange(len(terms))
    width = 0.38

    bars_c = ax.bar(x - width/2, c_freq, width, label="Client (B2B)",
                    color=C_CLIENT, alpha=0.88, edgecolor="white", linewidth=0.5)
    bars_w = ax.bar(x + width/2, w_freq, width, label="Worker (B2W)",
                    color=C_WORKER, alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=35, ha="right", fontsize=9, color=C_TEXT)
    ax.set_ylabel("Relative frequency (per 1,000 tokens)", **FONT_LABEL)
    ax.set_title("Relative Frequency of Theoretically Key Terms by Audience Register",
                 **FONT_TITLE, pad=12)

    # H-group separators
    group_boundaries = [5.5, 10.5]
    group_labels     = ["H1a — Labour\nvisibility", "H1b — Automation\nmyth",
                        "H1c — Strategic\nhypervisibility"]
    group_centres    = [2.5, 8.0, 13.0]

    for xb in group_boundaries:
        ax.axvline(xb, color=C_GRID, linewidth=1.2, linestyle="--")

    for xc, gl in zip(group_centres, group_labels):
        ax.text(xc, ax.get_ylim()[1] * 0.97, gl,
                ha="center", va="top", fontsize=7.5,
                color=C_SUBTEXT, style="italic")

    if style == "exp":
        for bar in list(bars_c) + list(bars_w):
            h = bar.get_height()
            if h > 0.1:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6.5,
                        color=C_SUBTEXT)

    ax.legend(frameon=False, fontsize=9)
    apply_base_style(ax, bg)
    fig.text(0.5, -0.02,
             "Terms grouped by hypothesis  •  Frequency per 1,000 lemmatized tokens  •  Cross-platform corpus",
             ha="center", **FONT_ANNOT)

    save(fig, "fig3_frequency_comparison", style)


# ---------------------------------------------------------------------------
# Figure 4: Within-pair comparison — appen and toloka
# ---------------------------------------------------------------------------

def fig_within_pair(conn, style: str):
    log.info(f"Figure 4 — Within-pair comparison ({style})")

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=bg)
    fig.subplots_adjust(wspace=0.5)

    for ax, company_id, pair_label in [
        (axes[0], "appen",  "Appen (appen.com B2B) vs CrowdGen (crowdgen.com B2W)"),
        (axes[1], "toloka", "Toloka (toloka.ai B2B) vs Mindrift (mindrift.ai B2W)"),
    ]:
        rows = fetch_keyness(conn, company_id)
        client_top = [r for r in rows if r["ll_score"] > 0
                      and r["term_type"] == "unigram"][:12]
        worker_top = [r for r in rows if r["ll_score"] < 0
                      and r["term_type"] == "unigram"][:12]
        worker_top = list(reversed(worker_top))

        all_terms  = [r["term"] for r in worker_top] + \
                     ["— — —"] + \
                     [r["term"] for r in client_top]
        all_scores = [-abs(r["ll_score"]) for r in worker_top] + \
                     [0] + \
                     [abs(r["ll_score"]) for r in client_top]

        colours = [C_WORKER] * len(worker_top) + \
                  ["none"] + \
                  [C_CLIENT] * len(client_top)

        y_pos = np.arange(len(all_terms))
        bars  = ax.barh(y_pos, all_scores, color=colours,
                        alpha=0.85, edgecolor="white", linewidth=0.4,
                        height=0.65)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_terms, fontsize=8.5, color=C_TEXT)
        ax.axvline(0, color=C_TEXT, linewidth=0.8)
        ax.set_xlabel("Log-likelihood G²  (← Worker | Client →)", **FONT_LABEL)
        ax.set_title(pair_label, **FONT_TITLE, pad=10)
        apply_base_style(ax, bg)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=C_GRID, linewidth=0.5, linestyle="--")

        if style == "exp":
            for bar, score in zip(bars, all_scores):
                if abs(score) > 0:
                    ax.text(score + (max(all_scores) * 0.02 if score > 0
                                     else min(all_scores) * 0.02),
                            bar.get_y() + bar.get_height() / 2,
                            f"{abs(score):.0f}",
                            va="center", fontsize=6.5, color=C_SUBTEXT)

    # Legend
    client_patch = mpatches.Patch(color=C_CLIENT, label="Client-distinctive (B2B)")
    worker_patch = mpatches.Patch(color=C_WORKER, label="Worker-distinctive (B2W)")
    fig.legend(handles=[client_patch, worker_patch],
               loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Within-Pair Keyness: Same Company, Different Audience Register",
                 **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01,
             "Paired comparison controls for company-level variation  •  Differences attributable to audience only",
             ha="center", **FONT_ANNOT)

    save(fig, "fig4_within_pair", style)


# ---------------------------------------------------------------------------
# Figure 5: Platform heatmap
# ---------------------------------------------------------------------------

def fig_platform_heatmap(conn, style: str):
    log.info(f"Figure 5 — Platform heatmap ({style})")

    # Select theoretically motivated terms for the heatmap
    heatmap_terms = [
        "human", "worker", "autonomous", "quality", "annotation",
        "model", "earn", "job", "oversight", "automate",
        "machine", "task", "label", "pay", "datum",
    ]

    term_data, audiences = fetch_platform_terms(conn, heatmap_terms)

    # Sort platforms: clients first, then workers
    domains_client = sorted([d for d, a in audiences.items() if a == "client"])
    domains_worker = sorted([d for d, a in audiences.items() if a == "worker"])
    domains_both   = sorted([d for d, a in audiences.items() if a == "both"])
    all_domains    = domains_client + domains_both + domains_worker

    # Build matrix
    matrix = np.zeros((len(all_domains), len(heatmap_terms)))
    for i, domain in enumerate(all_domains):
        for j, term in enumerate(heatmap_terms):
            matrix[i, j] = term_data.get(domain, {}).get(term, 0)

    # Normalise rows for pub style; raw for exp style
    if style == "pub":
        row_max = matrix.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        plot_matrix = matrix / row_max
        cbar_label  = "Normalised relative frequency (row max = 1)"
        cmap        = "Blues"
    else:
        plot_matrix = matrix
        cbar_label  = "Relative frequency per 1,000 tokens"
        cmap        = "YlOrRd"

    # Short domain labels
    short_labels = [d.replace(".com", "").replace(".ai", "")
                     .replace(".net", "").replace(".org", "")
                     .replace(".tech", "").replace(".me", "")
                     for d in all_domains]

    bg = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, ax = plt.subplots(figsize=(14, max(7, len(all_domains) * 0.45)),
                           facecolor=bg)
    ax.set_facecolor(bg)

    im = ax.imshow(plot_matrix, aspect="auto", cmap=cmap,
                   interpolation="nearest")

    ax.set_xticks(np.arange(len(heatmap_terms)))
    ax.set_xticklabels(heatmap_terms, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(all_domains)))
    ax.set_yticklabels(short_labels, fontsize=9)

    # Colour y-labels by audience
    for i, domain in enumerate(all_domains):
        aud = audiences.get(domain, "both")
        col = C_CLIENT if aud == "client" else \
              C_WORKER if aud == "worker" else C_SUBTEXT
        ax.get_yticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_fontweight(
            "bold" if aud != "both" else "normal")

    # Separator between client and worker sections
    n_client = len(domains_client) + len(domains_both)
    ax.axhline(n_client - 0.5, color=C_TEXT, linewidth=1.5, linestyle="--")
    ax.text(len(heatmap_terms) - 0.4, n_client - 0.7,
            "▲ Client  |  Worker ▼",
            ha="right", va="bottom", fontsize=8, color=C_TEXT)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(cbar_label, fontsize=8, color=C_SUBTEXT)
    cbar.ax.tick_params(labelsize=7)

    if style == "exp":
        # Add cell values
        for i in range(len(all_domains)):
            for j in range(len(heatmap_terms)):
                val = plot_matrix[i, j]
                if val > 0.1:
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=5.5,
                            color="white" if val > plot_matrix.max() * 0.6
                            else C_TEXT)

    ax.set_title("Term Frequency Heatmap Across Platforms",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.02,
             "Platforms sorted by audience  •  Blue labels = client  •  Red labels = worker  •  Theoretically key terms only",
             ha="center", **FONT_ANNOT)

    save(fig, "fig5_platform_heatmap", style)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("03_visualise_step1.py — Step 1 Figures")
    log.info("=" * 60)
    log.info(f"Output directory: {OUTPUT_DIR.resolve()}")

    conn = get_conn()

    # Verify results tables exist
    for table in ["keyness_results", "cooccurrence_results", "platform_term_counts"]:
        if not conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
        ).fetchone():
            raise RuntimeError(f"Table {table} not found — run 02_step1_frequency.py first.")

    for style in ["pub", "exp"]:
        log.info(f"\n{'='*30} Style: {style.upper()} {'='*30}")
        fig_keyness_bar(conn, style)
        fig_cooccurrence_network(conn, style)
        fig_frequency_comparison(conn, style)
        fig_within_pair(conn, style)
        fig_platform_heatmap(conn, style)

    conn.close()

    log.info("=" * 60)
    log.info(f"All figures saved to {OUTPUT_DIR.resolve()}")
    log.info("  fig1_keyness_bar_pub/exp.png")
    log.info("  fig2_cooccurrence_network_pub/exp.png")
    log.info("  fig3_frequency_comparison_pub/exp.png")
    log.info("  fig4_within_pair_pub/exp.png")
    log.info("  fig5_platform_heatmap_pub/exp.png")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
