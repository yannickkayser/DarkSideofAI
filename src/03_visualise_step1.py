"""
03_visualise_step1.py
=====================
Produces visualisations of Step 1 keyness and co-occurrence results.

Fixes applied over v1:
  - fetch_top_client / fetch_top_worker query correctly in opposite directions
  - Within-pair (fig4) now shows worker-distinctive terms on the left
  - Heatmap columns reordered by hypothesis group (H1a / H1b / H1c)
  - Heatmap colour scale capped at 95th percentile so datum doesn't dominate
  - Extended artifact filter covers residual noise terms from v1

Five figures, each in two styles (pub = clean thesis, exp = annotated):
  1. fig1_keyness_bar         — top client vs worker distinctive terms
  2. fig2_cooccurrence_network — 'human' collocates in client vs worker
  3. fig3_frequency_comparison — key terms grouped by H1a / H1b / H1c
  4. fig4_within_pair         — appen and toloka diverging bar
  5. fig5_platform_heatmap    — term frequency across all platforms

Usage:
    python3 src/03_visualise_step1.py
"""

import sqlite3
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH    = "data/scraping.db"
OUTPUT_DIR = Path("output/step_1/")
TOP_N      = 20      # terms per direction in keyness bar charts
TOP_PAIR_N = 12      # terms per direction in within-pair charts
MIN_COFREQ = 15      # minimum co-occurrence frequency for network edges
NETWORK_N  = 18      # max collocates shown per side in network

# Artifact terms — scraping noise identified during Step 1 interpretation
ARTIFACT_TERMS = {
    "cookie", "set_cookie", "cooky",
    "/hr", "/hr_remote", "remote_apply", "feb", "opportunity_feb",
    "faq", "faq_help", "help_desk", "desk", "subscribe",
    "website", "account", "access", "enable", "microworker", "shall", "youtube",
    "zeynep", "koouchnir", "gavrilov", "unga", "gary", "yalda",
    "monarch", "warhol", "fremont", "pittsburgh", "mpii",
    "experience.with", "rhml", "ead", "cc0", "ft",
    # Residual noise from v1 worker side
    "hole", "overfit", "surprised", "christmas", "morale", "high-quality",
    "slash", "500", "pickup", "loophole", "conceptually", "housing",
    "firefighting", "sidestep", "wary", "downward", "jira", "voluman",
    "squeeze", "retrofit", "yt", "ml",
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_CLIENT  = "#1B4F8A"
C_WORKER  = "#C0392B"
C_BG_PUB  = "#FFFFFF"
C_BG_EXP  = "#F7F9FC"
C_GRID    = "#DEE2E6"
C_TEXT    = "#1A1A2E"
C_SUBTEXT = "#6C757D"

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


def _ph():
    return ",".join("?" * len(ARTIFACT_TERMS))


def fetch_top_client(conn, comparison, n=TOP_N):
    rows = conn.execute(f"""
        SELECT term, ll_score, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = ?
          AND ll_score > 0
          AND term_type = 'unigram'
          AND term NOT IN ({_ph()})
        ORDER BY ll_score DESC
        LIMIT ?
    """, [comparison] + list(ARTIFACT_TERMS) + [n]).fetchall()
    return [dict(r) for r in rows]


def fetch_top_worker(conn, comparison, n=TOP_N):
    rows = conn.execute(f"""
        SELECT term, ll_score, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = ?
          AND ll_score < 0
          AND term_type = 'unigram'
          AND term NOT IN ({_ph()})
        ORDER BY ll_score ASC
        LIMIT ?
    """, [comparison] + list(ARTIFACT_TERMS) + [n]).fetchall()
    return [dict(r) for r in rows]


def fetch_cooccurrence(conn, comparison, audience, focus, min_freq=MIN_COFREQ):
    rows = conn.execute(f"""
        SELECT collocate, pmi, cofreq
        FROM cooccurrence_results
        WHERE comparison = ? AND audience = ? AND focus_term = ?
          AND cofreq >= ?
          AND collocate NOT IN ({_ph()})
        ORDER BY pmi DESC
        LIMIT ?
    """, [comparison, audience, focus, min_freq]
         + list(ARTIFACT_TERMS) + [NETWORK_N]).fetchall()
    return [dict(r) for r in rows]


def fetch_term_freqs(conn, terms, comparison="cross_platform"):
    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(f"""
        SELECT term, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = ?
          AND term IN ({placeholders})
          AND term_type = 'unigram'
    """, [comparison] + terms).fetchall()
    return {r["term"]: dict(r) for r in rows}


def fetch_platform_terms(conn, terms):
    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(f"""
        SELECT domain, audience, term, rel_freq
        FROM platform_term_counts
        WHERE term IN ({placeholders})
          AND term_type = 'unigram'
    """, terms).fetchall()
    result, audiences = defaultdict(dict), {}
    for r in rows:
        result[r["domain"]][r["term"]] = r["rel_freq"]
        audiences[r["domain"]] = r["audience"]
    return dict(result), audiences


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
    path = OUTPUT_DIR / f"{name}_{style}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    log.info(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Keyness bar chart
# ---------------------------------------------------------------------------

def fig_keyness_bar(conn, style):
    log.info(f"Figure 1 — Keyness bar chart ({style})")

    client_top = fetch_top_client(conn, "cross_platform", TOP_N)
    worker_top = list(reversed(fetch_top_worker(conn, "cross_platform", TOP_N)))

    log.info(f"  Client: {[r['term'] for r in client_top]}")
    log.info(f"  Worker: {[r['term'] for r in worker_top]}")

    bg  = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor=bg)
    fig.subplots_adjust(wspace=0.5)

    for ax, data, colour, label, reg in [
        (axes[0], client_top, C_CLIENT, "Client-distinctive (B2B)", "client"),
        (axes[1], worker_top, C_WORKER, "Worker-distinctive (B2W)", "worker"),
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
             "Log-likelihood G² statistic  •  Artifact terms excluded  •  Unigrams only",
             ha="center", **FONT_ANNOT)
    save(fig, "fig1_keyness_bar", style)


# ---------------------------------------------------------------------------
# Figure 2: Co-occurrence network for 'human'
# ---------------------------------------------------------------------------

def fig_cooccurrence_network(conn, style):
    log.info(f"Figure 2 — Co-occurrence network ({style})")

    client_cooc = fetch_cooccurrence(conn, "cross_platform", "client", "human")
    worker_cooc = fetch_cooccurrence(conn, "cross_platform", "worker", "human")

    bg  = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=bg)

    for ax, cooc, colour, title in [
        (axes[0], client_cooc, C_CLIENT, '"human" in Client texts (B2B)'),
        (axes[1], worker_cooc, C_WORKER, '"human" in Worker texts (B2W)'),
    ]:
        ax.set_facecolor(bg)
        ax.set_title(title, **FONT_TITLE, pad=12)
        ax.axis("off")

        if not cooc:
            ax.text(0.5, 0.5, "No collocates above threshold",
                    ha="center", va="center",
                    transform=ax.transAxes, **FONT_LABEL)
            continue

        G = nx.Graph()
        G.add_node("human")
        for row in cooc:
            G.add_node(row["collocate"])
            G.add_edge("human", row["collocate"],
                       weight=row["pmi"], freq=row["cofreq"])

        pos = nx.spring_layout(G, seed=42, k=2.5)

        edge_widths = [G[u][v]["weight"] * 0.35 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                               edge_color=colour + "88", alpha=0.8)

        node_sizes = [2000 if n == "human" else
                      300 + G["human"][n]["freq"] * 0.25
                      if G.has_edge("human", n) else 300
                      for n in G.nodes()]
        node_colors = [colour if n == "human" else colour + "44"
                       for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors,
                               edgecolors=colour, linewidths=1.0)

        for node, (x, y) in pos.items():
            ax.text(x, y, node,
                    fontsize=11 if node == "human" else 8,
                    fontweight="bold" if node == "human" else "normal",
                    ha="center", va="center",
                    color="white" if node == "human" else C_TEXT,
                    zorder=5)

        if style == "exp":
            for u, v, d in G.edges(data=True):
                mx, my = (pos[u] + pos[v]) / 2
                ax.text(mx, my, f"{d['weight']:.1f}",
                        fontsize=6, color=C_SUBTEXT, ha="center", zorder=6)

    fig.suptitle("Co-occurrence Network: 'human' Collocates by Audience Register",
                 **FONT_TITLE, y=1.01)
    fig.text(0.5, -0.01,
             f"Edge weight = PMI score  •  Node size ∝ co-occurrence frequency  •  Min freq={MIN_COFREQ}",
             ha="center", **FONT_ANNOT)
    save(fig, "fig2_cooccurrence_network", style)


# ---------------------------------------------------------------------------
# Figure 3: Frequency comparison grouped by hypothesis
# ---------------------------------------------------------------------------

def fig_frequency_comparison(conn, style):
    log.info(f"Figure 3 — Frequency comparison ({style})")

    groups = {
        "H1a — Labour visibility":        ["worker", "work", "job", "earn", "pay", "payment"],
        "H1b — Automation myth":           ["autonomous", "automate", "automation", "machine", "model"],
        "H1c — Strategic hypervisibility": ["human", "quality", "oversight", "annotation", "label"],
    }
    all_terms = [t for v in groups.values() for t in v]
    freq_data = fetch_term_freqs(conn, all_terms)
    terms   = [t for t in all_terms if t in freq_data]
    c_freq  = [freq_data[t]["rel_freq_client"] for t in terms]
    w_freq  = [freq_data[t]["rel_freq_worker"] for t in terms]

    bg  = C_BG_PUB if style == "pub" else C_BG_EXP
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

    # Group boundaries and labels
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


# ---------------------------------------------------------------------------
# Figure 4: Within-pair diverging bar
# ---------------------------------------------------------------------------

def fig_within_pair(conn, style):
    log.info(f"Figure 4 — Within-pair comparison ({style})")

    bg  = C_BG_PUB if style == "pub" else C_BG_EXP
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=bg)
    fig.subplots_adjust(wspace=0.55)

    for ax, company_id, pair_label in [
        (axes[0], "appen",  "Appen (B2B) vs CrowdGen (B2W)"),
        (axes[1], "toloka", "Toloka (B2B) vs Mindrift (B2W)"),
    ]:
        client_top = fetch_top_client(conn, company_id, TOP_PAIR_N)
        worker_top = fetch_top_worker(conn, company_id, TOP_PAIR_N)

        log.info(f"  {company_id} client: {[r['term'] for r in client_top]}")
        log.info(f"  {company_id} worker: {[r['term'] for r in worker_top]}")

        if not client_top and not worker_top:
            ax.text(0.5, 0.5, "No data", ha="center",
                    transform=ax.transAxes, **FONT_LABEL)
            ax.set_title(pair_label, **FONT_TITLE)
            continue

        worker_rev = list(reversed(worker_top))
        all_terms  = ([r["term"] for r in worker_rev] +
                      ["— — —"] +
                      [r["term"] for r in client_top])
        all_scores = ([-abs(r["ll_score"]) for r in worker_rev] +
                      [0] +
                      [abs(r["ll_score"]) for r in client_top])
        colours    = ([C_WORKER] * len(worker_rev) +
                      ["none"] +
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
# Figure 5: Platform heatmap — columns ordered by hypothesis
# ---------------------------------------------------------------------------

def fig_platform_heatmap(conn, style):
    log.info(f"Figure 5 — Platform heatmap ({style})")

    # Columns ordered by hypothesis group — mirrors fig3
    heatmap_terms = [
        "worker", "job", "earn", "pay",           # H1a
        "autonomous", "automate", "machine", "model",  # H1b
        "human", "quality", "oversight", "annotation", "label", "datum",  # H1c
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

    def shorten(d):
        for ext in [".com", ".ai", ".net", ".org", ".tech", ".me"]:
            d = d.replace(ext, "")
        return d.replace("www.", "")

    short_labels = [shorten(d) for d in all_domains]

    bg  = C_BG_PUB if style == "pub" else C_BG_EXP
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
        col = C_CLIENT if aud == "client" else \
              C_WORKER if aud == "worker" else C_SUBTEXT
        ax.get_yticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_fontweight(
            "bold" if aud != "both" else "normal")

    ax.axhline(n_client_rows - 0.5, color=C_TEXT, linewidth=1.5, linestyle="--")
    ax.text(len(heatmap_terms) - 0.4, n_client_rows - 0.7,
            "▲ Client  |  Worker ▼",
            ha="right", va="bottom", fontsize=8, color=C_TEXT)

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

    ax.set_title("Term Frequency Heatmap Across Platforms",
                 **FONT_TITLE, pad=12)
    fig.text(0.5, -0.03,
             "Blue labels = client  •  Red labels = worker  •  Columns grouped by hypothesis",
             ha="center", **FONT_ANNOT)
    save(fig, "fig5_platform_heatmap", style)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    log.info("=" * 60)
    log.info("03_visualise_step1.py v2 — Step 1 Figures")
    log.info("=" * 60)

    conn = get_conn()

    for table in ["keyness_results", "cooccurrence_results", "platform_term_counts"]:
        if not conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'"
        ).fetchone():
            raise RuntimeError(
                f"Table '{table}' not found — run 02_step1_frequency.py first.")

    for style in ["pub", "exp"]:
        log.info(f"\n{'='*30} Style: {style.upper()} {'='*30}")
        fig_keyness_bar(conn, style)
        fig_cooccurrence_network(conn, style)
        fig_frequency_comparison(conn, style)
        fig_within_pair(conn, style)
        fig_platform_heatmap(conn, style)

    conn.close()
    log.info("=" * 60)
    log.info(f"All 10 figures saved to {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
