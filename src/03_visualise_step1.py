"""
03_visualise_step1.py
=====================
Produces all Step 1 visualisations for keyness and co-occurrence results.

Pipeline position:
  Stage 3a — Visualisation (run after 02_step1_frequency.py)
  Prerequisites: keyness_results, cooccurrence_results, platform_term_counts
  Output: 12 .png files (6 figures × 2 styles) in output/step_1/

What this script does:
  Generates six figures, each exported in two styles:
    pub  — clean thesis-quality figure (white background, no annotations)
    exp  — exploratory/annotated figure with frequency labels and values
           (used during analysis; not included in thesis)

  Figure overview:
    fig1_keyness_bar
      Horizontal bar charts showing the top 20 B2B-distinctive and top 20
      B2W-distinctive unigrams ranked by Log-Likelihood G².  The primary
      evidence for H1b and H1c (client side) and H1a (worker side).

    fig2_cooccurrence_network
      PMI collocate profiles for four theory-selected focus terms
      (human, annotation, autonomous, earn).  Shows how the same term's
      discursive neighbourhood differs between B2B and B2W registers.
      These were selected because they produce theoretically contrasting
      profiles (e.g. "human" + "oversight" in B2B vs "human" + "task"
      in B2W).

    fig3_frequency_comparison
      Grouped bar chart of relative frequencies for hypothesis terms,
      grouped by H1a / H1b / H1c.  Directly visualises the magnitude of
      register differences for the theory-focus vocabulary.

    fig4_within_pair
      Diverging bar charts for appen/crowdgen and toloka/mindrift within-
      pair comparisons.  Controls for company-level style variation —
      differences are attributable to audience alone.

    fig5_platform_heatmap
      Heatmap of term frequencies across all platforms, columns ordered
      by hypothesis group.  Visualises which platforms are most strongly
      associated with each hypothesis-relevant term.

    fig6_theory_cooccurrence
      Co-occurrence PMI profiles for the full H1a/H1b/H1c theory
      vocabulary (14 terms).  Complements fig2 (which showed 4 terms
      selected by LL rank) by showing all theoretically motivated terms
      regardless of keyness score.

Dual-style rationale:
  The 'pub' style produces clean, thesis-ready figures at 150 dpi.
  The 'exp' style adds value annotations, frequency labels, and a grey
  background — useful during the analysis phase to read exact values
  without querying the database.

Artifact filtering:
  ARTIFACT_TERMS contains scraping noise identified during interpretation:
  UI boilerplate, proper nouns, German scraping residue.  These are
  filtered from all figure queries at display time (not from the DB
  tables themselves, which retain the raw statistical values for
  reference).  The DB-level filtering in 01_prepare_additions.py is
  more thorough; ARTIFACT_TERMS here is a secondary display filter.

Input (from data/scraping.db):
  keyness_results       — term, ll_score, rel_freq_client, rel_freq_worker
  cooccurrence_results  — focus_term, collocate, pmi, cofreq
  platform_term_counts  — domain, term, rel_freq

Output:
  output/step_1/fig1_keyness_bar_pub.png
  output/step_1/fig1_keyness_bar_exp.png
  output/step_1/fig2_cooccurrence_network_pub.png
  output/step_1/fig2_cooccurrence_network_exp.png
  ... (12 files total)

Usage:
    python3 src/03_visualise_step1.py

Fixes applied over v1:
  - fetch_top_client / fetch_top_worker query correctly in opposite
    directions (LL > 0 for client, LL < 0 for worker)
  - Within-pair (fig4) now shows worker-distinctive terms on the left
  - Heatmap columns reordered by hypothesis group (H1a / H1b / H1c)
  - Heatmap colour scale capped at 95th percentile so 'datum' doesn't
    dominate the colour range
  - Extended ARTIFACT_TERMS covers residual noise terms from v1
"""

import math
import sqlite3
import logging
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH    = "data/scraping.db"
OUTPUT_DIR = Path("output/step_1/")
TOP_N      = 20      # terms per direction in keyness bar charts
TOP_PAIR_N = 12      # terms per direction in within-pair charts
MIN_COFREQ = 10      # minimum co-occurrence frequency (= DB minimum)
NETWORK_N  = 18      # max collocates shown per side in network

# Focus terms for fig2 co-occurrence profiles.
# Selected because they (a) exist as focus_term in cooccurrence_results for
# both audiences and (b) produce theoretically contrasting collocate profiles:
#   human      - H1c: human-in-the-loop / oversight framing vs worker-centric use
#   annotation - H1c: core data-labour task; rich client side, sparse worker side
#   autonomous - H1b: automation myth vocabulary, almost absent on worker side
#   earn       - H1a: labour visibility; appears primarily in worker-facing register
COOC_FOCUS_TERMS = ["human", "annotation", "autonomous", "earn"]
COOC_TOP_N       = 8    # collocates shown per focus term per audience

# Artifact terms — scraping noise identified during Step 1 interpretation.
# These are filtered from figure queries at display time.
# Note: the DB-level exclusion in 01_prepare_additions.py is more thorough;
# this list handles display-time residual noise not caught before analysis.
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
    # Noise visible in co-occurrence profiles (scraping artefacts / proper nouns)
    "deciphering", "trafficking", "recap", "ueberwinden", "bildbearbeitung",
    "sicherstellung", "kunst", "human-le", "pto", "generous",
    "dhanesh", "ramachandram", "outlet", "daniela", "braga", "forbe",
}

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_CLIENT  = "#1B4F8A"   # dark blue — client (B2B) register
C_WORKER  = "#C0392B"   # dark red  — worker (B2W) register
C_BG_PUB  = "#FFFFFF"   # white background for pub style
C_BG_EXP  = "#F7F9FC"   # light grey background for exp style
C_GRID    = "#DEE2E6"   # grid line colour
C_TEXT    = "#1A1A2E"   # primary text
C_SUBTEXT = "#6C757D"   # secondary / annotation text

FONT_TITLE = {"fontsize": 13, "fontweight": "bold",   "color": C_TEXT}
FONT_LABEL = {"fontsize": 10, "fontweight": "normal", "color": C_SUBTEXT}
FONT_ANNOT = {"fontsize":  8, "color": C_SUBTEXT}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hypothesis vocabulary for register scatter (fig_register_scatter)
# ---------------------------------------------------------------------------
# Each entry: display label → {terms, color, marker}
# Matches the HYPOTHESIS_TERMS vocabulary in 02c_step1_topics.py.
# Color logic: H1a terms should appear BELOW the y=x diagonal (more B2W);
# H1b terms should appear ABOVE (more B2B); H1c terms may straddle both.
C_H1C = "#E67E22"   # orange for H1c — neither client nor worker color

HYPOTHESIS_VOCAB = {
    "H1a — Labour visibility": {
        "terms": {"worker", "labour", "task", "job", "earn", "pay", "payment",
                  "annotator", "gig", "contractor", "wage", "labeller",
                  "freelance", "income"},
        "color":  C_WORKER,
        "marker": "o",
    },
    "H1b — Automation myth": {
        "terms": {"autonomous", "machine", "automate", "automation", "algorithm",
                  "pipeline", "deploy", "inference", "neural", "llm",
                  "intelligent", "scalable"},
        "color":  C_CLIENT,
        "marker": "s",
    },
    "H1c — Strategic hypervisibility": {
        "terms": {"human", "quality", "oversight", "annotation", "label",
                  "expert", "accuracy", "datum", "review", "verification"},
        "color":  C_H1C,
        "marker": "^",
    },
}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    """Open a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ph():
    """Build a parameterised NOT IN placeholder string for ARTIFACT_TERMS."""
    return ",".join("?" * len(ARTIFACT_TERMS))


def fetch_top_client(conn, comparison, n=TOP_N):
    """
    Fetch the top N client-distinctive terms (positive LL score) for a
    comparison, excluding ARTIFACT_TERMS.

    Args:
        comparison : 'cross_platform' or a company_id for within-pair.
        n          : number of terms to return.

    Returns:
        List of row dicts with keys: term, ll_score, rel_freq_client,
        rel_freq_worker.
    """
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
    """
    Fetch the top N worker-distinctive terms (negative LL score) for a
    comparison, excluding ARTIFACT_TERMS.

    Args:
        comparison : 'cross_platform' or a company_id for within-pair.
        n          : number of terms to return.

    Returns:
        List of row dicts sorted by LL score ASC (most negative first).
    """
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
    """
    Fetch collocates for a focus term in a specific comparison and audience.

    Args:
        comparison : 'cross_platform' or company_id.
        audience   : 'client' or 'worker'.
        focus      : focus term string.
        min_freq   : minimum co-occurrence count (= minimum from 02 config).

    Returns:
        List of row dicts with keys: collocate, pmi, cofreq.
        Sorted by pmi DESC.
    """
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
    """
    Fetch relative frequencies for a list of terms.

    Used by fig3 to look up frequencies for the hypothesis-grouped term list.

    Args:
        terms      : list of term strings to look up.
        comparison : which comparison to query (default cross_platform).

    Returns:
        Dict {term: {rel_freq_client, rel_freq_worker}}.
    """
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
    """
    Fetch per-domain relative frequencies for a list of terms.

    Used by fig5 heatmap to build the platform × term frequency matrix.

    Args:
        terms: list of term strings to look up.

    Returns:
        Tuple (result, audiences) where:
          result    : {domain: {term: rel_freq}}
          audiences : {domain: audience}
    """
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
    """
    Apply consistent base styling to an Axes object.

    Removes top/right spines, colours remaining spines with C_GRID,
    and sets the axes background.  Called on every subplot before adding
    figure-specific elements.
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
    Save a figure to OUTPUT_DIR/<name>_<style>.png at 150 dpi.

    Creates OUTPUT_DIR if it does not exist.  Closes the figure after
    saving to release memory.

    Args:
        fig   : matplotlib Figure object.
        name  : base filename (without _style.png suffix).
        style : 'pub' or 'exp'.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_{style}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    log.info(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Keyness bar chart
# ---------------------------------------------------------------------------

def fig_keyness_bar(conn, style):
    """
    Figure 1 — Cross-platform keyness: top B2B and B2W distinctive terms.

    Two horizontal bar charts side by side:
      Left:  Top N client-distinctive terms (positive LL, B2B register)
      Right: Top N worker-distinctive terms (negative LL, B2W register)

    Both panels show absolute |LL| so bars extend in the same direction
    for visual symmetry.  The side-by-side layout invites direct
    comparison of what each register foregrounds.

    In exp style: each bar is annotated with the raw relative frequencies
    (per 1,000 tokens) for both sides, allowing the analyst to see the
    magnitude of the difference without querying the DB.

    Saves to: fig1_keyness_bar_{style}.png
    """
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
# Figure 2: Co-occurrence PMI profiles for top key terms
# ---------------------------------------------------------------------------

def fig_cooccurrence_network(conn, style):
    """
    Figure 2 — PMI collocate profiles for theory-selected key terms.

    One row per focus term (human, annotation, autonomous, earn), two
    columns (B2B left, B2W right).  Each panel shows the top PMI-scored
    collocates for the focus term in that register.

    Interpretation guide:
      - If the collocate profiles diverge strongly, the focus term is
        doing different rhetorical work in each register.
      - If they converge, the term has a shared discursive function.
      - Diverging profiles are the strongest candidates for Step 2 close
        reading investigation.

    In exp style: co-occurrence counts are annotated on each bar.

    Saves to: fig2_cooccurrence_network_{style}.png
    """
    log.info(f"Figure 2 — Co-occurrence PMI profiles ({style})")

    bg  = C_BG_PUB if style == "pub" else C_BG_EXP
    n_terms = len(COOC_FOCUS_TERMS)

    fig, axes = plt.subplots(n_terms, 2,
                             figsize=(16, n_terms * 3.2),
                             facecolor=bg)
    fig.subplots_adjust(hspace=0.55, wspace=0.55)

    for row_idx, focus in enumerate(COOC_FOCUS_TERMS):
        for col_idx, (audience, colour, reg_label) in enumerate([
            ("client", C_CLIENT, "B2B"),
            ("worker", C_WORKER, "B2W"),
        ]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(bg)

            cooc = fetch_cooccurrence(conn, "cross_platform", audience,
                                      focus, min_freq=MIN_COFREQ)
            cooc = cooc[:COOC_TOP_N]

            title = f'"{focus}"  —  {reg_label}'
            ax.set_title(title, fontsize=10, fontweight="bold",
                         color=colour, pad=8)

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
    fig.text(
        0.5, -0.01,
        "PMI = Pointwise Mutual Information  •  Reveals discursive neighbourhood of each key term"
        f"  •  Min co-freq={MIN_COFREQ}  •  Directed to Step 2 close reading",
        ha="center", **FONT_ANNOT)
    save(fig, "fig2_cooccurrence_network", style)


# ---------------------------------------------------------------------------
# Figure 3: Frequency comparison grouped by hypothesis
# ---------------------------------------------------------------------------

def fig_frequency_comparison(conn, style):
    """
    Figure 3 — Relative frequencies of hypothesis terms grouped by H1a/b/c.

    Grouped bar chart showing client and worker relative frequencies for
    the theory-focus vocabulary, arranged by hypothesis group.  Vertical
    dividers mark hypothesis boundaries.

    This figure directly visualises whether the hypothesis predictions are
    borne out in the frequency data:
      H1a: labour vocabulary (worker, job, earn) should be higher in B2W
      H1b: automation vocabulary (autonomous, machine) should be higher in B2B
      H1c: quality vocabulary (human, oversight) should be higher in B2B

    Saves to: fig3_frequency_comparison_{style}.png
    """
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
    """
    Figure 4 — Within-pair keyness for appen and toloka platform pairs.

    Diverging bar charts: worker-distinctive terms on the left (negative
    LL), client-distinctive terms on the right (positive LL).  One panel
    per pair.

    The within-pair design controls for company-level language variation:
    since both platforms in each pair belong to the same company, any
    vocabulary difference is attributable to audience rather than to
    different corporate cultures or industries.  This strengthens the
    causal attribution of register differences to audience-targeting.

    Saves to: fig4_within_pair_{style}.png
    """
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
# Figure 5: Platform heatmap
# ---------------------------------------------------------------------------

def fig_platform_heatmap(conn, style):
    """
    Figure 5 — Heatmap of term frequencies across all platforms.

    Columns are hypothesis-grouped (H1a / H1b / H1c), rows are platforms
    sorted client-first then worker.  A horizontal dashed line separates
    client from worker platforms.

    In pub style: values are row-normalised (max=1) to make within-platform
    patterns visible regardless of absolute frequency level.
    In exp style: raw clipped values with actual numbers annotated.

    Heatmap colour scale is capped at the 95th percentile to prevent
    high-frequency terms (like 'datum') from collapsing all other values
    into the bottom of the colour range.

    Saves to: fig5_platform_heatmap_{style}.png
    """
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
# Figure 6: Theory-driven co-occurrence profiles
# ---------------------------------------------------------------------------

# Full H1a/H1b/H1c vocabulary for theory-driven co-occurrence panels.
# Grouped by hypothesis for readability in the figure.
# Terms absent from cooccurrence_results are silently skipped.
FIG6_GROUPS = {
    "H1a — Labour visibility":         ["worker", "labour", "task", "earn", "pay"],
    "H1b — Automation myth":           ["autonomous", "machine", "automation", "intelligent"],
    "H1c — Strategic hypervisibility": ["human", "quality", "oversight", "annotation", "label"],
}
FIG6_TOP_N = 8   # collocates shown per panel


def fig_theory_cooccurrence(conn, style):
    """
    Figure 6 — Theory-driven co-occurrence profiles for all H1a-c terms.

    Complements fig2 (which showed 4 terms selected by LL rank) by
    showing all theoretically motivated terms regardless of keyness score.
    Some H1c terms like "human" may appear on both sides with low LL but
    have strongly divergent collocate profiles that reveal the rhetorical
    difference.

    Layout: one row per focus term, B2B left / B2W right, with hypothesis
    group labels.  Horizontal dividers separate hypothesis groups.

    Saves to: fig6_theory_cooccurrence_{style}.png
    """
    log.info(f"Figure 6 — Theory-driven co-occurrence profiles ({style})")

    bg = C_BG_PUB if style == "pub" else C_BG_EXP

    # Flatten ordered term list; skip any not in DB
    ordered_terms = [t for terms in FIG6_GROUPS.values() for t in terms]
    available = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT focus_term FROM cooccurrence_results "
            "WHERE comparison = 'cross_platform'"
        ).fetchall()
    }
    focus_terms = [t for t in ordered_terms if t in available]

    if not focus_terms:
        log.warning("  No theory-driven focus terms found in DB — "
                    "re-run 02_step1_frequency.py first.")
        return

    skipped = [t for t in ordered_terms if t not in available]
    if skipped:
        log.warning(f"  Terms not yet in DB (re-run 02): {skipped}")

    n_terms = len(focus_terms)
    fig, axes = plt.subplots(n_terms, 2,
                             figsize=(16, n_terms * 3.2),
                             facecolor=bg)
    if n_terms == 1:
        axes = [axes]   # ensure iterable
    fig.subplots_adjust(hspace=0.55, wspace=0.55)

    # Build group label positions for left margin annotations
    group_row_map = {}
    for group_label, terms in FIG6_GROUPS.items():
        rows = [i for i, t in enumerate(focus_terms) if t in terms]
        if rows:
            group_row_map[group_label] = (rows[0], rows[-1])

    for row_idx, focus in enumerate(focus_terms):
        for col_idx, (audience, colour, reg_label) in enumerate([
            ("client", C_CLIENT, "B2B"),
            ("worker", C_WORKER, "B2W"),
        ]):
            ax = axes[row_idx][col_idx] if n_terms > 1 else axes[col_idx]
            ax.set_facecolor(bg)

            cooc = fetch_cooccurrence(conn, "cross_platform", audience,
                                      focus, min_freq=MIN_COFREQ)
            cooc = cooc[:FIG6_TOP_N]

            # Determine hypothesis group for subtitle
            group_short = next(
                (k.split("—")[0].strip() for k, v in FIG6_GROUPS.items()
                 if focus in v), "")
            ax.set_title(f'"{focus}"  —  {reg_label}  [{group_short}]',
                         fontsize=10, fontweight="bold", color=colour, pad=8)

            if not cooc:
                ax.text(0.5, 0.5,
                        "No collocates above threshold\n(re-run 02_step1_frequency.py)",
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

    # Group boundary lines between hypothesis blocks
    if n_terms > 1:
        for group_label, (first_row, last_row) in group_row_map.items():
            if last_row < n_terms - 1:
                for col_idx in range(2):
                    ax = axes[last_row][col_idx]
                    ax.axhline(-0.5, color=C_GRID, linewidth=1.2, linestyle="--")

    fig.suptitle(
        "Theory-Driven Co-occurrence Profiles: Terms Central to H1a–H1c",
        **FONT_TITLE, y=1.01)
    fig.text(
        0.5, -0.01,
        "Terms selected for theoretical relevance, not statistical keyness  •  "
        "Same term, different discursive neighbourhood by audience register  •  "
        f"PMI, min co-freq={MIN_COFREQ}",
        ha="center", **FONT_ANNOT)
    save(fig, "fig6_theory_cooccurrence", style)


# ---------------------------------------------------------------------------
# Figure S1: Register gap scatter
# ---------------------------------------------------------------------------

def fig_register_scatter(conn, style):
    """
    Figure S1 — Register gap scatter: all unigrams by B2B vs B2W relative frequency.

    Plots every unigram in keyness_results as a dot on a log-log scale:
      x-axis: log10(relative frequency in B2W — worker)
      y-axis: log10(relative frequency in B2B — client)

    The y = x diagonal marks equal frequency in both registers.
    Terms above the diagonal are B2B-distinctive (more frequent in client texts);
    terms below are B2W-distinctive (more frequent in worker texts).

    Hypothesis vocabulary terms are highlighted with distinct shapes and colours:
      H1a (labour visibility): red circles — predicted to fall BELOW diagonal
        (pay, earn, task should appear more in B2W)
      H1b (automation myth): blue squares — predicted to fall ABOVE diagonal
        (autonomous, algorithm, deploy should appear more in B2B)
      H1c (hypervisibility): orange triangles — may straddle both registers
        because human, quality, oversight appear in both but differently framed

    Why this figure matters:
      Figures 1 and 3 show the TOP 20 terms per register — necessarily a
      curated selection.  This figure plots the FULL vocabulary (thousands of
      terms) so the analyst and reader can see that the register differentiation
      is not cherry-picked but systematic: hypothesis vocabulary consistently
      falls in the predicted regions of the frequency space.

    Terms with zero frequency in one register are plotted at the epsilon floor
    (0.001 per 1,000 tokens) — they appear as a band along the left or
    bottom edge, showing strong register exclusivity.

    Saves to: figS1_register_scatter_{style}.png
    """
    log.info(f"Figure S1 — Register gap scatter ({style})")

    all_rows = conn.execute(f"""
        SELECT term, rel_freq_client, rel_freq_worker
        FROM keyness_results
        WHERE comparison = 'cross_platform'
          AND term_type = 'unigram'
          AND term NOT IN ({_ph()})
    """, list(ARTIFACT_TERMS)).fetchall()

    if not all_rows:
        log.warning("  No keyness data — skipping register scatter.")
        return

    # Build flat lookup: term → hypothesis config (first match wins)
    term_to_hyp = {}
    for hyp_key, cfg in HYPOTHESIS_VOCAB.items():
        for t in cfg["terms"]:
            if t not in term_to_hyp:
                term_to_hyp[t] = (hyp_key, cfg)

    # Separate background from hypothesis-highlighted terms
    EPS = 0.001   # floor to avoid log10(0)
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

    # Layer 0: all background terms — small, grey, transparent
    ax.scatter(bg_x, bg_y, s=7, color=C_GRID, alpha=0.30,
               edgecolors="none", zorder=1)

    # Diagonal y = x reference line
    all_x = bg_x + [v for d in hyp_data.values() for v in d["x"]]
    all_y = bg_y + [v for d in hyp_data.values() for v in d["y"]]
    if all_x and all_y:
        lo = min(min(all_x), min(all_y)) - 0.15
        hi = max(max(all_x), max(all_y)) + 0.15
        ax.plot([lo, hi], [lo, hi], color=C_SUBTEXT, linewidth=1.0,
                linestyle="--", alpha=0.55, zorder=2)

    # Layer 1: hypothesis terms — larger, coloured, labelled
    for hyp_key, cfg in HYPOTHESIS_VOCAB.items():
        layer = hyp_data[hyp_key]
        if not layer["x"]:
            continue
        ax.scatter(layer["x"], layer["y"],
                   s=70, color=cfg["color"], alpha=0.92,
                   edgecolors="white", linewidths=0.6,
                   marker=cfg["marker"], zorder=4)
        for xv, yv, lbl in zip(layer["x"], layer["y"], layer["labels"]):
            ax.annotate(lbl, (xv, yv), fontsize=7.5,
                        color=cfg["color"], fontweight="bold",
                        textcoords="offset points", xytext=(5, 4),
                        zorder=5)

    # Directional region labels
    ylo, yhi = ax.get_ylim()
    xlo, xhi = ax.get_xlim()
    ax.text(0.97, 0.03, "← more B2W-distinctive →",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=C_WORKER, style="italic", alpha=0.75)
    ax.text(0.03, 0.97, "← more B2B-distinctive →",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=C_CLIENT, style="italic", alpha=0.75,
            rotation=90)

    ax.set_xlabel("log₁₀(relative frequency in B2W — worker register)",
                  **FONT_LABEL)
    ax.set_ylabel("log₁₀(relative frequency in B2B — client register)",
                  **FONT_LABEL)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.4, linestyle=":")
    apply_base_style(ax, bg_colour)

    # Legend
    legend_entries = [
        plt.Line2D([0], [0], color=C_SUBTEXT, linestyle="--", alpha=0.6,
                   label="Equal frequency (y = x)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C_GRID, markersize=7, alpha=0.6,
                   label=f"All terms (n={len(bg_x)+sum(len(d['x']) for d in hyp_data.values()):,})"),
    ]
    for hyp_key, cfg in HYPOTHESIS_VOCAB.items():
        short = hyp_key.split("—")[0].strip()
        legend_entries.append(
            plt.Line2D([0], [0], marker=cfg["marker"], color="w",
                       markerfacecolor=cfg["color"], markersize=9,
                       label=short)
        )
    ax.legend(handles=legend_entries, loc="upper left", frameon=True,
              fontsize=8.5, facecolor=bg_colour, edgecolor=C_GRID)

    ax.set_title(
        "Register Gap: Full Vocabulary Distribution by Audience Frequency",
        **FONT_TITLE, pad=12)
    fig.text(
        0.5, -0.02,
        "Each dot = one unigram  •  Above diagonal = B2B-distinctive  •  "
        "Below = B2W-distinctive  •  "
        "Hypothesis vocabulary should cluster in predicted regions",
        ha="center", **FONT_ANNOT)
    save(fig, "figS1_register_scatter", style)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Generate all 12 Step 1 figures (6 × 2 styles).

    Verifies required DB tables exist before generating any figures.
    Iterates over both styles (pub, exp) and calls each figure function.
    """
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
        fig_theory_cooccurrence(conn, style)
        fig_register_scatter(conn, style)   # NEW: full vocabulary register gap

    conn.close()
    log.info("=" * 60)
    log.info(f"All 12 figures saved to {OUTPUT_DIR.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
