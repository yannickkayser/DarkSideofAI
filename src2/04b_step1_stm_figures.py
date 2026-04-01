
"""
04b_step1_stm_figures.py
========================
STM visualisations for the DarkSideofAI thesis.

Reads from the SQLite tables populated by 03b_import_stm.py and produces
four publication-ready figures in the same visual style as 04_step1_figures.py.

Prerequisites:
  STMAnalysis/04_export.R  → writes stm_*.csv to STMAnalysis/output/step_1/stm/
  src2/03b_import_stm.py   → loads those CSVs into scraping_2.db as:
      stm_theta        (long format: one row per page × topic)
      stm_topic_terms  (top-N terms per topic)
      stm_prevalence   (audience prevalence effects)
      stm_content      (audience-specific terms, optional)

Figures produced (output/step_1/stm/):
  STM_A  — Topic overview bar chart
            All K topics ranked by expected corpus proportion, coloured by
            dominant audience, labelled with top-5 FREX terms.

  STM_B  — Audience separation forest plot
            Prevalence coefficients (worker vs client) with 95% CI per topic.
            Significant topics are labelled; direction encoded by colour.

  STM_C  — Hypothesis alignment heatmap
            Topics × hypotheses FREX-overlap matrix showing which topics
            map onto H1a–H5 theory vocabulary.

  STM_D  — Domain × topic heatmap
            Mean topic proportion per platform domain, revealing whether
            results are cross-platform or driven by a single source.

Dual style (same convention as 04_step1_figures.py):
  pub — clean thesis-ready (white background, minimal annotations)
  exp — exploratory (light background, value labels, centroids)

Usage (from project root):
    python3 src2/04b_step1_stm_figures.py

    # To run only specific figures:
    FIGURES = {"STM_A": True, "STM_B": True, "STM_C": False, "STM_D": False}
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH    = "data/scraping_2.db"
OUTPUT_DIR = Path("output/step_1/stm")
DPI        = 150
EXT        = "jpg"

# Toggle individual figures
FIGURES = {
    "STM_A": True,   # topic overview
    "STM_B": True,   # audience separation
    "STM_C": True,   # hypothesis alignment
    "STM_D": True,   # domain × topic heatmap
}

# How many FREX terms to show as labels on figures
LABEL_FREX_N = 5

# Minimum mean topic proportion to include a topic in STM_D (avoids clutter)
DOMAIN_MIN_PROP = 0.01

# ---------------------------------------------------------------------------
# Colour palette — identical to 04_step1_figures.py
# ---------------------------------------------------------------------------

C_CLIENT  = "#1B4F8A"
C_WORKER  = "#C0392B"
C_SHARED  = "#6C757D"
C_H1C     = "#E67E22"
C_BG_PUB  = "#FFFFFF"
C_BG_EXP  = "#F7F9FC"
C_GRID    = "#DEE2E6"
C_TEXT    = "#1A1A2E"
C_SUBTEXT = "#6C757D"

FONT_TITLE = {"fontsize": 13, "fontweight": "bold",   "color": C_TEXT}
FONT_LABEL = {"fontsize": 10, "fontweight": "normal", "color": C_SUBTEXT}
FONT_ANNOT = {"fontsize":  8, "color": C_SUBTEXT}

# Hypothesis vocabulary and display colours (mirrors 00_config.R + 04_step1_figures.py)
HYP_VOCAB = {
    "H1a": {"terms": {"worker","labour","task","job","earn","pay","payment",
                       "annotator","gig","contractor","wage","freelance"},
             "color": C_WORKER},
    "H1b": {"terms": {"autonomous","machine","automate","automation","algorithm",
                       "pipeline","deploy","inference","neural","llm","intelligent"},
             "color": C_CLIENT},
    "H1c": {"terms": {"human","quality","oversight","annotation","label",
                       "expert","accuracy","datum","review","verification"},
             "color": C_H1C},
    "H2":  {"terms": {"flexible","flexibility","freedom","schedule","anytime","balance"},
             "color": "#8E44AD"},
    "H3":  {"terms": {"income","opportunity","skill","earn","grow","career","develop"},
             "color": "#27AE60"},
    "H4":  {"terms": {"community","connect","together","support","network","belong"},
             "color": "#2980B9"},
    "H5":  {"terms": {"global","worldwide","region","south","africa","india",
                       "philippines","kenya","nigeria","pakistan"},
             "color": "#D35400"},
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
# Style helpers
# ---------------------------------------------------------------------------

def apply_base_style(ax, bg=C_BG_PUB):
    ax.set_facecolor(bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=C_GRID, linewidth=0.6)


def save(fig, name: str, style: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_{style}.{EXT}"
    fig.savefig(str(path), dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info("  Saved: %s", path)
    plt.close(fig)


def shorten_domain(d: str) -> str:
    for ext in [".com", ".ai", ".net", ".org", ".tech", ".me", ".io"]:
        d = d.replace(ext, "")
    return d.replace("www.", "")


# ---------------------------------------------------------------------------
# Data loading — queries scraping_2.db (populated by 03b_import_stm.py)
# ---------------------------------------------------------------------------

def require_table(con: sqlite3.Connection, name: str) -> None:
    exists = con.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone()
    if not exists:
        raise RuntimeError(
            f"Table '{name}' not found in {DB_PATH}.\n"
            "  -> Run src2/03b_import_stm.py first."
        )


def load_stm_data(con: sqlite3.Connection) -> dict:
    """
    Load all STM data from SQLite into the same in-memory structure that the
    figure functions expect.  Returns a dict with keys:

        K          -- int, number of topics
        terms      -- dict[topic_id] = {"frex": [...], "prob": [...]}
        prevalence -- dict[topic_id] = {estimate, ci_lower, ci_upper,
                                         significant, direction, frex_label}
        theta      -- list of dicts  {page_id, audience, domain,
                                      props: [float x K], dominant: int}
        content    -- dict[topic_id][audience] = [term, ...]  (may be empty)

    Note: stm_theta is stored in long format (one row per page x topic).
    This function pivots it back to a per-document props list by grouping
    on page_id ordered by topic_id.
    """
    for t in ("stm_theta", "stm_topic_terms", "stm_prevalence"):
        require_table(con, t)

    # -- K (number of topics) ------------------------------------------------
    K = con.execute(
        "SELECT MAX(topic_id) FROM stm_topic_terms"
    ).fetchone()[0]
    if K is None:
        raise RuntimeError("stm_topic_terms is empty -- re-run 03b_import_stm.py.")

    # -- Topic terms ---------------------------------------------------------
    terms = defaultdict(lambda: {"frex": [], "prob": []})
    for row in con.execute(
        "SELECT topic_id, prob_term, frex_term "
        "FROM stm_topic_terms ORDER BY topic_id, rank"
    ):
        topic_id, prob_term, frex_term = row
        terms[topic_id]["frex"].append(frex_term or "")
        terms[topic_id]["prob"].append(prob_term or "")

    # -- Prevalence effects --------------------------------------------------
    prevalence = {}
    for row in con.execute(
        "SELECT topic_id, frex_label, estimate, ci_lower, ci_upper, "
        "       significant, direction "
        "FROM stm_prevalence"
    ):
        topic_id, frex_label, estimate, ci_lower, ci_upper, significant, direction = row
        prevalence[topic_id] = {
            "estimate":    estimate  or 0.0,
            "ci_lower":    ci_lower  or 0.0,
            "ci_upper":    ci_upper  or 0.0,
            "significant": bool(significant),
            "direction":   direction or "",
            "frex_label":  frex_label or "",
        }

    # -- Theta: pivot long -> wide (one entry per document) ------------------
    #
    # stm_theta stores one row per (page_id, topic_id).
    # Rows arrive ordered by page_id, topic_id so we can stream-group them.

    theta_rows = []
    current_page  = None
    current_entry = None

    for row in con.execute(
        "SELECT page_id, audience, domain, topic_id, theta, "
        "       dominant_topic, dominant_prop "
        "FROM stm_theta "
        "ORDER BY page_id, topic_id"
    ):
        page_id, audience, domain, topic_id, theta_val, dominant_topic, dominant_prop = row

        if page_id != current_page:
            if current_entry is not None:
                theta_rows.append(current_entry)
            current_page  = page_id
            current_entry = {
                "page_id":  page_id,
                "audience": audience,
                "domain":   domain,
                "props":    [],
                "dominant": dominant_topic,
            }
        current_entry["props"].append(theta_val or 0.0)

    if current_entry is not None:
        theta_rows.append(current_entry)

    # -- Content covariate terms (optional) ----------------------------------
    content = {}
    content_exists = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stm_content'"
    ).fetchone()
    if content_exists:
        for row in con.execute(
            "SELECT topic_id, audience, term "
            "FROM stm_content ORDER BY topic_id, audience, rank"
        ):
            topic_id, audience, term = row
            content.setdefault(topic_id, {}).setdefault(audience, []).append(term or "")

    log.info("  K            : %d", K)
    log.info("  Documents    : %d", len(theta_rows))
    log.info("  Topics w/ prevalence : %d", len(prevalence))
    log.info("  Content table: %s", "yes" if content_exists else "no")

    return {
        "K":          K,
        "terms":      dict(terms),
        "prevalence": prevalence,
        "theta":      theta_rows,
        "content":    content,
    }


# ---------------------------------------------------------------------------
# Figure STM_A -- Topic overview bar chart
# ---------------------------------------------------------------------------

def fig_stm_topic_overview(data: dict, style: str):
    """
    Horizontal bar chart of all K topics ranked by expected corpus proportion.

    Colour encodes the dominant audience direction (from prevalence estimates):
      Blue  = significantly more client
      Red   = significantly more worker
      Grey  = no significant audience separation

    Label shows top-5 FREX terms to aid interpretation.
    """
    K          = data["K"]
    terms      = data["terms"]
    prevalence = data["prevalence"]
    theta      = data["theta"]

    mean_props = [
        sum(row["props"][t - 1] for row in theta) / len(theta)
        for t in range(1, K + 1)
    ]

    rows = []
    for t in range(1, K + 1):
        prev  = prevalence.get(t, {})
        sig   = prev.get("significant", False)
        dirn  = prev.get("direction", "")
        color = (C_CLIENT if sig and dirn == "client"
                 else C_WORKER if sig and dirn == "worker"
                 else C_SHARED)
        frex5 = ", ".join(terms.get(t, {}).get("frex", [])[:LABEL_FREX_N])
        rows.append({
            "topic": t,
            "prop":  mean_props[t - 1],
            "color": color,
            "frex5": frex5,
            "sig":   sig,
            "dirn":  dirn,
        })

    rows.sort(key=lambda r: r["prop"])   # ascending so highest is at top

    bg = C_BG_EXP if style == "exp" else C_BG_PUB
    fig, ax = plt.subplots(figsize=(10, max(6, K * 0.35)))
    fig.patch.set_facecolor(bg)
    apply_base_style(ax, bg)

    y_pos  = range(len(rows))
    colors = [r["color"] for r in rows]
    props  = [r["prop"]  for r in rows]

    bars = ax.barh(list(y_pos), props, color=colors, alpha=0.82,
                   edgecolor="none", height=0.7)

    for i, (bar, row) in enumerate(zip(bars, rows)):
        ax.text(bar.get_width() + 0.001, i,
                f"T{row['topic']:02d}: {row['frex5']}",
                va="center", **FONT_ANNOT)

    if style == "exp":
        for i, (bar, row) in enumerate(zip(bars, rows)):
            ax.text(bar.get_width() / 2, i,
                    f"{row['prop']:.3f}",
                    va="center", ha="center",
                    fontsize=7, color="white", fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("Expected topic proportion", **FONT_LABEL)
    ax.set_title(f"Final STM (K={K}) — Topic Proportions", **FONT_TITLE)
    ax.set_xlim(0, max(props) * 1.55)

    legend_patches = [
        mpatches.Patch(color=C_CLIENT, label="Client-dominant (sig.)"),
        mpatches.Patch(color=C_WORKER, label="Worker-dominant (sig.)"),
        mpatches.Patch(color=C_SHARED, label="No significant separation"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=8, framealpha=0.8)

    fig.tight_layout()
    save(fig, "STM_A_topic_overview", style)


# ---------------------------------------------------------------------------
# Figure STM_B -- Audience separation forest plot
# ---------------------------------------------------------------------------

def fig_stm_audience_forest(data: dict, style: str):
    """
    Forest plot of audience prevalence effects (worker vs client coefficient).

    Positive estimate = topic more prevalent in worker pages.
    Negative estimate = topic more prevalent in client pages.
    Significant topics are labelled with their top FREX term.
    """
    prevalence = data["prevalence"]
    terms      = data["terms"]
    K          = data["K"]

    rows = []
    for t in range(1, K + 1):
        p = prevalence.get(t)
        if p is None:
            continue
        rows.append({
            "topic": t,
            "est":   p["estimate"],
            "lo":    p["ci_lower"],
            "hi":    p["ci_upper"],
            "sig":   p["significant"],
            "dirn":  p["direction"],
            "frex1": terms.get(t, {}).get("frex", ["?"])[0],
        })

    rows.sort(key=lambda r: r["est"])

    bg = C_BG_EXP if style == "exp" else C_BG_PUB
    fig, ax = plt.subplots(figsize=(9, max(6, len(rows) * 0.38)))
    fig.patch.set_facecolor(bg)
    apply_base_style(ax, bg)
    ax.grid(axis="x", color=C_GRID, linewidth=0.6)
    ax.axvline(0, color=C_TEXT, linewidth=0.8, linestyle="--", alpha=0.5)

    for i, row in enumerate(rows):
        color = (C_CLIENT if row["sig"] and row["dirn"] == "client"
                 else C_WORKER if row["sig"] and row["dirn"] == "worker"
                 else C_SHARED)
        alpha = 1.0 if row["sig"] else 0.4
        ax.plot([row["lo"], row["hi"]], [i, i],
                color=color, linewidth=1.2, alpha=alpha)
        ax.scatter([row["est"]], [i], color=color, s=35, zorder=3, alpha=alpha)
        if row["sig"]:
            ax.text(row["hi"] + 0.002, i, row["frex1"],
                    va="center", fontsize=7.5, color=color)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"T{r['topic']:02d}" for r in rows], fontsize=8)
    ax.set_xlabel("Prevalence coefficient  (positive = worker, negative = client)",
                  **FONT_LABEL)
    ax.set_title(f"STM Audience Separation — K={K}  (95% CI)", **FONT_TITLE)

    n_sig = sum(r["sig"] for r in rows)
    ax.annotate(f"{n_sig}/{len(rows)} topics significant",
                xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", **FONT_ANNOT)

    legend_patches = [
        mpatches.Patch(color=C_CLIENT, label="Client-dominant"),
        mpatches.Patch(color=C_WORKER, label="Worker-dominant"),
        mpatches.Patch(color=C_SHARED, alpha=0.4, label="Not significant"),
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              fontsize=8, framealpha=0.8)

    fig.tight_layout()
    save(fig, "STM_B_audience_forest", style)


# ---------------------------------------------------------------------------
# Figure STM_C -- Hypothesis alignment heatmap
# ---------------------------------------------------------------------------

def fig_stm_hypothesis_heatmap(data: dict, style: str):
    """
    Heatmap of FREX-term overlap between each topic and each hypothesis vocab.

    Cell value = number of top-20 FREX terms matching the hypothesis vocabulary.
    """
    K          = data["K"]
    terms      = data["terms"]
    prevalence = data["prevalence"]
    hyp_keys   = list(HYP_VOCAB.keys())

    matrix = np.zeros((K, len(hyp_keys)), dtype=int)
    for t in range(1, K + 1):
        frex20 = {w.lower() for w in terms.get(t, {}).get("frex", [])}
        for j, hyp in enumerate(hyp_keys):
            matrix[t - 1, j] = len(frex20 & HYP_VOCAB[hyp]["terms"])

    best_col = matrix.argmax(axis=1)
    best_val = matrix.max(axis=1)
    order    = sorted(range(K), key=lambda i: (best_col[i], -best_val[i]))

    mat_sorted   = matrix[order, :]
    topic_labels = []
    for idx in order:
        t    = idx + 1
        prev = prevalence.get(t, {})
        dirn = prev.get("direction", "")[0].upper() if prev.get("significant") else "·"
        topic_labels.append(f"T{t:02d} [{dirn}]")

    bg = C_BG_EXP if style == "exp" else C_BG_PUB
    fig, ax = plt.subplots(figsize=(len(hyp_keys) * 1.2 + 2, max(6, K * 0.38)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    vmax = max(mat_sorted.max(), 1)
    for j, hyp in enumerate(hyp_keys):
        col_color = matplotlib.colors.to_rgb(HYP_VOCAB[hyp]["color"])
        for i in range(K):
            val   = mat_sorted[i, j]
            alpha = val / vmax
            rect  = plt.Rectangle([j - 0.5, i - 0.5], 1, 1,
                                   facecolor=col_color + (alpha,),
                                   edgecolor="white", linewidth=0.8)
            ax.add_patch(rect)
            if val > 0:
                txt_color = "white" if alpha > 0.5 else C_TEXT
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

    ax.set_xlim(-0.5, len(hyp_keys) - 0.5)
    ax.set_ylim(-0.5, K - 0.5)
    ax.set_xticks(range(len(hyp_keys)))
    ax.set_xticklabels(hyp_keys, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(K))
    ax.set_yticklabels(topic_labels, fontsize=8)
    ax.set_xlabel("Hypothesis", **FONT_LABEL)
    ax.set_title("Hypothesis vocabulary alignment\n"
                 "(FREX overlap count, top-20 terms per topic)",
                 **FONT_TITLE)

    legend_patches = [
        mpatches.Patch(color=HYP_VOCAB[h]["color"], label=h) for h in hyp_keys
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=7.5, framealpha=0.8, ncol=2)

    ax.spines[:].set_visible(False)
    fig.tight_layout()
    save(fig, "STM_C_hypothesis_alignment", style)


# ---------------------------------------------------------------------------
# Figure STM_D -- Domain x topic heatmap
# ---------------------------------------------------------------------------

def fig_stm_domain_topic(data: dict, style: str):
    """
    Heatmap of mean topic proportion per platform domain.

    Rows   = platforms (sorted by audience: client -> both -> worker)
    Columns = topics (sorted by mean proportion, most prevalent first)
    Cell   = mean theta value for that domain x topic combination
    """
    theta = data["theta"]
    K     = data["K"]

    domain_counts = defaultdict(int)
    domain_sums   = defaultdict(lambda: [0.0] * K)
    domain_aud    = {}

    for row in theta:
        d = row["domain"]
        domain_counts[d] += 1
        domain_aud[d] = row["audience"]
        for t in range(K):
            domain_sums[d][t] += row["props"][t]

    domains = sorted(domain_counts.keys())
    means   = {d: [domain_sums[d][t] / domain_counts[d] for t in range(K)]
               for d in domains}

    topic_means = [sum(means[d][t] for d in domains) / len(domains)
                   for t in range(K)]
    keep_topics = [t for t in range(K) if topic_means[t] >= DOMAIN_MIN_PROP]
    keep_topics.sort(key=lambda t: -topic_means[t])

    aud_order = {"client": 0, "both": 1, "worker": 2}
    domains   = sorted(domains,
                       key=lambda d: (aud_order.get(domain_aud.get(d, "both"), 1),
                                      shorten_domain(d)))
    mat = np.array([[means[d][t] for t in keep_topics] for d in domains])

    bg = C_BG_EXP if style == "exp" else C_BG_PUB
    fig, ax = plt.subplots(figsize=(max(8, len(keep_topics) * 0.55),
                                     max(5, len(domains) * 0.42)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=mat.max())

    if style == "exp":
        for i in range(len(domains)):
            for j in range(len(keep_topics)):
                v = mat[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if v > mat.max() * 0.6 else C_TEXT)

    ax.set_xticks(range(len(keep_topics)))
    ax.set_xticklabels([f"T{t+1:02d}" for t in keep_topics],
                       fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([shorten_domain(d) for d in domains], fontsize=8)

    prev_aud = None
    for i, d in enumerate(domains):
        aud = domain_aud.get(d, "both")
        if prev_aud and aud != prev_aud:
            ax.axhline(i - 0.5, color=C_GRID, linewidth=1.5)
        prev_aud = aud

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(domains)))
    ax2.set_yticklabels(
        [domain_aud.get(d, "")[:1].upper() for d in domains],
        fontsize=7, color=C_SUBTEXT,
    )
    ax2.spines[:].set_visible(False)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Mean topic proportion")

    ax.set_title("Domain x Topic distribution\n"
                 "(mean STM topic proportion per platform)",
                 **FONT_TITLE)
    ax.set_xlabel("Topic (ranked by corpus proportion)", **FONT_LABEL)
    ax.set_ylabel("Platform domain  [C=client  B=both  W=worker]", **FONT_LABEL)
    ax.spines[:].set_visible(False)

    fig.tight_layout()
    save(fig, "STM_D_domain_topic", style)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("04b_step1_stm_figures.py -- STM visualisations")
    log.info("  Database   : %s", Path(DB_PATH).resolve())
    log.info("  Output dir : %s", OUTPUT_DIR)
    log.info("=" * 60)

    with sqlite3.connect(DB_PATH) as con:
        data = load_stm_data(con)

        for style in ("pub", "exp"):
            log.info("-" * 40)
            log.info("Style: %s", style)

            if FIGURES.get("STM_A"):
                log.info("STM_A -- Topic overview")
                fig_stm_topic_overview(data, style)

            if FIGURES.get("STM_B"):
                log.info("STM_B -- Audience forest plot")
                fig_stm_audience_forest(data, style)

            if FIGURES.get("STM_C"):
                log.info("STM_C -- Hypothesis alignment heatmap")
                fig_stm_hypothesis_heatmap(data, style)

            if FIGURES.get("STM_D"):
                log.info("STM_D -- Domain x topic heatmap")
                fig_stm_domain_topic(data, style)

    log.info("=" * 60)
    log.info("Done.  Figures written to %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
