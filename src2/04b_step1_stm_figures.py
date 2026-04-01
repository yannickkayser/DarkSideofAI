"""
04b_step1_stm_figures.py
========================
STM visualisations for the DarkSideofAI thesis.

Reads the CSV exports produced by STMAnalysis/04_export.R and produces
four publication-ready figures in the same visual style as 04_step1_figures.py.

Prerequisites:
  STMAnalysis/03_fit_model.R  → stm_model.rds, prev_df.rds
  STMAnalysis/04_export.R     → stm_theta.csv, stm_topic_terms.csv,
                                 stm_prevalence.csv  (stm_content.csv optional)

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

import csv
import logging
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STM_DIR    = Path("STMAnalysis/output/step_1/stm")   # R export location
OUTPUT_DIR = Path("output/step_1/stm")                # Python figure output
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
# Helpers
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
    log.info(f"  Saved: {path}")
    plt.close(fig)


def load_csv(filename: str) -> list[dict]:
    """Load a CSV from the STM export directory."""
    path = STM_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"STM export file not found: {path}\n"
            f"Run STMAnalysis/04_export.R first."
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def shorten_domain(d: str) -> str:
    for ext in [".com", ".ai", ".net", ".org", ".tech", ".me", ".io"]:
        d = d.replace(ext, "")
    return d.replace("www.", "")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stm_data() -> dict:
    """
    Load all STM export CSVs into structured dicts.
    Returns a dict with keys: theta, terms, prevalence, content (optional).
    """
    log.info("Loading STM export CSVs from %s", STM_DIR)

    # ── Topic terms ──────────────────────────────────────────────────────────
    raw_terms = load_csv("stm_topic_terms.csv")
    terms = defaultdict(lambda: {"frex": [], "prob": []})
    for row in raw_terms:
        t = int(row["topic_id"])
        terms[t]["frex"].append(row["frex_term"])
        terms[t]["prob"].append(row["prob_term"])

    # ── Prevalence effects ───────────────────────────────────────────────────
    raw_prev  = load_csv("stm_prevalence.csv")
    prevalence = {}
    for row in raw_prev:
        t = int(row["topic_id"])
        prevalence[t] = {
            "estimate":    float(row["estimate"]),
            "ci_lower":    float(row["ci_lower"]),
            "ci_upper":    float(row["ci_upper"]),
            "significant": row["significant"].strip().upper() in ("TRUE", "1"),
            "direction":   row["direction"],
            "frex_label":  row["frex_label"],
        }

    # ── Document-topic proportions (theta) ───────────────────────────────────
    raw_theta = load_csv("stm_theta.csv")
    K = max(int(k.replace("topic_", ""))
            for k in raw_theta[0].keys() if k.startswith("topic_"))

    theta_rows = []
    for row in raw_theta:
        entry = {
            "page_id":  row["page_id"],
            "audience": row["audience"],
            "domain":   row["domain"],
            "props":    [float(row[f"topic_{t}"]) for t in range(1, K + 1)],
            "dominant": int(row["dominant_topic"]),
        }
        theta_rows.append(entry)

    # ── Content covariate (optional) ─────────────────────────────────────────
    content_path = STM_DIR / "stm_content.csv"
    content = {}
    if content_path.exists():
        for row in load_csv("stm_content.csv"):
            t   = int(row["topic_id"])
            aud = row["audience"]
            content.setdefault(t, {})[aud] = content.get(t, {}).get(aud, [])
            content[t][aud].append(row["term"])

    log.info("  Topics (K)   : %d", K)
    log.info("  Documents    : %d", len(theta_rows))
    log.info("  Prevalence   : %d topics with estimates", len(prevalence))

    return {"K": K, "terms": dict(terms), "prevalence": prevalence,
            "theta": theta_rows, "content": content}


# ---------------------------------------------------------------------------
# Figure STM_A — Topic overview bar chart
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

    # Expected proportion = mean of each topic column across all documents
    mean_props = [
        sum(row["props"][t - 1] for row in theta) / len(theta)
        for t in range(1, K + 1)
    ]

    # Build rows sorted by proportion descending
    rows = []
    for t in range(1, K + 1):
        prev  = prevalence.get(t, {})
        sig   = prev.get("significant", False)
        dirn  = prev.get("direction", "—")
        color = (C_CLIENT if sig and dirn == "client"
                 else C_WORKER if sig and dirn == "worker"
                 else C_SHARED)
        frex5 = ", ".join(terms.get(t, {}).get("frex", [])[:LABEL_FREX_N])
        rows.append({
            "topic":  t,
            "prop":   mean_props[t - 1],
            "color":  color,
            "frex5":  frex5,
            "sig":    sig,
            "dirn":   dirn,
        })

    rows.sort(key=lambda r: r["prop"])   # ascending so top is at top of chart

    bg = C_BG_EXP if style == "exp" else C_BG_PUB
    fig, ax = plt.subplots(figsize=(10, max(6, K * 0.35)))
    fig.patch.set_facecolor(bg)
    apply_base_style(ax, bg)

    y_pos  = range(len(rows))
    colors = [r["color"] for r in rows]
    props  = [r["prop"]  for r in rows]

    bars = ax.barh(list(y_pos), props, color=colors, alpha=0.82,
                   edgecolor="none", height=0.7)

    # Labels: "T07: frex1, frex2, ..."
    for i, (bar, row) in enumerate(zip(bars, rows)):
        label = f"T{row['topic']:02d}: {row['frex5']}"
        ax.text(bar.get_width() + 0.001, i, label,
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
# Figure STM_B — Audience separation forest plot
# ---------------------------------------------------------------------------

def fig_stm_audience_forest(data: dict, style: str):
    """
    Forest plot of audience prevalence effects (worker vs client coefficient).

    Each topic is a point with 95% CI bars.
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
            "topic":    t,
            "est":      p["estimate"],
            "lo":       p["ci_lower"],
            "hi":       p["ci_upper"],
            "sig":      p["significant"],
            "dirn":     p["direction"],
            "frex1":    (terms.get(t, {}).get("frex", ["?"])[0]),
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
        ax.scatter([row["est"]], [i],
                   color=color, s=35, zorder=3, alpha=alpha)

        if row["sig"]:
            x_off = row["hi"] + 0.002
            ax.text(x_off, i, row["frex1"],
                    va="center", fontsize=7.5, color=color)

    y_labels = [f"T{r['topic']:02d}" for r in rows]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_xlabel("Prevalence coefficient  (positive = worker, negative = client)",
                  **FONT_LABEL)
    ax.set_title(f"STM Audience Separation — K={K}  (95% CI)",
                 **FONT_TITLE)

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
# Figure STM_C — Hypothesis alignment heatmap
# ---------------------------------------------------------------------------

def fig_stm_hypothesis_heatmap(data: dict, style: str):
    """
    Heatmap of FREX-term overlap between each topic and each hypothesis vocab.

    Cell value = number of top-20 FREX terms matching the hypothesis vocabulary.
    Cell colour scales from white (0) to the hypothesis colour (max overlap).
    Rows are sorted by dominant hypothesis (highest overlap column).
    """
    K          = data["K"]
    terms      = data["terms"]
    prevalence = data["prevalence"]
    hyp_keys   = list(HYP_VOCAB.keys())

    # Build overlap matrix  (K × len(hyp_keys))
    matrix = np.zeros((K, len(hyp_keys)), dtype=int)
    for t in range(1, K + 1):
        frex20 = {w.lower() for w in terms.get(t, {}).get("frex", [])}
        for j, hyp in enumerate(hyp_keys):
            matrix[t - 1, j] = len(frex20 & HYP_VOCAB[hyp]["terms"])

    # Sort topics: first by max overlap column, then by max overlap value desc
    best_col = matrix.argmax(axis=1)
    best_val = matrix.max(axis=1)
    order    = sorted(range(K),
                      key=lambda i: (best_col[i], -best_val[i]))

    mat_sorted   = matrix[order, :]
    topic_labels = []
    for idx in order:
        t    = idx + 1
        prev = prevalence.get(t, {})
        dirn = prev.get("direction", "")[0].upper() if prev.get("significant") else "·"
        topic_labels.append(f"T{t:02d} [{dirn}]")

    bg = C_BG_EXP if style == "exp" else C_BG_PUB
    fig, ax = plt.subplots(figsize=(len(hyp_keys) * 1.2 + 2,
                                     max(6, K * 0.38)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    # Draw cells manually so each hypothesis column can have its own colour
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

    # Colour legend
    legend_patches = [
        mpatches.Patch(color=HYP_VOCAB[h]["color"], label=h)
        for h in hyp_keys
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              fontsize=7.5, framealpha=0.8, ncol=2)

    ax.spines[:].set_visible(False)
    fig.tight_layout()
    save(fig, "STM_C_hypothesis_alignment", style)


# ---------------------------------------------------------------------------
# Figure STM_D — Domain × topic heatmap
# ---------------------------------------------------------------------------

def fig_stm_domain_topic(data: dict, style: str):
    """
    Heatmap of mean topic proportion per platform domain.

    Reveals whether STM results are driven by a single platform or
    represent genuine cross-platform discourse patterns.

    Rows   = platforms (sorted by audience: client → both → worker)
    Columns = topics (sorted by mean proportion, most prevalent first)
    Cell   = mean theta value for that domain × topic combination
    """
    theta = data["theta"]
    K     = data["K"]

    # Aggregate mean proportion per domain per topic
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
    means   = {d: [domain_sums[d][t] / domain_counts[d]
                   for t in range(K)]
               for d in domains}

    # Only keep topics with mean prop > threshold (avoids blank columns)
    topic_means = [sum(means[d][t] for d in domains) / len(domains)
                   for t in range(K)]
    keep_topics = [t for t in range(K) if topic_means[t] >= DOMAIN_MIN_PROP]
    keep_topics.sort(key=lambda t: -topic_means[t])

    # Build matrix
    mat = np.array([[means[d][t] for t in keep_topics] for d in domains])

    # Sort domains by audience label (client first, then both, then worker)
    aud_order = {"client": 0, "both": 1, "worker": 2}
    domains   = sorted(domains,
                       key=lambda d: (aud_order.get(domain_aud.get(d, "both"), 1),
                                      shorten_domain(d)))
    mat       = np.array([[means[d][t] for t in keep_topics] for d in domains])

    col_labels = [f"T{t+1:02d}" for t in keep_topics]
    row_labels = [shorten_domain(d) for d in domains]

    bg = C_BG_EXP if style == "exp" else C_BG_PUB
    fig, ax = plt.subplots(figsize=(max(8, len(keep_topics) * 0.55),
                                     max(5, len(domains) * 0.42)))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    im = ax.imshow(mat, aspect="auto", cmap="Blues",
                   vmin=0, vmax=mat.max())

    if style == "exp":
        for i in range(len(domains)):
            for j in range(len(keep_topics)):
                v = mat[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if v > mat.max() * 0.6 else C_TEXT)

    ax.set_xticks(range(len(keep_topics)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Audience separator lines
    prev_aud = None
    for i, d in enumerate(domains):
        aud = domain_aud.get(d, "both")
        if prev_aud and aud != prev_aud:
            ax.axhline(i - 0.5, color=C_GRID, linewidth=1.5)
        prev_aud = aud

    # Audience labels on right y-axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(domains)))
    ax2.set_yticklabels(
        [domain_aud.get(d, "")[:1].upper() for d in domains],
        fontsize=7, color=C_SUBTEXT
    )
    ax2.spines[:].set_visible(False)

    plt.colorbar(im, ax=ax, shrink=0.6, label="Mean topic proportion")

    ax.set_title("Domain × Topic distribution\n"
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
    log.info("04b_step1_stm_figures.py — STM visualisations")
    log.info(f"  STM exports : {STM_DIR}")
    log.info(f"  Output dir  : {OUTPUT_DIR}")
    log.info("=" * 60)

    data = load_stm_data()

    for style in ("pub", "exp"):
        log.info("-" * 40)
        log.info("Style: %s", style)

        if FIGURES.get("STM_A"):
            log.info("STM_A — Topic overview")
            fig_stm_topic_overview(data, style)

        if FIGURES.get("STM_B"):
            log.info("STM_B — Audience forest plot")
            fig_stm_audience_forest(data, style)

        if FIGURES.get("STM_C"):
            log.info("STM_C — Hypothesis alignment heatmap")
            fig_stm_hypothesis_heatmap(data, style)

        if FIGURES.get("STM_D"):
            log.info("STM_D — Domain × topic heatmap")
            fig_stm_domain_topic(data, style)

    log.info("=" * 60)
    log.info("Done.  Figures written to %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
