"""
06_stm_results_figures.py
=========================
Chapter 4 STM results visualisation and export.

This is the "06_step1_stm_figures.py" script referenced (as future work)
in 02d_step1_stm.py. It reads the four STM result tables imported by
02d_step1_stm.py and produces everything Chapter 4 needs that the earlier
visualisation scripts do not cover.

Pipeline position
-----------------
  Stage 6 — STM results (run AFTER 03b_import_stm.py has completed)
  Prerequisites:
    stm_prevalence         — estimateEffect results (from 03b_import_stm.py)
    stm_topic_terms        — FREX / Prob terms per topic (from 03b_import_stm.py)
    stm_theta              — per-document topic proportions (from 03b_import_stm.py)
    stm_topic_profile      — convenience view (from 03b_import_stm.py)
    cooccurrence_results   — PMI profiles (from 02_step1_frequency.py)
    corpus_view            — raw pages with text (from 01_prepare_corpus.py)

Outputs (written to output/step_1/stm/)
----------------------------------------
  Figures (publication-ready at 300 dpi, PDF vector):
    fig_stm_effects_overview.pdf  — all 25 topics, sorted by estimate
    fig_stm_effects_h1a.pdf       — labour-centred topics (H1a evidence)
    fig_stm_effects_h1b.pdf       — automation-centred topics (H1b evidence)
    fig_stm_effects_h1c.pdf       — quality-signal topics (H1c evidence)
    fig_cooc_human_h1c.pdf        — 'human' PMI profile B2B vs B2W side-by-side

  Exploratory variants (PNG, 150 dpi, annotated):
    fig_stm_effects_overview_exp.png
    fig_stm_effects_h1a_exp.png
    ... (one per figure above)

  Tables:
    topic_table.tex    — LaTeX table: all 25 topics, FREX, estimate, CI
    topic_table.csv    — same, CSV for inspection

  KWIC (Key Words In Context / findThoughts output):
    kwic_h1a.txt       — top representative passages for H1a topics
    kwic_h1b.txt       — top representative passages for H1b topics
    kwic_h1c.txt       — top representative passages for H1c topics
    kwic_all.txt       — all topics combined, for close reading

Schema (matches 03b_import_stm.py exactly — no changes needed)
--------------------------------------------------------------
  stm_prevalence:  topic_id, frex_label, estimate, std_err,
                   ci_lower, ci_upper, significant (0/1), direction
  stm_topic_terms: topic_id, rank, prob_term, frex_term
  stm_theta:       page_id, audience, domain, topic_id, theta,
                   dominant_topic, dominant_prop  [LONG format]
  stm_topic_profile (VIEW): page_id, audience, domain, platform_type,
                   hq_region, company_id, topic_id, theta, dominant_topic,
                   dominant_prop, frex_label, prevalence_estimate,
                   prevalence_significant, prevalence_direction

Customisation
-------------
  TOPIC_LABELS      — interpretive labels (fall back to frex_label from DB
                      for any topic not listed here; run once with empty dict
                      to see what R already exported)
  TOPIC_HYPOTHESIS  — classify each topic as 'H1a', 'H1b', 'H1c', or None
  KWIC_TOP_N        — how many documents to retrieve per topic for KWIC
  H1C_FOCUS_TERMS   — terms to profile in the human co-occurrence comparison

Usage
-----
    python3 src/06_stm_results_figures.py

    # To run only specific outputs (edit OUTPUTS dict at top of CONFIG):
    #   set any key to False to skip that output
"""

import sqlite3
import logging
import textwrap
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

# =============================================================================
# ── CONFIG ────────────────────────────────────────────────────────────────────
# =============================================================================

DB_PATH    = "data/scraping_2.db"      # matches 03b_import_stm.py
OUTPUT_DIR = Path("output/step_1c/stm")
DPI_PUB    = 300      # publication figures
DPI_EXP    = 150      # exploratory / annotated figures

# ── which outputs to produce ───────────────────────────────────────────────────
OUTPUTS = {
    "fig_overview":    True,   # full 25-topic prevalence plot
    "fig_h1a":         True,   # H1a labour topics
    "fig_h1b":         True,   # H1b automation topics
    "fig_h1c":         True,   # H1c quality-signal topics
    "fig_cooc_human":  True,   # human PMI comparison B2B vs B2W
    "topic_table":     True,   # LaTeX + CSV topic table
    "kwic":            True,   # KWIC / findThoughts passages
}

# ── topic labels ──────────────────────────────────────────────────────────────
# Fill in your 25 interpretive labels. Keys are topic numbers (1-based integers).
# These labels appear on figure axes and in the LaTeX table.
# For any topic NOT listed here, the script falls back to the frex_label
# column already stored in stm_prevalence (computed from your R export).
# TIP: run the script once with an empty dict {} to see the R-generated labels,
# then override only the ones you want to rename.
TOPIC_LABELS = {
    1 : "Freelance AI training work",
    2 : "Survey & market research",
    3 : "NLP & text annotation",
    4 : "B2B outsourcing content",
    5 : "Technical AI evaluation tasks",
    6 : "AI governance & oversight",
    7 : "LLM evaluation & fine-tuning",
    8 : "Cookie & tracking notices",
    9 : "Computer vision labeling",
    10 : "Privacy policy & terms of service",
    11 : "Worker payments & earnings",
    12 : "Annotation platform tools",
    13 : "Government & compliance hiring",
    14 : "Expert specialist recruitment",
    15 : "Enterprise partnerships",
    16 : "B2B workforce solutions",
    17 : "Annotation API & developer tools",
    18 : "Human-AI discourse",
    19 : "Speech & audio data collection",
    20 : "Autonomous vehicles & drone sensing",
    21 : "AI deployment & synthetic data",
    22 : "Microwork platform campaigns",
    23 : "Medical image annotation",
    24 : "Search relevance & content tagging",
    25 : "Machine learning concepts"
}

# ── hypothesis classification ─────────────────────────────────────────────────
# Classify each topic as 'H1a', 'H1b', 'H1c', or None (shared/neutral).
# These drive the colour-coding in figures and the hypothesis-grouped plots.
# Edit after inspecting your FREX terms and prevalence estimates.
TOPIC_HYPOTHESIS = {
    # H1a  — labour-centred: expect NEGATIVE estimates (more worker-facing)
    # H1b  — automation-centred: expect POSITIVE estimates (more client-facing)
    # H1c  — quality-signal: expect POSITIVE but moderate estimates
    # None — shared institutional / neutral
    1:  None,   # ← fill in: 'H1a', 'H1b', 'H1c', or None
    2:  None,
    3:  None,
    4:  None,
    5:  None,
    6:  None,
    7:  None,
    8:  None,
    9:  None,
    10: None,
    11: None,
    12: None,
    13: None,
    14: None,
    15: None,
    16: None,
    17: None,
    18: None,
    19: None,
    20: None,
    21: None,
    22: None,
    23: None,
    24: None,
    25: None,
}

# ── KWIC settings ─────────────────────────────────────────────────────────────
KWIC_TOP_N           = 5     # passages per topic
KWIC_MAX_CHARS       = 600   # max chars per passage (truncated with …)
KWIC_HYPOTHESIS_ONLY = True  # if True, only retrieve for H1a/H1b/H1c topics

# ── H1c co-occurrence: which terms to compare ────────────────────────────────
# These are the terms whose PMI profiles are compared B2B vs B2W for Fig H1c.
H1C_FOCUS_TERMS = [
    "human",
    "human_in_the_loop",
    "expert",
    "oversight",
]
COOC_TOP_N = 15    # top N collocates to show per term per audience
COOC_MIN_COFREQ = 5  # minimum co-occurrence count to display

# ── palette (matches 03_visualise_step1.py and 04_step1_narrative_figures.py) ─
PAL = {
    "H1a":     "#C0392B",   # red — labour / worker
    "H1b":     "#1B4F8A",   # blue — automation / client
    "H1c":     "#E67E22",   # orange — strategic hypervisibility
    "neutral": "#95A5A6",   # grey — shared / unclassified
    "sig_pos": "#1B4F8A",   # significant positive effect (more client)
    "sig_neg": "#C0392B",   # significant negative effect (more worker)
    "insig":   "#BDC3C7",   # credible interval crosses zero
    "zero":    "#2C3E50",   # zero reference line
}

# =============================================================================
# ── LOGGING ───────────────────────────────────────────────────────────────────
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# ── HELPERS ───────────────────────────────────────────────────────────────────
# =============================================================================

def get_conn() -> sqlite3.Connection:
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()[0] > 0


def view_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='view' AND name=?", (name,)
    ).fetchone()[0] > 0


def apply_theme():
    """Apply consistent visual theme matching existing scripts."""
    if _HAS_SNS:
        sns.set_theme(
            style="white",
            font_scale=1.0,
            rc={
                "font.family": "serif",
                "axes.spines.top":   False,
                "axes.spines.right": False,
            }
        )
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "figure.facecolor":  "white",
            "axes.facecolor":    "white",
        })


def label(topic_num: int) -> str:
    """Return interpretive label for a topic number, with fallback."""
    return TOPIC_LABELS.get(topic_num, f"Topic {topic_num}")


def hyp(topic_num: int) -> str | None:
    """Return hypothesis classification for a topic number."""
    return TOPIC_HYPOTHESIS.get(topic_num)


def node_colour(topic_num: int) -> str:
    h = hyp(topic_num)
    return PAL.get(h, PAL["neutral"]) if h else PAL["neutral"]


def is_significant(row: dict) -> bool:
    """Return True if the topic has a significant prevalence effect.
    Uses the pre-computed 'significant' flag from stm_prevalence (set in R).
    """
    return bool(row.get("significant", False))


def bar_colour(row: dict) -> str:
    if not is_significant(row):
        return PAL["insig"]
    est = row.get("estimate", 0)
    return PAL["sig_pos"] if est > 0 else PAL["sig_neg"]


# =============================================================================
# ── DATA LOADING ──────────────────────────────────────────────────────────────
# =============================================================================

def load_prevalence(conn: sqlite3.Connection) -> list[dict]:
    """
    Load estimateEffect results from stm_prevalence.

    stm_prevalence has ONE row per topic — no covariate filtering needed.
    significance and direction are pre-computed by R and stored in the table.

    frex_label from the DB is used as a fallback label for topics not listed
    in TOPIC_LABELS, so this function merges both sources automatically.

    Returns a list of dicts sorted by estimate ascending (most worker-facing
    topics first), each with keys:
      topic, label, hypothesis, estimate, ci_lower, ci_upper,
      significant, direction, frex_label_db
    """
    if not table_exists(conn, "stm_prevalence"):
        raise RuntimeError(
            "stm_prevalence table not found. "
            "Run 03b_import_stm.py first."
        )

    rows = conn.execute("""
        SELECT topic_id, frex_label, estimate, ci_lower, ci_upper,
               significant, direction
        FROM   stm_prevalence
        ORDER  BY topic_id
    """).fetchall()

    result = []
    for r in rows:
        t          = int(r["topic_id"])
        frex_db    = r["frex_label"] or f"Topic {t}"
        # TOPIC_LABELS takes priority; fall back to R-generated frex_label
        lbl        = TOPIC_LABELS.get(t, frex_db)
        est        = float(r["estimate"])  if r["estimate"]  is not None else 0.0
        lo         = float(r["ci_lower"])  if r["ci_lower"]  is not None else est
        hi         = float(r["ci_upper"])  if r["ci_upper"]  is not None else est
        sig        = bool(r["significant"])
        direction  = r["direction"] or ("client" if est > 0 else "worker")

        result.append({
            "topic":         t,
            "label":         lbl,
            "frex_label_db": frex_db,
            "hypothesis":    hyp(t),
            "estimate":      est,
            "ci_lower":      lo,
            "ci_upper":      hi,
            "significant":   sig,
            "direction":     direction,
        })

    log.info(
        f"Loaded {len(result)} topics from stm_prevalence. "
        f"{sum(r['significant'] for r in result)} significant."
    )
    return result


def load_frex_terms(conn: sqlite3.Connection, n: int = 7) -> dict[int, str]:
    """
    Return top-n FREX terms per topic as a comma-separated string.
    Uses the frex_term column from stm_topic_terms (set by 03b_import_stm.py).
    Falls back to prob_term if frex_term is NULL for a given row.
    """
    if not table_exists(conn, "stm_topic_terms"):
        log.warning("stm_topic_terms not found; FREX terms unavailable.")
        return {}

    rows = conn.execute("""
        SELECT topic_id,
               COALESCE(frex_term, prob_term) AS term
        FROM   stm_topic_terms
        WHERE  rank <= ?
        ORDER  BY topic_id, rank
    """, (n,)).fetchall()

    terms_by_topic: dict[int, list] = defaultdict(list)
    for r in rows:
        if r["term"]:
            terms_by_topic[int(r["topic_id"])].append(r["term"])

    return {t: ", ".join(terms) for t, terms in terms_by_topic.items()}


def load_document_topics(conn: sqlite3.Connection) -> list[dict]:
    """
    Load per-page dominant topic assignments.

    Uses the stm_topic_profile VIEW (created by 03b_import_stm.py) when
    available — it already filters to one row per page (dominant topic only)
    and joins in frex_label and audience. Falls back to querying stm_theta
    directly with a WHERE topic_id = dominant_topic filter.
    """
    # Prefer the convenience view
    if view_exists(conn, "stm_topic_profile"):
        rows = conn.execute("""
            SELECT page_id, domain, audience,
                   dominant_topic, dominant_prop  AS theta_1,
                   frex_label,
                   prevalence_estimate    AS estimate,
                   prevalence_significant AS significant
            FROM   stm_topic_profile
        """).fetchall()
        return [dict(r) for r in rows]

    # Fallback: query stm_theta directly
    if not table_exists(conn, "stm_theta"):
        raise RuntimeError(
            "Neither stm_topic_profile view nor stm_theta table found. "
            "Run 03b_import_stm.py first."
        )
    rows = conn.execute("""
        SELECT page_id, domain, audience,
               dominant_topic,
               dominant_prop AS theta_1
        FROM   stm_theta
        WHERE  topic_id = dominant_topic
        ORDER  BY page_id
    """).fetchall()
    return [dict(r) for r in rows]


def load_page_text(conn: sqlite3.Connection, page_ids: list) -> dict:
    """
    Load raw text for specific page IDs from corpus_view or the pages table.
    Returns dict: page_id → {domain, audience, text}

    page_ids may be integers or strings (stm_theta stores page_id as TEXT).
    The function tries several common column names for the text field so it
    works regardless of the exact corpus_view definition.
    """
    if not page_ids:
        return {}

    # Normalise to strings (stm_theta.page_id is TEXT)
    str_ids = [str(pid) for pid in page_ids]
    placeholders = ",".join("?" * len(str_ids))

    # Candidate text columns in corpus_view / pages table
    text_col_candidates = ("text_content", "content", "text", "body", "clean_text")

    # 1. Try corpus_view with each candidate text column
    if view_exists(conn, "corpus_view"):
        # Inspect available columns
        try:
            pragma = conn.execute("PRAGMA table_info(corpus_view)").fetchall()
            cv_cols = {r[1] for r in pragma}
        except Exception:
            cv_cols = set()

        for col in text_col_candidates:
            if col not in cv_cols:
                continue
            try:
                rows = conn.execute(f"""
                    SELECT page_id, domain, audience, {col} AS text
                    FROM   corpus_view
                    WHERE  CAST(page_id AS TEXT) IN ({placeholders})
                """, str_ids).fetchall()
                if rows:
                    return {str(r["page_id"]): dict(r) for r in rows}
            except Exception:
                continue

    # 2. Try the pages table directly
    for col in text_col_candidates:
        try:
            rows = conn.execute(f"""
                SELECT CAST(p.id AS TEXT) AS page_id,
                       w.domain,
                       p.{col} AS text
                FROM   pages    p
                JOIN   websites w ON p.website_id = w.id
                WHERE  CAST(p.id AS TEXT) IN ({placeholders})
            """, str_ids).fetchall()
            if rows:
                return {r["page_id"]: dict(r) for r in rows}
        except Exception:
            continue

    log.warning(
        "Could not load raw page text for KWIC. "
        "Inspect corpus_view and pages table column names manually."
    )
    return {}


def load_cooccurrence(
    conn: sqlite3.Connection,
    focus_terms: list[str],
    comparison: str = "cross_platform",
) -> dict:
    """
    Load PMI profiles for focus_terms from cooccurrence_results.
    Returns: {term: {'client': [(collocate, pmi), ...], 'worker': [...]}}
    """
    if not table_exists(conn, "cooccurrence_results"):
        raise RuntimeError("cooccurrence_results not found. Run 02_step1_frequency.py first.")

    result = {}
    for term in focus_terms:
        result[term] = {"client": [], "worker": []}
        for audience in ("client", "worker"):
            rows = conn.execute("""
                SELECT collocate, pmi, cofreq
                FROM cooccurrence_results
                WHERE comparison = ?
                  AND focus_term = ?
                  AND audience   = ?
                  AND cofreq     >= ?
                ORDER BY pmi DESC
                LIMIT ?
            """, (comparison, term, audience, COOC_MIN_COFREQ, COOC_TOP_N)).fetchall()
            result[term][audience] = [(r["collocate"], r["pmi"]) for r in rows]

        if not result[term]["client"] and not result[term]["worker"]:
            log.warning(f"No co-occurrence data found for '{term}'.")

    return result


# =============================================================================
# ── FIGURE 1: FULL 25-TOPIC PREVALENCE OVERVIEW ───────────────────────────────
# =============================================================================

def fig_effects_overview(prevalence: list[dict], pub: bool = True):
    """
    Horizontal forest plot of all 25 topic prevalence effects.
    Topics sorted by estimate. Colour = significant positive / negative / insig.
    Hypothesis classification annotated as coloured label prefix.
    """
    data = sorted(prevalence, key=lambda x: x["estimate"])
    n = len(data)

    fig_h = max(6, n * 0.38)
    fig, ax = plt.subplots(figsize=(8, fig_h))

    ys = np.arange(n)

    for i, row in enumerate(data):
        colour = PAL["insig"]
        if row["significant"]:
            colour = PAL["sig_pos"] if row["estimate"] > 0 else PAL["sig_neg"]

        # CI line
        ax.plot(
            [row["ci_lower"], row["ci_upper"]],
            [i, i],
            color=colour, linewidth=1.5, solid_capstyle="round", zorder=2
        )
        # Point estimate
        ax.scatter(
            row["estimate"], i,
            color=colour, s=55, zorder=3, edgecolors="white", linewidths=0.5
        )

    # Zero reference
    ax.axvline(0, color=PAL["zero"], linewidth=0.8, linestyle="--", zorder=1)

    # Y-axis labels (hypothesis prefix + topic label)
    ytick_labels = []
    for row in data:
        h = row["hypothesis"]
        prefix = f"[{h}] " if h else ""
        ytick_labels.append(prefix + row["label"])

    ax.set_yticks(ys)
    ax.set_yticklabels(ytick_labels, fontsize=7.5)
    ax.set_xlabel(
        "Estimated effect of audience type on topic proportion\n"
        "(positive = higher in client-facing, negative = higher in worker-facing)",
        fontsize=9
    )
    ax.set_title(
        "STM Topic Prevalence Effects: Client-Facing vs. Worker-Facing",
        fontsize=10, fontweight="bold", pad=10
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=PAL["sig_pos"],  label="Higher in client-facing (sig.)"),
        mpatches.Patch(color=PAL["sig_neg"],  label="Higher in worker-facing (sig.)"),
        mpatches.Patch(color=PAL["insig"],    label="Not significant (95% CI ∋ 0)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right",
              frameon=True, framealpha=0.9)

    if not pub:
        # Annotate estimates on exploratory version
        for i, row in enumerate(data):
            ax.text(
                row["estimate"], i + 0.25,
                f"{row['estimate']:.3f}", fontsize=6, ha="center", va="bottom",
                color="#555555"
            )

    plt.tight_layout()
    return fig


# =============================================================================
# ── FIGURE 2–4: HYPOTHESIS-GROUPED PREVALENCE PLOTS ──────────────────────────
# =============================================================================

def fig_effects_hypothesis(
    prevalence: list[dict],
    hypothesis: str,
    title: str,
    pub: bool = True,
):
    """
    Forest plot for a single hypothesis cluster.
    Topics sorted by estimate. Colours match the hypothesis palette.
    """
    data = [row for row in prevalence if row["hypothesis"] == hypothesis]

    if not data:
        log.warning(
            f"No topics classified as '{hypothesis}'. "
            f"Update TOPIC_HYPOTHESIS in config."
        )
        return None

    data = sorted(data, key=lambda x: x["estimate"])
    n = len(data)
    hyp_colour = PAL.get(hypothesis, PAL["neutral"])

    fig, ax = plt.subplots(figsize=(7, max(2.5, n * 0.55 + 1.2)))

    for i, row in enumerate(data):
        sig = row["significant"]
        colour = hyp_colour if sig else PAL["insig"]

        ax.plot(
            [row["ci_lower"], row["ci_upper"]], [i, i],
            color=colour, linewidth=2.0, solid_capstyle="round", zorder=2
        )
        ax.scatter(
            row["estimate"], i,
            color=colour, s=70, zorder=3,
            edgecolors="white", linewidths=0.7,
            marker="D" if sig else "o"
        )

    ax.axvline(0, color=PAL["zero"], linewidth=0.8, linestyle="--", zorder=1)

    ax.set_yticks(range(n))
    ax.set_yticklabels([row["label"] for row in data], fontsize=9)
    ax.set_xlabel(
        "Δ Expected topic proportion (client − worker)",
        fontsize=9
    )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    if not pub:
        for i, row in enumerate(data):
            ax.annotate(
                f"Δ={row['estimate']:.3f}\n[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]",
                xy=(row["estimate"], i),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7, va="center", color="#444444"
            )
        ax.set_title(title + " [EXPLORATORY — NOT FOR THESIS]",
                     fontsize=9, fontweight="bold")

    plt.tight_layout()
    return fig


# =============================================================================
# ── FIGURE 5: HUMAN CO-OCCURRENCE COMPARISON (H1c) ───────────────────────────
# =============================================================================

def fig_cooc_human(cooc_data: dict, pub: bool = True):
    """
    Side-by-side horizontal bar charts showing PMI co-occurrence profiles
    for H1C_FOCUS_TERMS in the client vs worker register.

    Layout: one row per focus term, two columns (B2B left, B2W right).
    """
    terms = [t for t in H1C_FOCUS_TERMS if t in cooc_data]
    if not terms:
        log.warning("No co-occurrence data found for H1C_FOCUS_TERMS.")
        return None

    n_terms = len(terms)
    fig, axes = plt.subplots(
        n_terms, 2,
        figsize=(12, n_terms * 3.0),
        sharey=False,
    )
    if n_terms == 1:
        axes = [axes]   # ensure 2-D indexing

    for row_i, term in enumerate(terms):
        for col_i, (audience, colour, label_str) in enumerate([
            ("client", PAL["H1b"], "Client-facing (B2B)"),
            ("worker", PAL["H1a"], "Worker-facing (B2W)"),
        ]):
            ax = axes[row_i][col_i]
            pairs = cooc_data[term][audience]

            if not pairs:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#999999")
                ax.set_title(f"'{term}' — {label_str}", fontsize=9)
                continue

            collocates, pmis = zip(*pairs)
            ys = np.arange(len(collocates))

            bars = ax.barh(ys, pmis, color=colour, alpha=0.82, edgecolor="white")
            ax.set_yticks(ys)
            ax.set_yticklabels(collocates, fontsize=8)
            ax.set_xlabel("PMI", fontsize=8)
            ax.set_title(f"'{term}' — {label_str}", fontsize=9, fontweight="bold")
            ax.invert_yaxis()   # highest PMI at top

            if not pub:
                for bar, pmi in zip(bars, pmis):
                    ax.text(
                        pmi + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{pmi:.2f}", va="center", fontsize=7, color="#333333"
                    )

    plt.suptitle(
        "Co-occurrence Profiles: Labour Terms in Client- vs. Worker-Facing Discourse (H1c)",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    return fig


# =============================================================================
# ── TOPIC TABLE ───────────────────────────────────────────────────────────────
# =============================================================================

def make_topic_table(
    prevalence: list[dict],
    frex_terms: dict[int, str],
) -> tuple[str, str]:
    """
    Return (latex_str, csv_str) for the full topic table.

    LaTeX table suitable for Table 4.2 in the thesis.
    Columns: topic number | label | FREX terms | estimate | CI | significance | hypothesis
    """
    # Sort by hypothesis group then by estimate
    hyp_order = {"H1b": 0, "H1c": 1, "H1a": 2, None: 3}
    data = sorted(
        prevalence,
        key=lambda x: (hyp_order.get(x["hypothesis"], 99), x["estimate"])
    )

    # ── LaTeX ──────────────────────────────────────────────────────────────────
    latex_lines = [
        r"\begin{longtable}{clp{5.5cm}rrrl}",
        r"  \caption{All STM topics: interpretive label, top FREX terms, "
        r"estimated prevalence effect ($\Delta\hat{\theta}$, positive = higher "
        r"in client-facing), 95\,\% credible interval, and hypothesis relevance. "
        r"$^{*}$ 95\,\% CI excludes zero.}",
        r"  \label{tab:topic-labels} \\",
        r"  \toprule",
        r"  \# & Label & Top FREX terms & $\Delta\hat{\theta}$ & Lower & Upper & H \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{7}{c}{\small\textit{(continued)}} \\",
        r"  \toprule",
        r"  \# & Label & Top FREX terms & $\Delta\hat{\theta}$ & Lower & Upper & H \\",
        r"  \midrule",
        r"  \endhead",
        r"  \bottomrule",
        r"  \endfoot",
    ]

    prev_hyp = object()  # sentinel
    for row in data:
        # Horizontal rule between hypothesis groups
        if row["hypothesis"] != prev_hyp and prev_hyp is not object():
            latex_lines.append(r"  \midrule")
        prev_hyp = row["hypothesis"]

        t       = row["topic"]
        lbl     = row["label"].replace("&", r"\&").replace("%", r"\%")
        frex    = frex_terms.get(t, "—").replace("_", r"\_")
        est     = f"{row['estimate']:.3f}"
        lo      = f"{row['ci_lower']:.3f}"
        hi      = f"{row['ci_upper']:.3f}"
        sig_str = r"$^{*}$" if row["significant"] else ""
        hyp_str = row["hypothesis"] or "—"

        latex_lines.append(
            f"  {t} & {lbl} & \\small {frex} & "
            f"{est}{sig_str} & {lo} & {hi} & {hyp_str} \\\\"
        )

    latex_lines += [
        r"  \begin{tablenotes}",
        r"  \small",
        r"  \item \textit{Note.} STM fitted with $K=25$, Spectral initialisation, "
        r"prevalence\,\textasciitilde\,audience. "
        r"Effects estimated using \texttt{estimateEffect()} with Global uncertainty, "
        r"$n_{\mathrm{sim}}=500$. "
        r"Positive $\Delta\hat{\theta}$ = higher prevalence in client-facing documents.",
        r"  \end{tablenotes}",
        r"\end{longtable}",
    ]

    latex_str = "\n".join(latex_lines)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_lines = [
        "topic,label,frex_terms,estimate,ci_lower,ci_upper,significant,hypothesis"
    ]
    for row in data:
        t     = row["topic"]
        frex  = frex_terms.get(t, "")
        sig   = "TRUE" if row["significant"] else "FALSE"
        h     = row["hypothesis"] or ""
        csv_lines.append(
            f'{t},"{row["label"]}","{frex}",'
            f'{row["estimate"]:.4f},{row["ci_lower"]:.4f},'
            f'{row["ci_upper"]:.4f},{sig},{h}'
        )

    return latex_str, "\n".join(csv_lines)


# =============================================================================
# ── KWIC / FINDTHOUGHTS ───────────────────────────────────────────────────────
# =============================================================================

def make_kwic(
    conn: sqlite3.Connection,
    doc_topics: list[dict],
    prevalence: list[dict],
) -> dict[str, str]:
    """
    For each hypothesis-relevant topic, retrieve the KWIC_TOP_N documents
    with the highest theta_1 score for that dominant topic, then load and
    format their text.

    Returns dict: {'H1a': formatted_text, 'H1b': ..., 'H1c': ..., 'all': ...}
    """
    if KWIC_HYPOTHESIS_ONLY:
        target_topics = [
            row for row in prevalence
            if row["hypothesis"] in ("H1a", "H1b", "H1c")
        ]
    else:
        target_topics = prevalence

    # Group document assignments by dominant topic
    docs_by_topic: dict[int, list[dict]] = defaultdict(list)
    for doc in doc_topics:
        t = doc.get("dominant_topic")
        if t is not None:
            docs_by_topic[int(t)].append(doc)

    # For each topic, pick top-N by theta_1 (= dominant_prop in stm_theta)
    output_by_hyp: dict[str, list[str]] = defaultdict(list)
    all_passages: list[str] = []

    for topic_row in sorted(target_topics, key=lambda x: x["topic"]):
        t       = topic_row["topic"]
        h       = topic_row["hypothesis"] or "neutral"
        t_label = topic_row["label"]

        docs = docs_by_topic.get(t, [])
        docs_sorted = sorted(docs, key=lambda d: d.get("theta_1", 0), reverse=True)
        top_docs = docs_sorted[:KWIC_TOP_N]

        if not top_docs:
            continue

        page_ids = [d["page_id"] for d in top_docs if d.get("page_id") is not None]
        texts    = load_page_text(conn, page_ids)

        header = (
            f"\n{'='*70}\n"
            f"TOPIC {t}: {t_label}  [{h}]\n"
            f"Effect: Δθ={topic_row['estimate']:+.3f} "
            f"[{topic_row['ci_lower']:.3f}, {topic_row['ci_upper']:.3f}] "
            f"{'(sig.)' if topic_row['significant'] else '(n.s.)'}\n"
            f"{'='*70}\n"
        )

        block_lines = [header]

        for rank, doc in enumerate(top_docs, 1):
            pid      = doc.get("page_id")
            domain   = doc.get("domain", "unknown")
            audience = doc.get("audience", "unknown")
            theta    = doc.get("theta_1", 0.0)

            # texts dict is keyed by string page_id
            page_info = texts.get(str(pid), {})
            raw_text  = page_info.get("text", "") or ""

            # Truncate and clean
            text_clean = " ".join(raw_text.split())[:KWIC_MAX_CHARS]
            if len(text_clean) >= KWIC_MAX_CHARS:
                text_clean = text_clean[:KWIC_MAX_CHARS] + "…"

            wrapped = textwrap.fill(text_clean, width=72, initial_indent="    ",
                                    subsequent_indent="    ")

            snippet = (
                f"\n  [{rank}] Domain: {domain}  |  Audience: {audience}  "
                f"|  θ={theta:.3f}  |  page_id={pid}\n"
                f"{wrapped}\n"
            )
            block_lines.append(snippet)

        block = "".join(block_lines)
        output_by_hyp[h].append(block)
        all_passages.append(block)

    return {
        "H1a": "".join(output_by_hyp.get("H1a", ["(No H1a topics classified.)\n"])),
        "H1b": "".join(output_by_hyp.get("H1b", ["(No H1b topics classified.)\n"])),
        "H1c": "".join(output_by_hyp.get("H1c", ["(No H1c topics classified.)\n"])),
        "all": "".join(all_passages),
    }


# =============================================================================
# ── SAVE HELPERS ──────────────────────────────────────────────────────────────
# =============================================================================

def save_fig(fig, stem: str, pub: bool):
    """Save figure in both PDF (pub) and PNG (exp) as appropriate."""
    suffix = "pub" if pub else "exp"
    ext    = "pdf" if pub else "png"
    dpi    = DPI_PUB if pub else DPI_EXP

    path = OUTPUT_DIR / f"{stem}_{suffix}.{ext}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path.name}")


def save_text(content: str, filename: str):
    path = OUTPUT_DIR / filename
    path.write_text(content, encoding="utf-8")
    log.info(f"  Saved: {path.name}")


# =============================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("06_stm_results_figures.py — Chapter 4 STM Visualisation")
    log.info("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_theme()

    conn = get_conn()

    # ── Load shared data ───────────────────────────────────────────────────────
    log.info("Loading data from database…")
    prevalence = load_prevalence(conn)
    frex_terms = load_frex_terms(conn, n=7)

    sig_count = sum(1 for r in prevalence if r["significant"])
    log.info(
        f"  {len(prevalence)} topics loaded; "
        f"{sig_count} with significant prevalence effects."
    )

    # ── Figure 1: Overview (all topics) ───────────────────────────────────────
    if OUTPUTS["fig_overview"]:
        log.info("Producing fig_stm_effects_overview…")
        for pub in (True, False):
            fig = fig_effects_overview(prevalence, pub=pub)
            save_fig(fig, "fig_stm_effects_overview", pub=pub)

    # ── Figures 2–4: Hypothesis-grouped ───────────────────────────────────────
    hyp_specs = [
        ("fig_h1a", "H1a",
         "H1a: Labour Visibility Gap\n"
         "Topic prevalence in client-facing vs. worker-facing documents"),
        ("fig_h1b", "H1b",
         "H1b: The Automation Myth\n"
         "Topic prevalence in client-facing vs. worker-facing documents"),
        ("fig_h1c", "H1c",
         "H1c: Strategic Hypervisibility\n"
         "Topic prevalence in client-facing vs. worker-facing documents"),
    ]

    for key, hypothesis, title in hyp_specs:
        if OUTPUTS[key]:
            log.info(f"Producing {key}…")
            for pub in (True, False):
                fig = fig_effects_hypothesis(prevalence, hypothesis, title, pub=pub)
                if fig is not None:
                    save_fig(fig, key, pub=pub)

    # ── Figure 5: Human co-occurrence comparison ──────────────────────────────
    if OUTPUTS["fig_cooc_human"]:
        log.info("Producing fig_cooc_human_h1c…")
        try:
            cooc_data = load_cooccurrence(conn, H1C_FOCUS_TERMS)
            for pub in (True, False):
                fig = fig_cooc_human(cooc_data, pub=pub)
                if fig is not None:
                    save_fig(fig, "fig_cooc_human_h1c", pub=pub)
        except RuntimeError as e:
            log.warning(f"  Skipping fig_cooc_human: {e}")

    # ── Topic table ───────────────────────────────────────────────────────────
    if OUTPUTS["topic_table"]:
        log.info("Producing topic_table.tex / .csv…")
        latex_str, csv_str = make_topic_table(prevalence, frex_terms)
        save_text(latex_str, "topic_table.tex")
        save_text(csv_str,   "topic_table.csv")

    # ── KWIC ──────────────────────────────────────────────────────────────────
    if OUTPUTS["kwic"]:
        log.info("Producing KWIC passages…")
        try:
            doc_topics = load_document_topics(conn)
            kwic = make_kwic(conn, doc_topics, prevalence)
            for hyp_key, content in kwic.items():
                save_text(content, f"kwic_{hyp_key}.txt")
        except RuntimeError as e:
            log.warning(f"  Skipping KWIC: {e}")

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 65)
    log.info("DONE. Next steps:")
    log.info("")
    log.info("  1. Fill in TOPIC_HYPOTHESIS classifying each topic as")
    log.info("     'H1a', 'H1b', 'H1c', or None.")
    log.info("     (Tip: check kwic_all.txt and topic_table.csv first —")
    log.info("      frex_label from R is already shown as the default label.)")
    log.info("  2. Optionally override TOPIC_LABELS for any topic where")
    log.info("     you want a label different from the R-generated frex_label.")
    log.info("  3. Re-run — hypothesis-grouped figures populate automatically.")
    log.info("")
    log.info(f"  Outputs written to: {OUTPUT_DIR.resolve()}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
