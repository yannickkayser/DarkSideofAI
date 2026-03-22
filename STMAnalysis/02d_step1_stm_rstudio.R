# =============================================================================
# 02d_step1_stm_rstudio.R
# =======================
# Structural Topic Model — DarkSideofAI Master's thesis
# Designed for interactive use in RStudio.
#
# Prerequisites
# -------------
#   Run 02d_step1_stm_export.py first to produce:
#     output/step_1/stm/corpus_export.csv
#     output/step_1/stm/metadata_export.csv
#
# Required R packages
# -------------------
#   install.packages(c("stm", "dplyr", "ggplot2", "tidyr"))
#
# Structure
# ---------
#   0. Config & setup
#   1. Load data
#   2. Handle 'both' audience pages
#   3. Build STM input objects (vocabulary + documents)
#   4. Fit STM
#   5. Inspect topics (FREX labels, prevalence effects)
#   6. Export results to CSV
# =============================================================================


# =============================================================================
# 0. CONFIG & SETUP
# =============================================================================

# ── File paths ────────────────────────────────────────────────────────────────
# Set this to your project root (the folder containing data/, output/, src/)
PROJECT_ROOT <- "."                           # change if running from elsewhere

CORPUS_CSV <- file.path(PROJECT_ROOT, "output/step_1/stm/corpus_export.csv")
META_CSV   <- file.path(PROJECT_ROOT, "output/step_1/stm/metadata_export.csv")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output/step_1/stm")

# ── STM parameters ────────────────────────────────────────────────────────────
K           <- 35       # number of topics — try 25, 30, 35; compare with searchK()
MAX_ITER    <- 75       # EM iterations (75 is usually sufficient)
SEED        <- 42       # for reproducibility

# ── Vocabulary pruning ────────────────────────────────────────────────────────
# Mirrors the MIN_DF=5 / MAX_DF_FRAC=0.85 settings in 02c_step1_topics.py
MIN_DOCFREQ <- 5        # term must appear in at least this many documents
MAX_DOCFRAC <- 0.85     # term must appear in fewer than this fraction of docs

# ── 'Both' audience handling ──────────────────────────────────────────────────
# Pages labelled audience='both' address both clients and workers.
# Set BOTH_STRATEGY to one of:
#   "exclude"    — drop 'both' pages before fitting (cleanest binary contrast)
#   "third_level"— keep as a third factor level between client and worker
BOTH_STRATEGY <- "exclude"    # recommended default for H1a / H1b / H1c

# ── Export settings ───────────────────────────────────────────────────────────
N_TOP_TERMS <- 20       # terms per topic per metric in stm_topic_terms.csv
CI_LEVEL    <- 0.95     # confidence interval level for prevalence estimates

# ── Packages ──────────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(stm)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
set.seed(SEED)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

cat("Loading corpus ...\n")
corpus_df <- read.csv(CORPUS_CSV, stringsAsFactors = FALSE, encoding = "UTF-8")
meta_df   <- read.csv(META_CSV,   stringsAsFactors = FALSE, encoding = "UTF-8")

cat(sprintf("  %d pages loaded from corpus_export.csv\n", nrow(corpus_df)))
cat("  Audience distribution before filtering:\n")
print(table(corpus_df$audience))


# =============================================================================
# 2. HANDLE 'BOTH' AUDIENCE PAGES
# =============================================================================

cat(sprintf("\nHandling 'both' pages (strategy: %s) ...\n", BOTH_STRATEGY))
n_both <- sum(corpus_df$audience == "both")
cat(sprintf("  'both' pages found: %d (%.1f%% of corpus)\n",
            n_both, 100 * n_both / nrow(corpus_df)))

if (BOTH_STRATEGY == "exclude") {
  # ── Option A: remove 'both' pages ──────────────────────────────────────────
  # Gives a clean binary client / worker contrast.
  # This is the recommended choice for testing H1a (Labour Visibility Gap),
  # H1b (Automation Myth), and H1c (Strategic Hypervisibility).
  corpus_df <- corpus_df[corpus_df$audience != "both", ]
  meta_df   <- meta_df[meta_df$audience     != "both", ]
  meta_df$audience <- factor(meta_df$audience, levels = c("client", "worker"))
  cat(sprintf("  %d pages retained after excluding 'both'\n", nrow(corpus_df)))

} else if (BOTH_STRATEGY == "third_level") {
  # ── Option B: keep as third factor level ───────────────────────────────────
  # Allows checking whether 'both' pages sit between client and worker in
  # topic space.  Useful as a robustness check but not for the main STM run.
  # NOTE: content = ~ audience will not work cleanly with 3 levels because
  # STM's content covariate requires exactly 2 levels.  Remove content formula
  # below if you use this option.
  meta_df$audience <- factor(meta_df$audience,
                             levels = c("client", "both", "worker"))
  cat(sprintf("  %d pages retained (including 'both')\n", nrow(corpus_df)))
  cat("  NOTE: content covariate is disabled for three-level audience.\n")

} else {
  stop(sprintf("Unknown BOTH_STRATEGY: '%s'. Use 'exclude' or 'third_level'.",
               BOTH_STRATEGY))
}

cat("  Audience distribution after filtering:\n")
print(table(meta_df$audience))


# =============================================================================
# 3. BUILD STM INPUT OBJECTS
# =============================================================================
# The corpus is already tokenised by Python (SpaCy lemmatisation, stopwords
# removed, month names removed).  We bypass textProcessor() and build the
# STM documents list and vocabulary manually to avoid any double-processing.

cat("\nBuilding STM vocabulary and document objects ...\n")

# Split space-separated token strings into lists
token_lists <- strsplit(corpus_df$tokens, " ")

# ── Build vocabulary with frequency pruning ───────────────────────────────────
all_tokens <- unlist(token_lists)
n_docs     <- length(token_lists)

# Document frequency (how many documents contain each term)
# This is slightly slow for large corpora — progress is shown every 1000 terms
unique_terms <- unique(all_tokens)
cat(sprintf("  Unique terms before pruning: %d\n", length(unique_terms)))

doc_freq <- vapply(unique_terms, function(term) {
  sum(vapply(token_lists, function(tl) term %in% tl, logical(1)))
}, integer(1))
names(doc_freq) <- unique_terms

# Apply MIN_DOCFREQ and MAX_DOCFRAC thresholds
keep <- doc_freq >= MIN_DOCFREQ & doc_freq <= MAX_DOCFRAC * n_docs
vocab <- sort(unique_terms[keep])
cat(sprintf("  Vocabulary after pruning: %d terms  (MIN_DF=%d, MAX_DF=%.0f%%)\n",
            length(vocab), MIN_DOCFREQ, MAX_DOCFRAC * 100))

# ── Build STM documents list ──────────────────────────────────────────────────
# Each element: 2 × N_unique_terms_in_doc integer matrix
#   row 1: 1-indexed vocabulary positions
#   row 2: token counts
vocab_index <- setNames(seq_along(vocab), vocab)

build_doc <- function(tokens) {
  tokens <- tokens[tokens %in% vocab]
  if (length(tokens) == 0L) return(NULL)
  counts <- table(tokens)
  matrix(c(as.integer(vocab_index[names(counts)]),
           as.integer(counts)),
         nrow = 2, byrow = TRUE)
}

cat("  Building document objects ...\n")
documents <- lapply(token_lists, build_doc)

# Drop empty documents (rare after MIN_TOKEN_COUNT=30 in Python export)
non_empty  <- !vapply(documents, is.null, logical(1))
documents  <- documents[non_empty]
meta_stm   <- meta_df[non_empty, ]

cat(sprintf("  %d non-empty documents\n", length(documents)))
cat(sprintf("  %d empty documents dropped\n", sum(!non_empty)))


# =============================================================================
# 4. FIT STM
# =============================================================================
# prevalence = ~ audience  (H1a / H1b): which topics appear more in
#              client vs worker documents?
# content    = ~ audience  (H1c): which WORDS are used for a topic
#              differ by audience?
#
# init.type = "Spectral" (Arora et al. 2013) is preferred over "LDA" for
# reproducibility and faster convergence.

content_formula <- if (BOTH_STRATEGY == "exclude") ~ audience else NULL

cat(sprintf("\nFitting STM  (K=%d, max.em.its=%d, seed=%d) ...\n",
            K, MAX_ITER, SEED))
cat("  This may take several minutes.\n\n")

stm_model <- stm(
  documents  = documents,
  vocab      = vocab,
  K          = K,
  prevalence = ~ audience,
  content    = content_formula,
  data       = meta_stm,
  max.em.its = MAX_ITER,
  init.type  = "Spectral",
  seed       = SEED,
  verbose    = TRUE
)

cat("\nFitting complete.\n")

# Save model for re-use (avoids refitting)
model_path <- file.path(OUTPUT_DIR, "stm_model.rds")
saveRDS(stm_model, model_path)
cat(sprintf("  Model saved → %s\n", model_path))

# To reload later without refitting:
#   stm_model <- readRDS("output/step_1/stm/stm_model.rds")


# =============================================================================
# 5. INSPECT TOPICS
# =============================================================================

cat("\n── Topic labels (FREX, top 7 terms) ────────────────────────────────────\n")

labels <- labelTopics(stm_model, n = N_TOP_TERMS)

# Estimate prevalence effects first so we can show them alongside labels
prevalence_est <- estimateEffect(
  formula     = 1:K ~ audience,
  stmobj      = stm_model,
  metadata    = meta_stm,
  uncertainty = "Global",
  nsims       = 500
)

# Extract estimate for the audienceworker contrast
alpha   <- 1 - CI_LEVEL
z_crit  <- qnorm(1 - alpha / 2)

prev_df <- do.call(rbind, lapply(1:K, function(t) {
  s   <- summary(prevalence_est, topics = t)
  tbl <- s$tables[[1]]
  if (!"audienceworker" %in% rownames(tbl)) return(NULL)
  row <- tbl["audienceworker", ]
  data.frame(
    topic_id = t,
    estimate = row["Estimate"],
    std_err  = row["Std. Error"],
    ci_lower = row["Estimate"] - z_crit * row["Std. Error"],
    ci_upper = row["Estimate"] + z_crit * row["Std. Error"],
    stringsAsFactors = FALSE
  )
}))
prev_df <- Filter(Negate(is.null), prev_df)
prev_df <- do.call(rbind, prev_df)
rownames(prev_df) <- NULL

# Print topic summary table
cat(sprintf("  %-6s  %-9s  %-60s\n", "Topic", "Prev.est", "FREX terms"))
cat(sprintf("  %-6s  %-9s  %-60s\n",
            "──────", "─────────", paste(rep("─", 60), collapse="")))
for (t in 1:K) {
  frex  <- paste(labels$frex[t, 1:7], collapse=", ")
  est_r <- prev_df[prev_df$topic_id == t, ]
  est_s <- if (nrow(est_r) > 0) sprintf("%+.3f", est_r$estimate) else "    n/a"
  cat(sprintf("  T%-5d  %-9s  %s\n", t, est_s, frex))
}
cat("  (+ estimate = more prevalent in WORKER documents)\n")
cat("  (- estimate = more prevalent in CLIENT documents)\n")

# ── Top 10 topics by absolute effect ─────────────────────────────────────────
cat("\n── Top 10 topics by |prevalence effect| ────────────────────────────────\n")
top10 <- prev_df[order(abs(prev_df$estimate), decreasing = TRUE), ][1:10, ]
for (i in 1:nrow(top10)) {
  r  <- top10[i, ]
  ci <- sprintf("[%+.3f, %+.3f]", r$ci_lower, r$ci_upper)
  cat(sprintf("  T%-3d  est=%+.4f  SE=%.4f  %s CI=%s\n",
              r$topic_id, r$estimate, r$std_err,
              ifelse(r$estimate > 0, "(+worker)", "(-client)"), ci))
}


# =============================================================================
# 6. EXPORT RESULTS TO CSV
# =============================================================================

cat("\n── Exporting CSVs ───────────────────────────────────────────────────────\n")

# ── 6a. stm_theta.csv — per-document top-3 topic proportions ─────────────────
theta    <- stm_model$theta    # n_docs × K matrix
page_ids <- meta_stm$page_id

theta_df <- do.call(rbind, lapply(seq_len(nrow(theta)), function(i) {
  row  <- theta[i, ]
  top3 <- order(row, decreasing = TRUE)[1:3]
  data.frame(
    page_id        = page_ids[i],
    audience       = as.character(meta_stm$audience[i]),
    domain         = meta_stm$domain[i],
    topic_1_id     = top3[1],
    topic_1_weight = round(row[top3[1]], 6),
    topic_2_id     = top3[2],
    topic_2_weight = round(row[top3[2]], 6),
    topic_3_id     = top3[3],
    topic_3_weight = round(row[top3[3]], 6),
    stringsAsFactors = FALSE
  )
}))
write.csv(theta_df, file.path(OUTPUT_DIR, "stm_theta.csv"), row.names = FALSE)
cat(sprintf("  stm_theta.csv         → %d rows\n", nrow(theta_df)))

# ── 6b. stm_topic_terms.csv — FREX, Prob, Lift, Score ────────────────────────
terms_df <- do.call(rbind, lapply(c("prob", "frex", "lift", "score"), function(metric) {
  mat <- labels[[metric]]    # K × N_TOP_TERMS matrix
  do.call(rbind, lapply(1:K, function(t) {
    data.frame(
      topic_id = t,
      metric   = metric,
      rank     = seq_len(N_TOP_TERMS),
      term     = mat[t, ],
      stringsAsFactors = FALSE
    )
  }))
}))
write.csv(terms_df, file.path(OUTPUT_DIR, "stm_topic_terms.csv"), row.names = FALSE)
cat(sprintf("  stm_topic_terms.csv   → %d rows\n", nrow(terms_df)))

# ── 6c. stm_prevalence.csv — audience regression coefficients ─────────────────
write.csv(prev_df, file.path(OUTPUT_DIR, "stm_prevalence.csv"), row.names = FALSE)
cat(sprintf("  stm_prevalence.csv    → %d rows\n", nrow(prev_df)))

# ── 6d. stm_content.csv — per-audience top words per topic ────────────────────
# Only available when content = ~ audience (i.e. BOTH_STRATEGY = "exclude")
if (!is.null(content_formula)) {
  tryCatch({
    sage <- sageLabels(stm_model, n = N_TOP_TERMS)
    # sage$wordcov$audience[[1]] = client words × topics matrix
    # sage$wordcov$audience[[2]] = worker words × topics matrix
    aud_labels   <- c("client", "worker")
    content_rows <- list()
    for (aud_idx in 1:2) {
      mat <- sage$wordcov$audience[[aud_idx]]   # terms × K matrix
      for (t in 1:K) {
        ord        <- order(mat[, t], decreasing = TRUE)[1:N_TOP_TERMS]
        top_terms  <- rownames(mat)[ord]
        top_scores <- mat[ord, t]
        content_rows <- c(content_rows, lapply(seq_along(top_terms), function(r) {
          data.frame(
            topic_id = t,
            audience = aud_labels[aud_idx],
            rank     = r,
            term     = top_terms[r],
            score    = round(top_scores[r], 6),
            stringsAsFactors = FALSE
          )
        }))
      }
    }
    content_df <- do.call(rbind, content_rows)
    write.csv(content_df, file.path(OUTPUT_DIR, "stm_content.csv"), row.names = FALSE)
    cat(sprintf("  stm_content.csv       → %d rows\n", nrow(content_df)))
  }, error = function(e) {
    cat(sprintf("  WARNING: content export failed: %s\n", e$message))
  })
} else {
  cat("  stm_content.csv       → skipped (three-level audience; no content covariate)\n")
}

# =============================================================================
# DONE
# =============================================================================

cat("\n─────────────────────────────────────────────────────────────────────────\n")
cat("STM complete.\n")
cat(sprintf("  K = %d  |  documents = %d  |  vocabulary = %d\n",
            K, length(documents), length(vocab)))
cat(sprintf("  Output directory: %s\n", normalizePath(OUTPUT_DIR)))
cat("\nFiles written:\n")
for (f in c("stm_model.rds", "stm_theta.csv", "stm_topic_terms.csv",
            "stm_prevalence.csv", "stm_content.csv")) {
  p    <- file.path(OUTPUT_DIR, f)
  mark <- if (file.exists(p)) "✓" else "✗  (not written)"
  cat(sprintf("  %s  %s\n", mark, f))
}
cat("─────────────────────────────────────────────────────────────────────────\n")
