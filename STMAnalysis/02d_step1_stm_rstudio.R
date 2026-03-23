# =============================================================================
# 02d_step1_stm_rstudio.R
# =======================
# Structural Topic Model — DarkSideofAI Master's thesis
# Designed for interactive, step-by-step use in RStudio.
#
# Structure follows Trajceski (2021) "Structural Topic Modeling with R — Part I"
# https://jovantrajceski.medium.com/structural-topic-modeling-with-r-part-i-...
#
#   1. Load libraries
#   2. Load data
#   3. Handle 'both' audience pages
#   4. textProcessor()   — build vocabulary (text already clean; all flags FALSE)
#   5. prepDocuments()   — prune vocabulary, drop empty docs
#   6. Preliminary STM   — quick run at low K to check output and timing
#   7. searchK()         — three approaches to find optimal K:
#        Approach 1: default searchK over a range
#        Approach 2: searchK with N=500 and 75 iterations
#        Approach 3: Lee-Mimno (K=0, automatic topic number selection)
#   8. Final STM         — fit with chosen K
#   9. labelTopics()     — inspect topic labels
#  10. findThoughts()    — find representative documents
#  11. estimateEffect()  — audience prevalence effects (H1a / H1b)
#  12. sageLabels()      — audience word differences per topic (H1c)
#  13. topicCorr()       — topic correlation network
#  14. Export CSVs       — for import back into Python / SQLite
#
# NOTE ON PRE-PROCESSED DATA
# --------------------------
# The corpus has already been tokenised, lemmatised (SpaCy), and cleaned
# by 02d_step1_stm_export.py.  textProcessor() is called with ALL internal
# processing flags set to FALSE to prevent double-processing.
# The rest of the workflow is identical to a standard raw-text STM pipeline.
#
# Prerequisites
# -------------
#   Run 02d_step1_stm_export.py to produce corpus_export.csv + metadata_export.csv
#
# Install packages (once)
# -----------------------
#   install.packages(c("stm", "dplyr", "ggplot2", "igraph"))
#
# searchK() is slow — read section 7 carefully before running it.
# Each K value = one full STM fit.  Use parallel cores and save results.
# =============================================================================


# =============================================================================
# 0. CONFIG
# =============================================================================

PROJECT_ROOT <- "."   # set to your project root if not running from there

CORPUS_CSV <- file.path(PROJECT_ROOT, "output/step_1/stm/corpus_export.csv")
META_CSV   <- file.path(PROJECT_ROOT, "output/step_1/stm/metadata_export.csv")
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output/step_1/stm")

# Final model parameters — adjust after reviewing searchK() results
K        <- 20    # number of topics -> seems the best number
MAX_ITER <- 150    # EM iterations

SEED     <- 42

# Vocabulary pruning — mirrors 02c_step1_topics.py (MIN_DF = 5)
LOWER_THRESH <- 5

# 'Both' audience handling:
#   "exclude"      — clean binary client/worker contrast (recommended for H1)
#   "third_level"  — keep as third factor for robustness check
BOTH_STRATEGY <- "exclude"

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)


# =============================================================================
# 1. LOAD LIBRARIES
# =============================================================================

suppressPackageStartupMessages({
  library(stm)
  library(dplyr)
  library(ggplot2)
  library(igraph)
  library(geometry)
})

set.seed(SEED)


# =============================================================================
# 2. LOAD DATA
# =============================================================================

cat("── 2. Loading data ──────────────────────────────────────────────────────\n")

corpus_df <- read.csv(CORPUS_CSV, stringsAsFactors = FALSE)
meta_df   <- read.csv(META_CSV,   stringsAsFactors = FALSE)

cat(sprintf("  %d pages loaded\n", nrow(corpus_df)))
cat("  Audience distribution (raw):\n")
print(table(corpus_df$audience))

# Check for NAs
cat(sprintf("  NAs in tokens   : %d\n", sum(is.na(corpus_df$tokens))))
cat(sprintf("  NAs in audience : %d\n", sum(is.na(meta_df$audience))))


# =============================================================================
# 3. HANDLE 'BOTH' AUDIENCE PAGES
# =============================================================================

cat("\n── 3. Handling 'both' audience pages ────────────────────────────────────\n")

n_both <- sum(corpus_df$audience == "both", na.rm = TRUE)
cat(sprintf("  'both' pages: %d (%.1f%%)\n",
            n_both, 100 * n_both / nrow(corpus_df)))

if (BOTH_STRATEGY == "exclude") {
  # Recommended — gives the clean binary contrast needed for H1a / H1b / H1c.
  # The content covariate (which words differ by audience) requires exactly
  # two factor levels, so this is the only option that enables it.
  corpus_df <- corpus_df[corpus_df$audience != "both", ]
  meta_df   <- meta_df[meta_df$audience     != "both", ]
  meta_df$audience <- factor(meta_df$audience, levels = c("client", "worker"))
  cat(sprintf("  %d pages retained\n", nrow(corpus_df)))

} else if (BOTH_STRATEGY == "third_level") {
  # Robustness check — tests whether 'both' pages sit between client and worker.
  # NOTE: content covariate is disabled for 3 levels (see step 8).
  meta_df$audience <- factor(meta_df$audience,
                             levels = c("client", "both", "worker"))
  cat(sprintf("  %d pages retained (including 'both')\n", nrow(corpus_df)))
}

cat("  Audience distribution after filtering:\n")
print(table(meta_df$audience))


# =============================================================================
# 4. TEXT PROCESSING (textProcessor)
# =============================================================================
# Because the corpus is already lemmatised and cleaned by Python, we set ALL
# of textProcessor()'s internal processing flags to FALSE.
# We use it only to build the vocabulary / document-term matrix in STM format.
#
# Trajceski's article runs textProcessor() with full preprocessing on raw text.
# Our approach is equivalent but skips the R-side preprocessing step because
# SpaCy (Python) has already done it — and done it better (true lemmatisation
# vs R's stemming, accurate English stopwords, etc.).

cat("\n── 4. textProcessor() — building vocabulary ─────────────────────────────\n")
cat("  NOTE: all processing flags are FALSE — corpus is already clean.\n\n")

processed <- textProcessor(
  documents         = corpus_df$tokens,
  metadata          = meta_df,
  lowercase         = FALSE,   # already lowercase
  removestopwords   = FALSE,   # already removed
  removenumbers     = FALSE,   # already removed
  removepunctuation = FALSE,   # already removed
  stem              = FALSE,   # already lemmatised — stemming would degrade quality
  wordLengths       = c(2, Inf),
  verbose           = TRUE
)

cat(sprintf("\n  Vocabulary before prepDocuments() : %d terms\n",
            length(processed$vocab)))


# =============================================================================
# 5. PREPARE DOCUMENTS (prepDocuments)
# =============================================================================
# Removes terms below lower.thresh (mirrors MIN_DF=5 in 02c_step1_topics.py)
# and drops any documents that become empty after pruning.
#
# plotRemoved() shows vocabulary/document loss across different thresholds.
# Run this interactively to confirm lower.thresh=5 is appropriate.

cat("\n── 5. prepDocuments() — pruning vocabulary ──────────────────────────────\n")

# Uncomment to plot document/term loss at different thresholds before deciding:
# plotRemoved(processed$documents, lower.thresh.words = 1:30)

out <- prepDocuments(
  documents    = processed$documents,
  vocab        = processed$vocab,
  meta         = processed$meta,
  lower.thresh = LOWER_THRESH,
  verbose      = TRUE
)

cat(sprintf("\n  Documents  : %d\n", length(out$documents)))
cat(sprintf("  Vocabulary : %d terms\n", length(out$vocab)))
cat(sprintf("  Dropped    : %d empty documents after pruning\n",
            length(processed$documents) - length(out$documents)))

# Save out object so you can reload without re-running steps 4–5
saveRDS(out, file.path(OUTPUT_DIR, "out.rds"))
# Reload with: out <- readRDS("output/step_1/stm/out.rds")


# =============================================================================
# 6. PRELIMINARY STM (quick look before searchK)
# =============================================================================
# As in Trajceski: run a quick model at low K first to check that the pipeline
# works, assess runtime, and get a first impression of the topics.
# No content covariate yet — just prevalence for speed.

cat("\n── 6. Preliminary STM (K=15, quick look) ────────────────────────────────\n")
cat("  Running a quick model to check output and estimate runtime ...\n\n")

stm_prelim <- stm(
  documents  = out$documents,
  vocab      = out$vocab,
  K          = 15,
  prevalence = ~ audience,
  data       = out$meta,
  max.em.its = 30,       # fewer iterations — just for a quick look
  init.type  = "LDA",
  seed       = SEED,
  verbose    = TRUE
)

cat("\n  Preliminary topic labels (FREX):\n")
labelTopics(stm_prelim, n = 7)

# Topic proportion overview plot
plot(stm_prelim,
     type     = "summary",
     n        = 5,
     main     = "Preliminary STM (K=15) — Topic Proportions",
     text.cex = 0.75)


# =============================================================================
# 7. FINDING OPTIMAL K: THREE APPROACHES
# =============================================================================
# searchK() is slow because it fits one complete STM for every K value.
# To avoid staring at a frozen console:
#   - verbose = TRUE  prints EM iteration output per K so you can see progress
#   - cores           runs K values in parallel (set to your CPU core count - 1)
#   - results are saved to .rds after each approach — reload to avoid re-running
#
# Run the approaches in order.  Each one is faster than the previous.
# Start with the sanity check and Approach 3 (Lee-Mimno, ~30 sec) first.

cat("\n── 7. Finding optimal K ─────────────────────────────────────────────────\n")

# How many CPU cores to use for parallel searchK.
# Check yours with: parallel::detectCores()
# Set to detectCores() - 1 to leave one free for RStudio itself.
N_CORES <- max(1L, parallel::detectCores() - 1L)
cat(sprintf("  Using %d parallel cores for searchK.\n", N_CORES))


# ── Sanity check: 2 K values, few iterations ─────────────────────────────────
# Run this FIRST before any of the three approaches.
# It takes ~1 minute and proves the pipeline is working — you will see
# EM iteration output in the console for each K value.

cat("\n  [Sanity check] Fitting K=10 and K=15 with 10 iterations each ...\n")
cat("  You should see EM iteration output below this line:\n\n")

searchK_test <- searchK(
  documents  = out$documents,
  vocab      = out$vocab,
  K          = c(10, 15),
  prevalence = ~ audience,
  data       = out$meta,
  max.em.its = 10,        # very few iterations — just to confirm it runs
  cores      = 1L,        # single core for sanity check (cleaner console output)
  init.type  = "LDA",
  seed       = SEED,
  verbose    = TRUE       # <-- this is what shows EM progress in the console
)
cat("\n  Sanity check complete — pipeline is working.\n")
cat("  searchK column names in your stm version:\n")
print(names(searchK_test$results))
cat("  Full results (ignore values, only 10 iterations):\n")
print(searchK_test$results)


# ── Approach 3: Lee-Mimno (K=0 — fastest, ~30 sec) ────────────────────────────
# Run this BEFORE Approaches 1 and 2 because it is by far the fastest.
# stm(K=0) uses the Lee & Mimno (2014) NMF algorithm to select K automatically.
# The selected K gives a useful anchor for the other two approaches.


# ── Approach 3 replacement: manual coherence + exclusivity loop ───────────────
cat("\n  Approach 3 (replacement): semantic coherence + exclusivity per K ...\n")
cat("  Expected runtime: ~5–10 min. Progress printed per K.\n\n")

K_range <- c(10, 15, 20, 25, 30, 35)

manual_k_path <- file.path(OUTPUT_DIR, "manual_k_results.rds")
if (file.exists(manual_k_path)) {
    cat("  Loading saved result ...\n")
    k_results <- readRDS(manual_k_path)
} else {
    k_results <- do.call(rbind, lapply(K_range, function(k) {
        cat(sprintf("  Fitting K=%d ...\n", k))
        m <- stm(
            documents  = out$documents,
            vocab      = out$vocab,
            K          = k,
            prevalence = ~ audience,
            data       = out$meta,
            max.em.its = 75,
            init.type  = "LDA",
            seed       = SEED,
            verbose    = FALSE
        )
        data.frame(
            K      = k,
            semcoh = mean(semanticCoherence(m, out$documents)),
            exclus = mean(exclusivity(m)),
            stringsAsFactors = FALSE
        )
    }))
    saveRDS(k_results, manual_k_path)
}

print(k_results)

plot(k_results$semcoh, k_results$exclus,
     type = "b", pch = 16,
     xlab = "Semantic coherence (higher = better)",
     ylab = "Exclusivity (higher = better)",
     main = "K selection: coherence vs exclusivity")
text(k_results$semcoh, k_results$exclus,
     labels = paste0("K=", k_results$K),
     pos = 3, cex = 0.8)


# ── Approach 2: searchK with N=500, 75 iterations ─────────────────────────────
# Fewer K values but more reliable estimates.
# With N_CORES parallel, Trajceski's ~6 min becomes ~2–3 min.
# Console will be quiet during parallel runs — this is normal.
# Progress appears as each K completes and is printed to the console.

cat(sprintf("\n  Approach 2: searchK K=%s (N=500, 75 iter, %d cores) ...\n",
            paste(c(10,20,30,40,50), collapse=","), N_CORES))
cat("  Expected runtime: 3–8 min depending on corpus size and cores.\n")
cat("  Console output appears after each K completes.\n\n")

searchK2_path <- file.path(OUTPUT_DIR, "searchK_approach2.rds")
if (file.exists(searchK2_path)) {
  cat("  Loading saved result ...\n")
  K_search2 <- readRDS(searchK2_path)
} else {
  K_search2 <- searchK(
    documents  = out$documents,
    vocab      = out$vocab,
    K          = c(10, 20, 30, 40, 50),
    prevalence = ~ audience,
    data       = out$meta,
    N          = 500,
    max.em.its = 75,
    cores      = N_CORES,
    init.type  = "LDA",
    seed       = SEED,
    verbose    = TRUE
  )
  saveRDS(K_search2, searchK2_path)
  cat(sprintf("  Saved → %s\n", searchK2_path))
}
plot(K_search2)
cat("  Approach 2 results:\n")
print(K_search2$results)


# ── Approach 1: searchK over K = 10–30 (most thorough) ───────────────────────
# Tests every K value from 10 to 30.
# With parallel cores this is manageable (Trajceski's ~23 min → ~8–12 min).
# Run LAST since you may already have enough information from Approaches 2 & 3.

cat(sprintf("\n  Approach 1: searchK K=10:30 (default params, %d cores) ...\n",
            N_CORES))
cat("  Expected runtime: 8–20 min. Results auto-saved.\n\n")

searchK1_path <- file.path(OUTPUT_DIR, "searchK_approach1.rds")
if (file.exists(searchK1_path)) {
  cat("  Loading saved result ...\n")
  K_search1 <- readRDS(searchK1_path)
} else {
  K_search1 <- searchK(
    documents  = out$documents,
    vocab      = out$vocab,
    K          = 10:30,
    prevalence = ~ audience,
    data       = out$meta,
    cores      = N_CORES,
    init.type  = "LDA",
    seed       = SEED,
    verbose    = TRUE
  )
  saveRDS(K_search1, searchK1_path)
  cat(sprintf("  Saved → %s\n", searchK1_path))
}
plot(K_search1)
cat("  Approach 1 results:\n")
print(K_search1$results)


# ── K selection summary ───────────────────────────────────────────────────────
cat("\n  ── K selection summary ──────────────────────────────────────────────────\n")

best3 <- k_results$K[which.min(abs(k_results$semcoh - max(k_results$semcoh)) +
                                   abs(k_results$exclus - max(k_results$exclus)))]
cat(sprintf("  Approach 3 (coherence+exclusivity balance): K = %d\n", best3))

# held-out likelihood column is named "heldout" in older stm versions
# and stored as a list column in newer ones — unlist() handles both
heldout_col <- if ("heldout" %in% names(K_search2$results)) "heldout" else
               grep("held", names(K_search2$results), value = TRUE)[1]

best2 <- K_search2$results$K[which.max(unlist(K_search2$results[[heldout_col]]))]
cat(sprintf("  Approach 2 (best held-out likelihood): K = %d\n", best2[]))

heldout_col <- if ("heldout" %in% names(K_search1$results)) "heldout" else
               grep("held", names(K_search1$results), value = TRUE)[1]
best1 <- K_search1$results$K[which.max(unlist(K_search1$results[[heldout_col]]))]
cat(sprintf("  Approach 1 (best held-out likelihood): K = %d\n", best1[]))
cat("\n  Reading the plots:\n")
cat("    Held-out likelihood : higher is better — look for the elbow\n")
cat("    Residuals           : lower is better\n")
cat("    Semantic coherence  : higher = top words co-occur more (interpretable)\n")
cat("    Exclusivity         : higher = top words are unique to each topic\n")
cat("    Coherence vs exclusivity trade off — balance both when choosing K.\n")
cat("\n  → Update K at the top of the script (CONFIG section) and run step 8.\n")


# =============================================================================
# 8. FINAL STM
# =============================================================================
# prevalence = ~ audience  — H1a / H1b: which topics appear more in
#              client vs worker documents?
# content    = ~ audience  — H1c: which WORDS differ per topic by audience?
#              (requires exactly 2 factor levels → only with BOTH_STRATEGY="exclude")

cat(sprintf("\n── 8. Final STM (K=%d, max.em.its=%d) ──────────────────────────────────\n",
            K, MAX_ITER))

content_formula <- if (BOTH_STRATEGY == "exclude") ~ audience else NULL

if (!is.null(content_formula)) {
  cat("  Fitting with prevalence = ~audience AND content = ~audience\n\n")
} else {
  cat("  Fitting with prevalence = ~audience only (three-level audience)\n\n")
}

stm_model <- stm(
  documents  = out$documents,
  vocab      = out$vocab,
  K          = K,
  prevalence = ~ audience,
  content    = content_formula,
  data       = out$meta,
  max.em.its = MAX_ITER,
  init.type  = "Spectral",
  seed       = SEED,
  verbose    = TRUE
)

# Save model — reload without refitting:
#   stm_model <- readRDS("output/step_1/stm/stm_model.rds")
model_path <- file.path(OUTPUT_DIR, "stm_model.rds")
saveRDS(stm_model, model_path)
cat(sprintf("\n  Model saved → %s\n", model_path))

# Topic proportion overview
plot(stm_model,
     type     = "summary",
     n        = 5,
     main     = sprintf("Final STM (K=%d) — Topic Proportions", K),
     text.cex = 0.75)


# =============================================================================
# 9. LABEL TOPICS (labelTopics)
# =============================================================================
# Four ranked term lists per topic:
#   Prob  — highest probability (most frequent in topic)
#   FREX  — frequency × exclusivity (most distinctive to this topic)
#   Lift  — highest lift over corpus background
#   Score — log-likelihood based, similar to FREX
#
# FREX is the most interpretable for naming topics.
# For thesis Step 2 close reading: use FREX to name topics, then validate
# by reading the actual pages from findThoughts() below.

cat("\n── 9. Topic labels ───────────────────────────────────────────────────────\n")

labels <- labelTopics(stm_model, n = 10)
print(labels)

# Perspectives plot: compare word use between two specific topics
# Replace topic IDs after reading the labels above:
plot(stm_model, type = "perspectives", topics = c(1, 2))


# =============================================================================
# 10. REPRESENTATIVE DOCUMENTS (findThoughts)
# =============================================================================
# findThoughts() returns the documents most representative of each topic.
# These complement the Step 2 sample — use them to validate topic labels
# before coding (Nelson 2020: close reading should be theory-guided).

cat("\n── 10. Representative documents (findThoughts) ───────────────────────────\n")

# Create a text vector aligned with out$documents
# (after prepDocuments some rows may have been dropped)
aligned_texts <- corpus_df$tokens[match(out$meta$page_id,
                                        corpus_df$page_id)]

for (t in 1:K) {
  thoughts <- findThoughts(
    model  = stm_model,
    texts  = aligned_texts,
    topics = t,
    n      = 2
  )
  cat(sprintf("\nTopic %d  FREX: %s\n",
              t, paste(labels$frex[t, 1:5], collapse=", ")))
  for (i in seq_along(thoughts$docs[[1]])) {
    cat(sprintf("  [%d] %s...\n", i,
                substr(thoughts$docs[[1]][i], 1, 150)))
  }
}



# =============================================================================
# 11. PREVALENCE EFFECTS (estimateEffect)
# =============================================================================
# Tests H1a (Labour Visibility Gap) and H1b (Automation Myth):
# which topics appear significantly more in client vs worker documents?
#
# Positive estimate → topic more prevalent in WORKER (B2W) documents
# Negative estimate → topic more prevalent in CLIENT (B2B) documents

cat("\n── 11. Prevalence effects (H1a / H1b) ───────────────────────────────────\n")

prevalence_est <- estimateEffect(
  formula     = 1:K ~ audience,
  stmobj      = stm_model,
  metadata    = out$meta,
  uncertainty = "Global",
  nsims       = 500
)

# Full summary (all topics and coefficients)
summary(prevalence_est)

# ── Coefficient plot (difference method) ─────────────────────────────────────
# Each topic appears as a point with 95% CI.
# Left of zero = more client-prevalent; right = more worker-prevalent.
plot(
  prevalence_est,
  covariate      = "audience",
  topics         = 1:K,
  model          = stm_model,
  method         = "difference",
  cov.value1     = "worker",
  cov.value2     = "client",
  xlab           = "More client  ←                →  More worker",
  main           = "Audience prevalence effect per topic (95% CI)",
  labeltype      = "frex",
  n              = 4,
  verbose.labels = FALSE,
  width          = 50
)

# ── Extract to data frame ─────────────────────────────────────────────────────
z_crit  <- qnorm(0.975)
prev_df <- do.call(rbind, lapply(1:K, function(t) {
  s   <- summary(prevalence_est, topics = t)
  tbl <- s$tables[[1]]
  if (!"audienceworker" %in% rownames(tbl)) return(NULL)
  row <- tbl["audienceworker", ]
  data.frame(
    topic_id = t,
    frex_label = paste(labels$frex[t, 1:5], collapse=", "),
    estimate   = row["Estimate"],
    std_err    = row["Std. Error"],
    ci_lower   = row["Estimate"] - z_crit * row["Std. Error"],
    ci_upper   = row["Estimate"] + z_crit * row["Std. Error"],
    significant = abs(row["Estimate"]) > z_crit * row["Std. Error"],
    direction  = ifelse(row["Estimate"] > 0, "worker", "client"),
    stringsAsFactors = FALSE
  )
}))
prev_df <- do.call(rbind, Filter(Negate(is.null), prev_df))
rownames(prev_df) <- NULL

cat("\n  Significant audience differences (sorted by |effect|):\n")
sig <- prev_df[prev_df$significant, ]
sig <- sig[order(abs(sig$estimate), decreasing = TRUE), ]
print(sig[, c("topic_id", "frex_label", "estimate", "ci_lower", "ci_upper", "direction")])


# =============================================================================
# 12. CONTENT COVARIATE (sageLabels)
# =============================================================================
# Tests H1c (Strategic Hypervisibility):
# For the SAME topic, do client and worker pages use different words?
# sageLabels() uses the content = ~audience model to show per-audience
# word distributions for each topic.

if (!is.null(content_formula)) {
  cat("\n── 12. Content covariate labels (H1c) ────────────────────────────────────\n")
  sage <- sageLabels(stm_model, n = 10)
  print(sage)
  # sage$wordcov$audience[[1]] = words used by CLIENT pages for this topic
  # sage$wordcov$audience[[2]] = words used by WORKER pages for this topic
} else {
  cat("\n── 12. Content covariate: skipped (three-level audience)\n")
}


# =============================================================================
# 13. TOPIC CORRELATIONS (topicCorr)
# =============================================================================
# Identifies clusters of co-occurring topics — useful for finding whether
# H1a and H1b topics form a coherent "labour invisibility" cluster.

cat("\n── 13. Topic correlations ────────────────────────────────────────────────\n")

topic_corr <- topicCorr(stm_model, method = "simple", cutoff = 0.01)
plot(topic_corr,
     main = sprintf("Topic correlations (K=%d)", K))


# =============================================================================
# 14. EXPORT RESULTS TO CSV
# =============================================================================

cat("\n── 14. Exporting CSVs ───────────────────────────────────────────────────\n")

# ── stm_theta.csv — per-document top-3 topic proportions ─────────────────────
theta    <- stm_model$theta
page_ids <- out$meta$page_id

theta_df <- do.call(rbind, lapply(seq_len(nrow(theta)), function(i) {
  row  <- theta[i, ]
  top3 <- order(row, decreasing = TRUE)[1:3]
  data.frame(
    page_id        = page_ids[i],
    audience       = as.character(out$meta$audience[i]),
    domain         = out$meta$domain[i],
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

# ── stm_topic_terms.csv — FREX, Prob, Lift, Score (top 20 per metric) ─────────
all_labels <- labelTopics(stm_model, n = 20)
terms_df <- do.call(rbind, lapply(c("prob", "frex", "lift", "score"), function(metric) {
  mat <- all_labels[[metric]]
  do.call(rbind, lapply(1:K, function(t) {
    data.frame(
      topic_id = t,
      metric   = metric,
      rank     = seq_len(ncol(mat)),
      term     = mat[t, ],
      stringsAsFactors = FALSE
    )
  }))
}))
write.csv(terms_df, file.path(OUTPUT_DIR, "stm_topic_terms.csv"), row.names = FALSE)
cat(sprintf("  stm_topic_terms.csv   → %d rows\n", nrow(terms_df)))

# ── stm_prevalence.csv — audience regression coefficients ─────────────────────
write.csv(prev_df, file.path(OUTPUT_DIR, "stm_prevalence.csv"), row.names = FALSE)
cat(sprintf("  stm_prevalence.csv    → %d rows\n", nrow(prev_df)))

# ── stm_content.csv — per-audience top words per topic ────────────────────────
if (!is.null(content_formula)) {
  tryCatch({
    sage <- sageLabels(stm_model, n = 20)
    content_df <- do.call(rbind, lapply(1:2, function(aud_idx) {
      aud_name <- c("client", "worker")[aud_idx]
      mat <- sage$wordcov$audience[[aud_idx]]   # terms × K
      do.call(rbind, lapply(1:K, function(t) {
        ord <- order(mat[, t], decreasing = TRUE)[1:20]
        data.frame(
          topic_id = t,
          audience = aud_name,
          rank     = seq_along(ord),
          term     = rownames(mat)[ord],
          score    = round(mat[ord, t], 6),
          stringsAsFactors = FALSE
        )
      }))
    }))
    write.csv(content_df, file.path(OUTPUT_DIR, "stm_content.csv"),
              row.names = FALSE)
    cat(sprintf("  stm_content.csv       → %d rows\n", nrow(content_df)))
  }, error = function(e) {
    cat(sprintf("  WARNING: stm_content.csv not written: %s\n", e$message))
  })
} else {
  cat("  stm_content.csv       → skipped (three-level audience)\n")
}


# =============================================================================
# DONE
# =============================================================================

cat("\n─────────────────────────────────────────────────────────────────────────\n")
cat("STM complete.\n")
cat(sprintf("  Final model  : K=%d | documents=%d | vocabulary=%d\n",
            K, length(out$documents), length(out$vocab)))
cat(sprintf("  Output dir   : %s\n", normalizePath(OUTPUT_DIR)))
cat("\nFiles:\n")
for (f in c("stm_model.rds", "out.rds",
            "searchK_approach1.rds", "searchK_approach2.rds", "stm_leemimno.rds",
            "stm_theta.csv", "stm_topic_terms.csv",
            "stm_prevalence.csv", "stm_content.csv")) {
  p <- file.path(OUTPUT_DIR, f)
  cat(sprintf("  %s  %s\n", if (file.exists(p)) "✓" else "○ (not yet)", f))
}
cat("─────────────────────────────────────────────────────────────────────────\n")
