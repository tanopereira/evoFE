#!/usr/bin/env Rscript
# Runnable End-to-End Competition Pipeline Script for evoFE
#
# Implements the Two-Phase Decoupled Playbook from ML_STRATEGY.md:
#   Stage 0: Recon, Adversarial Drift Check & Metric/Target Setup
#   Phase 1: Smart Structure-Preserving Subsampling (when nrow(train) > max_search_rows)
#   Stage 2: Multi-Evaluator Leak-Free Feature Discovery (LightGBM + XGBoost + RealMLP)
#   Stage 3: Mega-Matrix Synthesis onto 100% of train and test
#   Stage 4: High-Capacity 1-Gen Retraining with K-Fold Model Bagging (bag_folds = TRUE)
#   Stage 5: Nested OOF Elastic-Net Stacking & Submission Export
#
# CLI Usage:
#   Rscript demo/competition_pipeline.R <train.csv> <test.csv> <target_col> \
#     [submission.csv] [id_col] [metric] [preset] [max_search_rows]
#
# Example:
#   Rscript demo/competition_pipeline.R train.csv test.csv target submission.csv id auc standard 200000
#
# Interactive / Demo Usage (runs a fast synthetic benchmark when no CLI args are passed):
#   demo("competition_pipeline", package = "evoFE")

if (!("evoFE" %in% loadedNamespaces())) {
  library(evoFE)
}
library(data.table)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 3L) {
  train_path  <- args[1L]
  test_path   <- args[2L]
  target_col  <- args[3L]
  sub_path    <- if (length(args) >= 4L && nzchar(args[4L])) args[4L] else "submission.csv"
  id_col      <- if (length(args) >= 5L && nzchar(args[5L]) && args[5L] != "NULL") args[5L] else NULL
  metric_arg  <- if (length(args) >= 6L && nzchar(args[6L])) args[6L] else "auto"
  preset_arg  <- if (length(args) >= 7L && nzchar(args[7L])) args[7L] else "standard"
  max_rows    <- if (length(args) >= 8L && nzchar(args[8L])) as.numeric(args[8L]) else 200000L

  message(sprintf(">>> Loading train (%s) and test (%s)...", train_path, test_path))
  train_dt <- fread(train_path)
  test_dt  <- fread(test_path)

  res <- run_competition_pipeline(
    train           = train_dt,
    test            = test_dt,
    target_col      = target_col,
    id_col          = id_col,
    metric          = metric_arg,
    preset          = preset_arg,
    max_search_rows = max_rows,
    submission_file = sub_path,
    verbose         = TRUE
  )

  print(summary(res))
  message("\n>>> Preview of generated submission:")
  print(head(res$submission))
} else {
  message(">>> No CLI CSV arguments supplied; running fast synthetic competition demo...")
  set.seed(42)
  n_tr <- 240L
  n_te <- 50L
  sim_table <- function(n, offset_id) {
    x1 <- stats::rnorm(n)
    x2 <- stats::rnorm(n)
    price <- round(stats::runif(n, 10, 100), 2)
    cohort <- sample(c("enterprise", "smb", "consumer"), n, replace = TRUE)
    eta <- 1.1 * x1 - 0.8 * x2 + 0.4 * sin(price / 10) + ifelse(cohort == "enterprise", 0.7, -0.2)
    data.table(
      id = seq.int(offset_id, length.out = n),
      x1 = x1,
      x2 = x2,
      price = price,
      cohort = cohort,
      eta = eta
    )
  }

  train_demo <- sim_table(n_tr, 1L)
  train_demo[, target := as.integer(stats::rbinom(.N, 1L, stats::plogis(eta)))]
  train_demo[, eta := NULL]

  test_demo <- sim_table(n_te, 10001L)[, eta := NULL]

  res <- run_competition_pipeline(
    train           = train_demo,
    test            = test_demo,
    target_col      = "target",
    id_col          = "id",
    preset          = "fast",
    cv_folds        = 3L,
    max_search_rows = 160L,
    verbose         = TRUE
  )

  message("\n>>> Competition Run Summary:")
  print(summary(res))
  message("\n>>> First 6 Submission Rows:")
  print(head(res$submission))
}
