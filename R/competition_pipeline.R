# High-level competition pipeline orchestrating the Two-Phase Decoupled Strategy
# documented in ML_STRATEGY.md:
#   Stage 0: Recon, Drift Check & Target/Metric Setup
#   Stage 2: Leak-Free Unsupervised Feature Discovery (on subsampled rows when N > max_search_rows)
#   Stage 3: Mega-Matrix Synthesis onto 100% of train and test
#   Stage 4: High-Capacity 1-Gen Retraining with K-Fold Model Bagging
#   Stage 5: Stacked Ensembling & Submission Export

#' Preset hyperparameter and budget configurations for run_competition_pipeline
#' @keywords internal
#' @noRd
.competition_preset_config <- function(preset = c("standard", "fast", "max")) {
  preset <- match.arg(preset)
  switch(preset,
    fast = list(
      islands = 2L,
      s2_gen = 2L,
      s2_pop = 4L,
      s2_mlp_gen = 2L,
      s2_mlp_pop = 3L,
      s2_nrounds = 60L,
      s2_es = 10L,
      s2_mlp_epochs = 12L,
      s2_mf = FALSE,
      s2_holdout = 0.0,
      s2_sd_penalty = 0.10,
      s2_confirm_k = 0L,
      clust_cap = 2000L,
      s4_lgb_lr = 0.05,
      s4_lgb_nrounds = 200L,
      s4_lgb_es = 25L,
      s4_xgb_lr = 0.05,
      s4_xgb_nrounds = 180L,
      s4_xgb_es = 25L,
      s4_mlp_epochs = 30L,
      s4_mlp_lr = 0.008,
      s4_mlp_es = 8L,
      s4_mlp_hidden = 64L
    ),
    standard = list(
      islands = 4L,
      s2_gen = 10L,
      s2_pop = 8L,
      s2_mlp_gen = 8L,
      s2_mlp_pop = 6L,
      s2_nrounds = 300L,
      s2_es = 25L,
      s2_mlp_epochs = 30L,
      s2_mf = TRUE,
      s2_holdout = 0.08,
      s2_sd_penalty = 0.15,
      s2_confirm_k = 3L,
      clust_cap = 25000L,
      s4_lgb_lr = 0.005,
      s4_lgb_nrounds = 50000L,
      s4_lgb_es = 250L,
      s4_xgb_lr = 0.008,
      s4_xgb_nrounds = 35000L,
      s4_xgb_es = 200L,
      s4_mlp_epochs = 250L,
      s4_mlp_lr = 0.005,
      s4_mlp_es = 30L,
      s4_mlp_hidden = 256L
    ),
    max = list(
      islands = 6L,
      s2_gen = 12L,
      s2_pop = 10L,
      s2_mlp_gen = 10L,
      s2_mlp_pop = 8L,
      s2_nrounds = 400L,
      s2_es = 30L,
      s2_mlp_epochs = 40L,
      s2_mf = TRUE,
      s2_holdout = 0.08,
      s2_sd_penalty = 0.20,
      s2_confirm_k = 3L,
      clust_cap = 50000L,
      s4_lgb_lr = 0.003,
      s4_lgb_nrounds = 50000L,
      s4_lgb_es = 300L,
      s4_xgb_lr = 0.005,
      s4_xgb_nrounds = 40000L,
      s4_xgb_es = 250L,
      s4_mlp_epochs = 400L,
      s4_mlp_lr = 0.004,
      s4_mlp_es = 40L,
      s4_mlp_hidden = 384L
    )
  )
}

#' Filter requested evaluators to registered backends whose packages are installed
#' @keywords internal
#' @noRd
.resolve_competition_evaluators <- function(evaluators) {
  if (!is.character(evaluators) || length(evaluators) == 0L) {
    stop("'evaluators' must be a non-empty character vector.")
  }
  evaluators <- unique(evaluators)
  pkg_map <- c(lightgbm = "lightgbm", xgboost = "xgboost", catboost = "catboost", keras3 = "keras3")
  kept <- character(0)
  for (ev in evaluators) {
    if (!has_evaluator(ev)) {
      warning(sprintf("Evaluator '%s' is not registered in evo_evaluators; skipping.", ev))
      next
    }
    base_ev <- get_base_evaluator(ev)
    req_pkg <- unname(pkg_map[base_ev])
    if (!is.na(req_pkg) && nzchar(req_pkg) && !requireNamespace(req_pkg, quietly = TRUE)) {
      warning(sprintf("Package '%s' required for evaluator '%s' is not installed; skipping.", req_pkg, ev))
      next
    }
    kept <- c(kept, ev)
  }
  if (length(kept) == 0L) {
    stop("None of the requested 'evaluators' are available in this R installation.")
  }
  kept
}

#' Auto-select regression target transformation based on skewness and positivity
#' @keywords internal
#' @noRd
.resolve_target_transform <- function(y, task, target_transform) {
  target_transform <- match.arg(target_transform, c("auto", "none", "log1p", "yeo_johnson", "rank_gauss"))
  if (task != "regression") return("none")
  if (target_transform != "auto") return(target_transform)
  y_num <- as.numeric(y[is.finite(y)])
  if (length(y_num) < 20L) return("none")
  s <- stats::sd(y_num)
  if (!is.finite(s) || s < 1e-12) return("none")
  skew <- mean(((y_num - mean(y_num)) / s)^3)
  if (is.finite(skew) && skew > 1.5 && min(y_num) >= 0) "log1p" else "none"
}

#' Build evaluator-specific training arguments for Stage 2 or Stage 4
#' @keywords internal
#' @noRd
.evaluator_stage_args <- function(ev, stage, cfg, n_rows) {
  base_ev <- get_base_evaluator(ev)
  if (stage == 2L) {
    if (base_ev == "lightgbm") {
      return(list(
        learning_rate = 0.08,
        num_leaves = 63L,
        nrounds = cfg$s2_nrounds,
        early_stopping_rounds = cfg$s2_es
      ))
    }
    if (base_ev == "xgboost") {
      return(list(
        tree_method = "hist",
        learning_rate = 0.08,
        max_depth = 6L,
        nrounds = cfg$s2_nrounds,
        early_stopping_rounds = cfg$s2_es
      ))
    }
    if (base_ev == "realmlp") {
      bs <- if (n_rows >= 8192L) 2048L else -1L
      return(list(
        realmlp_epochs = cfg$s2_mlp_epochs,
        realmlp_lr = 0.01,
        early_stopping_rounds = min(10L, cfg$s2_mlp_epochs),
        hidden_dim = cfg$s4_mlp_hidden,
        batch_size = bs
      ))
    }
    return(list(nrounds = cfg$s2_nrounds, early_stopping_rounds = cfg$s2_es))
  }

  # Stage 4 high-capacity retraining
  if (base_ev == "lightgbm") {
    return(list(
      learning_rate = cfg$s4_lgb_lr,
      num_leaves = 63L,
      feature_fraction = 0.75,
      bagging_fraction = 0.80,
      bagging_freq = 1L,
      lambda_l1 = 0.5,
      lambda_l2 = 2.0,
      nrounds = cfg$s4_lgb_nrounds,
      early_stopping_rounds = cfg$s4_lgb_es
    ))
  }
  if (base_ev == "xgboost") {
    return(list(
      tree_method = "hist",
      learning_rate = cfg$s4_xgb_lr,
      max_depth = 6L,
      subsample = 0.80,
      colsample_bytree = 0.75,
      lambda = 2.0,
      nrounds = cfg$s4_xgb_nrounds,
      early_stopping_rounds = cfg$s4_xgb_es
    ))
  }
  if (base_ev == "realmlp") {
    bs <- if (n_rows >= 8192L) 2048L else -1L
    return(list(
      epochs = cfg$s4_mlp_epochs,
      lr = cfg$s4_mlp_lr,
      early_stopping_rounds = cfg$s4_mlp_es,
      hidden_dim = cfg$s4_mlp_hidden,
      batch_size = bs
    ))
  }
  list(nrounds = cfg$s4_xgb_nrounds, early_stopping_rounds = cfg$s4_xgb_es)
}

#' Run the Two-Phase ML Competition Pipeline End-to-End
#'
#' Executes the complete high-capacity tabular competition workflow from
#' \code{ML_STRATEGY.md} given \code{train}, \code{test}, and \code{target_col}:
#' \enumerate{
#'   \item \strong{Stage 0 (Recon & Drift Diagnostics)}: Infers task type, selects
#'     competition-grade search metrics (\code{"ts_refinement"} for probability/ranking
#'     classification, \code{"default"} RMSE for regression), auto-detects \code{"log1p"}
#'     target transformation on skewed non-negative regression targets, and runs
#'     \code{\link{compute_adversarial_weights}} between \code{train} and \code{test}.
#'   \item \strong{Phase-1 Smart Downsampling}: When \code{nrow(train) > max_search_rows},
#'     downsamples \strong{only} the feature-discovery table via
#'     \code{\link{subsample_for_discovery}} (preserving groups, temporal contiguity,
#'     rare-event counts, and test propensity weights) while keeping 100\% of \code{train}
#'     for Stage 3--5 retraining.
#'   \item \strong{Stage 2 (Multi-Evaluator Leak-Free Discovery)}: Evolves features
#'     independently for each active evaluator using strictly unsupervised transformers
#'     so no target encoding leaks into Stage 4 cross-validation.
#'   \item \strong{Stage 3 (Mega-Matrix Synthesis)}: Combines all Stage 2 scout recipes via
#'     \code{ensemble_islands(..., method = "equal")} and projects learned states onto
#'     100\% of \code{train} and \code{test}.
#'   \item \strong{Stage 4 (High-Capacity Retraining)}: Trains each model family on full
#'     \code{train_mega} with low learning rates, deep early-stopping ceilings, and
#'     \code{bag_folds = TRUE} (optionally adding fold-local supervised target encoders
#'     when \code{supervised_stage4 = TRUE}).
#'   \item \strong{Stage 5 (Stacking & Submission)}: Blends Stage 4 models via nested
#'     out-of-fold elastic-net stacking (\code{ensemble_islands(..., method = "stack")})
#'     and builds a ready-to-submit \code{data.table} (writing \code{submission_file} if given).
#' }
#'
#' @param train Training \code{data.frame} or \code{data.table} containing \code{target_col}.
#' @param test Test \code{data.frame} or \code{data.table}.
#' @param target_col Character string naming the target column in \code{train}.
#' @param id_col Optional character string naming an identifier column to exclude from
#'   features and prepend to \code{$submission}.
#' @param task Modeling task: \code{"auto"} (default), \code{"classification"},
#'   \code{"multiclass"}, or \code{"regression"}.
#' @param metric Optimization metric name (e.g. \code{"auto"}, \code{"ts_refinement"},
#'   \code{"auc"}, \code{"aucpr"}, \code{"f1"}, \code{"mcc"}, \code{"qwk"}, \code{"rmsle"},
#'   \code{"mae"}, \code{"smape"}) or custom metric function.
#' @param preset Compute budget preset: \code{"standard"} (competition playbook defaults),
#'   \code{"fast"} (quick smoke/baseline run), or \code{"max"} (deep schedule).
#' @param evaluators Character vector of model families to run (default
#'   \code{c("lightgbm", "xgboost", "realmlp")}). Unavailable packages are skipped with a warning.
#' @param cv_folds Number of cross-validation folds for Stage 4 retraining and stacking (default \code{5L}).
#' @param cv_strategy Cross-validation split strategy: \code{"random"},
#'   \code{"stratified_group"}, \code{"group"}, or \code{"time"}.
#' @param group_col Optional group column name when \code{cv_strategy} is grouped.
#' @param time_col Optional timestamp/order column name when \code{cv_strategy = "time"}.
#' @param target_transform Regression target transform: \code{"auto"} (selects \code{"log1p"}
#'   when skewness > 1.5 and \code{min(y) >= 0}), \code{"none"}, \code{"log1p"},
#'   \code{"yeo_johnson"}, or \code{"rank_gauss"}.
#' @param max_search_rows Maximum rows used in Stage 2 feature discovery (default \code{200000L};
#'   set \code{NULL} or \code{Inf} to disable Phase-1 subsampling). Stage 4 always trains on all rows.
#' @param drop_drifting Logical (default \code{FALSE}). If \code{TRUE}, drops raw columns whose
#'   single-feature train-vs-test AUC exceeds \code{drift_auc_threshold}.
#' @param drift_auc_threshold Single-feature adversarial AUC cutoff (default \code{0.85}).
#' @param supervised_stage4 Logical (default \code{FALSE}). If \code{TRUE}, runs a fast
#'   surrogate search over supervised encoders on \code{train_mega} (subsampled to
#'   \code{max_search_rows} if huge) and passes only the \strong{unfitted} winning gene
#'   formulas via \code{seed_genes} into Stage 4's \code{generations = 1, pop_size = 1}
#'   deep fold-bagged refit (zero target leakage and zero deep-schedule search overhead).
#' @param bag_folds Logical (default \code{TRUE}). Retains and averages all \code{cv_folds}
#'   fold models during test prediction.
#' @param submission_file Optional file path to write \code{$submission} via \code{data.table::fwrite()}.
#' @param threads Integer thread count (default \code{\link{default_threads}()}).
#' @param seed Integer RNG seed for reproducibility (caller's \code{.Random.seed} is preserved).
#' @param verbose Logical or integer verbosity level.
#' @return An S3 object of class \code{"evo_competition_run"} containing:
#'   \code{predictions}, \code{submission}, \code{final_model} (\code{evo_ensemble} or
#'   \code{evo_recipe}), \code{stage2_recipes}, \code{stage4_recipes}, \code{scout_ensemble},
#'   \code{train_mega}, \code{test_mega}, \code{adversarial}, \code{task}, \code{metric},
#'   \code{target_transform}, \code{discovery_rows}, \code{train_rows}, and \code{stage4_fitness}.
#' @export
run_competition_pipeline <- function(
  train,
  test,
  target_col,
  id_col = NULL,
  task = c("auto", "classification", "multiclass", "regression"),
  metric = "auto",
  preset = c("standard", "fast", "max"),
  evaluators = c("lightgbm", "xgboost", "realmlp"),
  cv_folds = 5L,
  cv_strategy = c("random", "stratified_group", "group", "time"),
  group_col = NULL,
  time_col = NULL,
  target_transform = c("auto", "none", "log1p", "yeo_johnson", "rank_gauss"),
  max_search_rows = 200000L,
  drop_drifting = FALSE,
  drift_auc_threshold = 0.85,
  supervised_stage4 = FALSE,
  bag_folds = TRUE,
  submission_file = NULL,
  threads = default_threads(),
  seed = 42L,
  verbose = TRUE
) {
  if (!is.data.frame(train) || nrow(train) < 10L) {
    stop("'train' must be a data.frame or data.table with at least 10 rows.")
  }
  if (!is.data.frame(test) || nrow(test) < 1L) {
    stop("'test' must be a data.frame or data.table with at least 1 row.")
  }
  if (!is.character(target_col) || length(target_col) != 1L || !target_col %in% names(train)) {
    stop(sprintf("Target column '%s' not found in 'train'.", as.character(target_col)[1L]))
  }
  if (anyNA(train[[target_col]])) {
    stop(sprintf("Target column '%s' contains missing (NA) values.", target_col))
  }
  if (!is.null(id_col)) {
    if (!is.character(id_col) || length(id_col) != 1L || !id_col %in% names(test)) {
      stop("'id_col' must be NULL or name an existing column in 'test'.")
    }
  }

  preset <- match.arg(preset)
  cv_strategy <- match.arg(cv_strategy)
  cv_folds <- as.integer(cv_folds)
  if (is.na(cv_folds) || cv_folds < 2L) stop("'cv_folds' must be an integer >= 2.")
  cfg <- .competition_preset_config(preset)
  active_evaluators <- .resolve_competition_evaluators(evaluators)

  env_state <- setup_core_env(threads = threads, max_clustering_size = cfg$clust_cap, seed = seed)
  on.exit(restore_core_env(env_state), add = TRUE)

  old_red <- getOption("evoFE.redundancy_cor_threshold")
  old_imp <- getOption("evoFE.importance_threshold")
  options(evoFE.redundancy_cor_threshold = 0.99, evoFE.importance_threshold = 0.002)
  on.exit(options(evoFE.redundancy_cor_threshold = old_red, evoFE.importance_threshold = old_imp), add = TRUE)

  y_vec <- train[[target_col]]
  resolved_task <- .infer_task_type(y_vec, task)
  resolved_metric <- if (identical(metric, "auto")) {
    if (resolved_task %in% c("classification", "multiclass")) "ts_refinement" else "default"
  } else {
    metric
  }
  validate_metric(resolved_metric, resolved_task)
  resolved_tt <- .resolve_target_transform(y_vec, resolved_task, target_transform)

  test_ids <- if (!is.null(id_col)) test[[id_col]] else NULL
  excl_cols <- unique(c(target_col, id_col))
  shared_features <- intersect(setdiff(names(train), excl_cols), setdiff(names(test), excl_cols))
  if (length(shared_features) == 0L) {
    stop("No shared feature columns found between 'train' and 'test'.")
  }

  train_dt <- shallow_copy_dt(data.table::as.data.table(train)[, c(shared_features, target_col), with = FALSE])
  test_dt <- shallow_copy_dt(data.table::as.data.table(test)[, shared_features, with = FALSE])

  # Stage 0: Adversarial covariate-shift check
  adv_info <- compute_adversarial_weights(
    train_dt, test_dt,
    target_col = target_col,
    drift_auc_threshold = drift_auc_threshold
  )
  dropped_drift_cols <- character(0)
  if (isTRUE(drop_drifting) && length(adv_info$drifting_features) > 0L) {
    protected <- unique(c(group_col, time_col))
    cand_drop <- setdiff(adv_info$drifting_features, protected)
    if (length(cand_drop) > 0L && length(cand_drop) < length(shared_features)) {
      dropped_drift_cols <- cand_drop
      keep_tr <- setdiff(names(train_dt), dropped_drift_cols)
      keep_te <- setdiff(names(test_dt), dropped_drift_cols)
      train_dt <- train_dt[, keep_tr, with = FALSE]
      test_dt <- test_dt[, keep_te, with = FALSE]
      shared_features <- setdiff(shared_features, dropped_drift_cols)
    }
  }

  show_msg <- isTRUE(verbose) || (is.numeric(verbose) && verbose >= 1L)
  if (show_msg) {
    message(sprintf(
      "[Stage 0] Task: %s | Metric: %s | TargetTransform: %s | Adversarial AUC: %.4f",
      resolved_task,
      if (is.character(resolved_metric)) resolved_metric else "custom",
      resolved_tt,
      adv_info$adversarial_auc
    ))
  }

  # Phase-1 Smart Downsampling (Stage 2 Discovery ONLY; Stage 3-5 use 100% of train_dt)
  eff_adv_w <- if (is.finite(adv_info$adversarial_auc) && adv_info$adversarial_auc >= 0.60) {
    adv_info$weights
  } else {
    NULL
  }
  search_dt <- subsample_for_discovery(
    data = train_dt,
    target_col = target_col,
    max_rows = max_search_rows,
    task = resolved_task,
    cv_strategy = cv_strategy,
    group_col = group_col,
    time_col = time_col,
    adv_weights = eff_adv_w,
    seed = seed
  )
  n_discovery <- nrow(search_dt)
  if (show_msg && isTRUE(attr(search_dt, "subsampled"))) {
    message(sprintf(
      "[Phase 1] Subsampled %d -> %d rows for Stage 2 feature discovery (full %d rows retained for Stage 4).",
      nrow(train_dt), n_discovery, nrow(train_dt)
    ))
  }

  # Stage 2: Leak-free unsupervised exploration per model family
  all_t <- names(evo_transformers)
  unsup_t <- all_t[!vapply(all_t, is_supervised_transformer, logical(1L))]
  sup_t <- all_t[vapply(all_t, is_supervised_transformer, logical(1L))]

  use_metacv <- (n_discovery >= 1000L) && (cv_strategy != "time") && (cfg$islands >= 2L)
  mig_cfg <- if (cfg$islands >= 2L) {
    migration_config(
      topology = topology_complete(islands = cfg$islands),
      policy = policy_gibbs_pull(stagnation_threshold = 2L, temperature = 1.0),
      payload = "full_individual"
    )
  } else {
    NULL
  }

  stage2_recipes <- vector("list", length(active_evaluators))
  names(stage2_recipes) <- active_evaluators
  base_seed <- if (!is.null(seed)) as.integer(seed) else 42L

  for (idx in seq_along(active_evaluators)) {
    ev <- active_evaluators[idx]
    base_ev <- get_base_evaluator(ev)
    is_mlp <- identical(base_ev, "realmlp")
    if (show_msg) {
      message(sprintf("[Stage 2] Discovering features with '%s' (%d/%d)...", ev, idx, length(active_evaluators)))
    }

    s2_common <- list(
      data = search_dt,
      target_col = target_col,
      task = resolved_task,
      evaluator = ev,
      metric = resolved_metric,
      allowed_transformers = unsup_t,
      evaluation_strategy = if (use_metacv) "metacv" else "cv",
      metacv_selection = "tournament",
      cv_folds = if (use_metacv) cfg$islands else min(cv_folds, 5L),
      cv_strategy = cv_strategy,
      group_col = group_col,
      time_col = time_col,
      holdout_frac = if (n_discovery >= 500L) cfg$s2_holdout else 0.0,
      test_data = if (n_discovery >= 500L && cfg$s2_holdout > 0) test_dt else NULL,
      sd_penalty = cfg$s2_sd_penalty,
      confirm_top_k = if (!use_metacv && cv_strategy != "time") cfg$s2_confirm_k else 0L,
      migration = mig_cfg,
      islands = cfg$islands,
      migration_interval = 1L,
      migration_rate = 1L,
      gene_migration_prob = 0.25,
      generations = if (is_mlp) cfg$s2_mlp_gen else cfg$s2_gen,
      pop_size = if (is_mlp) cfg$s2_mlp_pop else cfg$s2_pop,
      multi_fidelity = isTRUE(cfg$s2_mf) && (n_discovery >= 1000L),
      mf_sample_frac = 0.4,
      mf_warmup_frac = 0.4,
      complexity_penalty = if (is_mlp) 0.30 else 0.25,
      complexity_mode = if (n_discovery < 10000L) "pac_bayes_dynamic" else "bic_dynamic",
      complexity_target = "all_features",
      target_transform = resolved_tt,
      threads = threads,
      max_clustering_size = min(cfg$clust_cap, n_discovery),
      seed = base_seed + 100L * idx,
      verbose = verbose
    )
    ev_args <- .evaluator_stage_args(ev, stage = 2L, cfg = cfg, n_rows = n_discovery)
    stage2_recipes[[ev]] <- do.call(evolve_features, c(s2_common, ev_args))
  }

  # Stage 3: Mega-Matrix Synthesis onto 100% of train_dt and test_dt
  if (length(stage2_recipes) >= 2L) {
    scout_ens <- ensemble_islands(
      recipe = stage2_recipes,
      data = search_dt,
      target_col = target_col,
      method = "equal",
      threads = threads,
      verbose = FALSE
    )
    train_mega <- stats::predict(scout_ens, newdata = train_dt)
    test_mega <- stats::predict(scout_ens, newdata = test_dt)
  } else {
    scout_ens <- NULL
    train_mega <- stats::predict(stage2_recipes[[1L]], newdata = train_dt)
    test_mega <- stats::predict(stage2_recipes[[1L]], newdata = test_dt)
  }

  # Preserve group_col / time_col if Stage 2 active masks pruned them
  for (aux_col in unique(c(group_col, time_col))) {
    if (!is.null(aux_col) && nzchar(aux_col) && aux_col %in% names(train_dt) && !aux_col %in% names(train_mega)) {
      train_mega[[aux_col]] <- train_dt[[aux_col]]
      if (aux_col %in% names(test_dt)) {
        test_mega[[aux_col]] <- test_dt[[aux_col]]
      }
    }
  }
  train_mega[[target_col]] <- train_dt[[target_col]]

  if (show_msg) {
    message(sprintf(
      "[Stage 3] Mega-Matrix synthesized on %d full training rows: %d raw -> %d engineered features.",
      nrow(train_mega), length(shared_features), ncol(train_mega) - 1L
    ))
  }

  # Optional Stage 3.5: Fast decoupled supervised discovery on train_mega
  # Searches at fast Stage-2 surrogate speeds (subsampled if > max_search_rows),
  # then extracts ONLY unfitted symbolic gene formulas (state = NULL) for Stage 4!
  stage3_sup_recipes <- NULL
  sup_seed_genes <- stats::setNames(vector("list", length(active_evaluators)), active_evaluators)
  mega_feature_names <- setdiff(names(train_mega), target_col)

  if (isTRUE(supervised_stage4) && length(sup_t) > 0L) {
    sup_search_dt <- subsample_for_discovery(
      data = train_mega,
      target_col = target_col,
      max_rows = max_search_rows,
      task = resolved_task,
      cv_strategy = cv_strategy,
      group_col = group_col,
      time_col = time_col,
      adv_weights = eff_adv_w,
      seed = base_seed + 500L
    )
    stage3_sup_recipes <- vector("list", length(active_evaluators))
    names(stage3_sup_recipes) <- active_evaluators
    sup_gens <- if (preset == "fast") 2L else 3L
    sup_pop <- if (preset == "fast") 4L else 6L

    for (idx in seq_along(active_evaluators)) {
      ev <- active_evaluators[idx]
      if (show_msg) {
        message(sprintf(
          "[Stage 3.5] Fast supervised encoder search on %d rows for '%s'...",
          nrow(sup_search_dt), ev
        ))
      }
      s35_common <- list(
        data = sup_search_dt,
        target_col = target_col,
        task = resolved_task,
        evaluator = ev,
        metric = resolved_metric,
        generations = sup_gens,
        pop_size = sup_pop,
        islands = 1L,
        allowed_transformers = sup_t,
        evaluation_strategy = "cv",
        cv_folds = min(cv_folds, 5L),
        cv_strategy = cv_strategy,
        group_col = group_col,
        time_col = time_col,
        sd_penalty = cfg$s2_sd_penalty,
        confirm_top_k = if (preset != "fast" && cv_strategy != "time") 2L else 0L,
        raw_toggle_prob = 0.0,
        bag_folds = FALSE,
        target_transform = resolved_tt,
        threads = threads,
        seed = base_seed + 500L + idx,
        verbose = verbose
      )
      ev_fast_args <- .evaluator_stage_args(ev, stage = 2L, cfg = cfg, n_rows = nrow(sup_search_dt))
      rec_sup <- do.call(evolve_features, c(s35_common, ev_fast_args))
      stage3_sup_recipes[[ev]] <- rec_sup
      sup_seed_genes[[ev]] <- normalize_seed_genes(rec_sup, mega_feature_names)
    }
  }

  # Stage 4: High-capacity 1-Gen retraining on 100% of train_mega
  stage4_recipes <- vector("list", length(active_evaluators))
  names(stage4_recipes) <- active_evaluators
  s4_seed <- base_seed + 1000L

  for (idx in seq_along(active_evaluators)) {
    ev <- active_evaluators[idx]
    if (show_msg) {
      message(sprintf("[Stage 4] High-capacity %d-fold retraining for '%s'...", cv_folds, ev))
    }
    s4_common <- list(
      data = train_mega,
      target_col = target_col,
      task = resolved_task,
      evaluator = ev,
      metric = resolved_metric,
      generations = 1L,
      pop_size = 1L,
      seed_genes = sup_seed_genes[[ev]],
      evaluation_strategy = "cv",
      cv_folds = cv_folds,
      cv_strategy = cv_strategy,
      group_col = group_col,
      time_col = time_col,
      bag_folds = isTRUE(bag_folds),
      target_transform = resolved_tt,
      threads = threads,
      seed = s4_seed,
      verbose = verbose
    )
    ev_args <- .evaluator_stage_args(ev, stage = 4L, cfg = cfg, n_rows = nrow(train_mega))
    stage4_recipes[[ev]] <- do.call(evolve_features, c(s4_common, ev_args))
  }

  stage4_fitness <- vapply(stage4_recipes, function(r) {
    val <- r$best_individual$raw_fitness
    if (is.null(val) || !is.finite(val)) r$best_individual$fitness else val
  }, numeric(1L))

  # Stage 5: Stacked Ensembling & Test Inference
  if (length(stage4_recipes) >= 2L) {
    if (show_msg) message("[Stage 5] Stacking Stage 4 models via nested OOF elastic-net...")
    final_model <- ensemble_islands(
      recipe = stage4_recipes,
      data = train_mega,
      target_col = target_col,
      method = "stack",
      stack_folds = cv_folds,
      stack_alpha = 0.5,
      bag_folds = isTRUE(bag_folds),
      threads = threads,
      seed = s4_seed,
      verbose = show_msg
    )
  } else {
    final_model <- stage4_recipes[[1L]]
  }

  preds <- predict_model(final_model, test_mega, bag_folds = isTRUE(bag_folds), threads = threads)

  if (is.matrix(preds)) {
    sub_dt <- data.table::as.data.table(preds)
    if (!is.null(id_col)) {
      id_dt <- stats::setNames(data.table::data.table(test_ids), id_col)
      sub_dt <- cbind(id_dt, sub_dt)
    }
  } else {
    pred_vec <- as.numeric(preds)
    if (!is.null(id_col)) {
      sub_dt <- data.table::data.table(id_val = test_ids, target_val = pred_vec)
      data.table::setnames(sub_dt, c("id_val", "target_val"), c(id_col, target_col))
    } else {
      sub_dt <- stats::setNames(data.table::data.table(pred_vec), target_col)
    }
  }

  if (!is.null(submission_file) && is.character(submission_file) && length(submission_file) == 1L && nzchar(submission_file)) {
    data.table::fwrite(sub_dt, submission_file)
    if (show_msg) message(sprintf("[Stage 6] Wrote submission file: %s", submission_file))
  }

  ens_fitness <- if (inherits(final_model, "evo_ensemble")) {
    if (!is.null(final_model$stack_cv_fitness) && is.finite(final_model$stack_cv_fitness)) {
      final_model$stack_cv_fitness
    } else {
      final_model$ensemble_val_fitness
    }
  } else {
    unname(stage4_fitness[1L])
  }

  resolved_weights <- if (inherits(final_model, "evo_ensemble") && !is.null(final_model$weights)) {
    w_raw <- final_model$weights
    stats::setNames(vapply(active_evaluators, function(ev) {
      hit <- names(w_raw) == ev | startsWith(names(w_raw), paste0(ev, "_island_"))
      if (any(hit)) sum(w_raw[hit]) else 0.0
    }, numeric(1L)), active_evaluators)
  } else {
    stats::setNames(1.0, active_evaluators[1L])
  }

  structure(
    list(
      predictions = preds,
      submission = sub_dt,
      final_model = final_model,
      stage2_recipes = stage2_recipes,
      stage3_supervised_recipes = stage3_sup_recipes,
      stage4_recipes = stage4_recipes,
      scout_ensemble = scout_ens,
      train_mega = train_mega,
      test_mega = test_mega,
      adversarial = adv_info,
      dropped_drifting_features = dropped_drift_cols,
      task = resolved_task,
      metric = resolved_metric,
      target_col = target_col,
      id_col = id_col,
      preset = preset,
      evaluators = active_evaluators,
      cv_folds = cv_folds,
      cv_strategy = cv_strategy,
      target_transform = resolved_tt,
      bag_folds = isTRUE(bag_folds),
      raw_feature_count = length(shared_features),
      mega_feature_count = ncol(train_mega) - 1L,
      train_rows = nrow(train_dt),
      discovery_rows = n_discovery,
      test_rows = nrow(test_dt),
      stage4_fitness = stage4_fitness,
      final_cv_fitness = ens_fitness,
      weights = resolved_weights
    ),
    class = "evo_competition_run"
  )
}
