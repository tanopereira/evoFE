# High-level competition pipeline orchestrating the Two-Phase Decoupled Strategy
# documented in ML_STRATEGY.md:
#   Stage 0: Recon, Drift Check & Target/Metric Setup
#   Stage 2: Leak-Free Multi-Evaluator Discovery (on subsampled rows when N > max_search_rows)
#   Stage 3: Unsupervised Mega-Matrix Synthesis onto 100% of train and test + View Tournament
#   Stage 4: High-Capacity 1-Gen Retraining with K-Fold Model Bagging (& Fold-Local Supervised Seeds)
#   Stage 5: Stacked Ensembling & Submission Export

#' Preset hyperparameter and budget configurations for run_competition_pipeline
#' @keywords internal
#' @noRd
.competition_preset_config <- function(preset = c("standard", "fast", "max"), overrides = list()) {
  preset <- match.arg(preset)
  cfg <- switch(preset,
    fast = list(
      islands = 2L,
      islands_per_evaluator = 1L,
      s2_gen = 2L,
      s2_pop = 4L,
      s2_mlp_gen = 2L,
      s2_mlp_pop = 3L,
      s2_nrounds = 60L,
      s2_es = 10L,
      s2_mlp_epochs = 12L,
      s2_mlp_es = 12L,
      s2_holdout = 0.0,
      s2_sd_penalty = 0.10,
      s2_complexity_penalty = NULL,
      s2_complexity_mode = "auto",
      s2_confirm_k = 0L,
      clust_cap = 2000L,
      s2_mbo_iters = 1L,
      s2_mbo_init_design = 2L,
      mbo_iters = 3L,
      mbo_init_design = 4L,
      s4_lgb_lr = 0.05,
      s4_lgb_nrounds = 200L,
      s4_lgb_es = 25L,
      s4_xgb_lr = 0.05,
      s4_xgb_nrounds = 180L,
      s4_xgb_es = 25L,
      s4_mlp_epochs = 30L,
      s4_mlp_es = 30L,
      s4_mlp_lr = -1.0,
      s4_mlp_hidden = 64L
    ),
    standard = list(
      islands = 4L,
      islands_per_evaluator = 2L,
      s2_gen = 10L,
      s2_pop = 8L,
      s2_mlp_gen = 8L,
      s2_mlp_pop = 6L,
      s2_nrounds = 300L,
      s2_es = 25L,
      s2_mlp_epochs = 30L,
      s2_mlp_es = 30L,
      s2_holdout = 0.08,
      s2_sd_penalty = 0.15,
      s2_complexity_penalty = NULL,
      s2_complexity_mode = "auto",
      s2_confirm_k = 3L,
      clust_cap = 25000L,
      s2_mbo_iters = 3L,
      s2_mbo_init_design = 4L,
      mbo_iters = 10L,
      mbo_init_design = 8L,
      s4_lgb_lr = 0.005,
      s4_lgb_nrounds = 50000L,
      s4_lgb_es = 250L,
      s4_xgb_lr = 0.008,
      s4_xgb_nrounds = 35000L,
      s4_xgb_es = 200L,
      s4_mlp_epochs = 250L,
      s4_mlp_es = 250L,
      s4_mlp_lr = -1.0,
      s4_mlp_hidden = 256L
    ),
    max = list(
      islands = 6L,
      islands_per_evaluator = 3L,
      s2_gen = 12L,
      s2_pop = 10L,
      s2_mlp_gen = 10L,
      s2_mlp_pop = 8L,
      s2_nrounds = 400L,
      s2_es = 30L,
      s2_mlp_epochs = 40L,
      s2_mlp_es = 40L,
      s2_holdout = 0.08,
      s2_sd_penalty = 0.20,
      s2_complexity_penalty = NULL,
      s2_complexity_mode = "auto",
      s2_confirm_k = 3L,
      clust_cap = 50000L,
      s2_mbo_iters = 5L,
      s2_mbo_init_design = 6L,
      mbo_iters = 25L,
      mbo_init_design = 12L,
      s4_lgb_lr = 0.003,
      s4_lgb_nrounds = 50000L,
      s4_lgb_es = 300L,
      s4_xgb_lr = 0.005,
      s4_xgb_nrounds = 40000L,
      s4_xgb_es = 250L,
      s4_mlp_epochs = 400L,
      s4_mlp_es = 400L,
      s4_mlp_lr = -1.0,
      s4_mlp_hidden = 384L
    )
  )

  if (!is.null(overrides) && length(overrides) > 0L) {
    if (!is.list(overrides) || is.null(names(overrides)) || any(!nzchar(names(overrides)))) {
      stop("'preset_overrides' must be a named list of hyperparameter values.")
    }
    unknown <- setdiff(names(overrides), names(cfg))
    if (length(unknown) > 0L) {
      stop(sprintf(
        "Unknown preset_overrides key(s): %s. Allowed keys: %s",
        paste(unknown, collapse = ", "),
        paste(names(cfg), collapse = ", ")
      ))
    }
    for (nm in names(overrides)) {
      cfg[[nm]] <- overrides[[nm]]
    }
    # Keep early stopping patience tracking epochs unless explicitly overridden
    if ("s2_mlp_epochs" %in% names(overrides) && !"s2_mlp_es" %in% names(overrides)) {
      cfg$s2_mlp_es <- cfg$s2_mlp_epochs
    }
    if ("s4_mlp_epochs" %in% names(overrides) && !"s4_mlp_es" %in% names(overrides)) {
      cfg$s4_mlp_es <- cfg$s4_mlp_epochs
    }
  }
  cfg
}

#' Filter requested evaluators to registered backends whose packages are installed
#' @keywords internal
#' @noRd
.resolve_competition_evaluators <- function(evaluators) {
  if (!is.character(evaluators) || length(evaluators) == 0L) {
    stop("'evaluators' must be a non-empty character vector.")
  }
  evaluators <- unique(evaluators)
  pkg_map <- c(
    lightgbm = "lightgbm",
    lightgbm_rf = "lightgbm",
    xgboost = "xgboost",
    catboost = "catboost",
    keras3 = "keras3",
    lm = "glmnet",
    spline_lm = "glmnet",
    ranger = "ranger"
  )
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

#' Remap evaluator stage args for `evolve_features()` calls
#'
#' Translates `s` to `lm_s` because `evolve_features()` has multiple formal
#' arguments starting with `s` (`split_ratio`, `split_ids`, `seed`, `sd_penalty`,
#' `seed_genes`), so passing `s = ...` via `do.call()` fails partial matching.
#' @keywords internal
#' @noRd
.map_evaluator_args_for_evolve <- function(ev_args) {
  if (length(ev_args) == 0L) return(ev_args)
  if ("s" %in% names(ev_args)) {
    if (!("lm_s" %in% names(ev_args)) || is.null(ev_args[["lm_s"]])) {
      ev_args[["lm_s"]] <- ev_args[["s"]]
    }
    ev_args[["s"]] <- NULL
  }
  ev_args
}

#' Resolve Stage 4 tuned evaluator or fallback to untuned base
#' @keywords internal
#' @noRd
.resolve_stage4_evaluator <- function(ev, tune_stage4 = c("none", "tabarena", "mbo")) {
  .resolve_tuned_counterpart(ev, tune = tune_stage4)
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
.evaluator_stage_args <- function(ev, stage, cfg, n_rows, custom_evaluator_args = NULL) {
  base_ev <- get_base_evaluator(ev)
  is_tuned <- is_tuned_evaluator(ev)
  args <- if (stage == 2L) {
    if (base_ev == "lightgbm") {
      list(
        learning_rate = 0.08,
        num_leaves = 63L,
        nrounds = cfg$s2_nrounds,
        early_stopping_rounds = cfg$s2_es
      )
    } else if (base_ev == "xgboost") {
      list(
        tree_method = "hist",
        learning_rate = 0.08,
        max_depth = 6L,
        nrounds = cfg$s2_nrounds,
        early_stopping_rounds = cfg$s2_es
      )
    } else if (base_ev == "catboost") {
      list(
        learning_rate = 0.08,
        depth = 6L,
        nrounds = cfg$s2_nrounds,
        early_stopping_rounds = cfg$s2_es
      )
    } else if (base_ev == "realmlp") {
      bs <- if (n_rows >= 8192L) 2048L else -1L
      list(
        realmlp_epochs = cfg$s2_mlp_epochs,
        realmlp_lr = -1.0,
        # Patience == epochs: early stopping stays on (tracks best epoch) but never truncates cyclic schedule.
        early_stopping_rounds = cfg$s2_mlp_es,
        hidden_dim = cfg$s4_mlp_hidden,
        batch_size = bs
      )
    } else if (base_ev == "lm") {
      list(alpha = 0.5, s = "lambda.min")
    } else if (base_ev == "lightgbm_rf") {
      list(
        boosting = "rf",
        bagging_freq = 1L,
        bagging_fraction = 0.80,
        feature_fraction = 0.80,
        num_leaves = 63L,
        nrounds = cfg$s2_nrounds,
        early_stopping_rounds = cfg$s2_es
      )
    } else if (base_ev == "ranger") {
      list(num.trees = 100L, save.memory = FALSE)
    } else if (base_ev == "spline_lm") {
      list(alpha = 0.5, s = "lambda.min")
    } else if (base_ev == "rff") {
      list(alpha = 0.0, s = "lambda.min", n_components = 128L)
    } else if (base_ev == "tabm") {
      bs <- if (n_rows >= 65536L) 512L else if (n_rows >= 8192L) 256L else -1L
      s2_n_blocks <- if (cfg$s2_mlp_epochs <= 16L || cfg$s4_mlp_hidden <= 64L) 1L else 2L
      list(
        epochs = cfg$s2_mlp_epochs,
        early_stopping_rounds = cfg$s2_mlp_es,
        d_block = 128L,
        n_blocks = s2_n_blocks,
        lr = if (bs >= 256L) 0.003 else 0.002,
        batch_size = bs
      )
    } else {
      list(nrounds = cfg$s2_nrounds, early_stopping_rounds = cfg$s2_es)
    }
  } else {
    # Stage 4 high-capacity retraining
    if (is_tuned) {
      mbo_it <- if (!is.null(cfg$mbo_iters)) as.integer(cfg$mbo_iters) else 10L
      mbo_id <- if (!is.null(cfg$mbo_init_design)) as.integer(cfg$mbo_init_design) else 8L
      if (base_ev == "lightgbm") {
        list(
          learning_rate = cfg$s4_lgb_lr,
          nrounds = cfg$s4_lgb_nrounds,
          early_stopping_rounds = cfg$s4_lgb_es,
          bagging_freq = 1L,
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      } else if (base_ev == "xgboost") {
        list(
          tree_method = "hist",
          learning_rate = cfg$s4_xgb_lr,
          nrounds = cfg$s4_xgb_nrounds,
          early_stopping_rounds = cfg$s4_xgb_es,
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      } else if (base_ev == "catboost") {
        list(
          learning_rate = cfg$s4_xgb_lr,
          nrounds = cfg$s4_xgb_nrounds,
          early_stopping_rounds = cfg$s4_xgb_es,
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      } else if (base_ev == "realmlp") {
        bs <- if (n_rows >= 8192L) 2048L else -1L
        list(
          epochs = cfg$s4_mlp_epochs,
          early_stopping_rounds = cfg$s4_mlp_es,
          batch_size = bs,
          mbo_iters = max(2L, as.integer(ceiling(mbo_it * 0.7))),
          mbo_init_design = max(3L, as.integer(ceiling(mbo_id * 0.75)))
        )
      } else if (base_ev == "lm") {
        list(
          alpha = 0.5,
          s = "lambda.min",
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      } else if (base_ev == "lightgbm_rf") {
        list(
          boosting = "rf",
          bagging_freq = 1L,
          bagging_fraction = 0.80,
          feature_fraction = 0.80,
          nrounds = cfg$s4_lgb_nrounds,
          early_stopping_rounds = cfg$s4_lgb_es,
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      } else if (base_ev == "ranger") {
        list(
          num.trees = 300L,
          save.memory = FALSE,
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      } else if (base_ev == "spline_lm") {
        list(
          alpha = 0.5,
          s = "lambda.min",
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      } else if (base_ev == "rff") {
        list(
          alpha = 0.0,
          s = "lambda.min",
          n_components = 256L,
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      } else if (base_ev == "tabm") {
        bs <- if (n_rows >= 65536L) 512L else if (n_rows >= 8192L) 256L else -1L
        s4_es <- if (cfg$s4_mlp_es < cfg$s4_mlp_epochs) {
          as.integer(cfg$s4_mlp_es)
        } else {
          min(as.integer(cfg$s4_mlp_es), max(24L, as.integer(ceiling(cfg$s4_mlp_epochs * 0.25))))
        }
        list(
          epochs = cfg$s4_mlp_epochs,
          early_stopping_rounds = s4_es,
          batch_size = bs,
          mbo_iters = max(2L, as.integer(ceiling(mbo_it * 0.7))),
          mbo_init_design = max(3L, as.integer(ceiling(mbo_id * 0.75)))
        )
      } else {
        list(
          learning_rate = cfg$s4_lgb_lr,
          nrounds = cfg$s4_lgb_nrounds,
          early_stopping_rounds = cfg$s4_lgb_es,
          mbo_iters = mbo_it,
          mbo_init_design = mbo_id
        )
      }
    } else {
      if (base_ev == "lightgbm") {
        list(
          learning_rate = cfg$s4_lgb_lr,
          num_leaves = 63L,
          feature_fraction = 0.75,
          bagging_fraction = 0.80,
          bagging_freq = 1L,
          lambda_l1 = 0.5,
          lambda_l2 = 2.0,
          nrounds = cfg$s4_lgb_nrounds,
          early_stopping_rounds = cfg$s4_lgb_es
        )
      } else if (base_ev == "xgboost") {
        list(
          tree_method = "hist",
          learning_rate = cfg$s4_xgb_lr,
          max_depth = 6L,
          subsample = 0.80,
          colsample_bytree = 0.75,
          lambda = 2.0,
          nrounds = cfg$s4_xgb_nrounds,
          early_stopping_rounds = cfg$s4_xgb_es
        )
      } else if (base_ev == "catboost") {
        list(
          learning_rate = cfg$s4_xgb_lr,
          depth = 6L,
          subsample = 0.80,
          rsm = 0.75,
          l2_leaf_reg = 3.0,
          nrounds = cfg$s4_xgb_nrounds,
          early_stopping_rounds = cfg$s4_xgb_es
        )
      } else if (base_ev == "realmlp") {
        bs <- if (n_rows >= 8192L) 2048L else -1L
        list(
          epochs = cfg$s4_mlp_epochs,
          lr = cfg$s4_mlp_lr,
          # Patience == epochs: early stopping stays on (tracks best epoch) but never truncates cyclic schedule.
          early_stopping_rounds = cfg$s4_mlp_es,
          hidden_dim = cfg$s4_mlp_hidden,
          batch_size = bs
        )
      } else if (base_ev == "lm") {
        list(
          alpha = 0.5,
          s = "lambda.min"
        )
      } else if (base_ev == "lightgbm_rf") {
        list(
          boosting = "rf",
          bagging_freq = 1L,
          bagging_fraction = 0.80,
          feature_fraction = 0.80,
          num_leaves = 63L,
          nrounds = cfg$s4_lgb_nrounds,
          early_stopping_rounds = cfg$s4_lgb_es
        )
      } else if (base_ev == "ranger") {
        list(
          num.trees = 300L,
          save.memory = FALSE
        )
      } else if (base_ev == "spline_lm") {
        list(
          alpha = 0.5,
          s = "lambda.min"
        )
      } else if (base_ev == "rff") {
        list(
          alpha = 0.0,
          s = "lambda.min",
          n_components = 256L
        )
      } else if (base_ev == "tabm") {
        bs <- if (n_rows >= 65536L) 512L else if (n_rows >= 8192L) 256L else -1L
        s4_es <- if (cfg$s4_mlp_es < cfg$s4_mlp_epochs) {
          as.integer(cfg$s4_mlp_es)
        } else {
          min(as.integer(cfg$s4_mlp_es), max(24L, as.integer(ceiling(cfg$s4_mlp_epochs * 0.25))))
        }
        list(
          epochs = cfg$s4_mlp_epochs,
          early_stopping_rounds = s4_es,
          d_block = as.integer(cfg$s4_mlp_hidden),
          n_blocks = 2L,
          lr = if (bs >= 256L) 0.003 else 0.002,
          batch_size = bs
        )
      } else {
        list(nrounds = cfg$s4_xgb_nrounds, early_stopping_rounds = cfg$s4_xgb_es)
      }
    }
  }

  if (stage == 2L && is_tuned) {
    s2_it <- if (!is.null(cfg$s2_mbo_iters)) as.integer(cfg$s2_mbo_iters) else 2L
    s2_id <- if (!is.null(cfg$s2_mbo_init_design)) as.integer(cfg$s2_mbo_init_design) else 3L
    if (base_ev %in% c("realmlp", "tabm")) {
      args$mbo_iters <- max(1L, as.integer(ceiling(s2_it * 0.7)))
      args$mbo_init_design <- max(2L, as.integer(ceiling(s2_id * 0.75)))
    } else {
      args$mbo_iters <- s2_it
      args$mbo_init_design <- s2_id
    }
  }

  if (!is.null(custom_evaluator_args) && is.list(custom_evaluator_args) && length(custom_evaluator_args) > 0L) {
    stage_key <- paste0("stage", stage)

    # 1. Global / top-level arguments (not keyed by evaluator or stage)
    section_keys <- c(
      "stage2", "stage4", "stage3_5",
      "lightgbm", "lightgbm_rf", "xgboost", "realmlp", "catboost",
      "lm", "spline_lm", "rff", "ranger", "tabm"
    )
    for (k in names(custom_evaluator_args)) {
      if (!k %in% section_keys && !has_evaluator(k)) {
        args[[k]] <- custom_evaluator_args[[k]]
      }
    }

    # 2. Model-keyed arguments: evaluator_args[[ev]] or evaluator_args[[base_ev]]
    ev_custom <- if (ev %in% names(custom_evaluator_args)) {
      custom_evaluator_args[[ev]]
    } else if (base_ev %in% names(custom_evaluator_args)) {
      custom_evaluator_args[[base_ev]]
    } else {
      NULL
    }
    if (is.list(ev_custom)) {
      for (k in names(ev_custom)) {
        if (k %in% c("stage2", "stage4", "stage3_5")) {
          if (k == stage_key && is.list(ev_custom[[k]])) {
            for (sub_k in names(ev_custom[[k]])) {
              args[[sub_k]] <- ev_custom[[k]][[sub_k]]
            }
          }
        } else {
          args[[k]] <- ev_custom[[k]]
        }
      }
    }

    # 2. Stage-keyed arguments: evaluator_args[[stage_key]][[ev]] or evaluator_args[[stage_key]][[base_ev]]
    if (stage_key %in% names(custom_evaluator_args) && is.list(custom_evaluator_args[[stage_key]])) {
      st_custom <- custom_evaluator_args[[stage_key]]
      st_ev_custom <- if (ev %in% names(st_custom)) {
        st_custom[[ev]]
      } else if (base_ev %in% names(st_custom)) {
        st_custom[[base_ev]]
      } else {
        NULL
      }
      if (is.list(st_ev_custom)) {
        for (k in names(st_ev_custom)) {
          args[[k]] <- st_ev_custom[[k]]
        }
      }
    }
  }

  args
}

#' Prune Heavy Internal Caches from Stage Recipes While Preserving Documented Fields
#' @keywords internal
#' @noRd
.prune_stage_recipes <- function(recipes) {
  if (is.null(recipes) || !is.list(recipes)) return(recipes)
  for (i in seq_along(recipes)) {
    r <- recipes[[i]]
    if (inherits(r, "evo_recipe")) {
      r$population <- NULL
      r$history <- NULL
      r$fold_models <- NULL
      r$alignment_cache <- NULL
      recipes[[i]] <- r
    }
  }
  recipes
}

#' Split an individual's genes into untainted unsupervised genes and unfitted supervised seed genes
#' @keywords internal
#' @noRd
.split_individual_tainted_genes <- function(ind) {
  if (is.null(ind) || length(ind$genes) == 0L) {
    return(list(
      unsup_ind = ind,
      sup_genes = list(),
      sup_output_cols = character(0),
      sup_input_cols = character(0)
    ))
  }
  tainted_cols <- character(0)
  unsup_genes <- list()
  sup_genes <- list()
  sup_input_cols <- character(0)

  for (g in ind$genes) {
    is_tainted <- is_supervised_transformer(g) || any(g$input_cols %in% tainted_cols)
    if (is_tainted) {
      if (!is.null(g$output_col) && nzchar(g$output_col)) {
        tainted_cols <- c(tainted_cols, g$output_col)
      }
      sup_input_cols <- c(sup_input_cols, setdiff(g$input_cols, tainted_cols))
      g_clean <- g
      g_clean$state <- NULL
      sup_genes <- c(sup_genes, list(g_clean))
    } else {
      unsup_genes <- c(unsup_genes, list(g))
    }
  }

  sup_input_cols <- unique(sup_input_cols)
  unsup_ind <- ind
  unsup_ind$genes <- unsup_genes
  if (length(sup_input_cols) > 0L) {
    if (!is.null(unsup_ind$all_numeric_cols)) {
      unsup_ind$numeric_cols <- unique(c(unsup_ind$numeric_cols, intersect(sup_input_cols, unsup_ind$all_numeric_cols)))
    }
    if (!is.null(unsup_ind$all_categorical_cols)) {
      unsup_ind$categorical_cols <- unique(c(unsup_ind$categorical_cols, intersect(sup_input_cols, unsup_ind$all_categorical_cols)))
    }
    if (!is.null(unsup_ind$all_datetime_cols)) {
      unsup_ind$datetime_cols <- unique(c(unsup_ind$datetime_cols, intersect(sup_input_cols, unsup_ind$all_datetime_cols)))
    }
  }

  list(
    unsup_ind = unsup_ind,
    sup_genes = sup_genes,
    sup_output_cols = unique(tainted_cols),
    sup_input_cols = sup_input_cols
  )
}

#' Extract Stage 3 feature columns originating from a given Stage 2 recipe
#' @keywords internal
#' @noRd
.extract_recipe_view_cols <- function(ev, stage2_recipes, scout_ens, available_cols) {
  cols <- character(0)
  rec <- stage2_recipes[[ev]]
  if (!is.null(rec) && !is.null(rec$best_individual)) {
    cols <- c(cols, extract_individual_features(rec$best_individual, target_col = rec$target_col, unique = TRUE))
  }
  if (!is.null(rec) && !is.null(rec$island_bests) && is.list(rec$island_bests)) {
    for (ind_i in rec$island_bests) {
      if (!is.null(ind_i)) {
        cols <- c(cols, extract_individual_features(ind_i, target_col = rec$target_col, unique = TRUE))
      }
    }
  }
  if (!is.null(scout_ens) && is.list(scout_ens$active_recipes)) {
    prefix <- paste0(ev, "_island_")
    for (nm in names(scout_ens$active_recipes)) {
      if (identical(nm, ev) || startsWith(nm, prefix)) {
        cols <- c(cols, extract_individual_features(scout_ens$active_recipes[[nm]], target_col = scout_ens$target_col, unique = TRUE))
      }
    }
  }
  cols <- intersect(available_cols, unique(cols))
  if (length(cols) == 0L) available_cols else cols
}

#' Run a fast CV view tournament per evaluator across own, full, and pruned mega-matrix views
#' @keywords internal
#' @noRd
.select_stage3_views <- function(train_mega, stage2_recipes, scout_ens, active_evaluators,
                                 target_col, task, metric, cv_folds, cv_strategy,
                                 group_col, time_col, max_search_rows, adv_weights,
                                 target_transform, cfg, evaluator_args, threads, seed,
                                 verbose, sup_seed_genes = NULL, sup_input_cols = NULL) {
  show_msg <- isTRUE(verbose) || (is.numeric(verbose) && verbose >= 1L)
  view_dt <- subsample_for_discovery(
    data = train_mega,
    target_col = target_col,
    max_rows = max_search_rows,
    task = task,
    cv_strategy = cv_strategy,
    group_col = group_col,
    time_col = time_col,
    adv_weights = adv_weights,
    seed = seed + 300L
  )

  v_folds <- max(2L, min(as.integer(cv_folds), 5L, nrow(view_dt)))
  fold_ids <- .build_cv_folds(
    view_dt, v_folds, cv_strategy, time_col, group_col,
    target_col = target_col, task = task
  )
  shared_folds <- .build_shared_folds(view_dt, fold_ids)
  shared_full <- shallow_copy_dt(data.table::as.data.table(view_dt))

  available_cols <- setdiff(names(view_dt), target_col)
  all_dt_cols <- available_cols[vapply(view_dt[, available_cols, with = FALSE], .is_datetime_col, logical(1L))]
  all_num_cols <- setdiff(available_cols[vapply(view_dt[, available_cols, with = FALSE], is.numeric, logical(1L))], all_dt_cols)
  all_cat_cols <- setdiff(available_cols, c(all_num_cols, all_dt_cols))

  .eval_view_cols <- function(cols, base_ev, ev_fast_evolve_args, seed_genes_ev = NULL, need_imp = FALSE) {
    active_seed_genes <- if (!is.null(seed_genes_ev) && length(seed_genes_ev) > 0L) {
      normalize_seed_genes(seed_genes_ev, cols)
    } else {
      list()
    }
    ind <- create_individual(
      genes = active_seed_genes,
      numeric_cols = intersect(cols, all_num_cols),
      categorical_cols = intersect(cols, all_cat_cols),
      datetime_cols = intersect(cols, all_dt_cols),
      all_numeric_cols = all_num_cols,
      all_categorical_cols = all_cat_cols,
      all_datetime_cols = all_dt_cols
    )
    call_args <- list(
      ind = ind,
      data = view_dt,
      target_col = target_col,
      task = task,
      cv_folds = v_folds,
      evaluation_strategy = "cv",
      evaluator = base_ev,
      fold_ids = fold_ids,
      shared_folds = shared_folds,
      shared_full = shared_full,
      state_cache = NULL,
      threads = threads,
      metric = metric,
      verbose = FALSE,
      allow_prune = TRUE,
      complexity_penalty = 0,
      sd_penalty = 0,
      cv_strategy = cv_strategy,
      time_col = time_col,
      group_col = group_col,
      target_transform = target_transform,
      n_samples = nrow(view_dt)
    )
    for (k in names(ev_fast_evolve_args)) {
      call_args[[k]] <- ev_fast_evolve_args[[k]]
    }
    call_args$compute_importances <- isTRUE(need_imp)
    evaluated <- tryCatch(do.call(evaluate_fitness, call_args), error = function(e) ind)
    score <- if (!is.null(evaluated$raw_fitness) && is.finite(evaluated$raw_fitness)) {
      evaluated$raw_fitness
    } else {
      evaluated$fitness
    }
    if (!is.finite(score)) score <- -Inf
    list(cols = cols, score = score, importances = evaluated$importances, n_cols = length(cols))
  }

  views_list <- vector("list", length(active_evaluators))
  names(views_list) <- active_evaluators
  view_types_vec <- stats::setNames(character(length(active_evaluators)), active_evaluators)

  for (ev in active_evaluators) {
    base_ev <- get_base_evaluator(ev)
    ev_fast_args <- .evaluator_stage_args(
      base_ev, stage = 2L, cfg = cfg, n_rows = nrow(view_dt),
      custom_evaluator_args = evaluator_args
    )
    ev_fast_evolve_args <- .map_evaluator_args_for_evolve(ev_fast_args)
    ev_seed_genes <- if (!is.null(sup_seed_genes)) sup_seed_genes[[ev]] else NULL
    ev_req_cols <- if (!is.null(sup_input_cols)) intersect(sup_input_cols[[ev]], available_cols) else character(0)

    res_mega <- .eval_view_cols(available_cols, base_ev, ev_fast_evolve_args, seed_genes_ev = ev_seed_genes, need_imp = TRUE)

    own_cols <- unique(c(.extract_recipe_view_cols(ev, stage2_recipes, scout_ens, available_cols), ev_req_cols))
    res_own <- if (setequal(own_cols, available_cols)) {
      list(cols = available_cols, score = res_mega$score, n_cols = length(available_cols))
    } else {
      .eval_view_cols(own_cols, base_ev, ev_fast_evolve_args, seed_genes_ev = ev_seed_genes, need_imp = FALSE)
    }

    res_pruned <- NULL
    imp <- res_mega$importances
    if (!is.null(imp) && length(imp) > 0L) {
      imp <- imp[intersect(names(imp), available_cols)]
      imp <- imp[is.finite(imp)]
      if (length(imp) > 0L) {
        imp_pos <- pmax(imp, 0)
        imp_sum <- sum(imp_pos)
        rel_imp <- if (imp_sum > 0) imp_pos / imp_sum else imp
        imp_thresh <- getOption("evoFE.importance_threshold", 0.002)
        eff_thresh <- max(imp_thresh, 0.10 / length(available_cols))
        pruned_cols <- names(rel_imp)[rel_imp >= eff_thresh]
        min_keep <- min(length(own_cols), length(available_cols))
        if (length(pruned_cols) < min_keep) {
          ord <- names(sort(rel_imp, decreasing = TRUE))
          pruned_cols <- ord[seq_len(min(min_keep, length(ord)))]
        }
        pruned_cols <- intersect(available_cols, unique(c(pruned_cols, ev_req_cols)))
        if (length(pruned_cols) > 0L && length(pruned_cols) < length(available_cols)) {
          if (setequal(pruned_cols, own_cols)) {
            res_pruned <- list(cols = own_cols, score = res_own$score, n_cols = length(own_cols))
          } else {
            res_pruned <- .eval_view_cols(pruned_cols, base_ev, ev_fast_evolve_args, seed_genes_ev = ev_seed_genes, need_imp = FALSE)
          }
        }
      }
    }

    candidates <- list(
      own = list(type = "own", cols = own_cols, score = res_own$score, n_cols = length(own_cols)),
      mega = list(type = "mega", cols = available_cols, score = res_mega$score, n_cols = length(available_cols))
    )
    if (!is.null(res_pruned)) {
      candidates$mega_pruned <- list(
        type = "mega_pruned",
        cols = res_pruned$cols,
        score = res_pruned$score,
        n_cols = res_pruned$n_cols
      )
    }

    best_cand <- candidates[[1L]]
    for (nm in names(candidates)[-1L]) {
      cand <- candidates[[nm]]
      is_better <- if (!is.finite(best_cand$score)) {
        is.finite(cand$score) || cand$n_cols < best_cand$n_cols
      } else if (!is.finite(cand$score)) {
        FALSE
      } else if (cand$score > best_cand$score + 1e-12) {
        TRUE
      } else if (abs(cand$score - best_cand$score) <= 1e-12 && cand$n_cols < best_cand$n_cols) {
        TRUE
      } else {
        FALSE
      }
      if (is_better) {
        best_cand <- cand
      }
    }

    views_list[[ev]] <- list(
      evaluator = ev,
      base_evaluator = base_ev,
      selected = best_cand$type,
      cols = best_cand$cols,
      scores = list(
        own = res_own$score,
        mega = res_mega$score,
        mega_pruned = if (!is.null(res_pruned)) res_pruned$score else NA_real_
      ),
      n_cols = list(
        own = length(own_cols),
        mega = length(available_cols),
        mega_pruned = if (!is.null(res_pruned)) res_pruned$n_cols else NA_integer_
      )
    )
    view_types_vec[ev] <- best_cand$type

    if (show_msg) {
      pr_str <- if (!is.null(res_pruned)) sprintf(" | mega_pruned(%d): %.5f", res_pruned$n_cols, res_pruned$score) else ""
      message(sprintf(
        "[Stage 3 View Tournament] '%s': selected '%s' (%d cols) [own(%d): %.5f | mega(%d): %.5f%s]",
        ev, best_cand$type, length(best_cand$cols),
        length(own_cols), res_own$score,
        length(available_cols), res_mega$score,
        pr_str
      ))
    }
  }

  list(views = views_list, view_types = view_types_vec)
}

#' Harmonize column data types and categorical levels between train and test
#' @keywords internal
#' @noRd
.harmonize_train_test_types <- function(train_dt, test_dt, shared_features) {
  for (col in shared_features) {
    tr_col <- train_dt[[col]]
    te_col <- test_dt[[col]]

    # 1. Logical vs Character / Factor (e.g. fread vs read.csv boolean parsing differences)
    if (is.logical(tr_col) && (is.character(te_col) || is.factor(te_col))) {
      te_str <- trimws(tolower(as.character(te_col)))
      te_log <- ifelse(te_str %in% c("true", "t", "1"), TRUE,
                ifelse(te_str %in% c("false", "f", "0"), FALSE, NA))
      if (sum(is.na(te_log)) == sum(is.na(te_col))) {
        test_dt[, (col) := te_log]
      }
    } else if (is.logical(te_col) && (is.character(tr_col) || is.factor(tr_col))) {
      tr_str <- trimws(tolower(as.character(tr_col)))
      tr_log <- ifelse(tr_str %in% c("true", "t", "1"), TRUE,
                ifelse(tr_str %in% c("false", "f", "0"), FALSE, NA))
      if (sum(is.na(tr_log)) == sum(is.na(tr_col))) {
        train_dt[, (col) := tr_log]
      }
    } else if (is.numeric(tr_col) && (is.character(te_col) || is.factor(te_col))) {
      num_conv <- suppressWarnings(as.numeric(as.character(te_col)))
      if (sum(is.na(num_conv)) == sum(is.na(te_col))) {
        test_dt[, (col) := num_conv]
      }
    } else if (is.numeric(te_col) && (is.character(tr_col) || is.factor(tr_col))) {
      num_conv <- suppressWarnings(as.numeric(as.character(tr_col)))
      if (sum(is.na(num_conv)) == sum(is.na(tr_col))) {
        train_dt[, (col) := num_conv]
      }
    } else if (is.factor(tr_col) && !is.factor(te_col)) {
      test_dt[, (col) := factor(as.character(te_col), levels = levels(tr_col))]
    } else if (is.factor(te_col) && !is.factor(tr_col)) {
      train_dt[, (col) := factor(as.character(tr_col), levels = levels(te_col))]
    } else if (is.integer(tr_col) && is.double(te_col)) {
      train_dt[, (col) := as.numeric(tr_col)]
    } else if (is.double(tr_col) && is.integer(te_col)) {
      test_dt[, (col) := as.numeric(te_col)]
    }
  }
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
#'   \item \strong{Stage 2 (Unified Multi-Evaluator Discovery)}: Evolves features
#'     in a unified multi-evaluator island topology (\code{evaluation_strategy = "metacv"}
#'     when \code{nrow >= 1000}, non-time splits, and \code{islands >= 2}, else \code{"cv"})
#'     allowing cross-pollination across models and islands. When
#'     \code{supervised_stage4 = FALSE}, only unsupervised transformers run; when
#'     \code{supervised_stage4 = TRUE}, supervised transformers also participate in
#'     Stage 2 and are stripped to unfitted symbolic \code{seed_genes} before Stage 3
#'     so zero target encoding leaks into Stage 3/4.
#'   \item \strong{Stage 3 (Mega-Matrix Synthesis & View Tournament)}: Combines all
#'     unsupervised Stage 2 scout recipes via \code{ensemble_islands(..., method = "equal")},
#'     projects learned states onto 100\% of \code{train} and \code{test}, and selects the
#'     best feature view per evaluator via a fast CV View Tournament.
#'   \item \strong{Stage 4 (High-Capacity Retraining)}: Trains each model family on 100\%
#'     of its winning \code{train_mega} view with low learning rates, deep early-stopping
#'     ceilings (or \code{_mbo} / \code{_tabarena} tuning), and \code{bag_folds = TRUE},
#'     refitting any extracted \code{supervised_stage4} seed genes strictly inside each fold.
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
#' @param ensemble_metric Optional evaluation metric used for Stage 5 ensembling
#'   and final stacked CV fitness reporting (default \code{NULL}, inheriting
#'   \code{metric}). Set to \code{"auc"} when optimizing calibration metrics like
#'   \code{"ts_refinement"} during Stage 2 & 4 feature evolution/training, so that
#'   Stage 5 ensembling automatically uses rank averaging and reports an honest CV AUC.
#' @param preset Compute budget preset: \code{"standard"} (competition playbook defaults),
#'   \code{"fast"} (quick smoke/baseline run), or \code{"max"} (deep schedule).
#' @param preset_overrides Optional named list of hyperparameter values overriding preset
#'   defaults (e.g. \code{list(s4_mlp_epochs = 500L, s4_lgb_lr = 0.01)}). When RealMLP epochs
#'   are overridden and \code{s2_mlp_es} or \code{s4_mlp_es} is not explicitly specified,
#'   early stopping patience automatically tracks total epochs to ensure all cyclic annealing
#'   phases complete while retaining optimal validation checkpointing.
#' @param evaluators Character vector of model families to run (default
#'   \code{c("lightgbm", "xgboost", "realmlp")}). Unavailable packages are skipped with a warning.
#' @param islands_per_evaluator Optional integer specifying the number of Stage 2 discovery
#'   islands to spawn per evaluator in the shared multi-evaluator island topology. When
#'   \code{NULL} (default), uses the preset default (\code{1L} for \code{"fast"}, \code{2L}
#'   for \code{"standard"}, \code{3L} for \code{"max"}; single-evaluator runs use at least
#'   \code{cfg$islands} islands). Setting \code{islands_per_evaluator >= 2L} enables simultaneous
#'   intra-evaluator and inter-evaluator gene cross-pollination.
#' @param tune_stage4 Optional Stage 4 hyperparameter optimization strategy:
#'   \code{"none"} (default, uses high-capacity fixed schedules), \code{"tabarena"} (tunes
#'   each model family using its TabArena parameter space via \pkg{mlr3mbo} Bayesian
#'   Optimization), or \code{"mbo"} (uses standard MBO tuning spaces). Stage 2 feature
#'   discovery always uses fast base surrogates so genetic search runs at peak throughput.
#' @param evaluator_args Optional nested list of model-specific or stage-specific training
#'   arguments passed into \code{\link{evolve_features}} (e.g.
#'   \code{list(realmlp = list(n_cycles = 6L), lightgbm = list(max_depth = 8L))} or
#'   \code{list(realmlp = list(stage4 = list(early_stopping_rounds = 50L)))}).
#' @param cv_folds Number of cross-validation folds for Stage 4 retraining and stacking (default \code{5L}).
#' @param cv_strategy Cross-validation split strategy: \code{"random"},
#'   \code{"stratified_group"}, \code{"group"}, or \code{"time"}.
#' @param group_col Optional group column name when \code{cv_strategy} is grouped.
#' @param time_col Optional timestamp/order column name when \code{cv_strategy = "time"}.
#' @param target_transform Regression target transform: \code{"auto"} (selects \code{"log1p"}
#'   when skewness > 1.5 and \code{min(y) >= 0}), \code{"none"}, \code{"log1p"},
#'   \code{"yeo_johnson"}, or \code{"rank_gauss"}.
#' @param complexity_penalty Optional Stage 2 complexity penalty multiplier: \code{NULL}
#'   (default, uses \code{0.30} for all-neural runs or \code{0.25} otherwise, or
#'   \code{preset_overrides$s2_complexity_penalty}), a non-negative number, or
#'   \code{"none"} (disables complexity penalty).
#' @param complexity_mode Stage 2 complexity penalty strategy: \code{"auto"} (default,
#'   uses \code{"pac_bayes_dynamic"} when discovery rows < 10000 and
#'   \code{"bic_dynamic"} otherwise), \code{"bic_dynamic"}, \code{"bic"},
#'   \code{"pac_bayes_dynamic"}, \code{"pac_bayes"}, or \code{"none"}.
#' @param sd_penalty Optional Stage 2 fold standard-error penalty multiplier: \code{NULL}
#'   (default, uses the preset's \code{s2_sd_penalty}), a non-negative number, or
#'   \code{"none"} (disables fold SD penalty).
#' @param max_search_rows Maximum rows used in Stage 2 feature discovery (default \code{200000L};
#'   set \code{NULL} or \code{Inf} to disable Phase-1 subsampling). Stage 4 always trains on all rows.
#' @param drop_drifting Logical (default \code{FALSE}). If \code{TRUE}, drops raw columns whose
#'   single-feature train-vs-test AUC exceeds \code{drift_auc_threshold}.
#' @param drift_auc_threshold Single-feature adversarial AUC cutoff (default \code{0.85}).
#' @param supervised_stage4 Logical (default \code{FALSE}). If \code{TRUE}, enables
#'   supervised encoders directly inside Stage 2 discovery, strips supervised (and
#'   downstream) genes before Stage 3 Mega-Matrix synthesis so \code{train_mega} contains
#'   zero target leakage, and passes those \strong{unfitted} symbolic gene formulas via
#'   \code{seed_genes} into Stage 3 View Tournament and Stage 4's
#'   \code{generations = 1, pop_size = 1} deep fold-bagged refit.
#' @param bag_folds Logical (default \code{TRUE}). Retains and averages all \code{cv_folds}
#'   fold models during test prediction.
#' @param rank_average Logical or NULL. If \code{TRUE}, ranks predictions to uniform
#'   percentiles prior to stacking and inference. If \code{NULL} (default), automatically
#'   defaults to \code{TRUE} if \code{metric} is rank-invariant (e.g. \code{"auc"},
#'   \code{"aucpr"}), and \code{FALSE} otherwise. Set to \code{TRUE} when optimizing
#'   calibration metrics like \code{"ts_refinement"} during feature evolution while
#'   submitting to a rank-based competition evaluation metric like AUC.
#' @param stage5_stack_space Character string. Blending space for Stage 5 stacking:
#'   \code{"auto"} (default, uses \code{"logit"} for classification/multiclass when
#'   \code{rank_average = FALSE}, else \code{"prob"}), \code{"prob"}, or \code{"logit"}.
#' @param stage5_residual_blender Character string. Optional second-stage residual model
#'   fit on top of Stage 5 ensemble margins: \code{"none"} (default), \code{"lm"}, or
#'   \code{"spline_lm"}.
#' @param submission_file Optional file path to write \code{$submission} via \code{data.table::fwrite()}.
#' @param threads Integer thread count (default \code{default_threads()}).
#' @param seed Integer RNG seed for reproducibility (caller's \code{.Random.seed} is preserved).
#' @param verbose Logical or integer verbosity level.
#' @return An S3 object of class \code{"evo_competition_run"} containing:
#'   \code{predictions}, \code{submission}, \code{final_model} (\code{evo_ensemble} or
#'   \code{evo_recipe}), \code{islands_per_evaluator}, \code{stage2_island_evaluators},
#'   \code{stage2_recipes}, \code{stage3_views}, \code{stage3_view_types},
#'   \code{stage4_recipes}, \code{scout_ensemble}, \code{train_mega}, \code{test_mega},
#'   \code{adversarial}, \code{task}, \code{metric}, \code{target_transform},
#'   \code{discovery_rows}, \code{train_rows}, and \code{stage4_fitness}.
#' @export
run_competition_pipeline <- function(
  train,
  test,
  target_col,
  id_col = NULL,
  task = c("auto", "classification", "multiclass", "regression"),
  metric = "auto",
  ensemble_metric = NULL,
  preset = c("standard", "fast", "max"),
  preset_overrides = list(),
  evaluators = c("lightgbm", "xgboost", "realmlp"),
  islands_per_evaluator = NULL,
  tune_stage4 = c("none", "tabarena", "mbo"),
  evaluator_args = list(),
  cv_folds = 5L,
  cv_strategy = c("random", "stratified_group", "group", "time"),
  group_col = NULL,
  time_col = NULL,
  target_transform = c("auto", "none", "log1p", "yeo_johnson", "rank_gauss"),
  complexity_penalty = NULL,
  complexity_mode = c("auto", "bic_dynamic", "bic", "pac_bayes_dynamic", "pac_bayes", "none"),
  sd_penalty = NULL,
  max_search_rows = 200000L,
  drop_drifting = FALSE,
  drift_auc_threshold = 0.85,
  supervised_stage4 = FALSE,
  bag_folds = TRUE,
  rank_average = NULL,
  stage5_stack_space = c("auto", "prob", "logit"),
  stage5_residual_blender = c("none", "lm", "spline_lm"),
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
  tune_stage4 <- match.arg(tune_stage4)
  complexity_mode <- match.arg(complexity_mode)
  if (!is.null(complexity_penalty) && !identical(complexity_penalty, "none")) {
    if (!is.numeric(complexity_penalty) || length(complexity_penalty) != 1L ||
        is.na(complexity_penalty) || !is.finite(complexity_penalty) || complexity_penalty < 0) {
      stop("'complexity_penalty' must be NULL, 'none', or a non-negative number.")
    }
  }
  if (!is.null(sd_penalty) && !identical(sd_penalty, "none")) {
    if (!is.numeric(sd_penalty) || length(sd_penalty) != 1L ||
        is.na(sd_penalty) || !is.finite(sd_penalty) || sd_penalty < 0) {
      stop("'sd_penalty' must be NULL, 'none', or a non-negative number.")
    }
  }
  stage5_stack_space <- match.arg(stage5_stack_space)
  stage5_residual_blender <- match.arg(stage5_residual_blender)
  any_tuned_input <- any(vapply(evaluators, function(e) {
    is.character(e) && length(e) == 1L && has_evaluator(e) && is_tuned_evaluator(e)
  }, logical(1L)))
  if (tune_stage4 %in% c("tabarena", "mbo") || any_tuned_input) {
    missing_pkgs <- character(0)
    for (pkg in c("mlr3mbo", "paradox", "bbotk")) {
      if (!requireNamespace(pkg, quietly = TRUE)) {
        missing_pkgs <- c(missing_pkgs, pkg)
      }
    }
    if (length(missing_pkgs) > 0L) {
      stop(sprintf(
        "Tuned evaluation (tune_stage4 = '%s') requires packages: %s. Please install them.",
        tune_stage4,
        paste(missing_pkgs, collapse = ", ")
      ))
    }
  }
  cv_strategy <- match.arg(cv_strategy)
  cv_folds <- as.integer(cv_folds)
  if (is.na(cv_folds) || cv_folds < 2L) stop("'cv_folds' must be an integer >= 2.")
  if (!is.null(islands_per_evaluator)) {
    if (!is.numeric(islands_per_evaluator) || length(islands_per_evaluator) != 1L ||
        is.na(islands_per_evaluator) || !is.finite(islands_per_evaluator) ||
        islands_per_evaluator < 1) {
      stop("'islands_per_evaluator' must be NULL or a positive integer >= 1.")
    }
  }
  cfg <- .competition_preset_config(preset, overrides = preset_overrides)
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
  resolved_metric <- canonical_metric_name(resolved_metric, resolved_task)

  resolved_ens_metric <- if (!is.null(ensemble_metric)) {
    validate_metric(ensemble_metric, resolved_task)
    canonical_metric_name(ensemble_metric, resolved_task)
  } else if (isTRUE(rank_average) && !metric_is_rank_invariant(resolved_metric, resolved_task) && resolved_task == "classification") {
    "auc"
  } else {
    resolved_metric
  }

  resolved_rank_avg <- if (!is.null(rank_average)) {
    if (!is.logical(rank_average) || length(rank_average) != 1L) {
      stop("'rank_average' must be a logical scalar (TRUE or FALSE) or NULL.")
    }
    rank_average
  } else {
    metric_is_rank_invariant(resolved_ens_metric, resolved_task)
  }
  resolved_tt <- .resolve_target_transform(y_vec, resolved_task, target_transform)

  test_ids <- if (!is.null(id_col)) test[[id_col]] else NULL
  excl_cols <- unique(c(target_col, id_col))
  shared_features <- intersect(setdiff(names(train), excl_cols), setdiff(names(test), excl_cols))
  if (length(shared_features) == 0L) {
    stop("No shared feature columns found between 'train' and 'test'.")
  }

  train_dt <- shallow_copy_dt(data.table::as.data.table(train)[, c(shared_features, target_col), with = FALSE])
  test_dt <- shallow_copy_dt(data.table::as.data.table(test)[, shared_features, with = FALSE])
  .harmonize_train_test_types(train_dt, test_dt, shared_features)

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

  # Stage 2: Unified multi-evaluator island discovery
  all_t <- names(evo_transformers)
  unsup_t <- all_t[!vapply(all_t, is_supervised_transformer, logical(1L))]
  sup_t <- all_t[vapply(all_t, is_supervised_transformer, logical(1L))]

  if (!is.null(islands_per_evaluator)) {
    eff_ipe <- as.integer(islands_per_evaluator)
    s2_island_evaluators <- rep(active_evaluators, each = eff_ipe)
  } else if ("islands_per_evaluator" %in% names(preset_overrides)) {
    eff_ipe <- max(1L, as.integer(cfg$islands_per_evaluator))
    s2_island_evaluators <- rep(active_evaluators, each = eff_ipe)
  } else if ("islands" %in% names(preset_overrides)) {
    total_req <- max(1L, as.integer(cfg$islands))
    s2_island_evaluators <- rep(active_evaluators, length.out = total_req)
    eff_ipe <- as.integer(ceiling(total_req / length(active_evaluators)))
  } else {
    eff_ipe <- max(1L, as.integer(cfg$islands_per_evaluator))
    if (length(active_evaluators) == 1L && eff_ipe < as.integer(cfg$islands)) {
      eff_ipe <- as.integer(cfg$islands)
    }
    s2_island_evaluators <- rep(active_evaluators, each = eff_ipe)
  }
  total_s2_islands <- length(s2_island_evaluators)

  base_evals <- vapply(active_evaluators, get_base_evaluator, character(1L))
  all_mlp <- all(base_evals %in% c("realmlp", "tabm"))

  raw_s2_comp_pen <- if (!is.null(complexity_penalty)) complexity_penalty else cfg$s2_complexity_penalty
  raw_s2_comp_mode <- if (!identical(complexity_mode, "auto")) {
    complexity_mode
  } else if (!is.null(cfg$s2_complexity_mode) && !identical(cfg$s2_complexity_mode, "auto")) {
    cfg$s2_complexity_mode
  } else {
    "auto"
  }
  raw_s2_comp_mode <- match.arg(
    raw_s2_comp_mode,
    c("auto", "bic_dynamic", "bic", "pac_bayes_dynamic", "pac_bayes", "none")
  )
  if (identical(raw_s2_comp_pen, "none") || identical(raw_s2_comp_mode, "none")) {
    s2_comp_pen <- 0
    s2_comp_mode <- "none"
  } else {
    s2_comp_pen <- if (!is.null(raw_s2_comp_pen)) {
      if (!is.numeric(raw_s2_comp_pen) || length(raw_s2_comp_pen) != 1L ||
          is.na(raw_s2_comp_pen) || !is.finite(raw_s2_comp_pen) || raw_s2_comp_pen < 0) {
        stop("'complexity_penalty' / 's2_complexity_penalty' must be NULL, 'none', or a non-negative number.")
      }
      as.numeric(raw_s2_comp_pen)
    } else if (all_mlp) {
      0.30
    } else {
      0.25
    }
    s2_comp_mode <- if (identical(raw_s2_comp_mode, "auto")) {
      if (n_discovery < 10000L) "pac_bayes_dynamic" else "bic_dynamic"
    } else {
      raw_s2_comp_mode
    }
  }

  raw_s2_sd_pen <- if (!is.null(sd_penalty)) sd_penalty else cfg$s2_sd_penalty
  if (is.null(raw_s2_sd_pen) || identical(raw_s2_sd_pen, "none")) {
    s2_sd_pen <- 0
  } else {
    if (!is.numeric(raw_s2_sd_pen) || length(raw_s2_sd_pen) != 1L ||
        is.na(raw_s2_sd_pen) || !is.finite(raw_s2_sd_pen) || raw_s2_sd_pen < 0) {
      stop("'sd_penalty' / 's2_sd_penalty' must be NULL, 'none', or a non-negative number.")
    }
    s2_sd_pen <- as.numeric(raw_s2_sd_pen)
  }

  use_metacv <- (n_discovery >= 1000L) && (cv_strategy != "time") && (total_s2_islands >= 2L)
  mig_cfg <- if (total_s2_islands >= 2L) {
    migration_config(
      topology = topology_complete(islands = total_s2_islands),
      policy = policy_gibbs_pull(stagnation_threshold = 2L, temperature = 1.0),
      payload = "gene_only"
    )
  } else {
    NULL
  }

  base_seed <- if (!is.null(seed)) as.integer(seed) else 42L
  s2_ev_args_map <- stats::setNames(lapply(active_evaluators, function(ev) {
    ev_args <- .evaluator_stage_args(ev, stage = 2L, cfg = cfg, n_rows = n_discovery, custom_evaluator_args = evaluator_args)
    .map_evaluator_args_for_evolve(ev_args)
  }), active_evaluators)

  if (show_msg) {
    message(sprintf(
      "[Stage 2] Discovering features in unified %d-island topology (%d/evaluator): %s...",
      total_s2_islands, eff_ipe, paste(s2_island_evaluators, collapse = ", ")
    ))
  }

  s2_common <- list(
    data = search_dt,
    target_col = target_col,
    task = resolved_task,
    evaluator = s2_island_evaluators,
    metric = resolved_metric,
    allowed_transformers = if (isTRUE(supervised_stage4)) all_t else unsup_t,
    evaluation_strategy = if (use_metacv) "metacv" else "cv",
    metacv_selection = "tournament",
    cv_folds = if (use_metacv) total_s2_islands else min(cv_folds, 5L),
    cv_strategy = cv_strategy,
    group_col = group_col,
    time_col = time_col,
    split_ratio = NULL,
    split_ids = NULL,
    holdout_frac = if (n_discovery >= 500L) cfg$s2_holdout else 0.0,
    test_data = if (n_discovery >= 500L && cfg$s2_holdout > 0) test_dt else NULL,
    sd_penalty = s2_sd_pen,
    confirm_top_k = if (length(unique(s2_island_evaluators)) == 1L && !use_metacv && cv_strategy != "time") cfg$s2_confirm_k else 0L,
    seed_genes = NULL,
    model_all_final_genes = TRUE,
    migration = mig_cfg,
    islands = total_s2_islands,
    migration_interval = 1L,
    migration_rate = 1L,
    gene_migration_prob = 0.25,
    generations = if (all_mlp) cfg$s2_mlp_gen else cfg$s2_gen,
    pop_size = if (all_mlp) cfg$s2_mlp_pop else cfg$s2_pop,
    complexity_penalty = s2_comp_pen,
    complexity_mode = s2_comp_mode,
    complexity_target = "all_features",
    target_transform = resolved_tt,
    threads = threads,
    max_clustering_size = min(cfg$clust_cap, n_discovery),
    seed = base_seed + 100L,
    evaluator_extra_args = s2_ev_args_map,
    verbose = verbose
  )
  stage2_unified_recipe <- do.call(evolve_features, s2_common)

  stage2_recipes <- vector("list", length(active_evaluators))
  names(stage2_recipes) <- active_evaluators
  sup_seed_genes <- stats::setNames(vector("list", length(active_evaluators)), active_evaluators)
  sup_input_cols <- stats::setNames(vector("list", length(active_evaluators)), active_evaluators)
  for (ev in active_evaluators) {
    idx_j <- which(s2_island_evaluators == ev)
    ev_extra <- s2_ev_args_map[[ev]]
    if (length(active_evaluators) == 1L) {
      rec_ev <- stage2_unified_recipe
      rec_ev$evaluator <- ev
      rec_ev$extra_args <- ev_extra
      if (!is.null(rec_ev$best_individual)) {
        rec_ev$best_individual$extra_args <- ev_extra
      }
      if (!is.null(rec_ev$island_bests)) {
        for (k in seq_along(rec_ev$island_bests)) {
          rec_ev$island_bests[[k]]$extra_args <- ev_extra
        }
      }
    } else {
      ev_inds <- stage2_unified_recipe$island_bests[idx_j]
      for (k in seq_along(ev_inds)) {
        ev_inds[[k]]$extra_args <- ev_extra
      }
      ev_fits <- vapply(ev_inds, function(ind) {
        if (!is.null(ind$fitness) && is.finite(ind$fitness)) ind$fitness else -Inf
      }, numeric(1L))
      best_ev_ind <- ev_inds[[which.max(ev_fits)]]
      u_best <- stage2_unified_recipe$best_individual
      u_matches <- !is.null(u_best) &&
        identical(u_best$evaluator, ev) &&
        is.finite(u_best$fitness) &&
        (!is.finite(best_ev_ind$fitness) || u_best$fitness >= best_ev_ind$fitness)
      if (u_matches) {
        best_ev_ind <- u_best
        best_ev_ind$extra_args <- ev_extra
      }
      rec_ev <- stage2_unified_recipe
      rec_ev$evaluator <- ev
      rec_ev$best_individual <- best_ev_ind
      rec_ev$island_bests <- ev_inds
      rec_ev$island_baselines <- stage2_unified_recipe$island_baselines[idx_j]
      rec_ev$island_best_fitness <- ev_fits
      rec_ev$island_improvements <- stage2_unified_recipe$island_improvements[idx_j]
      rec_ev$island_headroom_closed <- stage2_unified_recipe$island_headroom_closed[idx_j]
      rec_ev$oof_preds <- best_ev_ind$val_preds
      rec_ev$extra_args <- ev_extra
      rec_ev$alignment_cache <- new.env(hash = TRUE, parent = emptyenv())
      if (!u_matches) {
        rec_ev$best_model <- NULL
        rec_ev$fold_models <- NULL
      }
    }

    ev_sup_raw <- list()
    ev_req_inputs <- character(0)
    if (!is.null(rec_ev$best_individual)) {
      split_best <- .split_individual_tainted_genes(rec_ev$best_individual)
      rec_ev$best_individual <- split_best$unsup_ind
      if (length(split_best$sup_genes) > 0L) {
        ev_sup_raw <- c(ev_sup_raw, split_best$sup_genes)
        ev_req_inputs <- c(ev_req_inputs, split_best$sup_input_cols)
        rec_ev$best_model <- NULL
        rec_ev$fold_models <- NULL
      }
    }
    if (!is.null(rec_ev$island_bests) && is.list(rec_ev$island_bests)) {
      for (k in seq_along(rec_ev$island_bests)) {
        split_k <- .split_individual_tainted_genes(rec_ev$island_bests[[k]])
        rec_ev$island_bests[[k]] <- split_k$unsup_ind
        if (length(split_k$sup_genes) > 0L) {
          ev_sup_raw <- c(ev_sup_raw, split_k$sup_genes)
          ev_req_inputs <- c(ev_req_inputs, split_k$sup_input_cols)
        }
      }
    }
    sup_seed_genes[[ev]] <- if (length(ev_sup_raw) > 0L) normalize_seed_genes(ev_sup_raw, available_cols = NULL) else list()
    sup_input_cols[[ev]] <- unique(ev_req_inputs)
    stage2_recipes[[ev]] <- rec_ev
  }

  # Stage 3: Mega-Matrix Synthesis onto 100% of train_dt and test_dt
  if (length(stage2_recipes) >= 2L || total_s2_islands >= 2L) {
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
  if (!data.table::is.data.table(train_mega)) train_mega <- data.table::as.data.table(train_mega)
  if (!data.table::is.data.table(test_mega)) test_mega <- data.table::as.data.table(test_mega)

  for (aux_col in unique(c(group_col, time_col))) {
    if (!is.null(aux_col) && nzchar(aux_col) && aux_col %in% names(train_dt) && !aux_col %in% names(train_mega)) {
      train_mega[[aux_col]] <- train_dt[[aux_col]]
      if (aux_col %in% names(test_dt)) {
        test_mega[[aux_col]] <- test_dt[[aux_col]]
      }
    }
  }

  # Ensure train_mega and test_mega have strictly identical feature schemas and column ordering
  mega_feat_cols <- setdiff(names(train_mega), target_col)
  missing_in_test <- setdiff(mega_feat_cols, names(test_mega))
  if (length(missing_in_test) > 0L) {
    for (mfc in missing_in_test) {
      train_v <- train_mega[[mfc]]
      def_val <- if (is.numeric(train_v)) {
        m_val <- stats::median(train_v, na.rm = TRUE)
        if (is.na(m_val) || !is.finite(m_val)) 0.0 else m_val
      } else if (is.factor(train_v)) {
        levels(train_v)[1L]
      } else {
        "__missing__"
      }
      test_mega[, (mfc) := def_val]
    }
  }
  extra_in_test <- setdiff(names(test_mega), mega_feat_cols)
  if (length(extra_in_test) > 0L) {
    test_mega[, (extra_in_test) := NULL]
  }
  data.table::setcolorder(test_mega, mega_feat_cols)

  train_mega[[target_col]] <- train_dt[[target_col]]

  if (show_msg) {
    message(sprintf(
      "[Stage 3] Mega-Matrix synthesized on %d full training rows: %d raw -> %d engineered features.",
      nrow(train_mega), length(shared_features), ncol(train_mega) - 1L
    ))
  }

  stage3_view_res <- .select_stage3_views(
    train_mega = train_mega,
    stage2_recipes = stage2_recipes,
    scout_ens = scout_ens,
    active_evaluators = active_evaluators,
    target_col = target_col,
    task = resolved_task,
    metric = resolved_metric,
    cv_folds = cv_folds,
    cv_strategy = cv_strategy,
    group_col = group_col,
    time_col = time_col,
    max_search_rows = max_search_rows,
    adv_weights = eff_adv_w,
    target_transform = resolved_tt,
    cfg = cfg,
    evaluator_args = evaluator_args,
    threads = threads,
    seed = base_seed,
    verbose = verbose,
    sup_seed_genes = sup_seed_genes,
    sup_input_cols = sup_input_cols
  )
  stage3_views <- stage3_view_res$views
  stage3_view_types <- stage3_view_res$view_types

  if (nrow(train_dt) >= 10000L) {
    stage2_recipes <- .prune_stage_recipes(stage2_recipes)
    gc(verbose = FALSE)
  }

  # Supervised genes discovered directly in Stage 2 were already split to unfitted
  # symbolic seed_genes before Stage 3 Mega-Matrix synthesis (zero target leakage).
  stage3_sup_recipes <- if (isTRUE(supervised_stage4)) sup_seed_genes else NULL

  # Stage 4: High-capacity 1-Gen retraining on 100% of winning Stage 3 feature view
  stage4_recipes <- vector("list", length(active_evaluators))
  names(stage4_recipes) <- active_evaluators
  s4_seed <- base_seed + 1000L

  for (idx in seq_along(active_evaluators)) {
    ev <- active_evaluators[idx]
    s4_ev <- .resolve_stage4_evaluator(ev, tune_stage4)
    base_ev <- get_base_evaluator(ev)
    s4_cols_ev <- unique(c(
      stage3_views[[ev]]$cols,
      intersect(sup_input_cols[[ev]], names(train_mega)),
      intersect(c(group_col, time_col), names(train_mega)),
      target_col
    ))
    train_mega_ev <- train_mega[, s4_cols_ev, with = FALSE]
    s4_seed_genes <- if (length(sup_seed_genes[[ev]]) > 0L) {
      normalize_seed_genes(sup_seed_genes[[ev]], setdiff(names(train_mega_ev), target_col))
    } else {
      NULL
    }

    ev_args <- .evaluator_stage_args(s4_ev, stage = 4L, cfg = cfg, n_rows = nrow(train_mega_ev), custom_evaluator_args = evaluator_args)

    if (show_msg) {
      if (is_tuned_evaluator(s4_ev)) {
        mbo_trials <- if (!is.null(ev_args$mbo_iters)) ev_args$mbo_iters else cfg$mbo_iters
        tune_tag <- if (grepl("tabarena", s4_ev)) "TabArena" else "MBO"
        message(sprintf("[Stage 4] High-capacity %d-fold %s Bayesian optimization retraining for '%s' (%s view, %d cols, iters: %d)...",
                        cv_folds, tune_tag, s4_ev, stage3_view_types[[ev]], ncol(train_mega_ev) - 1L, mbo_trials))
      } else {
        message(sprintf("[Stage 4] High-capacity %d-fold retraining for '%s' (%s view, %d cols)...",
                        cv_folds, s4_ev, stage3_view_types[[ev]], ncol(train_mega_ev) - 1L))
      }
    }
    s4_common <- list(
      data = train_mega_ev,
      target_col = target_col,
      task = resolved_task,
      evaluator = s4_ev,
      metric = resolved_metric,
      generations = 1L,
      pop_size = 1L,
      seed_genes = if (length(s4_seed_genes) > 0L) s4_seed_genes else NULL,
      evaluation_strategy = "cv",
      cv_folds = cv_folds,
      cv_strategy = cv_strategy,
      group_col = group_col,
      time_col = time_col,
      split_ratio = NULL,
      split_ids = NULL,
      sd_penalty = 0,
      bag_folds = isTRUE(bag_folds),
      target_transform = resolved_tt,
      threads = threads,
      seed = s4_seed,
      verbose = verbose
    )
    ev_evolve_args <- .map_evaluator_args_for_evolve(ev_args)
    for (k in names(ev_evolve_args)) s4_common[[k]] <- ev_evolve_args[[k]]
    if (is.null(s4_common$compute_importances)) s4_common$compute_importances <- FALSE
    stage4_recipes[[ev]] <- do.call(evolve_features, s4_common)
  }

  stage4_fitness <- vapply(stage4_recipes, function(r) {
    val <- r$best_individual$raw_fitness
    if (is.null(val) || !is.finite(val)) r$best_individual$fitness else val
  }, numeric(1L))

  # Stage 5: Stacked Ensembling & Test Inference
  if (length(stage4_recipes) >= 2L) {
    if (show_msg) {
      rank_lbl <- if (isTRUE(resolved_rank_avg)) " (rank-averaged)" else ""
      message(sprintf("[Stage 5] Stacking Stage 4 models via nested OOF elastic-net%s...", rank_lbl))
    }
    final_model <- ensemble_islands(
      recipe = stage4_recipes,
      data = train_mega,
      target_col = target_col,
      method = "stack",
      metric = resolved_ens_metric,
      stack_folds = cv_folds,
      stack_alpha = 0.5,
      rank_average = resolved_rank_avg,
      stack_space = stage5_stack_space,
      residual_blender = stage5_residual_blender,
      bag_folds = isTRUE(bag_folds),
      threads = threads,
      seed = s4_seed,
      verbose = show_msg
    )
  } else {
    final_model <- stage4_recipes[[1L]]
  }

  preds <- predict_model(final_model, test_mega, bag_folds = isTRUE(bag_folds), threads = threads)

  if (nrow(train_dt) >= 10000L && length(stage4_recipes) >= 2L) {
    stage4_recipes <- .prune_stage_recipes(stage4_recipes)
    gc(verbose = FALSE)
  }

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
      islands_per_evaluator = eff_ipe,
      stage2_island_evaluators = s2_island_evaluators,
      stage2_recipes = stage2_recipes,
      stage3_views = stage3_views,
      stage3_view_types = stage3_view_types,
      stage3_supervised_recipes = stage3_sup_recipes,
      stage4_recipes = stage4_recipes,
      scout_ensemble = scout_ens,
      train_mega = train_mega,
      test_mega = test_mega,
      adversarial = adv_info,
      dropped_drifting_features = dropped_drift_cols,
      task = resolved_task,
      metric = resolved_metric,
      ensemble_metric = resolved_ens_metric,
      target_col = target_col,
      id_col = id_col,
      preset = preset,
      preset_config = cfg,
      preset_overrides = preset_overrides,
      evaluator_args = evaluator_args,
      evaluators = active_evaluators,
      cv_folds = cv_folds,
      cv_strategy = cv_strategy,
      target_transform = resolved_tt,
      bag_folds = isTRUE(bag_folds),
      rank_average = resolved_rank_avg,
      raw_feature_count = length(shared_features),
      mega_feature_count = ncol(train_mega) - 1L,
      train_rows = nrow(train_dt),
      discovery_rows = n_discovery,
      test_rows = nrow(test_dt),
      tune_stage4 = tune_stage4,
      stage4_evaluators = stats::setNames(vapply(stage4_recipes, function(r) r$evaluator, character(1L)), active_evaluators),
      stage4_best_params = stats::setNames(lapply(stage4_recipes, function(r) {
        if (!is.null(r$best_individual$best_params)) r$best_individual$best_params else NULL
      }), active_evaluators),
      stage4_fitness = stage4_fitness,
      final_cv_fitness = ens_fitness,
      weights = resolved_weights
    ),
    class = "evo_competition_run"
  )
}
