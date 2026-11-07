#' Train a boosted tree model
#'
#' Internal helper that encapsulates LightGBM / XGBoost parameter construction
#' and training. Returns the fitted model, optional predictions on validation
#' data, and feature importances.
#'
#' @param x_train Numeric matrix of training features.
#' @param y_train Numeric vector of training labels.
#' @param x_val Optional numeric matrix of validation features.
#' @param y_val Optional numeric vector of validation labels.
#' @param task Task type: "classification", "multiclass", or "regression".
#' @param evaluator Model type: "lightgbm" or "xgboost".
#' @param threads Number of threads.
#' @param num_class Number of classes (required for multiclass).
#' @param nrounds Number of boosting rounds.
#' @param ... Additional arguments passed to the evaluator training function.
#' @return A list with elements \code{model},
#'   \code{predictions} (NULL when \code{x_val} is NULL),
#'   and \code{importances} (named numeric vector or NULL).
#' @keywords internal
train_model <- function(x_train = NULL, y_train = NULL, x_val = NULL, y_val = NULL,
                        task = "classification", evaluator = "lightgbm",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {

  evaluator_entry <- get_evaluator(evaluator)

  resolved <- resolve_param_aliases(list(...), defaults = list(threads = threads, nrounds = nrounds))
  threads <- resolved$threads
  nrounds <- resolved$nrounds

  x_train <- .sanitize_feature_matrix(x_train, threads = threads)
  x_val   <- .sanitize_feature_matrix(x_val, threads = threads)

  evaluator_entry$train_func(
    x_train = x_train,
    y_train = y_train,
    x_val = x_val,
    y_val = y_val,
    task = task,
    threads = threads,
    num_class = num_class,
    nrounds = nrounds,
    ...
  )
}

#' Fit a K-fold bagged ensemble of fold-specific feature pipelines and models
#'
#' Each fold fits its own supervised feature states strictly on its training split
#' and trains its model with exact early-stopping on its validation split.
#'
#' @keywords internal
#' @noRd
.fit_fold_models_bag <- function(ind, data_dt, fold_ids, target_col, task,
                                 classes = NULL, num_class = NULL, metric = "default",
                                 evaluator = "lightgbm", best_params = NULL,
                                 extra_args = list(), threads = default_threads(),
                                 seed = NULL, state_cache = NULL, shared_full = NULL,
                                 target_transform = getOption("evoFE.target_transform", "none"),
                                 verbose = FALSE) {
  if (is.null(fold_ids) || length(fold_ids) != nrow(data_dt)) return(NULL)
  u_folds <- sort(unique(fold_ids[!is.na(fold_ids)]))
  if (length(u_folds) < 2L) return(NULL)

  use_tt <- identical(task, "regression") && !is.null(target_transform) && !identical(target_transform, "none")
  use_smear <- !identical(canonical_metric_name(metric, task), "mae")

  resolved <- resolve_tuned_evaluator(evaluator, best_params, extra_args, always_unwrap = TRUE)
  final_eval <- resolved$evaluator
  model_args <- resolved$extra_args
  clean_ind <- strip_individual_state(ind)
  if (is.null(state_cache)) state_cache <- new.env(hash = TRUE, parent = emptyenv())
  if (is.null(shared_full)) shared_full <- shallow_copy_dt(data.table::as.data.table(data_dt))

  fold_models <- vector("list", length(u_folds))
  for (idx in seq_along(u_folds)) {
    f <- u_folds[idx]
    tr_idx <- which(fold_ids != f)
    va_idx <- which(fold_ids == f)
    if (length(tr_idx) == 0L || length(va_idx) == 0L) next

    tr_dt <- shallow_copy_dt(data.table::as.data.table(data_dt[tr_idx, ]))
    va_dt <- shallow_copy_dt(data.table::as.data.table(data_dt[va_idx, ]))
    data.table::setattr(tr_dt, ".row_id", tr_idx)
    data.table::setattr(va_dt, ".row_id", va_idx)

    res_f <- apply_individual(
      clean_ind, tr_dt, va_dt, target_col = target_col,
      state_cache = state_cache, allow_prune = TRUE,
      full_data = shared_full, validate_genes = FALSE
    )
    if (!is.null(res_f$full)) shared_full <- res_f$full
    ind_f <- strip_loo_state(res_f$ind)
    features <- extract_individual_features(ind_f, target_col = target_col)

    x_tr <- .sanitize_feature_matrix(res_f$train[, features, with = FALSE], threads = threads)
    x_va <- .sanitize_feature_matrix(res_f$val[, features, with = FALSE], threads = threads)
    y_tr <- res_f$train[[target_col]]
    y_va <- res_f$val[[target_col]]
    if (task == "multiclass") {
      y_tr <- encode_multiclass_target(y_tr, classes)
      y_va <- encode_multiclass_target(y_va, classes)
    } else if (task == "classification") {
      y_tr <- encode_binary_target(y_tr)
      y_va <- encode_binary_target(y_va)
    }

    tt_state_f <- NULL
    y_va_raw <- y_va
    if (use_tt) {
      tt_fit <- .fit_target_transform(y_tr, target_transform)
      y_tr <- tt_fit$y_trans
      y_va <- .apply_target_transform(y_va, tt_fit$state)
      tt_state_f <- tt_fit$state
    }

    fold_seed <- if (!is.null(seed) && is.finite(seed)) as.integer(seed) + as.integer(idx) else NULL
    res_m <- do.call(train_model, c(
      list(
        x_train = x_tr, y_train = y_tr,
        x_val = x_va, y_val = y_va,
        task = task, evaluator = final_eval,
        threads = threads, num_class = num_class,
        metric = metric, verbose = FALSE,
        best_params = best_params, seed = fold_seed
      ),
      model_args
    ))
    if (use_tt && !is.null(tt_state_f)) {
      tt_state_f <- .update_target_transform_smearing(tt_state_f, y_va_raw, res_m$predictions)
    }
    fold_models[[idx]] <- list(
      ind = ind_f,
      model = res_m$model,
      evaluator = final_eval,
      best_iteration = extract_best_iteration(res_m),
      target_transform_state = tt_state_f,
      use_smearing = use_smear
    )
  }

  fold_models[!vapply(fold_models, is.null, logical(1))]
}

#' Predict with a K-fold bagged list of fold models
#' @keywords internal
#' @noRd
.predict_fold_models_bag <- function(fold_models, dt_input, target_col = NULL,
                                     task = "classification", classes = NULL,
                                     threads = default_threads(), user_args = list(),
                                     state_cache = NULL) {
  if (is.null(fold_models) || length(fold_models) == 0L) return(NULL)
  dt_base <- shallow_copy_dt(data.table::as.data.table(dt_input))
  avg_preds <- NULL
  n_valid <- 0L

  for (fm in fold_models) {
    if (is.null(fm) || is.null(fm$model) || is.null(fm$ind)) next
    eval_entry <- get_evaluator(fm$evaluator)
    dt_f <- shallow_copy_dt(dt_base)
    res_f <- apply_individual(
      fm$ind, dt_f, val_data = NULL, target_col = NULL,
      allow_prune = TRUE, state_cache = NULL, full_data = NULL
    )
    feats <- extract_individual_features(res_f$ind, target_col = target_col, unique = TRUE)
    x_new <- .sanitize_feature_matrix(res_f$train[, feats, with = FALSE], threads = threads)
    preds_f <- do.call(eval_entry$predict_func, c(
      list(model = fm$model, x_new = x_new, task = task, threads = threads),
      user_args
    ))
    if (!is.null(fm$target_transform_state)) {
      preds_f <- .invert_target_transform(
        preds_f,
        fm$target_transform_state,
        use_smearing = !identical(fm$use_smearing, FALSE)
      )
    }
    if (!is.null(classes)) {
      preds_f <- format_multiclass_predictions(preds_f, classes)
    }
    if (is.null(avg_preds)) {
      avg_preds <- preds_f
    } else {
      avg_preds <- avg_preds + preds_f
    }
    n_valid <- n_valid + 1L
  }

  if (n_valid > 0L) avg_preds / n_valid else NULL
}
