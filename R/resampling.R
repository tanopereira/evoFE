# Resampling utilities: stratified splitting, quantile-stratified regression folds,
# stratified group K-fold, and adversarial covariate-shift holdout weighting.

#' Construct discrete target strata for classification or continuous regression
#'
#' For categorical or low-cardinality numeric targets (\code{<= 10} unique values),
#' returns \code{as.factor(y)}. For continuous numeric targets, partitions \code{y}
#' into \code{n_bins} empirical rank-quantile strata so ties or point-masses never
#' cause non-unique quantile break errors.
#'
#' @param y Target vector.
#' @param n_bins Integer maximum number of quantile bins for continuous targets.
#' @return Factor vector of length \code{length(y)} representing stratum membership.
#' @keywords internal
#' @noRd
.target_strata <- function(y, n_bins = 10L) {
  n <- length(y)
  if (n <= 1L) return(factor(rep(1L, n)))
  u_vals <- unique(y[!is.na(y)])
  if (length(u_vals) <= 1L) return(factor(rep(1L, n)))
  if (!is.numeric(y) || length(u_vals) <= 10L) {
    fac <- as.character(y)
    fac[is.na(fac)] <- ".NA_STRATUM"
    return(as.factor(fac))
  }
  valid <- !is.na(y) & is.finite(y)
  n_valid <- sum(valid)
  b <- as.integer(min(as.integer(n_bins), max(2L, floor(n_valid / 5L))))
  if (b <= 1L || n_valid <= 2L) return(factor(rep(1L, n)))
  bins <- integer(n)
  r <- rank(y[valid], ties.method = "first")
  bins[valid] <- as.integer(ceiling(r / n_valid * b))
  bins[bins < 1L] <- 1L
  bins[bins > b] <- b
  if (any(!valid)) bins[!valid] <- b + 1L
  as.factor(bins)
}

#' Allocate integer counts across strata using Hamilton's largest-remainder method
#' @keywords internal
#' @noRd
.hamilton_quota <- function(stratum_sizes, target_total) {
  total_n <- sum(stratum_sizes)
  if (total_n <= 0L || target_total <= 0L) return(integer(length(stratum_sizes)))
  if (target_total >= total_n) return(as.integer(stratum_sizes))
  exact <- stratum_sizes * (as.numeric(target_total) / as.numeric(total_n))
  quotas <- as.integer(floor(exact))
  quotas <- pmin(quotas, as.integer(stratum_sizes))
  rem_needed <- as.integer(target_total) - sum(quotas)
  if (rem_needed > 0L) {
    frac <- exact - quotas
    frac[quotas >= stratum_sizes] <- -1.0
    ord <- order(frac, decreasing = TRUE)
    for (idx in ord) {
      if (rem_needed <= 0L) break
      if (quotas[idx] < stratum_sizes[idx]) {
        quotas[idx] <- quotas[idx] + 1L
        rem_needed <- rem_needed - 1L
      }
    }
  }
  quotas
}

#' Stratified, group-aware, and importance-weighted splitting helper
#'
#' @param y Target vector.
#' @param ratio Numeric vector of length 2 (\code{c(train, val)}) or 3 (\code{c(train, val, holdout)}).
#' @param group Optional vector of group IDs. When provided, all rows sharing a group ID
#'   are placed in the same split partition.
#' @param weights Optional numeric vector of positive sampling weights for selecting
#'   validation/holdout rows (e.g. adversarial test-to-train density ratios).
#' @return Character vector of length \code{length(y)} with labels \code{"train"}, \code{"val"},
#'   and optionally \code{"holdout"}.
#' @keywords internal
stratified_split <- function(y, ratio, group = NULL, weights = NULL) {
  n <- length(y)
  if (n == 0L) return(character(0))
  ratios <- ratio / sum(ratio)
  is_three_way <- (length(ratios) >= 3L)

  # Group-intact stratified splitting when group is provided
  if (!is.null(group) && length(group) == n) {
    ux <- unique(group)
    n_g <- length(ux)
    if (n_g >= 2L) {
      g_match <- match(group, ux)
      gsizes <- tabulate(g_match, nbins = n_g)
      strata <- .target_strata(y, n_bins = min(10L, max(2L, floor(n_g / 3L))))
      n_s <- nlevels(strata)
      s_int <- as.integer(strata)

      # Build group x stratum count matrix
      g_strata <- matrix(0, nrow = n_g, ncol = n_s)
      for (i in seq_len(n)) {
        g_strata[g_match[i], s_int[i]] <- g_strata[g_match[i], s_int[i]] + 1.0
      }

      n_parts <- if (is_three_way) 3L else 2L
      part_names <- if (is_three_way) c("train", "val", "holdout") else c("train", "val")
      part_ratios <- ratios[seq_len(n_parts)]
      target_loads <- pmax(1.0, n * part_ratios)
      target_strata <- outer(part_ratios, colSums(g_strata))
      target_strata[target_strata < 1e-6] <- 1.0

      part_load <- numeric(n_parts)
      part_strata <- matrix(0, nrow = n_parts, ncol = n_s)
      g_part <- integer(n_g)

      # Shuffle ties randomly before sorting by group size descending
      perm <- sample.int(n_g)
      ord <- perm[order(gsizes[perm], decreasing = TRUE)]

      for (gi in ord) {
        sz <- gsizes[gi]
        gs <- g_strata[gi, ]
        best_p <- 1L
        best_cost <- Inf
        for (p in seq_len(n_parts)) {
          # Ensure each non-empty partition gets at least 1 group first if possible
          load_rel <- (part_load[p] + sz) / target_loads[p]
          strata_rel <- (part_strata[p, ] + gs) / target_strata[p, ]
          cost <- load_rel^2 + mean(strata_rel^2)
          if (cost < best_cost) {
            best_cost <- cost
            best_p <- p
          }
        }
        g_part[gi] <- best_p
        part_load[best_p] <- part_load[best_p] + sz
        part_strata[best_p, ] <- part_strata[best_p, ] + gs
      }

      # Ensure both train and val have at least 1 group
      if (!any(g_part == 1L) && n_g >= 2L) g_part[ord[1L]] <- 1L
      if (!any(g_part == 2L) && n_g >= 2L) g_part[ord[n_g]] <- 2L

      return(part_names[g_part[g_match]])
    }
  }

  # Target-total sizes matching exact overall rounding
  n_train_total <- round(n * ratios[1])
  if (n_train_total < 1L && n >= 1L) n_train_total <- 1L
  if (n_train_total > n) n_train_total <- n

  n_val_total <- if (!is_three_way) {
    n - n_train_total
  } else {
    nv <- round(n * ratios[2])
    if (nv < 1L && (n - n_train_total) >= 1L) nv <- 1L
    if (n_train_total + nv > n) nv <- max(0L, n - n_train_total)
    nv
  }

  strata <- .target_strata(y, n_bins = 10L)
  levels_y <- levels(strata)
  lvl_indices <- lapply(levels_y, function(lvl) which(strata == lvl))
  lvl_sizes <- vapply(lvl_indices, length, integer(1))

  train_quotas <- .hamilton_quota(lvl_sizes, n_train_total)
  rem_sizes <- lvl_sizes - train_quotas
  val_quotas <- if (!is_three_way) rem_sizes else .hamilton_quota(rem_sizes, n_val_total)

  use_weights <- !is.null(weights) && is.numeric(weights) && length(weights) == n &&
    all(is.finite(weights)) && any(weights > 0) && stats::sd(weights) > 1e-9

  res <- character(n)
  for (s_idx in seq_along(levels_y)) {
    lvl_idx <- lvl_indices[[s_idx]]
    n_lvl <- length(lvl_idx)
    if (n_lvl == 0L) next

    n_tr <- train_quotas[s_idx]
    n_va <- val_quotas[s_idx]
    n_non_tr <- n_lvl - n_tr

    if (use_weights && n_non_tr > 0L && n_tr > 0L) {
      # Sample validation/holdout rows proportional to adversarial test weights
      w_sub <- pmax(1e-9, weights[lvl_idx])
      non_tr_idx <- if (n_lvl == 1L) {
        lvl_idx
      } else {
        sample(lvl_idx, size = n_non_tr, replace = FALSE, prob = w_sub)
      }
      train_idx <- setdiff(lvl_idx, non_tr_idx)
      shuffled_non_tr <- sample(non_tr_idx)
    } else {
      shuffled_lvl_idx <- if (n_lvl == 1L) lvl_idx else sample(lvl_idx)
      train_idx <- if (n_tr > 0L) shuffled_lvl_idx[seq_len(n_tr)] else integer(0)
      shuffled_non_tr <- if (n_non_tr > 0L) shuffled_lvl_idx[(n_tr + 1L):n_lvl] else integer(0)
    }

    if (!is_three_way) {
      res[train_idx] <- "train"
      res[shuffled_non_tr] <- "val"
    } else {
      val_idx <- if (n_va > 0L && length(shuffled_non_tr) >= n_va) {
        shuffled_non_tr[seq_len(n_va)]
      } else {
        integer(0)
      }
      holdout_idx <- setdiff(shuffled_non_tr, val_idx)
      res[train_idx] <- "train"
      res[val_idx] <- "val"
      res[holdout_idx] <- "holdout"
    }
  }

  res[res == ""] <- "train"
  res
}

#' Build cross-validation fold assignments
#'
#' Internal helper supporting four strategies:
#' \itemize{
#'   \item \code{random}: target-stratified folds (class-stratified for classification/multiclass,
#'     quantile-stratified for regression).
#'   \item \code{time}: rows ordered by \code{time_col} and split into k
#'     contiguous chronological blocks (blocked CV; each block is held out once).
#'     Rows with missing time values sort last.
#'   \item \code{group} / \code{stratified_group}: all rows of a group (\code{group_col})
#'     are assigned to a single fold via greedy largest-first balancing while
#'     simultaneously balancing target strata when \code{target_col} is provided.
#' }
#' @noRd
.build_cv_folds <- function(data, cv_folds, strategy = "random",
                            time_col = NULL, group_col = NULL,
                            target_col = NULL, task = NULL,
                            shuffle_groups = FALSE) {
  n <- nrow(data)
  if (strategy == "time") {
    ord <- order(data[[time_col]], method = "radix")
    f <- integer(n)
    f[ord] <- as.integer(cut(seq_len(n), breaks = cv_folds, labels = FALSE))
    return(f)
  }
  if (strategy %in% c("group", "stratified_group")) {
    g <- data[[group_col]]
    ux <- unique(g)
    n_g <- length(ux)
    g_match <- match(g, ux)
    gsizes <- tabulate(g_match, nbins = n_g)
    fold_load <- integer(cv_folds)
    gfold <- integer(n_g)
    g_order <- if (isTRUE(shuffle_groups) && n_g > 1L) {
      order(gsizes, sample.int(n_g), decreasing = TRUE)
    } else {
      order(gsizes, decreasing = TRUE)
    }

    has_target <- !is.null(target_col) && target_col %in% names(data)
    if (has_target && (strategy == "stratified_group" || length(unique(data[[target_col]])) > 1L)) {
      strata <- .target_strata(data[[target_col]], n_bins = min(10L, max(2L, floor(n_g / cv_folds))))
      n_s <- nlevels(strata)
      s_int <- as.integer(strata)
      g_strata <- matrix(0L, nrow = n_g, ncol = n_s)
      for (i in seq_len(n)) {
        g_strata[g_match[i], s_int[i]] <- g_strata[g_match[i], s_int[i]] + 1L
      }
      fold_strata <- matrix(0, nrow = cv_folds, ncol = n_s)
      strata_totals <- pmax(1.0, colSums(g_strata) / cv_folds)

      for (gi in g_order) {
        sz <- gsizes[gi]
        gs <- g_strata[gi, ]
        # Restrict candidate folds to those with minimum current load (or within 10% of min load
        # for stratified_group) so max(fold_totals) - min(fold_totals) <= max(gsizes) always holds
        min_load <- min(fold_load)
        load_tol <- if (strategy == "stratified_group") max(1L, as.integer(floor(0.05 * (n / cv_folds)))) else 0L
        cand_folds <- which(fold_load <= min_load + load_tol)
        if (length(cand_folds) == 1L) {
          best_f <- cand_folds
        } else {
          if (isTRUE(shuffle_groups)) {
            cand_folds <- sample(cand_folds)
          }
          best_f <- cand_folds[1L]
          best_cost <- Inf
          for (cf in cand_folds) {
            s_cost <- sum(((fold_strata[cf, ] + gs) / strata_totals)^2)
            l_cost <- ((fold_load[cf] + sz) / max(1.0, n / cv_folds))^2
            cost <- s_cost + l_cost
            if (cost < best_cost) {
              best_cost <- cost
              best_f <- cf
            }
          }
        }
        gfold[gi] <- best_f
        fold_load[best_f] <- fold_load[best_f] + sz
        fold_strata[best_f, ] <- fold_strata[best_f, ] + gs
      }
      return(gfold[g_match])
    }

    for (gi in g_order) {
      if (isTRUE(shuffle_groups)) {
        min_f <- which(fold_load == min(fold_load))
        f <- if (length(min_f) > 1L) sample(min_f, 1L) else min_f
      } else {
        f <- which.min(fold_load)
      }
      gfold[gi] <- f
      fold_load[f] <- fold_load[f] + gsizes[gi]
    }
    return(gfold[g_match])
  }
  if (!is.null(target_col) && target_col %in% names(data)) {
    return(stratified_kfold_ids(data[[target_col]], cv_folds, task = if (!is.null(task)) task else "regression"))
  }
  f <- cut(seq_len(n), breaks = cv_folds, labels = FALSE)
  sample(f)
}

#' Materialize pre-sliced train/val data.tables with .row_id tags from fold_ids
#' @keywords internal
#' @noRd
.build_shared_folds <- function(data, fold_ids) {
  u_folds <- sort(unique(fold_ids[!is.na(fold_ids)]))
  shared_folds <- vector("list", length(u_folds))
  for (idx in seq_along(u_folds)) {
    f <- u_folds[idx]
    tr_idx <- which(fold_ids != f)
    va_idx <- which(fold_ids == f)
    tr_dt <- data.table::as.data.table(data[tr_idx, ])
    va_dt <- data.table::as.data.table(data[va_idx, ])
    data.table::setattr(tr_dt, ".row_id", tr_idx)
    data.table::setattr(va_dt, ".row_id", va_idx)
    shared_folds[[idx]] <- list(train = tr_dt, val = va_dt)
  }
  shared_folds
}

#' Build stratified K-fold indices for classification, multiclass, and regression
#'
#' Stratifies fold assignments by class level for classification/multiclass tasks,
#' and by empirical target quantile bins for continuous regression tasks.
#'
#' @param y Target vector.
#' @param k Integer number of folds.
#' @param task Task type ("classification", "multiclass", or "regression").
#' @return Integer vector of fold assignments in \code{1:k} of length \code{length(y)}.
#' @keywords internal
#' @noRd
stratified_kfold_ids <- function(y, k, task = "regression") {
  n <- length(y)
  if (k <= 1L || n <= 1L) return(rep(1L, n))
  strata <- if (task %in% c("classification", "multiclass")) {
    as.factor(y)
  } else {
    .target_strata(y, n_bins = min(20L, max(2L, floor(n / max(5L, 2L * k)))))
  }
  fid <- integer(n)
  # Randomize initial fold offset per stratum so strata whose size is not divisible by k
  # distribute remainder items evenly across all k folds
  rem_offset <- 0L
  for (lv in levels(strata)) {
    ids <- which(strata == lv)
    m <- length(ids)
    if (m > 0L) {
      shuffled <- if (m == 1L) ids else sample(ids)
      fold_seq <- ((seq_len(m) - 1L + rem_offset) %% k) + 1L
      fid[shuffled] <- fold_seq
      rem_offset <- (rem_offset + m) %% k
    }
  }
  un <- which(fid == 0L)
  if (length(un) > 0L) fid[un] <- sample(rep(seq_len(k), length.out = length(un)))
  fid
}

#' Compute Adversarial Validation Propensity Weights & Feature Drift Diagnostics
#'
#' Fits a fast 3-fold L2-regularized linear probability model to distinguish
#' \code{train_data} (\code{is_test = 0}) from \code{test_data} (\code{is_test = 1})
#' using shared numeric and frequency-encoded categorical features.
#'
#' @param train_data Training \code{data.frame} or \code{data.table}.
#' @param test_data Test \code{data.frame} or \code{data.table}.
#' @param target_col Optional target column name to exclude from features.
#' @param drift_auc_threshold Single-feature AUC threshold above which a feature is flagged as drifting.
#' @return A list with \code{weights} (test-to-train odds ratio for each training row),
#'   \code{adversarial_auc} (out-of-fold ROC-AUC), \code{feature_auc} (named numeric vector of
#'   per-feature train-vs-test AUCs), and \code{drifting_features} (character vector).
#' @export
compute_adversarial_weights <- function(train_data, test_data, target_col = NULL,
                                        drift_auc_threshold = 0.80) {
  n_tr <- nrow(train_data)
  n_te <- if (!is.null(test_data)) nrow(test_data) else 0L
  if (n_tr < 10L || n_te < 5L) {
    return(list(
      weights = rep(1.0, n_tr),
      adversarial_auc = 0.5,
      feature_auc = numeric(0),
      drifting_features = character(0)
    ))
  }

  shared_cols <- intersect(names(train_data), names(test_data))
  if (!is.null(target_col)) shared_cols <- setdiff(shared_cols, target_col)
  if (length(shared_cols) == 0L) {
    return(list(
      weights = rep(1.0, n_tr),
      adversarial_auc = 0.5,
      feature_auc = numeric(0),
      drifting_features = character(0)
    ))
  }

  n_tot <- n_tr + n_te
  is_test <- c(rep(0.0, n_tr), rep(1.0, n_te))
  x_list <- vector("list", length(shared_cols))
  names(x_list) <- shared_cols
  feat_aucs <- numeric(length(shared_cols))
  names(feat_aucs) <- shared_cols

  for (col in shared_cols) {
    v_tr <- train_data[[col]]
    v_te <- test_data[[col]]
    if (is.numeric(v_tr) || is.numeric(v_te) || inherits(v_tr, c("Date", "POSIXt"))) {
      vec <- c(as.numeric(v_tr), as.numeric(v_te))
    } else {
      c_all <- c(as.character(v_tr), as.character(v_te))
      c_all[is.na(c_all)] <- ".NA"
      tab <- table(c_all)
      vec <- as.numeric(tab[c_all]) / n_tot
    }
    fin <- is.finite(vec)
    mu <- if (any(fin)) mean(vec[fin]) else 0.0
    vec[!fin] <- mu
    s <- stats::sd(vec)
    if (is.na(s) || s < 1e-12) {
      x_list[[col]] <- rep(0.0, n_tot)
      feat_aucs[[col]] <- 0.5
    } else {
      z <- pmin(pmax((vec - mu) / s, -10.0), 10.0)
      x_list[[col]] <- z
      # Mann-Whitney U statistic for single-feature drift AUC
      r_sum <- sum(rank(z, ties.method = "average")[(n_tr + 1L):n_tot])
      u_stat <- r_sum - (as.numeric(n_te) * (as.numeric(n_te) + 1.0)) / 2.0
      auc_raw <- u_stat / (as.numeric(n_tr) * as.numeric(n_te))
      feat_aucs[[col]] <- max(auc_raw, 1.0 - auc_raw)
    }
  }

  X <- do.call(cbind, x_list)
  keep_cols <- which(colSums(abs(X)) > 0)
  if (length(keep_cols) == 0L) {
    return(list(
      weights = rep(1.0, n_tr),
      adversarial_auc = 0.5,
      feature_auc = feat_aucs,
      drifting_features = character(0)
    ))
  }
  X <- cbind(1.0, X[, keep_cols, drop = FALSE])
  p_dim <- ncol(X)

  # 3-fold stratified OOF ridge propensity scoring
  folds <- stratified_kfold_ids(is_test, k = 3L, task = "classification")
  oof_pred <- numeric(n_tot)
  pen_diag <- diag(c(0.0, rep(10.0, p_dim - 1L)), nrow = p_dim, ncol = p_dim)

  for (f in 1:3) {
    tr_f <- which(folds != f)
    va_f <- which(folds == f)
    X_tr <- X[tr_f, , drop = FALSE]
    y_tr <- is_test[tr_f]
    xtx <- crossprod(X_tr) + pen_diag
    xty <- crossprod(X_tr, y_tr)
    beta <- tryCatch(solve(xtx, xty), error = function(e) rep(0.0, p_dim))
    oof_pred[va_f] <- as.numeric(X[va_f, , drop = FALSE] %*% beta)
  }

  # Compute OOF adversarial ROC-AUC
  r_all <- rank(oof_pred, ties.method = "average")
  u_oof <- sum(r_all[(n_tr + 1L):n_tot]) - (as.numeric(n_te) * (as.numeric(n_te) + 1.0)) / 2.0
  adv_auc <- u_oof / (as.numeric(n_tr) * as.numeric(n_te))
  if (!is.finite(adv_auc)) adv_auc <- 0.5

  # Convert raw linear predictions on train into bounded importance weights p(test|x) / p(train|x)
  prior_te <- n_te / n_tot
  p_tr <- pmin(pmax(oof_pred[seq_len(n_tr)], 0.05 * prior_te), 1.0 - 0.05 * (1.0 - prior_te))
  odds <- (p_tr / (1.0 - p_tr)) * (as.numeric(n_tr) / as.numeric(n_te))
  weights <- pmin(pmax(odds, 0.2), 5.0)
  weights <- weights / mean(weights)

  drifting <- names(feat_aucs)[feat_aucs >= drift_auc_threshold]

  list(
    weights = weights,
    adversarial_auc = adv_auc,
    feature_auc = sort(feat_aucs, decreasing = TRUE),
    drifting_features = drifting
  )
}

#' Smart Structure-Preserving Subsampling for Phase-1 Feature Discovery
#'
#' Downsamples large training tables specifically for Phase-1 unsupervised feature
#' discovery while enforcing four competition-critical invariants:
#' \enumerate{
#'   \item \strong{Temporal contiguity} (\code{cv_strategy = "time"}): retains the most
#'     recent trailing \code{max_rows} observations ordered by \code{time_col}.
#'   \item \strong{Group integrity} (\code{cv_strategy = "group"} or \code{"stratified_group"}):
#'     samples whole groups without splitting cohorts across kept vs. dropped rows.
#'   \item \strong{Rare-event preservation} (\code{classification} / \code{multiclass}):
#'     protects minority classes up to \code{min_minority_count} rows before downsampling
#'     majority strata so split gains on rare events do not vanish.
#'   \item \strong{Covariate-shift propensity weighting}: when \code{test_data} exhibits
#'     measurable covariate shift (\code{adversarial_auc >= 0.60}) or \code{adv_weights}
#'     are supplied, biases within-stratum sampling toward test-like rows.
#' }
#'
#' @param data Training \code{data.frame} or \code{data.table}.
#' @param target_col Name of the target column in \code{data}.
#' @param max_rows Maximum rows to retain (\code{NULL}, \code{<= 0}, or \code{Inf} disables subsampling).
#' @param task Modeling task: \code{"auto"}, \code{"classification"}, \code{"multiclass"}, or \code{"regression"}.
#' @param cv_strategy Validation split structure: \code{"random"}, \code{"stratified_group"},
#'   \code{"group"}, or \code{"time"}.
#' @param group_col Optional group column name for grouped data.
#' @param time_col Optional timestamp/order column name when \code{cv_strategy = "time"}.
#' @param test_data Optional unlabeled test table used to compute propensity weights when \code{adv_weights} is \code{NULL}.
#' @param adv_weights Optional numeric vector of positive row weights of length \code{nrow(data)}.
#' @param min_minority_count Minimum count floor for rare classes during classification subsampling (default \code{3000L}).
#' @param seed Optional integer RNG seed (restores caller's \code{.Random.seed} on exit).
#' @return A \code{data.table} subset with attributes \code{"subsampled"} (\code{TRUE}/\code{FALSE})
#'   and \code{"selected_indices"} (integer row indices into \code{data}).
#' @export
subsample_for_discovery <- function(data, target_col, max_rows = 200000L,
                                    task = c("auto", "classification", "multiclass", "regression"),
                                    cv_strategy = c("random", "stratified_group", "group", "time"),
                                    group_col = NULL, time_col = NULL,
                                    test_data = NULL, adv_weights = NULL,
                                    min_minority_count = 3000L,
                                    seed = NULL) {
  if (!is.data.frame(data)) stop("'data' must be a data.frame or data.table.")
  if (!is.character(target_col) || length(target_col) != 1L || !target_col %in% names(data)) {
    stop(sprintf("Target column '%s' not found in 'data'.", as.character(target_col)[1L]))
  }

  n <- nrow(data)
  if (is.null(max_rows) || length(max_rows) != 1L || is.na(max_rows) ||
      !is.finite(max_rows) || max_rows <= 0 || n <= as.integer(max_rows)) {
    out <- shallow_copy_dt(data.table::as.data.table(data))
    data.table::setattr(out, "subsampled", FALSE)
    data.table::setattr(out, "selected_indices", seq_len(n))
    return(out)
  }

  max_rows <- as.integer(max_rows)
  cv_strategy <- match.arg(cv_strategy, c("random", "stratified_group", "group", "time"))
  y <- data[[target_col]]
  task <- .infer_task_type(y, task)

  sel_idx <- with_seed(seed, function() {
    # Rule 1: Temporal contiguity -> keep trailing chronological window
    if (cv_strategy == "time") {
      if (is.null(time_col) || length(time_col) != 1L || !time_col %in% names(data)) {
        stop("'cv_strategy = \"time\"' requires 'time_col' to name an existing column in 'data'.")
      }
      ord <- order(data[[time_col]], method = "radix")
      return(utils::tail(ord, max_rows))
    }

    # Rule 2: Group integrity -> sample whole cohorts without splitting groups
    has_group <- (!is.null(group_col) && length(group_col) == 1L && group_col %in% names(data))
    if (cv_strategy %in% c("group", "stratified_group") || has_group) {
      if (!has_group) {
        stop(sprintf("'cv_strategy = \"%s\"' requires 'group_col' to name an existing column in 'data'.", cv_strategy))
      }
      sp <- stratified_split(y, ratio = c(max_rows, n - max_rows), group = data[[group_col]])
      return(sort(which(sp == "train")))
    }

    # Rule 3 & 4: Stratified / quantile sampling + rare-class floor + adversarial shift weights
    strata <- if (task %in% c("classification", "multiclass")) {
      fac <- as.character(y)
      fac[is.na(fac)] <- ".NA_STRATUM"
      as.factor(fac)
    } else {
      .target_strata(y, n_bins = 10L)
    }

    levels_y <- levels(strata)
    lvl_indices <- lapply(levels_y, function(lvl) which(strata == lvl))
    lvl_sizes <- vapply(lvl_indices, length, integer(1L))
    quotas <- .hamilton_quota(lvl_sizes, max_rows)

    if (task %in% c("classification", "multiclass") &&
        is.numeric(min_minority_count) && length(min_minority_count) == 1L &&
        !is.na(min_minority_count) && min_minority_count > 0) {
      min_floor <- pmin(lvl_sizes, as.integer(min_minority_count))
      is_rare <- quotas < min_floor
      if (any(is_rare) && any(!is_rare)) {
        rare_cap <- as.integer(floor(0.5 * max_rows))
        rare_targets <- min_floor[is_rare]
        if (sum(rare_targets) > rare_cap && rare_cap > 0L) {
          rare_targets <- .hamilton_quota(rare_targets, rare_cap)
        }
        rem_budget <- max_rows - sum(rare_targets)
        if (rem_budget > 0L && sum(lvl_sizes[!is_rare]) >= rem_budget) {
          quotas[is_rare] <- rare_targets
          quotas[!is_rare] <- .hamilton_quota(lvl_sizes[!is_rare], rem_budget)
        }
      }
    }

    eff_w <- adv_weights
    if (is.null(eff_w) && !is.null(test_data) && is.data.frame(test_data)) {
      adv <- compute_adversarial_weights(data, test_data, target_col = target_col)
      if (is.finite(adv$adversarial_auc) && adv$adversarial_auc >= 0.60) {
        eff_w <- adv$weights
      }
    }
    use_w <- !is.null(eff_w) && is.numeric(eff_w) && length(eff_w) == n &&
      all(is.finite(eff_w)) && any(eff_w > 0) && stats::sd(eff_w) > 1e-9

    chosen <- vector("list", length(levels_y))
    for (s_idx in seq_along(levels_y)) {
      q <- quotas[s_idx]
      if (q <= 0L) next
      idx_s <- lvl_indices[[s_idx]]
      if (q >= length(idx_s)) {
        chosen[[s_idx]] <- idx_s
      } else if (use_w) {
        chosen[[s_idx]] <- sample(idx_s, size = q, replace = FALSE, prob = pmax(1e-9, eff_w[idx_s]))
      } else {
        chosen[[s_idx]] <- sample(idx_s, size = q, replace = FALSE)
      }
    }
    sort(unlist(chosen, use.names = FALSE))
  })

  out <- data.table::as.data.table(data[sel_idx, , drop = FALSE])
  data.table::setattr(out, "subsampled", TRUE)
  data.table::setattr(out, "selected_indices", sel_idx)
  out
}


