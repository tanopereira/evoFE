#' Apply a single gene to a dataset
#'
#' @param gene A gene list representing a feature transformation.
#' @param train_data A data.frame or data.table representing the training data.
#' @param val_data Optional validation data.frame or data.table.
#' @param target_col Name of the target column.
#' @param state_cache Optional environment to cache full-dataset fitted states of stateful transformers.
#' @param data_hash Optional pre-computed xxhash64 digest of the target column, to avoid redundant hashing when applying multiple genes.
#' @param full_data Optional full dataset (all rows of X) for global fitting of unsupervised stateful transformers.
#' @param global_unsupervised Logical. If TRUE, unsupervised stateful transformers fit globally on full_data (default TRUE).
#' @return A list with three elements: \code{train} (the modified training
#'   \code{data.table} with the new gene column appended), \code{val} (the
#'   modified validation \code{data.table} or \code{NULL}), \code{full} (the
#'   modified full \code{data.table} or \code{NULL}), and \code{gene}
#'   (the gene list, with its \code{state} element populated if the transformer
#'   is stateful).
#' @export
apply_gene <- function(gene, train_data, val_data = NULL, target_col = NULL, state_cache = NULL, data_hash = NULL,
                       full_data = NULL, global_unsupervised = getOption("evoFE.global_unsupervised", TRUE)) {
  t_def <- evo_transformers[[gene$transformer_name]]

  col_exists_train <- gene$output_col %in% names(train_data)
  col_exists_val <- if (!is.null(val_data)) gene$output_col %in% names(val_data) else TRUE
  col_exists_full <- if (!is.null(full_data)) gene$output_col %in% names(full_data) else TRUE

  if (col_exists_train && col_exists_val && col_exists_full && (!is.null(gene$state) || is.null(t_def$fit_func))) {
    return(list(train = train_data, val = val_data, full = full_data, gene = gene))
  }

  is_supervised <- is_supervised_transformer(gene, t_def)
  use_global_fit <- !is_supervised && !is.null(full_data) && isTRUE(global_unsupervised)

  eff_global_cache <- if (!is.null(full_data)) attr(full_data, ".global_state_cache") else NULL
  if (is.null(eff_global_cache)) eff_global_cache <- state_cache

  state <- NULL
  has_cached_state <- FALSE
  cache_key <- NULL

  if (use_global_fit) {
    if (!is.null(gene$state)) {
      state <- gene$state
      has_cached_state <- TRUE
    } else {
      cache_key <- digest::digest(paste0(gene_to_state_formula(gene), "_global"), algo = "md5", serialize = FALSE)
      target_cache <- if (!is.null(eff_global_cache)) eff_global_cache else state_cache
      if (!is.null(target_cache) && exists(cache_key, envir = target_cache, inherits = TRUE)) {
        state <- get(cache_key, envir = target_cache, inherits = TRUE)
        gene$state <- state
        has_cached_state <- TRUE
      }
    }
  } else if (is.null(target_col)) {
    # Predicting without target_col: reuse fitted state from training time
    if (!is.null(gene$state)) {
      state <- gene$state
      has_cached_state <- TRUE
    }
  } else if (!is.null(state_cache)) {
    if (is.null(data_hash)) {
      data_hash <- digest::digest(train_data[[target_col]], algo = "xxhash64")
    }
    cache_key <- digest::digest(paste0(gene_to_state_formula(gene), "_", data_hash), algo = "md5", serialize = FALSE)
    if (exists(cache_key, envir = state_cache, inherits = TRUE)) {
      state <- get(cache_key, envir = state_cache, inherits = TRUE)
      gene$state <- state
      has_cached_state <- TRUE
    }
  }

  # If we are fitting and it's stateful
  if (!has_cached_state && !is.null(t_def$fit_func)) {
    if (use_global_fit) {
      if (!col_exists_full && all(gene$input_cols %in% names(full_data))) {
        state <- t_def$fit_func(full_data, gene, target_col = NULL)
        gene$state <- state
        target_cache <- if (!is.null(eff_global_cache)) eff_global_cache else state_cache
        if (!is.null(cache_key) && !is.null(target_cache)) {
          assign(cache_key, state, envir = target_cache)
        }
      } else if (!is.null(target_col)) {
        state <- t_def$fit_func(train_data, gene, target_col)
        gene$state <- state
      }
    } else if (!is.null(target_col)) {
      # Skip fitting in CV folds if columns already exist
      if (is.null(state_cache) && col_exists_train && col_exists_val) {
        # Skip fitting, state remains NULL
      } else {
        state <- t_def$fit_func(train_data, gene, target_col)
        gene$state <- state
        if (!is.null(cache_key) && !is.null(state_cache)) {
          assign(cache_key, state, envir = state_cache)
        }
      }
    }
  }

  out_type <- if (!is.null(t_def$output_type)) t_def$output_type else "numeric"
  can_use_global <- use_global_fit && all(gene$input_cols %in% names(full_data))
  new_col_full <- NULL

  # Apply to train
  if (!col_exists_train) {
    if (can_use_global) {
      if (col_exists_full) {
        new_col_full <- full_data[[gene$output_col]]
      } else {
        new_col_full <- t_def$apply_func(full_data, gene, state)
        if (out_type == "categorical") {
          new_col_full <- as.factor(new_col_full)
        } else if (is.double(new_col_full)) {
          new_col_full[!is.finite(new_col_full) | abs(new_col_full) > 3.402823e38] <- NA_real_
        }
      }
      tr_idx <- attr(train_data, ".row_id")
      if (!is.null(tr_idx) && length(new_col_full) >= max(tr_idx)) {
        new_col_train <- new_col_full[tr_idx]
      } else {
        new_col_train <- t_def$apply_func(train_data, gene, state)
      }
    } else {
      new_col_train <- t_def$apply_func(train_data, gene, state)
    }

    if (is.double(new_col_train)) {
      new_col_train[!is.finite(new_col_train) | abs(new_col_train) > 3.402823e38] <- NA_real_
    }

    # Reject constant columns (0 variance)
    if (!is.null(target_col) && length(unique(new_col_train[!is.na(new_col_train)])) <= 1) {
      stop("Constant column generated")
    }

    # Reject columns that are near-perfect duplicates of existing features.
    # Guard with !is.null(target_col): during inference (holdout/predict) we
    # must apply every gene that was accepted at training time — correlation on
    # a different data split must never prune a gene the model depends on.
    cor_threshold <- getOption("evoFE.redundancy_cor_threshold", 0.95)
    if (!is.null(target_col) && is.numeric(new_col_train) && cor_threshold < 1) {
      num_mask <- vapply(train_data, is.numeric, logical(1))
      existing_num_cols <- setdiff(names(train_data)[num_mask], c(gene$output_col, target_col))
      if (gene$transformer_name %in% c("robust_scale", "smooth_clip")) {
        existing_num_cols <- setdiff(existing_num_cols, gene$input_cols)
      }
      if (length(existing_num_cols) > 0) {
        new_is_finite <- is.finite(new_col_train)
        if (sum(new_is_finite) > 2 && suppressWarnings(stats::sd(new_col_train[new_is_finite])) > 0) {
          for (ecol in existing_num_cols) {
            ev <- train_data[[ecol]]
            # Find indices where both vectors are finite to preserve row alignment
            valid_idx <- new_is_finite & is.finite(ev)
            if (sum(valid_idx) > 2) {
              r <- tryCatch(
                suppressWarnings(abs(stats::cor(new_col_train[valid_idx], ev[valid_idx],
                  use = "complete.obs"
                ))),
                error = function(e) 0
              )
              if (!is.na(r) && r >= cor_threshold) stop("Redundant column")
            }
          }
        }
      }
    }

    if (out_type == "categorical") {
      new_col_train <- as.factor(new_col_train)
    } else if (is.double(new_col_train)) {
      new_col_train[!is.finite(new_col_train) | abs(new_col_train) > 3.402823e38] <- NA_real_
    }

    if (data.table::is.data.table(train_data)) {
      train_data[, (gene$output_col) := new_col_train]
    } else {
      train_data[[gene$output_col]] <- new_col_train
    }

    # Commit globally computed column to full_data after train passes validation
    if (can_use_global && !col_exists_full && !is.null(new_col_full)) {
      if (data.table::is.data.table(full_data)) {
        full_data[, (gene$output_col) := new_col_full]
      } else {
        full_data[[gene$output_col]] <- new_col_full
      }
      col_exists_full <- TRUE
    }
  }

  # Apply to val
  if (!is.null(val_data) && !col_exists_val) {
    if (can_use_global) {
      if (is.null(new_col_full)) {
        new_col_full <- if (col_exists_full) full_data[[gene$output_col]] else t_def$apply_func(full_data, gene, state)
        if (out_type == "categorical") {
          new_col_full <- as.factor(new_col_full)
        } else if (is.double(new_col_full)) {
          new_col_full[!is.finite(new_col_full) | abs(new_col_full) > 3.402823e38] <- NA_real_
        }
      }
      va_idx <- attr(val_data, ".row_id")
      if (!is.null(va_idx) && length(new_col_full) >= max(va_idx)) {
        new_col_val <- new_col_full[va_idx]
      } else {
        new_col_val <- t_def$apply_func(val_data, gene, state)
      }
    } else {
      new_col_val <- t_def$apply_func(val_data, gene, state)
    }

    if (out_type == "categorical") {
      # Fallback level alignment in case train_data has already been converted to factor
      train_factor <- train_data[[gene$output_col]]
      train_levels <- if (is.factor(train_factor)) levels(train_factor) else unique(as.character(train_factor))
      new_col_val <- factor(new_col_val, levels = train_levels)
    } else if (is.double(new_col_val)) {
      new_col_val[!is.finite(new_col_val) | abs(new_col_val) > 3.402823e38] <- NA_real_
    }
    if (data.table::is.data.table(val_data)) {
      val_data[, (gene$output_col) := new_col_val]
    } else {
      val_data[[gene$output_col]] <- new_col_val
    }
  }

  # Apply to full_data (if provided and column doesn't already exist, e.g. for supervised transformers)
  if (!is.null(full_data) && !col_exists_full) {
    if (is.null(new_col_full)) {
      new_col_full <- t_def$apply_func(full_data, gene, state)
      if (out_type == "categorical") {
        train_factor <- train_data[[gene$output_col]]
        train_levels <- if (is.factor(train_factor)) levels(train_factor) else unique(as.character(train_factor))
        new_col_full <- factor(new_col_full, levels = train_levels)
      } else if (is.double(new_col_full)) {
        new_col_full[!is.finite(new_col_full) | abs(new_col_full) > 3.402823e38] <- NA_real_
      }
    }
    if (data.table::is.data.table(full_data)) {
      full_data[, (gene$output_col) := new_col_full]
    } else {
      full_data[[gene$output_col]] <- new_col_full
    }
  }

  list(train = train_data, val = val_data, full = full_data, gene = gene)
}

#' Apply an entire individual's recipe to data
#'
#' @param ind An evo_individual object.
#' @param train_data A data.frame or data.table representing the training data.
#' @param val_data Optional validation data.frame or data.table.
#' @param target_col Name of the target column.
#' @param state_cache Optional environment to cache full-dataset fitted states of stateful transformers.
#' @param allow_prune Logical. If TRUE, genes that fail application are skipped instead of failing the entire individual.
#' @param full_data Optional full dataset (all rows of X) for global fitting of unsupervised stateful transformers.
#' @param global_unsupervised Logical. If TRUE, unsupervised stateful transformers fit globally on full_data (default TRUE).
#' @return A list with elements: \code{train} (the transformed training
#'   \code{data.table} with all gene columns applied), \code{val} (the
#'   transformed validation \code{data.table} or \code{NULL}), \code{full} (the
#'   transformed full \code{data.table} or \code{NULL}), and \code{ind}
#'   (the updated \code{evo_individual} whose genes now carry fitted states).
#' @export
apply_individual <- function(ind, train_data, val_data = NULL, target_col = NULL, state_cache = NULL, allow_prune = TRUE,
                             full_data = NULL, global_unsupervised = getOption("evoFE.global_unsupervised", TRUE)) {
  dt_train <- if (data.table::is.data.table(train_data)) train_data else data.table::as.data.table(train_data)
  dt_val <- if (!is.null(val_data)) {
    if (data.table::is.data.table(val_data)) val_data else data.table::as.data.table(val_data)
  } else {
    NULL
  }
  dt_full <- if (!is.null(full_data) && isTRUE(global_unsupervised)) {
    if (data.table::is.data.table(full_data)) full_data else data.table::as.data.table(full_data)
  } else {
    NULL
  }

  # Pre-compute target column hash once for all genes (avoids redundant hashing)
  pre_hash <- if (!is.null(state_cache) && !is.null(target_col)) {
    digest::digest(dt_train[[target_col]], algo = "xxhash64")
  } else {
    NULL
  }

  new_genes <- list()
  for (gene in ind$genes) {
    res <- tryCatch(
      {
        if (!all(gene$input_cols %in% names(dt_train))) {
          stop("Input column missing")
        }
        apply_gene(gene, dt_train, dt_val, target_col, state_cache = state_cache, data_hash = pre_hash,
                   full_data = dt_full, global_unsupervised = global_unsupervised)
      },
      error = function(e) {
        NULL
      }
    )

    if (is.null(res) || !is.null(res$skip)) {
      if (allow_prune) {
        next
      } else {
        return(NULL)
      }
    }

    dt_train <- res$train
    dt_val <- res$val
    if (!is.null(res$full)) dt_full <- res$full
    new_genes[[length(new_genes) + 1L]] <- res$gene
  }

  ind$genes <- new_genes

  # Safety check: if genes were pruned and total active columns fell below min_active,
  # restore raw columns to meet the safety floor
  all_num <- ind$all_numeric_cols
  all_cat <- ind$all_categorical_cols
  all_date <- ind$all_datetime_cols
  total_avail <- length(all_num) + length(all_cat) + length(all_date)
  if (total_avail > 0) {
    total_active <- length(ind$numeric_cols) + length(ind$categorical_cols) + length(ind$datetime_cols) + length(ind$genes)
    min_active <- if (total_avail >= 2) 2 else 1
    if (total_active < min_active) {
      all_cols <- c(all_num, all_cat, all_date)
      active_cols <- c(ind$numeric_cols, ind$categorical_cols, ind$datetime_cols)
      inactive_cols <- setdiff(all_cols, active_cols)
      needed <- min_active - total_active
      if (length(inactive_cols) > 0) {
        to_activate <- sample(inactive_cols, min(length(inactive_cols), needed))
        for (col in to_activate) {
          if (col %in% all_num) {
            ind$numeric_cols <- unique(c(ind$numeric_cols, col))
          } else if (col %in% all_cat) {
            ind$categorical_cols <- unique(c(ind$categorical_cols, col))
          } else if (col %in% all_date) ind$datetime_cols <- unique(c(ind$datetime_cols, col))
        }
      }
    }
  }

  list(train = dt_train, val = dt_val, full = dt_full, ind = ind)
}
