#' Sample active features mask
#'
#' @noRd
sample_active_mask <- function(cols, importances, temperature, fallback_rate = 0.7) {
  if (length(cols) == 0) return(character(0))
  
  has_imps <- length(importances) > 0 && any(importances > 0)
  
  if (has_imps) {
    threshold <- 1.0 / length(cols)
    vals <- importances[cols]
    vals[is.na(vals) | !is.finite(vals)] <- 0.0
    probs <- 1.0 / (1.0 + exp(-(vals - threshold) / temperature))
    keep <- stats::runif(length(cols)) < probs
  } else {
    keep <- stats::runif(length(cols)) < fallback_rate
  }
  cols[keep]
}

#' Sample gene inputs with importance weighting
#'
#' @noRd
sample_gene_inputs <- function(cols, n, importances = numeric(0), temperature = 1.0) {
  if (length(cols) == 0) return(character(0))
  if (length(cols) <= n) return(cols)
  
  has_imps <- length(importances) > 0 && any(importances > 0)
  if (has_imps) {
    vals <- importances[cols]
    vals[is.na(vals) | !is.finite(vals)] <- 0.0
    weights <- exp(vals / temperature)
    if (sum(weights) == 0 || any(is.na(weights))) {
      weights <- NULL
    }
    sample(cols, size = n, replace = FALSE, prob = weights)
  } else {
    sample(cols, size = n, replace = FALSE)
  }
}

#' Recalculate feature mask using feature importances
#'
#' @noRd
recalculate_mask <- function(ind, importances = numeric(0), temperature = 1.0, verbose = FALSE) {
  has_imps <- length(importances) > 0 && any(importances > 0)
  if (!has_imps) return(ind)
  
  all_num <- ind$all_numeric_cols
  all_cat <- ind$all_categorical_cols
  all_date <- ind$all_datetime_cols
  
  total_avail <- length(all_num) + length(all_cat) + length(all_date)
  if (total_avail == 0) return(ind)
  threshold <- 1.0 / total_avail
  
  sample_sigmoid <- function(cols) {
    if (length(cols) == 0) return(character(0))
    vals <- importances[cols]
    vals[is.na(vals) | !is.finite(vals)] <- 0.0
    probs <- 1.0 / (1.0 + exp(-(vals - threshold) / temperature))
    keep <- stats::runif(length(cols)) < probs
    cols[keep]
  }
  
  new_num <- sample_sigmoid(all_num)
  new_cat <- sample_sigmoid(all_cat)
  new_date <- sample_sigmoid(all_date)
  
  total_active <- length(new_num) + length(new_cat) + length(new_date) + length(ind$genes)
  min_active <- if (total_avail >= 2) 2 else 1
  
  if (total_active < min_active) {
    all_cols <- c(all_num, all_cat, all_date)
    active_cols <- c(new_num, new_cat, new_date)
    inactive_cols <- setdiff(all_cols, active_cols)
    needed <- min_active - total_active
    if (length(inactive_cols) > 0) {
      to_activate <- sample(inactive_cols, min(length(inactive_cols), needed))
      for (col in to_activate) {
        if (col %in% all_num) new_num <- unique(c(new_num, col))
        else if (col %in% all_cat) new_cat <- unique(c(new_cat, col))
        else if (col %in% all_date) new_date <- unique(c(new_date, col))
      }
    }
  }
  
  mask_changed <- !identical(sort(new_num), sort(ind$numeric_cols)) ||
                  !identical(sort(new_cat), sort(ind$categorical_cols)) ||
                  !identical(sort(new_date), sort(ind$datetime_cols))
  
  if (mask_changed) {
    ind$numeric_cols <- new_num
    ind$categorical_cols <- new_cat
    ind$datetime_cols <- new_date
    ind$fitness <- NA_real_
    if (verbose) {
      message("  Recalculated active mask based on feature importances.")
    }
  }
  
  ind
}

#' Toggle active raw features
#'
#' @noRd
toggle_raw_feature <- function(ind, importances = numeric(0), temperature = 1.0, verbose = FALSE) {
  all_num <- ind$all_numeric_cols
  all_cat <- ind$all_categorical_cols
  all_date <- ind$all_datetime_cols
  
  total_avail <- length(all_num) + length(all_cat) + length(all_date)
  if (total_avail == 0) return(ind)
  
  # Determine number of features to toggle using geometric distribution
  p <- 1.0 / (1.0 + log(total_avail))
  k <- 1 + stats::rgeom(1, p)
  k <- min(k, total_avail)
  
  toggled_active <- character(0)
  toggled_inactive <- character(0)
  
  for (step in 1:k) {
    active_num <- ind$numeric_cols
    active_cat <- ind$categorical_cols
    active_date <- ind$datetime_cols
    
    inactive_num <- setdiff(all_num, active_num)
    inactive_cat <- setdiff(all_cat, active_cat)
    inactive_date <- setdiff(all_date, active_date)
    
    # Exclude already toggled columns from candidate lists
    already_toggled <- c(toggled_active, toggled_inactive)
    active_cols <- setdiff(c(active_num, active_cat, active_date), already_toggled)
    inactive_cols <- setdiff(c(inactive_num, inactive_cat, inactive_date), already_toggled)
    
    # Safety checks based on current actual active list
    total_active <- length(active_num) + length(active_cat) + length(active_date) + length(ind$genes)
    min_active <- if (total_avail >= 2) 2 else 1
    
    can_deactivate <- (total_active > min_active) && (length(active_cols) > 0)
    can_activate <- length(inactive_cols) > 0
    
    if (!can_deactivate && !can_activate) {
      break
    }
    
    op <- if (can_deactivate && can_activate) {
      sample(c("deactivate", "activate"), 1)
    } else if (can_deactivate) {
      "deactivate"
    } else {
      "activate"
    }
    
    if (op == "deactivate") {
      weights <- sapply(active_cols, function(c) {
        val <- if (c %in% names(importances)) importances[[c]] else 0.0
        if (is.na(val) || !is.finite(val)) val <- 0.0
        exp(-val / temperature)
      })
      
      weights[is.na(weights) | !is.finite(weights)] <- 0
      if (sum(weights) == 0) weights <- rep(1, length(weights))
      
      col_to_deactivate <- sample(active_cols, 1, prob = weights)
      
      if (col_to_deactivate %in% active_num) {
        ind$numeric_cols <- setdiff(ind$numeric_cols, col_to_deactivate)
      } else if (col_to_deactivate %in% active_cat) {
        ind$categorical_cols <- setdiff(ind$categorical_cols, col_to_deactivate)
      } else if (col_to_deactivate %in% active_date) {
        ind$datetime_cols <- setdiff(ind$datetime_cols, col_to_deactivate)
      }
      
      toggled_inactive <- c(toggled_inactive, col_to_deactivate)
      
    } else {
      col_to_activate <- sample(inactive_cols, 1)
      
      if (col_to_activate %in% inactive_num) {
        ind$numeric_cols <- c(ind$numeric_cols, col_to_activate)
      } else if (col_to_activate %in% inactive_cat) {
        ind$categorical_cols <- c(ind$categorical_cols, col_to_activate)
      } else if (col_to_activate %in% inactive_date) {
        ind$datetime_cols <- c(ind$datetime_cols, col_to_activate)
      }
      
      toggled_active <- c(toggled_active, col_to_activate)
    }
  }
  
  if (length(toggled_active) > 0 || length(toggled_inactive) > 0) {
    ind$fitness <- NA_real_
    if (verbose) {
      msg <- "    [Mutation] Toggled raw features:"
      if (length(toggled_active) > 0) {
        msg <- paste0(msg, " Activated: ", paste(toggled_active, collapse = ", "))
      }
      if (length(toggled_inactive) > 0) {
        msg <- paste0(msg, " Deactivated: ", paste(toggled_inactive, collapse = ", "))
      }
      message(msg)
    }
  }
  
  ind
}
