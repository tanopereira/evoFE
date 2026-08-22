# Resampling utilities: stratified splitting and CV fold construction.


#' Stratified or random splitting helper
#' @keywords internal
stratified_split <- function(y, ratio) {
  n <- length(y)
  ratios <- ratio / sum(ratio)

  # For regression or if y has only 1 level, do standard random split
  if ((is.numeric(y) && length(unique(y)) > 10) || length(unique(y)) <= 1) {
    shuffled_idx <- sample(seq_len(n))
    if (length(ratios) == 2) {
      n_train <- round(n * ratios[1])
      if (n_train < 1 && n >= 1) n_train <- 1
      train_idx <- shuffled_idx[seq_len(n_train)]
      res <- rep("val", n)
      res[train_idx] <- "train"
    } else {
      n_train <- round(n * ratios[1])
      if (n_train < 1 && n >= 1) n_train <- 1
      n_val <- round(n * ratios[2])
      if (n_val < 1 && (n - n_train) >= 1) n_val <- 1
      if (n_train + n_val > n) {
        n_val <- max(0, n - n_train)
      }
      train_idx <- shuffled_idx[seq_len(n_train)]
      if (n_val > 0) {
        val_idx <- shuffled_idx[(n_train + 1):(n_train + n_val)]
      } else {
        val_idx <- integer(0)
      }
      res <- rep("holdout", n)
      res[train_idx] <- "train"
      res[val_idx] <- "val"
    }
    return(res)
  }

  # For classification/multiclass, split class-by-class
  y_factor <- as.factor(y)
  levels_y <- levels(y_factor)

  res <- character(n)

  for (lvl in levels_y) {
    lvl_idx <- which(y_factor == lvl)
    n_lvl <- length(lvl_idx)
    shuffled_lvl_idx <- sample(lvl_idx)

    if (length(ratios) == 2) {
      n_train <- round(n_lvl * ratios[1])
      if (n_train < 1 && n_lvl >= 1) n_train <- 1
      if (n_train > n_lvl) n_train <- n_lvl

      train_idx <- shuffled_lvl_idx[seq_len(n_train)]
      val_idx <- setdiff(shuffled_lvl_idx, train_idx)

      res[train_idx] <- "train"
      res[val_idx] <- "val"
    } else {
      n_train <- round(n_lvl * ratios[1])
      if (n_train < 1 && n_lvl >= 1) n_train <- 1

      n_val <- round(n_lvl * ratios[2])
      if (n_val < 1 && (n_lvl - n_train) >= 1) n_val <- 1

      if (n_train + n_val > n_lvl) {
        n_val <- max(0, n_lvl - n_train)
      }

      train_idx <- shuffled_lvl_idx[seq_len(n_train)]
      if (n_val > 0) {
        val_idx <- shuffled_lvl_idx[(n_train + 1):(n_train + n_val)]
      } else {
        val_idx <- integer(0)
      }
      holdout_idx <- setdiff(shuffled_lvl_idx, c(train_idx, val_idx))

      res[train_idx] <- "train"
      res[val_idx] <- "val"
      res[holdout_idx] <- "holdout"
    }
  }

  # Fill any remaining unassigned elements (due to rounding) with "train"
  res[res == ""] <- "train"
  res
}

#' Build cross-validation fold assignments
#'
#' Internal helper supporting three strategies:
#' \itemize{
#'   \item \code{random}: rows shuffled into k folds.
#'   \item \code{time}: rows ordered by \code{time_col} and split into k
#'     contiguous chronological blocks (blocked CV; each block is held out once).
#'     Rows with missing time values sort last.
#'   \item \code{group}: all rows of a group (\code{group_col}) are assigned to a
#'     single fold via greedy largest-first balancing, so groups never straddle folds.
#' }
#' @noRd
.build_cv_folds <- function(data, cv_folds, strategy = "random",
                            time_col = NULL, group_col = NULL) {
  n <- nrow(data)
  if (strategy == "time") {
    ord <- order(data[[time_col]], method = "radix")
    f <- integer(n)
    f[ord] <- as.integer(cut(seq_len(n), breaks = cv_folds, labels = FALSE))
    return(f)
  }
  if (strategy == "group") {
    g <- data[[group_col]]
    ux <- unique(g)
    gsizes <- tabulate(match(g, ux))
    fold_load <- integer(cv_folds)
    gfold <- integer(length(ux))
    for (gi in order(gsizes, decreasing = TRUE)) {
      f <- which.min(fold_load)
      gfold[gi] <- f
      fold_load[f] <- fold_load[f] + gsizes[gi]
    }
    return(gfold[match(g, ux)])
  }
  f <- cut(seq_len(n), breaks = cv_folds, labels = FALSE)
  sample(f)
}
