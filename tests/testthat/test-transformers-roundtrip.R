# Contract tests: every registered transformer must round-trip
# fit -> apply(train) -> apply(test) without erroring, return one value per row,
# and tolerate unseen categories / NA inputs at apply time.

test_that("all registered transformers round-trip fit/apply", {
  skip_if_not_installed("data.table")
  set.seed(42)
  n <- 120

  make_data <- function(n_rows, unseen = FALSE) {
    dt <- data.table::data.table(
      num_a = rnorm(n_rows),
      num_b = rnorm(n_rows),
      num_c = runif(n_rows),
      cat_a = sample(c("red", "green", "blue"), n_rows, replace = TRUE),
      cat_b = sample(c("low", "mid", "high"), n_rows, replace = TRUE),
      dt_col = as.POSIXct("2024-01-01") + seq_len(n_rows) * 3600,
      y = rnorm(n_rows)
    )
    if (unseen) {
      dt$cat_a[1] <- "zzz_unseen"
      dt$cat_b[2] <- "zzz_unseen"
    }
    # Exercise NA handling paths
    dt$num_a[sample.int(n_rows, 5)] <- NA
    dt
  }

  train <- make_data(n)
  test <- make_data(30, unseen = TRUE)

  # Binary target for supervised encoders
  train[, y := factor(y > median(y), levels = c("FALSE", "TRUE"))]

  input_cols_for <- function(name, t_def) {
    it <- t_def$input_type
    ty <- t_def$type
    if (it == "datetime") {
      "dt_col"
    } else if (it == "categorical") {
      if (ty %in% c("multivariate", "supervised_binary")) {
        if (name == "concat") c("cat_a", "cat_b") else c("cat_a", "cat_b", "cat_a")
      } else {
        "cat_a"
      }
    } else if (it == "mixed") {
      if (ty == "multivariate") {
        c("cat_a", "num_a", "num_b")
      } else {
        c("cat_a", "num_a")
      }
    } else { # numeric
      switch(ty,
        unary = "num_a",
        binary = c("num_a", "num_b"),
        multivariate = c("num_a", "num_b", "num_c"),
        "num_a"
      )
    }
  }

  registered <- names(evo_transformers)
  expect_gt(length(registered), 40)

  failures <- character(0)
  for (nm in registered) {
    t_def <- evo_transformers[[nm]]
    cols <- input_cols_for(nm, t_def)

    res <- tryCatch({
      gene <- create_gene(nm, cols)
      state <- NULL
      if (!is.null(t_def$fit_func)) {
        state <- t_def$fit_func(train, gene, "y")
      }
      out_train <- t_def$apply_func(train, gene, state)
      out_test <- t_def$apply_func(test, gene, state)

      stopifnot(
        length(out_train) == nrow(train),
        length(out_test) == nrow(test),
        !anyNA(out_train) || TRUE # NAs allowed; errors are not
      )
      NULL
    }, error = function(e) conditionMessage(e))

    if (!is.null(res)) failures[[nm]] <- res
  }

  expect_length(failures, 0)
})

test_that("groupby_max returns group maxima consistent with raw data", {
  library(data.table)
  set.seed(7)
  df <- data.table(cat = rep(c("a", "b", "c"), each = 10), num = rnorm(30))

  gene <- create_gene("groupby_max", c("cat", "num"))
  state <- evo_transformers$groupby_max$fit_func(df, gene)
  out <- evo_transformers$groupby_max$apply_func(df, gene, state)

  grp_max <- df[, .(mx = max(num, na.rm = TRUE)), by = cat]
  merged <- merge(data.table(cat = df$cat, out = out), grp_max, by = "cat")
  expect_equal(merged$out, merged$mx)
})

test_that("stateful encoders use training statistics, not batch statistics", {
  library(data.table)
  set.seed(11)
  train <- data.table(
    cat = rep(c("a", "b", "c"), each = 20),
    y = rep(c(10, 20, 30), each = 20) + rnorm(60)
  )
  test <- data.table(cat = c("a", "b", "zzz_unseen"))

  gene <- create_gene("target_encode", "cat")
  state <- evo_transformers$target_encode$fit_func(train, gene, "y")
  out <- evo_transformers$target_encode$apply_func(test, gene, state)

  # Encoded values are shrunk toward the training global mean but must
  # preserve the group ordering (a < b < c).
  expect_lt(out[1], out[2])
  expect_lt(out[2], out[3])
  # Unseen category falls back to the *training* global mean
  global_mean <- mean(train$y)
  expect_equal(out[3], global_mean, tolerance = 0.05 * abs(global_mean))
})

test_that("create_gene gives a helpful error for unknown transformers", {
  expect_error(
    create_gene("no_such_transformer", "x1"),
    regexp = "no_such_transformer.*Registered transformers",
  )
})
