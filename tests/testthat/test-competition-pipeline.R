test_that("subsample_for_discovery enforces no-op, temporal, group, rare-class, and drift rules", {
  set.seed(101)
  n <- 300L
  dt <- data.table::data.table(
    ts = seq_len(n),
    grp = rep(paste0("g", seq_len(30L)), each = 10L),
    x1 = stats::rnorm(n),
    x2 = stats::runif(n),
    y = as.integer(seq_len(n) %% 20L == 0L) # 15 positives (5% rare class)
  )

  # 1. No-op when nrow <= max_rows
  same_dt <- subsample_for_discovery(dt, target_col = "y", max_rows = 500L, seed = 1L)
  expect_equal(nrow(same_dt), n)
  expect_false(isTRUE(attr(same_dt, "subsampled")))

  # 2. Temporal contiguity: keeps most recent trailing window
  time_dt <- subsample_for_discovery(
    dt, target_col = "y", max_rows = 80L,
    cv_strategy = "time", time_col = "ts", seed = 1L
  )
  expect_equal(nrow(time_dt), 80L)
  expect_true(isTRUE(attr(time_dt, "subsampled")))
  expect_equal(time_dt$ts, seq.int(n - 80L + 1L, n))

  # 3. Group integrity: zero group overlap between kept and dropped rows
  grp_dt <- subsample_for_discovery(
    dt, target_col = "y", max_rows = 100L,
    cv_strategy = "stratified_group", group_col = "grp", seed = 2L
  )
  kept_idx <- attr(grp_dt, "selected_indices")
  dropped_grps <- unique(dt$grp[-kept_idx])
  expect_length(intersect(unique(grp_dt$grp), dropped_grps), 0L)

  # 4. Rare-event preservation + covariate-shift weighting
  test_shifted <- data.table::data.table(
    x1 = stats::rnorm(60L, mean = 2.5, sd = 0.5),
    x2 = stats::runif(60L)
  )
  rare_dt <- subsample_for_discovery(
    dt, target_col = "y", max_rows = 90L,
    task = "classification", test_data = test_shifted,
    min_minority_count = 12L, seed = 3L
  )
  expect_equal(nrow(rare_dt), 90L)
  expect_gte(sum(rare_dt$y == 1L), 12L)
  expect_gt(mean(rare_dt$x1), mean(dt$x1))
})

test_that("run_competition_pipeline executes full 2-phase strategy on classification and regression", {
  skip_if_not_installed("lightgbm")

  set.seed(2026)
  n_tr <- 140L
  n_te <- 30L
  make_df <- function(n, start_id) {
    x1 <- stats::rnorm(n)
    x2 <- stats::rnorm(n)
    x3 <- stats::runif(n, 0, 10)
    cat1 <- sample(c("A", "B", "C"), n, replace = TRUE)
    lin <- 1.2 * x1 - 0.9 * x2 + 0.3 * sin(x3) + ifelse(cat1 == "A", 0.6, -0.2)
    data.table::data.table(
      id = seq.int(start_id, length.out = n),
      x1 = x1,
      x2 = x2,
      x3 = x3,
      cat1 = cat1,
      lin = lin
    )
  }

  tr_raw <- make_df(n_tr, 1L)
  te_raw <- make_df(n_te, 1001L)

  # Binary classification with Phase-1 subsampling (90 discovery rows -> 140 Stage-4 rows)
  train_cls <- data.table::copy(tr_raw)
  train_cls[, target := as.integer(stats::rbinom(.N, 1L, stats::plogis(lin)))]
  train_cls[, lin := NULL]
  test_cls <- data.table::copy(te_raw)[, lin := NULL]

  res_cls <- run_competition_pipeline(
    train = train_cls,
    test = test_cls,
    target_col = "target",
    id_col = "id",
    task = "auto",
    preset = "fast",
    evaluators = c("lightgbm", "realmlp"),
    cv_folds = 3L,
    max_search_rows = 90L,
    bag_folds = TRUE,
    threads = 1L,
    seed = 42L,
    verbose = FALSE
  )

  expect_s3_class(res_cls, "evo_competition_run")
  expect_equal(res_cls$task, "classification")
  expect_equal(res_cls$discovery_rows, 90L)
  expect_equal(nrow(res_cls$train_mega), n_tr)
  expect_equal(nrow(res_cls$test_mega), n_te)
  expect_false("id" %in% names(res_cls$train_mega))
  expect_length(res_cls$predictions, n_te)
  expect_true(all(is.finite(res_cls$predictions)))
  expect_true(all(res_cls$predictions >= 0 & res_cls$predictions <= 1))
  expect_equal(names(res_cls$submission), c("id", "target"))
  expect_equal(res_cls$submission$id, test_cls$id)
  expect_false(anyNA(res_cls$weights))
  expect_equal(sum(res_cls$weights), 1.0, tolerance = 1e-6)

  out_print <- capture.output(print(res_cls))
  expect_true(any(grepl("Competition Pipeline", out_print)))
  out_sum <- capture.output(print(summary(res_cls)))
  expect_true(any(grepl("Stage 4", out_sum)))

  # Skewed positive regression -> auto log1p target transform + CSV export
  train_reg <- data.table::copy(tr_raw)
  train_reg[, price := exp(2.0 + 0.8 * lin + stats::rnorm(.N, sd = 0.25))]
  train_reg[, lin := NULL]
  test_reg <- data.table::copy(te_raw)[, lin := NULL]
  sub_path <- tempfile(fileext = ".csv")
  on.exit(unlink(sub_path), add = TRUE)

  res_reg <- run_competition_pipeline(
    train = train_reg,
    test = test_reg,
    target_col = "price",
    id_col = "id",
    task = "auto",
    preset = "fast",
    evaluators = "lightgbm",
    cv_folds = 3L,
    target_transform = "auto",
    submission_file = sub_path,
    threads = 1L,
    seed = 77L,
    verbose = FALSE
  )

  expect_equal(res_reg$task, "regression")
  expect_equal(res_reg$target_transform, "log1p")
  expect_true(file.exists(sub_path))
  sub_disk <- data.table::fread(sub_path)
  expect_equal(nrow(sub_disk), n_te)
  expect_equal(names(sub_disk), c("id", "price"))
  expect_true(all(is.finite(sub_disk$price) & sub_disk$price > 0))
})
