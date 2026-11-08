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

test_that("run_competition_pipeline handles character/factor binary targets with xgboost and lazy ensemble refits", {
  skip_if_not_installed("lightgbm")
  skip_if_not_installed("xgboost")

  set.seed(808)
  n_tr <- 120L
  n_te <- 25L
  mk <- function(n) data.table::data.table(
    x1 = stats::rnorm(n), x2 = stats::rnorm(n),
    grp = sample(c("a", "b", "c"), n, replace = TRUE)
  )
  tr <- mk(n_tr)
  tr[, Will_Buy := ifelse(stats::rbinom(.N, 1L, stats::plogis(1.3 * x1 - x2)) == 1L, "Yes", "No")]
  te <- mk(n_te)

  res <- run_competition_pipeline(
    train = tr, test = te, target_col = "Will_Buy",
    preset = "fast", evaluators = c("lightgbm", "xgboost"),
    cv_folds = 3L, threads = 1L, seed = 5L, verbose = FALSE
  )
  expect_equal(res$task, "classification")
  expect_length(res$predictions, n_te)
  expect_true(all(is.finite(res$predictions) & res$predictions >= 0 & res$predictions <= 1))
})

test_that("seed_genes and decoupled supervised_stage4 search fast and refit 1-gen per fold leak-free", {
  skip_if_not_installed("lightgbm")

  set.seed(303)
  n_tr <- 140L
  n_te <- 30L
  cats <- paste0("lvl_", seq_len(8L))
  cat_eff <- stats::setNames(seq(-2.0, 2.0, length.out = 8L), cats)

  tr_cat <- sample(cats, n_tr, replace = TRUE)
  te_cat <- sample(cats, n_te, replace = TRUE)
  tr_x1 <- stats::rnorm(n_tr)
  te_x1 <- stats::rnorm(n_te)
  tr_x2 <- stats::rnorm(n_tr)
  te_x2 <- stats::rnorm(n_te)

  train_sup <- data.table::data.table(
    id = seq_len(n_tr),
    x1 = tr_x1,
    x2 = tr_x2,
    cohort = tr_cat,
    target = as.integer(stats::rbinom(n_tr, 1L, stats::plogis(cat_eff[tr_cat] + 0.7 * tr_x1 - 0.5 * tr_x2)))
  )
  test_sup <- data.table::data.table(
    id = 1000L + seq_len(n_te),
    x1 = te_x1,
    x2 = te_x2,
    cohort = te_cat
  )

  # Direct seed_genes test: strip state and refit per-fold in 1-gen mode
  g_sup <- create_gene("pooled_target_encode", "cohort")
  g_sup$state <- list(fake_leaked_state = 999)
  cleaned <- normalize_seed_genes(list(g_sup), c("x1", "x2", "cohort"))
  expect_length(cleaned, 1L)
  expect_null(cleaned[[1L]]$state)

  rec_seeded <- evolve_features(
    data = train_sup[, .(x1, x2, cohort, target)],
    target_col = "target",
    task = "classification",
    evaluator = "lightgbm",
    generations = 1L,
    pop_size = 1L,
    cv_folds = 3L,
    bag_folds = TRUE,
    seed_genes = list(g_sup),
    nrounds = 40L,
    early_stopping_rounds = 10L,
    threads = 1L,
    seed = 11L,
    verbose = FALSE
  )
  expect_length(rec_seeded$best_individual$genes, 1L)
  expect_null(rec_seeded$best_individual$genes[[1L]]$state$fake_leaked_state)
  expect_length(rec_seeded$fold_models, 3L)
  p_seeded <- predict_model(rec_seeded, test_sup[, .(x1, x2, cohort)], threads = 1L)
  expect_length(p_seeded, n_te)
  expect_true(all(is.finite(p_seeded)))

  # End-to-end decoupled supervised_stage4 in run_competition_pipeline
  res_sup <- run_competition_pipeline(
    train = train_sup,
    test = test_sup,
    target_col = "target",
    id_col = "id",
    preset = "fast",
    evaluators = "lightgbm",
    cv_folds = 3L,
    max_search_rows = 80L,
    supervised_stage4 = TRUE,
    bag_folds = TRUE,
    threads = 1L,
    seed = 55L,
    verbose = FALSE
  )

  expect_s3_class(res_sup, "evo_competition_run")
  expect_true(!is.null(res_sup$stage3_supervised_recipes))
  expect_equal(res_sup$discovery_rows, 80L)
  # Stage 4 itself ran strictly as a 1-gen, pop_size = 1 deep refit on all 140 rows
  expect_length(res_sup$stage4_recipes$lightgbm$history, 1L)
  expect_length(res_sup$stage4_recipes$lightgbm$oof_preds, n_tr)
  expect_length(res_sup$predictions, n_te)
  expect_true(all(is.finite(res_sup$predictions)))
})

test_that("preset_overrides and evaluator_args customize pipeline and enforce RealMLP cyclic early stopping", {
  # 1. Preset config defaults: RealMLP patience equals total epochs
  cfg_fast <- .competition_preset_config("fast")
  expect_equal(cfg_fast$s2_mlp_es, cfg_fast$s2_mlp_epochs)
  expect_equal(cfg_fast$s4_mlp_es, cfg_fast$s4_mlp_epochs)

  cfg_std <- .competition_preset_config("standard")
  expect_equal(cfg_std$s2_mlp_es, cfg_std$s2_mlp_epochs)
  expect_equal(cfg_std$s4_mlp_es, cfg_std$s4_mlp_epochs)

  cfg_max <- .competition_preset_config("max")
  expect_equal(cfg_max$s2_mlp_es, cfg_max$s2_mlp_epochs)
  expect_equal(cfg_max$s4_mlp_es, cfg_max$s4_mlp_epochs)

  # Overriding epochs automatically updates patience when patience is not explicitly provided
  cfg_ovr <- .competition_preset_config("fast", overrides = list(s4_mlp_epochs = 120L))
  expect_equal(cfg_ovr$s4_mlp_epochs, 120L)
  expect_equal(cfg_ovr$s4_mlp_es, 120L)

  # Overriding patience explicitly is respected
  cfg_ovr2 <- .competition_preset_config("fast", overrides = list(s4_mlp_epochs = 120L, s4_mlp_es = 40L))
  expect_equal(cfg_ovr2$s4_mlp_epochs, 120L)
  expect_equal(cfg_ovr2$s4_mlp_es, 40L)

  # Unknown keys raise informative error
  expect_error(.competition_preset_config("fast", overrides = list(unknown_param = 1)), "Unknown preset_overrides key")
  expect_error(.competition_preset_config("fast", overrides = "not_a_list"), "must be a named list")

  # 2. .evaluator_stage_args resolves RealMLP cyclic patience and merges custom evaluator_args
  # Stage 2 defaults
  s2_args <- .evaluator_stage_args("realmlp", stage = 2L, cfg = cfg_fast, n_rows = 100L)
  expect_equal(s2_args$realmlp_epochs, cfg_fast$s2_mlp_epochs)
  expect_equal(s2_args$early_stopping_rounds, cfg_fast$s2_mlp_epochs)

  # Stage 4 defaults
  s4_args <- .evaluator_stage_args("realmlp", stage = 4L, cfg = cfg_fast, n_rows = 100L)
  expect_equal(s4_args$epochs, cfg_fast$s4_mlp_epochs)
  expect_equal(s4_args$early_stopping_rounds, cfg_fast$s4_mlp_epochs)

  # Custom evaluator_args: top-level evaluator key
  custom_ev <- list(
    realmlp = list(n_cycles = 6L, stage4 = list(early_stopping_rounds = 45L))
  )
  s2_custom <- .evaluator_stage_args("realmlp", stage = 2L, cfg = cfg_fast, n_rows = 100L, custom_evaluator_args = custom_ev)
  expect_equal(s2_custom$n_cycles, 6L)
  expect_equal(s2_custom$early_stopping_rounds, cfg_fast$s2_mlp_epochs) # untouched in stage 2

  s4_custom <- .evaluator_stage_args("realmlp", stage = 4L, cfg = cfg_fast, n_rows = 100L, custom_evaluator_args = custom_ev)
  expect_equal(s4_custom$n_cycles, 6L)
  expect_equal(s4_custom$early_stopping_rounds, 45L) # overridden in stage 4

  # Custom evaluator_args: stage-level key
  custom_stage <- list(
    stage4 = list(lightgbm = list(num_leaves = 127L))
  )
  s4_lgb <- .evaluator_stage_args("lightgbm", stage = 4L, cfg = cfg_fast, n_rows = 100L, custom_evaluator_args = custom_stage)
  expect_equal(s4_lgb$num_leaves, 127L)

  # 3. End-to-end integration with run_competition_pipeline
  set.seed(999)
  n_tr <- 50L
  n_te <- 10L
  tr <- data.table::data.table(
    x1 = stats::rnorm(n_tr),
    x2 = stats::rnorm(n_tr),
    y = stats::rbinom(n_tr, 1L, 0.5)
  )
  te <- data.table::data.table(
    x1 = stats::rnorm(n_te),
    x2 = stats::rnorm(n_te)
  )

  res_cust <- run_competition_pipeline(
    train = tr,
    test = te,
    target_col = "y",
    preset = "fast",
    preset_overrides = list(s2_mlp_epochs = 4L, s4_mlp_epochs = 5L),
    evaluators = "realmlp",
    evaluator_args = list(realmlp = list(hidden_dim = 32L)),
    cv_folds = 2L,
    threads = 1L,
    seed = 42L,
    verbose = FALSE
  )

  expect_s3_class(res_cust, "evo_competition_run")
  expect_equal(res_cust$preset_config$s2_mlp_epochs, 4L)
  expect_equal(res_cust$preset_config$s2_mlp_es, 4L)
  expect_equal(res_cust$preset_config$s4_mlp_epochs, 5L)
  expect_equal(res_cust$preset_config$s4_mlp_es, 5L)
  expect_equal(res_cust$evaluator_args$realmlp$hidden_dim, 32L)
  expect_length(res_cust$predictions, n_te)
  expect_true(all(is.finite(res_cust$predictions)))
})

test_that("run_competition_pipeline handles train/test categorical type mismatch and supervised stage 4 without schema drift", {
  skip_if_not_installed("lightgbm")

  set.seed(4242)
  n_tr <- 60L
  n_te <- 15L

  # cat_col is integer in train, but character in test
  tr <- data.table::data.table(
    id = seq_len(n_tr),
    num1 = stats::rnorm(n_tr),
    num2 = stats::runif(n_tr, 1, 10),
    cat_code = rep(1:3, length.out = n_tr), # integer
    y = stats::rbinom(n_tr, 1L, 0.5)
  )

  te <- data.table::data.table(
    id = seq.int(101L, length.out = n_te),
    num1 = stats::rnorm(n_te),
    num2 = stats::runif(n_te, 1, 10),
    cat_code = as.character(rep(1:4, length.out = n_te)) # character, including novel category "4"
  )

  res <- run_competition_pipeline(
    train = tr,
    test = te,
    target_col = "y",
    id_col = "id",
    preset = "fast",
    evaluators = "lightgbm",
    supervised_stage4 = TRUE,
    cv_folds = 2L,
    threads = 1L,
    seed = 42L,
    verbose = FALSE
  )

  expect_s3_class(res, "evo_competition_run")
  expect_length(res$predictions, n_te)
  expect_true(all(is.finite(res$predictions)))
  expect_true(all(setdiff(names(res$train_mega), "y") %in% names(res$test_mega)))
  expect_equal(setdiff(names(res$train_mega), "y"), names(res$test_mega))
})

test_that("run_competition_pipeline correctly resolves and applies rank_average for ts_refinement vs auc", {
  skip_if_not_installed("lightgbm")

  set.seed(88)
  n_tr <- 60L
  n_te <- 12L
  tr <- data.table::data.table(
    x1 = stats::rnorm(n_tr),
    x2 = stats::rnorm(n_tr),
    y = rep(c(0L, 1L), length.out = n_tr)
  )
  te <- data.table::data.table(
    x1 = stats::rnorm(n_te),
    x2 = stats::rnorm(n_te)
  )

  # 1. Default ts_refinement (rank_average = NULL) resolves to FALSE
  res_def <- run_competition_pipeline(
    train = tr, test = te, target_col = "y",
    preset = "fast", evaluators = c("lightgbm", "realmlp"),
    cv_folds = 2L, threads = 1L, seed = 1L, verbose = FALSE
  )
  expect_false(res_def$rank_average)
  expect_false(res_def$final_model$rank_average)

  # 2. ts_refinement with explicit rank_average = TRUE overrides to TRUE
  res_rank <- run_competition_pipeline(
    train = tr, test = te, target_col = "y",
    metric = "ts_refinement", rank_average = TRUE,
    preset = "fast", evaluators = c("lightgbm", "realmlp"),
    cv_folds = 2L, threads = 1L, seed = 1L, verbose = FALSE
  )
  expect_true(res_rank$rank_average)
  expect_true(res_rank$final_model$rank_average)
  expect_true(all(res_rank$predictions >= 0 & res_rank$predictions <= 1))
})

test_that("run_competition_pipeline harmonizes logical vs string representations across train and test", {
  skip_if_not_installed("lightgbm")

  set.seed(123)
  n_tr <- 50L
  n_te <- 10L

  # Train read with fread (boolean parsed as logical)
  tr <- data.table::data.table(
    x_num = stats::rnorm(n_tr),
    flag_active = sample(c(TRUE, FALSE), n_tr, replace = TRUE), # logical
    y = stats::rbinom(n_tr, 1L, 0.5)
  )

  # Test read with read.csv (boolean parsed as strings: "True", "False")
  te <- data.frame(
    x_num = stats::rnorm(n_te),
    flag_active = sample(c("True", "False"), n_te, replace = TRUE), # character strings
    stringsAsFactors = FALSE
  )

  res <- run_competition_pipeline(
    train = tr,
    test = te,
    target_col = "y",
    preset = "fast",
    evaluators = "lightgbm",
    cv_folds = 2L,
    threads = 1L,
    seed = 42L,
    verbose = FALSE
  )

  expect_s3_class(res, "evo_competition_run")
  expect_length(res$predictions, n_te)
  expect_true(all(is.finite(res$predictions)))
  # Confirm harmonized type is logical
  expect_type(res$train_mega$flag_active, "logical")
  expect_type(res$test_mega$flag_active, "logical")
})



