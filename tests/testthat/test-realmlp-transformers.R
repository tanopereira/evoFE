test_that("fourier_basis computes multi-component harmonic basis correctly and handles params", {
  dt_train <- data.table::data.table(x = c(0, pi / 2, pi, 3 * pi / 2, 2 * pi))
  gene <- create_gene("fourier_basis", "x")
  expect_true(!is.null(gene$params$comp_idx))
  expect_true(!is.null(gene$params$scale))
  expect_true(!is.null(gene$params$phase))

  # Test comp_idx = 1 (sine 1st harmonic)
  gene$params$comp_idx <- 1L
  gene$params$scale <- 1.0
  gene$params$phase <- 0.0
  out1 <- evo_transformers$fourier_basis$apply_func(dt_train, gene)
  expect_equal(out1, sin(dt_train$x), tolerance = 1e-6)

  # Test comp_idx = 2 (cosine 1st harmonic)
  gene$params$comp_idx <- 2L
  out2 <- evo_transformers$fourier_basis$apply_func(dt_train, gene)
  expect_equal(out2, cos(dt_train$x), tolerance = 1e-6)

  # Test comp_idx = 3 (sine 2nd harmonic)
  gene$params$comp_idx <- 3L
  out3 <- evo_transformers$fourier_basis$apply_func(dt_train, gene)
  expect_equal(out3, sin(2 * dt_train$x), tolerance = 1e-6)

  # Test comp_idx = 4 (cosine 2nd harmonic)
  gene$params$comp_idx <- 4L
  out4 <- evo_transformers$fourier_basis$apply_func(dt_train, gene)
  expect_equal(out4, cos(2 * dt_train$x), tolerance = 1e-6)

  # Test comp_idx = 5 & 6 (dyadic harmonic order k = 4)
  gene$params$comp_idx <- 5L
  out5 <- evo_transformers$fourier_basis$apply_func(dt_train, gene)
  expect_equal(out5, sin(4 * dt_train$x), tolerance = 1e-6)
  gene$params$comp_idx <- 6L
  out6 <- evo_transformers$fourier_basis$apply_func(dt_train, gene)
  expect_equal(out6, cos(4 * dt_train$x), tolerance = 1e-6)

  # Test data-adaptive fit_func and stateful apply
  x_vals <- c(10, 20, 30, 40, 50, 60, 70, 80, 90)
  dt_adaptive <- data.table::data.table(x = x_vals, target = rnorm(length(x_vals)))
  state_adaptive <- evo_transformers$fourier_basis$fit_func(dt_adaptive, gene, target_col = "target")
  expect_equal(state_adaptive$center, 50)
  expect_equal(state_adaptive$scale, 40) # IQR = 70 - 30 = 40

  gene$params$comp_idx <- 1L
  gene$params$scale <- 1.0
  gene$params$phase <- 0.0
  out_state <- evo_transformers$fourier_basis$apply_func(dt_adaptive, gene, state = state_adaptive)
  expect_equal(out_state, sin(2 * pi * (x_vals - 50) / 40), tolerance = 1e-6)

  # Test state formula caching sharing across components
  g_c1 <- create_gene("fourier_basis", "x")
  g_c1$params$comp_idx <- 1L
  g_c2 <- create_gene("fourier_basis", "x")
  g_c2$params$comp_idx <- 2L
  expect_equal(gene_to_state_formula(g_c1), gene_to_state_formula(g_c2))

  # Test NA / Inf safety
  dt_na <- data.table::data.table(x = c(NA, Inf, -Inf, 1))
  out_na <- evo_transformers$fourier_basis$apply_func(dt_na, gene)
  expect_equal(out_na[1:3], c(0, 0, 0))
  expect_true(is.finite(out_na[4]))

  # Test formula string with comp_idx
  formula_str <- gene_to_formula(gene)
  expect_match(formula_str, "fourier_basis1_s.*_p.*\\(x\\)")
})

test_that("robust_scale computes median/IQR scaling on train and applies state to test", {
  set.seed(123)
  x_train <- c(10, 20, 30, 40, 50, 60, 70, 80, 90)
  dt_train <- data.table::data.table(x = x_train, y = rnorm(length(x_train)))
  gene <- create_gene("robust_scale", "x")

  state <- evo_transformers$robust_scale$fit_func(dt_train, gene, target_col = "y")
  expect_equal(state$center, 50)
  expect_equal(state$scale, 40) # 70 - 30 = 40

  out_train <- evo_transformers$robust_scale$apply_func(dt_train, gene, state)
  expect_equal(out_train, (x_train - 50) / 40)

  # Apply state to test set (must use train center & scale)
  x_test <- c(50, 90, 130)
  dt_test <- data.table::data.table(x = x_test)
  out_test <- evo_transformers$robust_scale$apply_func(dt_test, gene, state)
  expect_equal(out_test, (x_test - 50) / 40)

  # Test zero IQR fallback (constant or sparse values)
  dt_const <- data.table::data.table(x = rep(5, 10))
  state_const <- evo_transformers$robust_scale$fit_func(dt_const, gene)
  expect_equal(state_const$scale, 1.0)
  out_const <- evo_transformers$robust_scale$apply_func(dt_const, gene, state_const)
  expect_equal(out_const, rep(0, 10))

  # Test formula string
  expect_equal(gene_to_formula(gene), "robust_scale(x)")
})

test_that("smooth_clip dampens outliers smoothly while preserving bulk distribution", {
  x_vals <- seq(-50, 50, length.out = 101)
  dt_train <- data.table::data.table(x = x_vals, y = rnorm(length(x_vals)))
  gene <- create_gene("smooth_clip", "x")
  expect_true(!is.null(gene$params$low_pct))
  expect_true(!is.null(gene$params$high_pct))

  gene$params$low_pct <- 0.10
  gene$params$high_pct <- 0.90

  state <- evo_transformers$smooth_clip$fit_func(dt_train, gene, target_col = "y")
  expect_true(state$q_low < state$q_high)
  expect_true(state$margin > 0)

  out <- evo_transformers$smooth_clip$apply_func(dt_train, gene, state)

  # Bulk region should be untouched (identical to x)
  bulk_idx <- which(x_vals >= state$q_low & x_vals <= state$q_high)
  expect_equal(out[bulk_idx], x_vals[bulk_idx])

  # Extreme tails should be compressed (out < x for upper tail, out > x for lower tail)
  high_idx <- which(x_vals > state$q_high)
  expect_true(all(out[high_idx] <= x_vals[high_idx]))
  expect_true(all(out[high_idx] >= state$q_high))

  low_idx <- which(x_vals < state$q_low)
  expect_true(all(out[low_idx] >= x_vals[low_idx]))
  expect_true(all(out[low_idx] <= state$q_low))

  # Test formula string
  expect_match(gene_to_formula(gene), "smooth_clip_l0.1_h0.9\\(x\\)")
})

test_that("apply_individual applies fourier_basis, robust_scale, and smooth_clip without pruning", {
  set.seed(42)
  dt <- data.table::data.table(
    a = rnorm(100, mean = 20, sd = 5),
    b = runif(100, 1, 10),
    target = rnorm(100)
  )

  g1 <- create_gene("fourier_basis", "a")
  g2 <- create_gene("robust_scale", "b")
  g3 <- create_gene("smooth_clip", "a")

  ind <- create_individual(
    genes = list(g1, g2, g3),
    numeric_cols = c("a", "b"),
    all_numeric_cols = c("a", "b")
  )

  res <- apply_individual(ind, data.table::copy(dt), target_col = "target", allow_prune = TRUE)
  expect_equal(length(res$ind$genes), 3)
  expect_true(g1$output_col %in% names(res$train))
  expect_true(g2$output_col %in% names(res$train))
  expect_true(g3$output_col %in% names(res$train))
})

test_that("mutate can tweak fourier_basis and smooth_clip parameters", {
  set.seed(999)
  g_fbr <- create_gene("fourier_basis", "x")
  g_fbr$params$scale <- 1.0
  g_fbr$params$phase <- 0.0
  ind <- create_individual(
    genes = list(g_fbr),
    numeric_cols = "x",
    all_numeric_cols = "x"
  )

  # Run mutation multiple times to exercise mutate_params
  mutated_params <- FALSE
  for (i in 1:30) {
    mut_ind <- mutate(ind, raw_toggle_prob = 0, recalculate_mask_prob = 0)
    if (length(mut_ind$genes) >= 1 && mut_ind$genes[[1]]$transformer_name == "fourier_basis") {
      p <- mut_ind$genes[[1]]$params
      if (!is.null(p$scale) && (p$scale != 1.0 || p$phase != 0.0)) {
        mutated_params = TRUE
        break
      }
    }
  }
  expect_true(mutated_params)
})

test_that("yeo_johnson fits MLE lambda and standardizes positive/negative skewed distributions", {
  set.seed(101)
  z <- rnorm(500)
  # Right-skewed distribution spanning negative and positive values
  x_train <- exp(z) - 1.5
  dt_train <- data.table::data.table(x = x_train, y = z)
  gene <- create_gene("yeo_johnson", "x")

  state <- evo_transformers$yeo_johnson$fit_func(dt_train, gene, target_col = "y")
  expect_true(is.numeric(state$lambda) && is.finite(state$lambda))
  expect_true(state$lambda < 0.6) # Should compress right tail (lambda < 1)
  expect_true(state$sd > 0)

  out_train <- evo_transformers$yeo_johnson$apply_func(dt_train, gene, state)
  expect_equal(mean(out_train), 0.0, tolerance = 1e-6)
  expect_equal(stats::sd(out_train), 1.0, tolerance = 1e-6)

  # Strictly monotonic with input x
  ord <- order(x_train)
  expect_true(all(diff(out_train[ord]) >= -1e-12))

  # Test stateful application on test data + non-finite handling
  dt_test <- data.table::data.table(x = c(-1.0, 0.0, 2.0, NA_real_, Inf))
  out_test <- evo_transformers$yeo_johnson$apply_func(dt_test, gene, state)
  expect_true(out_test[1] < out_test[2] && out_test[2] < out_test[3])
  expect_equal(out_test[4:5], c(0.0, 0.0))

  # Constant column fallback
  dt_const <- data.table::data.table(x = rep(3.0, 20))
  st_const <- evo_transformers$yeo_johnson$fit_func(dt_const, gene)
  expect_equal(st_const$sd, 1.0)
  expect_equal(evo_transformers$yeo_johnson$apply_func(dt_const, gene, st_const), rep(0.0, 20))
  expect_equal(gene_to_formula(gene), "yeo_johnson(x)")
})

test_that("box_cox fits MLE lambda on positive and shifted negative data", {
  set.seed(202)
  z <- rnorm(600, mean = 0, sd = 0.8)
  # Pure log-normal data -> theoretical Box-Cox lambda = 0, shift = 0
  x_pos <- exp(z)
  dt_pos <- data.table::data.table(x = x_pos, y = z)
  gene <- create_gene("box_cox", "x")

  state_pos <- evo_transformers$box_cox$fit_func(dt_pos, gene, target_col = "y")
  expect_equal(state_pos$shift, 0.0)
  expect_equal(state_pos$lambda, 0.0, tolerance = 0.15)

  out_pos <- evo_transformers$box_cox$apply_func(dt_pos, gene, state_pos)
  expect_equal(mean(out_pos), 0.0, tolerance = 1e-6)
  expect_equal(stats::sd(out_pos), 1.0, tolerance = 1e-6)

  # Shifted data containing negative and zero values
  x_neg <- x_pos - 5.0
  dt_neg <- data.table::data.table(x = x_neg, y = z)
  state_neg <- evo_transformers$box_cox$fit_func(dt_neg, gene, target_col = "y")
  expect_true(state_neg$shift > -min(x_neg))

  out_neg <- evo_transformers$box_cox$apply_func(dt_neg, gene, state_neg)
  expect_equal(mean(out_neg), 0.0, tolerance = 1e-6)
  expect_equal(stats::sd(out_neg), 1.0, tolerance = 1e-6)
  ord <- order(x_neg)
  expect_true(all(diff(out_neg[ord]) >= -1e-12))

  # Test unseen extreme negative value & NA safety on test split
  dt_test <- data.table::data.table(x = c(-1000.0, NA_real_, Inf))
  out_test <- evo_transformers$box_cox$apply_func(dt_test, gene, state_neg)
  expect_true(all(is.finite(out_test)))
  expect_equal(out_test[2:3], c(0.0, 0.0))
  expect_equal(gene_to_formula(gene), "box_cox(x)")
})

test_that("yeo_johnson and box_cox survive parent correlation check and default threshold is 0.99", {
  set.seed(303)
  old_opt <- getOption("evoFE.redundancy_cor_threshold")
  on.exit(options(evoFE.redundancy_cor_threshold = old_opt), add = TRUE)
  options(evoFE.redundancy_cor_threshold = NULL) # Exercise 0.99 default

  # Nearly Gaussian positive data where Box-Cox and Yeo-Johnson have |r(T(x), x)| > 0.99 against raw x
  dt <- data.table::data.table(
    a = rnorm(200, mean = 50, sd = 2),
    b = exp(rnorm(200, mean = 0, sd = 0.5)),
    target = rnorm(200)
  )

  g_yj <- create_gene("yeo_johnson", "a")
  g_bc <- create_gene("box_cox", "b")
  g_rk <- create_gene("rank_transform", "a")

  ind <- create_individual(
    genes = list(g_yj, g_bc, g_rk),
    numeric_cols = c("a", "b"),
    all_numeric_cols = c("a", "b")
  )

  res <- apply_individual(ind, data.table::copy(dt), target_col = "target", allow_prune = TRUE)
  expect_equal(length(res$ind$genes), 3)
  expect_true(all(c(g_yj$output_col, g_bc$output_col, g_rk$output_col) %in% names(res$train)))
})

