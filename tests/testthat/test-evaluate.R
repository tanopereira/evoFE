test_that("compute_calibrated_rmse works correctly", {
  # 1. Standard case
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(2, 4, 6, 8, 10) # perfect linear prediction (y = 0.5 * y_pred)
  expect_equal(compute_calibrated_rmse(y_true, y_pred), 0.0, tolerance = 1e-7)

  # 2. Complete noise case (uncorrelated)
  set.seed(42)
  y_true <- rnorm(100, mean = 10, sd = 2)
  y_pred <- rnorm(100, mean = 0, sd = 1)
  # Calibrated RMSE should be very close to SD(y_true) * sqrt(1 - R^2)
  r <- cor(y_pred, y_true)
  expected <- sd(y_true) * sqrt(1 - r^2)
  expect_equal(compute_calibrated_rmse(y_true, y_pred), expected, tolerance = 1e-9)

  # 3. Constant prediction edge case (no variance)
  y_pred_const <- rep(5, 5)
  y_true <- c(1, 2, 3, 4, 5)
  expected_const <- sqrt(mean((y_true - 5)^2))
  expect_equal(compute_calibrated_rmse(y_true, y_pred_const), expected_const, tolerance = 1e-9)

  # 4. Single element edge case
  expect_equal(compute_calibrated_rmse(c(1), c(2)), 0.0)
})

test_that("compute_calibrated_mae works correctly", {
  # 1. Perfect linear relation
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.1, 2.1, 3.1, 4.1, 5.1) # exact shift of +0.1
  # Optimal calibration: a = -0.1, b = 1.0, MAE = 0
  expect_equal(compute_calibrated_mae(y_true, y_pred), 0.0, tolerance = 1e-5)

  # 2. Standard case
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.5, 1.8, 3.2, 4.5, 4.8)
  val <- compute_calibrated_mae(y_true, y_pred)
  expect_true(val >= 0.0)
  expect_true(val <= mean(abs(y_true - y_pred))) # Calibrated MAE should be <= raw MAE

  # 3. Constant prediction edge case
  y_pred_const <- rep(5, 5)
  y_true <- c(1, 2, 3, 4, 5)
  # Optimal L1 calibration: a = median(y_true) = 3, b = 0. MAE = mean(abs(y_true - 3)) = (2+1+0+1+2)/5 = 1.2
  expect_equal(compute_calibrated_mae(y_true, y_pred_const), 1.2, tolerance = 1e-5)

  # 4. NA handling
  y_true <- c(1, 2, NA, 4, 5)
  y_pred <- c(1, 2, 3, 4, NA)
  val_na <- compute_calibrated_mae(y_true, y_pred)
  # valid pairs are c(1, 2, 4) vs c(1, 2, 4) -> perfect
  expect_equal(val_na, 0.0, tolerance = 1e-5)
})
