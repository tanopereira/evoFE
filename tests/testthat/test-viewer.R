test_that("record = TRUE successfully populates the evolution log", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  # Check that evolution log is NULL when record = FALSE
  set.seed(42)
  res_no_record <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 1,
    pop_size = 2,
    record = FALSE,
    verbose = FALSE
  )
  expect_null(res_no_record$evolution_log)
  expect_error(view(res_no_record), "No evolution log found")

  # Check that evolution log is populated when record = TRUE
  res_record <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 2,
    pop_size = 2,
    record = TRUE,
    verbose = FALSE
  )

  expect_type(res_record$evolution_log, "list")
  expect_named(res_record$evolution_log, c("config", "baseline", "generations", "tournament", "pooled", "historical", "final"))
  expect_equal(res_record$evolution_log$config$generations, 2)
  expect_length(res_record$evolution_log$generations, 2)

  # Verify baseline logged
  expect_type(res_record$evolution_log$baseline$fitness, "double")
  expect_type(res_record$evolution_log$baseline$recipe, "character")
})

test_that("start_evolution_viewer starts and stops httpuv server correctly", {
  # Skip if httpuv or jsonlite is missing
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")

  viewer <- start_evolution_viewer()
  expect_type(viewer, "list")
  expect_named(viewer, c("url", "server", "send", "stop"))
  expect_type(viewer$url, "character")
  expect_true(grepl("^http://127.0.0.1:", viewer$url))

  # Shutdown
  expect_silent(viewer$stop())
})
