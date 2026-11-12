test_that("record = TRUE successfully populates the evolution log", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")
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
  expect_named(viewer, c("url", "server", "get_connection", "send", "stop"))
  expect_type(viewer$url, "character")
  expect_true(grepl("^http://127.0.0.1:", viewer$url))

  # Shutdown
  expect_silent(viewer$stop())
})

test_that("start_evolution_viewer respects custom port parameter and global option", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")

  # 1. Custom port parameter
  test_port <- 14567
  viewer <- start_evolution_viewer(port = test_port)
  expect_equal(viewer$url, paste0("http://127.0.0.1:", test_port))
  expect_silent(viewer$stop())

  # 2. Global option evoFE.viewer_port
  options(evoFE.viewer_port = 14568)
  on.exit(options(evoFE.viewer_port = NULL), add = TRUE)
  
  viewer_opt <- start_evolution_viewer()
  expect_equal(viewer_opt$url, "http://127.0.0.1:14568")
  expect_silent(viewer_opt$stop())
})

test_that("start_evolution_viewer handles port in use by falling back to another port with warning", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")

  # 1. Bind to a port first using raw httpuv (so it's not in .viewer_env)
  busy_port <- 14569
  server1 <- httpuv::startServer("127.0.0.1", busy_port, list(call = function(req) list(status = 200L)))
  
  # 2. Try to bind to the same port, expect it to fallback with warning
  expect_warning(
    {
      viewer2 <- start_evolution_viewer(port = busy_port)
    },
    "was already in use. Falling back to port"
  )
  
  # Check that the second viewer chose a different port and successfully started
  expect_true(viewer2$url != paste0("http://127.0.0.1:", busy_port))
  
  # Cleanup
  httpuv::stopServer(server1)
  expect_silent(viewer2$stop())
})

test_that("start_evolution_viewer automatically stops the previous active viewer session", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")

  # 1. Start first viewer session
  v1 <- start_evolution_viewer(port = 14570)
  expect_true(exists("active_viewer", envir = .viewer_env))
  expect_identical(.viewer_env$active_viewer$server, v1$server)

  # 2. Start second viewer session on the same port
  # It should succeed without warning because v1 is stopped automatically
  v2 <- start_evolution_viewer(port = 14570)
  
  # Check that v2 is now the active one
  expect_identical(.viewer_env$active_viewer$server, v2$server)
  expect_equal(v2$url, "http://127.0.0.1:14570")

  # Shutdown v2
  expect_silent(v2$stop())
  expect_false(exists("active_viewer", envir = .viewer_env))
})

test_that("evolution log records min_fitness_threshold for tiered admission policies", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  mig_cfg <- migration_config(
    topology = topology_tiered(islands = 4, tiers = 2),
    policy = policy_tiered_admission(min_fitness_threshold = "mean_peer")
  )

  res <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 1,
    pop_size = 2,
    islands = 4,
    migration = mig_cfg,
    record = TRUE,
    verbose = FALSE
  )

  expect_equal(res$evolution_log$config$migration_policy, "tiered_admission")
  expect_equal(res$evolution_log$config$min_fitness_threshold, "mean_peer")
})

test_that("record = TRUE and record = FALSE produce identical evolutionary recipes and fitness", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  res_f <- evolve_features(
    data = df, target_col = "am", task = "classification",
    evaluator = "lightgbm", generations = 2, pop_size = 4,
    seed = 42, record = FALSE, verbose = FALSE
  )
  res_t <- evolve_features(
    data = df, target_col = "am", task = "classification",
    evaluator = "lightgbm", generations = 2, pop_size = 4,
    seed = 42, record = TRUE, verbose = FALSE
  )

  expect_equal(res_f$best_fitness, res_t$best_fitness)
  expect_identical(
    individual_to_recipe_string(res_f$best_individual),
    individual_to_recipe_string(res_t$best_individual)
  )
})

test_that("record and port validation reject invalid inputs safely", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")
  data(mtcars)
  df <- mtcars[, c("mpg", "cyl", "disp", "am")]
  df$am <- as.integer(df$am)

  expect_error(
    evolve_features(df, "am", task = "classification", evaluator = "lightgbm", generations = 1, pop_size = 2, record = NA, verbose = FALSE),
    "record must be a logical scalar"
  )

  # Invalid ports should fail cleanly without disturbing an active viewer
  v_active <- start_evolution_viewer()
  on.exit(v_active$stop(), add = TRUE)

  for (bad_port in list("8080", -1L, 0L, 65536L, NA_integer_, 1.5, c(8080L, 8081L), Inf)) {
    expect_error(start_evolution_viewer(port = bad_port), "port must be NULL or a single integer")
  }
  expect_true(exists("active_viewer", envir = .viewer_env))
  expect_identical(.viewer_env$active_viewer$server, v_active$server)
})

test_that("record = FALSE leaves .viewer_env untouched even with islands and pooling", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")
  if (exists("active_viewer", envir = .viewer_env)) {
    tryCatch(.viewer_env$active_viewer$stop(), error = function(e) NULL)
  }

  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  res <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    islands = 2,
    generations = 2,
    pop_size = 4,
    migration_interval = 1,
    migration_rate = 1,
    model_all_final_genes = TRUE,
    model_all_historical_genes = TRUE,
    record = FALSE,
    verbose = FALSE
  )

  expect_null(res$evolution_log)
  expect_identical(length(ls(.viewer_env)), 0L)
})

test_that("start_evolution_viewer and record = TRUE preserve .Random.seed", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")

  # Case 1: .Random.seed does not exist yet
  if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    rm(".Random.seed", envir = globalenv())
  }
  v1 <- start_evolution_viewer()
  expect_false(exists(".Random.seed", envir = globalenv(), inherits = FALSE))
  v1$send(list(type = "config", data = list(numeric_cols = c("mpg"))))
  viewer_flush(v1, 10L)
  expect_false(exists(".Random.seed", envir = globalenv(), inherits = FALSE))
  v1$stop()
  v1$stop() # Idempotent stop

  # Case 2: .Random.seed already exists and seed = NULL runs match bit-for-bit
  data(mtcars)
  df <- mtcars[, c("mpg", "cyl", "disp", "hp", "am")]
  df$am <- as.integer(df$am)

  set.seed(101)
  res_f <- evolve_features(
    df, "am", task = "classification", evaluator = "lightgbm",
    generations = 2, pop_size = 4, seed = NULL, record = FALSE, verbose = FALSE
  )
  seed_after_f <- .Random.seed

  set.seed(101)
  res_t <- evolve_features(
    df, "am", task = "classification", evaluator = "lightgbm",
    generations = 2, pop_size = 4, seed = NULL, record = TRUE, verbose = FALSE
  )
  seed_after_t <- .Random.seed

  expect_identical(seed_after_f, seed_after_t)
  expect_equal(res_f$best_fitness, res_t$best_fitness)
  expect_identical(
    individual_to_recipe_string(res_f$best_individual),
    individual_to_recipe_string(res_t$best_individual)
  )
})

test_that("viewer, browseURL, service, and websocket failures never crash or alter evolve_features", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")
  data(mtcars)
  df <- mtcars[, c("mpg", "cyl", "disp", "hp", "am")]
  df$am <- as.integer(df$am)

  baseline_run <- evolve_features(
    df, "am", task = "classification", evaluator = "lightgbm",
    generations = 1, pop_size = 3, seed = 77, record = FALSE, verbose = FALSE
  )

  # 1. start_evolution_viewer throws -> warns and continues cleanly with evolution_log
  local({
    testthat::local_mocked_bindings(
      start_evolution_viewer = function(...) stop("simulated bind failure")
    )
    expect_warning(
      res_fail_start <- evolve_features(
        df, "am", task = "classification", evaluator = "lightgbm",
        generations = 1, pop_size = 3, seed = 77, record = TRUE, verbose = FALSE
      ),
      "Failed to start evolution viewer"
    )
    expect_equal(res_fail_start$best_fitness, baseline_run$best_fitness)
    expect_type(res_fail_start$evolution_log, "list")
    expect_length(res_fail_start$evolution_log$generations, 1L)
  })

  # 2. browseURL fails -> warns, skips connection wait timeout immediately, finishes fast
  local({
    old_opts <- options(evoFE.viewer_browse = TRUE, evoFE.viewer_timeout = 5.0)
    on.exit(options(old_opts), add = TRUE)
    testthat::local_mocked_bindings(
      browseURL = function(...) stop("no headless browser"),
      .package = "utils"
    )
    t0 <- proc.time()[["elapsed"]]
    expect_warning(
      res_fail_browse <- evolve_features(
        df, "am", task = "classification", evaluator = "lightgbm",
        generations = 1, pop_size = 3, seed = 77, record = TRUE, verbose = FALSE
      ),
      "Failed to open evolution viewer"
    )
    elapsed <- proc.time()[["elapsed"]] - t0
    expect_lt(elapsed, 4.0)
    expect_equal(res_fail_browse$best_fitness, baseline_run$best_fitness)
  })

  # 3. viewer$send / ws$send / httpuv::service failures are completely isolated
  local({
    testthat::local_mocked_bindings(
      start_evolution_viewer = function(...) {
        list(
          url = "http://127.0.0.1:19999",
          server = NULL,
          get_connection = function() NULL,
          send = function(...) stop("ws send failure"),
          stop = function() stop("stop failure")
        )
      }
    )
    res_fail_send <- evolve_features(
      df, "am", task = "classification", evaluator = "lightgbm",
      generations = 1, pop_size = 3, seed = 77, record = TRUE, verbose = FALSE
    )
    expect_equal(res_fail_send$best_fitness, baseline_run$best_fitness)
    expect_identical(
      individual_to_recipe_string(res_fail_send$best_individual),
      individual_to_recipe_string(baseline_run$best_individual)
    )
  })

  # 4. jsonlite::toJSON / httpuv::service throwing inside viewer$send is swallowed cleanly
  local({
    testthat::local_mocked_bindings(
      service = function(...) stop("service error"),
      .package = "httpuv"
    )
    v <- start_evolution_viewer()
    on.exit(v$stop(), add = TRUE)
    expect_silent(v$send(list(type = "status", data = list(island = 1L, status = "evaluating", generation = 1L))))
    expect_silent(viewer_flush(v, 25L))
  })
})

test_that("viewer HTTP routes, structured late-reconnect replay, and closed socket pruning work", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")

  captured_app <- NULL
  testthat::local_mocked_bindings(
    startServer = function(host, port, app) {
      captured_app <<- app
      list(stop = function() NULL)
    },
    stopServer = function(srv) NULL,
    service = function(...) TRUE,
    randomPort = function() 14601L,
    .package = "httpuv"
  )

  v <- start_evolution_viewer()
  on.exit(v$stop(), add = TRUE)
  expect_true(is.list(captured_app))

  # HTTP routes: /favicon.ico -> 204, / -> 200 UTF-8 no-store
  fav_res <- captured_app$call(list(PATH_INFO = "/favicon.ico"))
  expect_identical(fav_res$status, 204L)
  expect_identical(fav_res$body, "")

  root_res <- captured_app$call(list(PATH_INFO = "/"))
  expect_identical(root_res$status, 200L)
  expect_identical(root_res$headers[["Content-Type"]], "text/html; charset=utf-8")
  expect_true(grepl("evolution-data", root_res$body, fixed = TRUE))

  make_mock_ws <- function(fail_after = Inf) {
    sent <- character(0)
    close_cb <- NULL
    ws <- list(
      handle = new.env(parent = emptyenv()),
      onClose = function(cb) {
        close_cb <<- cb
      },
      send = function(msg) {
        if (length(sent) >= fail_after) {
          stop("broken pipe")
        }
        sent <<- c(sent, msg)
      },
      close = function() {
        ws$handle <<- NULL
        if (is.function(close_cb)) close_cb()
      },
      get_sent = function() sent
    )
    ws
  }

  # Emit generation 1 lifecycle including transient status & island_evaluated
  v$send(list(type = "config", data = list(islands = 1L, numeric_cols = c("mpg"))))
  v$send(list(type = "baseline", data = list(fitness = 0.5, recipe = "baseline", sample = list(mpg = c(21)), importances = list())))
  v$send(list(type = "status", data = list(island = 1L, status = "evaluating", generation = 1L)))
  v$send(list(type = "island_evaluated", data = list(island = 1L, generation = 1L, best_fitness = 0.7, stagnation = 0L, all_fitness = c(0.7))))
  v$send(list(type = "generation", data = list(
    generation = 1L,
    islands = list(list(island = 1L, best_fitness = 0.7, baseline_fitness = 0.5, headroom_closed = 0.4, stagnation = 0L, pop_size = 1L, population = list(list(fitness = 0.7, n_genes = 1L)), all_fitness = c(0.7))),
    global_best_fitness = 0.7, global_best_island = 1L, global_best_island_baseline = 0.5,
    global_headroom_closed = 0.4, global_best_recipe = "g1", global_best_n_genes = 1L,
    global_best_importances = list(), global_best_genes = list(), sample = list(mpg = c(21))
  )))
  # Emitted for generation 2 (uncommitted)
  v$send(list(type = "status", data = list(island = 1L, status = "evaluating", generation = 2L)))
  v$send(list(type = "island_evaluated", data = list(island = 1L, generation = 2L, best_fitness = 0.8, stagnation = 0L, all_fitness = c(0.8))))

  # Client 1 connects mid-run: should NOT receive stale gen-1 status/island_evaluated,
  # SHOULD receive gen-2 uncommitted status/island_evaluated
  ws1 <- make_mock_ws()
  captured_app$onWSOpen(ws1)
  replayed1 <- lapply(ws1$get_sent(), function(s) jsonlite::fromJSON(s, simplifyVector = FALSE))
  types1 <- vapply(replayed1, function(x) x$type, character(1))
  expect_identical(types1, c("config", "baseline", "generation", "status", "island_evaluated"))
  expect_identical(replayed1[[4]]$data$generation, 2L)
  expect_identical(replayed1[[5]]$data$generation, 2L)

  # Broken client failing during replay stops early and gets pruned
  ws_broken <- make_mock_ws(fail_after = 1L)
  captured_app$onWSOpen(ws_broken)
  expect_length(ws_broken$get_sent(), 1L)

  # Client 1 disconnects via onClose; sending after disconnect prunes broken/closed sockets
  ws1$close()
  expect_null(v$get_connection())

  # Complete arrives; a late client gets clean ordered summary without stale transient status
  v$send(list(type = "confirmation", data = list(top_k = 1L, winner_rank = 1L, split1_fitness = c(0.8), split2_fitness = c(0.79), mean_fitness = c(0.795))))
  v$send(list(type = "complete", data = list(best_fitness = 0.8, baseline_fitness = 0.5, global_best_importances = list(), global_best_genes = list(), sample = list(mpg = c(21)))))

  ws_late <- make_mock_ws()
  captured_app$onWSOpen(ws_late)
  replayed_late <- lapply(ws_late$get_sent(), function(s) jsonlite::fromJSON(s, simplifyVector = FALSE))
  types_late <- vapply(replayed_late, function(x) x$type, character(1))
  expect_identical(types_late, c("config", "baseline", "generation", "confirmation", "complete"))
})

test_that("viewer_to_json preserves array types, empty objects, na='null', and escapes script tags", {
  skip_if_not_installed("jsonlite")

  log_obj <- list(
    config = list(
      numeric_cols = c("x1"),
      categorical_cols = character(0),
      datetime_cols = NULL,
      adj_list = list(c(2L), c(1L))
    ),
    baseline = list(
      fitness = 0.5,
      recipe = "baseline",
      sample = list(x1 = c(1.2345)),
      importances = list(),
      islands = list(list(island = 1L, evaluator = "lightgbm", fitness = NA_real_))
    ),
    generations = list(
      list(
        generation = 1L,
        islands = list(
          list(
            island = 1L,
            best_fitness = NaN,
            baseline_fitness = Inf,
            headroom_closed = -Inf,
            stagnation = 0L,
            pop_size = 1L,
            population = list(list(fitness = NA_real_, n_genes = 1L)),
            all_fitness = c(0.6)
          )
        ),
        global_best_fitness = 0.6,
        global_best_recipe = "bad</script><!--x",
        global_best_importances = list(),
        global_best_genes = list(list(formula = "x1</script>", output_col = "g1", importance = 1.0)),
        migrations = list(list(generation = 1L, from = 1L, to = 2L, migrated_genes = list(list(formula = "a", output_col = "g1", importance = 1.0)))),
        sample = list(x1 = c(1.23))
      )
    ),
    tournament = list(candidates = list(list(island = 1L, local_fitness = 0.6, global_fitness = 0.6, recipe = "g1", n_genes = 1L))),
    confirmation = list(top_k = 1L, winner_rank = 1L, split1_fitness = c(0.6), split2_fitness = c(0.59), mean_fitness = c(0.595)),
    final = list(
      best_fitness = 0.6,
      global_best_importances = list(),
      global_best_genes = list(list(formula = "x1</script>", output_col = "g1", importance = 1.0)),
      sample = list(x1 = c(1.23))
    )
  )

  raw_json <- viewer_to_json(log_obj, pretty = TRUE, is_full_log = TRUE, escape_script = TRUE)
  expect_false(grepl("</script>", raw_json, fixed = TRUE))
  expect_false(grepl("<!--", raw_json, fixed = TRUE))
  expect_false(grepl('"NaN"', raw_json, fixed = TRUE))
  expect_false(grepl('"Inf"', raw_json, fixed = TRUE))
  expect_false(grepl('"-Inf"', raw_json, fixed = TRUE))
  expect_false(grepl('"NA"', raw_json, fixed = TRUE))
  expect_true(grepl('"importances": \\{\\}', raw_json))
  expect_true(grepl('"global_best_importances": \\{\\}', raw_json))

  parsed <- jsonlite::fromJSON(raw_json, simplifyVector = FALSE)
  expect_type(parsed$config$numeric_cols, "list")
  expect_length(parsed$config$numeric_cols, 1L)
  expect_type(parsed$config$categorical_cols, "list")
  expect_length(parsed$config$categorical_cols, 0L)
  expect_type(parsed$config$datetime_cols, "list")
  expect_length(parsed$config$adj_list, 2L)
  expect_type(parsed$config$adj_list[[1]], "list")
  expect_length(parsed$config$adj_list[[1]], 1L)
  expect_type(parsed$baseline$islands, "list")
  expect_null(parsed$baseline$islands[[1]]$fitness)
  expect_type(parsed$baseline$sample$x1, "list")
  expect_length(parsed$generations, 1L)
  expect_null(parsed$generations[[1]]$islands[[1]]$best_fitness)
  expect_null(parsed$generations[[1]]$islands[[1]]$baseline_fitness)
  expect_null(parsed$generations[[1]]$islands[[1]]$headroom_closed)
  expect_type(parsed$generations[[1]]$islands[[1]]$all_fitness, "list")
  expect_type(parsed$generations[[1]]$migrations, "list")
  expect_type(parsed$generations[[1]]$migrations[[1]]$migrated_genes, "list")
  expect_type(parsed$tournament$candidates, "list")
  expect_type(parsed$confirmation$split1_fitness, "list")
  expect_identical(parsed$generations[[1]]$global_best_recipe, "bad</script><!--x")
})

test_that("view() emits valid standalone HTML and schema parity across islands, pooling, and confirmation", {
  skip_if_not_installed("httpuv")
  skip_if_not_installed("jsonlite")
  data(mtcars)
  df <- mtcars[, c("mpg", "cyl", "disp", "hp", "wt", "am")]
  df$am <- as.integer(df$am)

  # Single-island schema parity check
  res_1 <- evolve_features(
    df, "am", task = "classification", evaluator = "lightgbm",
    generations = 1, pop_size = 3, seed = 9, record = TRUE, verbose = FALSE
  )
  g1_single <- res_1$evolution_log$generations[[1]]
  expect_identical(res_1$evolution_log$config$metric, "default")
  expect_equal(res_1$evolution_log$config$ideal_ceiling, 1.0)
  expect_identical(g1_single$global_best_island, 1L)
  expect_equal(g1_single$global_best_island_baseline, res_1$evolution_log$baseline$fitness)
  expect_true("headroom_closed" %in% names(g1_single$islands[[1]]))
  expect_true("baseline_fitness" %in% names(g1_single$islands[[1]]))
  expect_true("global_headroom_closed" %in% names(g1_single))
  # Sample preview does not leak columns from other individuals
  expect_setequal(intersect(names(g1_single$sample), names(df)), names(df))

  # Multi-island + CV + migration + pooling + confirmation parity + view() HTML output
  res_m <- evolve_features(
    df, "am", task = "classification", evaluator = "lightgbm",
    evaluation_strategy = "cv", cv_folds = 3,
    islands = 2, generations = 2, pop_size = 4,
    migration_interval = 1, migration_rate = 1, migration_topology = "gibbs_stagnation",
    model_all_final_genes = TRUE, model_all_historical_genes = TRUE,
    confirm_top_k = 2, seed = 11, record = TRUE, verbose = FALSE
  )

  log_m <- res_m$evolution_log
  expect_identical(log_m$config$migration_policy, "gibbs_push_stagnation")
  expect_identical(log_m$config$migration_rate, 1L)
  expect_identical(log_m$config$cv_folds, 3L)
  expect_true(length(log_m$generations[[1]]$migrations) >= 1L)
  expect_identical(log_m$generations[[1]]$migrations[[1]]$generation, 1L)
  expect_true(all(c("n_genes", "fitness", "adopted", "recipe", "baseline_fitness", "improvement") %in% names(log_m$pooled)))
  expect_true(all(c("n_genes", "fitness", "adopted", "recipe", "baseline_fitness", "improvement") %in% names(log_m$historical)))
  expect_true(all(c("top_k", "winner_rank", "split1_fitness", "split2_fitness", "mean_fitness") %in% names(log_m$confirmation)))

  html_file <- view(res_m)
  on.exit(unlink(html_file), add = TRUE)
  expect_true(file.exists(html_file))

  html_text <- paste(readLines(html_file, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
  expect_false(grepl('<script id="evolution-data" type="application/json">__EVOLUTION_DATA__</script>', html_text, fixed = TRUE))

  m <- regexec('<script id="evolution-data" type="application/json">(.*?)</script>', html_text)
  caps <- regmatches(html_text, m)[[1]]
  expect_length(caps, 2L)

  embedded <- jsonlite::fromJSON(caps[2], simplifyVector = FALSE)
  expect_identical(embedded$config$migration_policy, "gibbs_push_stagnation")
  expect_length(embedded$generations, 2L)
  expect_identical(embedded$final$best_recipe, log_m$final$best_recipe)
  expect_identical(embedded$confirmation$winner_rank, log_m$confirmation$winner_rank)

  if (requireNamespace("xml2", quietly = TRUE)) {
    doc <- xml2::read_html(html_text)
    expect_s3_class(doc, "xml_document")
  }
})

