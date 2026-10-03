#' Truncate columns for console display
#'
#' @param cols Vector of column names
#' @param max_show Maximum number of columns to show
#' @return Formatted character string
#' @noRd
truncate_cols <- function(cols, max_show = 10) {
  if (length(cols) <= max_show) {
    return(paste(cols, collapse = ", "))
  }
  paste0(paste(cols[1:max_show], collapse = ", "), ", ... (+ ", length(cols) - max_show, " more)")
}

#' Check if terminal supports ANSI colors
#' @keywords internal
supports_color <- function() {
  term <- Sys.getenv("TERM")
  if (term %in% c("dumb", "")) {
    return(FALSE)
  }
  if (.Platform$OS.type == "windows") {
    return(interactive() || !is.na(Sys.getenv("RSTUDIO", unset = NA)))
  }
  isatty(stdout()) || !is.na(Sys.getenv("RSTUDIO", unset = NA))
}

#' Resolve thread count and aliases
#'
#' @param threads Thread count argument
#' @param extra_args List of extra arguments
#' @return Normalized thread count
#' @noRd
resolve_thread_count <- function(threads = 2L, extra_args = list()) {
  resolve_param_aliases(extra_args, defaults = list(threads = threads))$threads
}

#' Setup core environment configuration, threads, and RNG seed
#'
#' @param threads Thread count to configure
#' @param max_clustering_size Max clustering size option
#' @param seed Optional random seed
#' @return Environment state list used by restore_core_env
#' @noRd
setup_core_env <- function(threads = 2L, max_clustering_size = 5000L, seed = NULL) {
  # Temporarily configure max clustering size and threads options
  old_max_size <- getOption("evoFE.max_clustering_size")
  old_threads <- getOption("evoFE.threads")
  old_opt_dt <- getOption("datatable.threads")
  options(evoFE.max_clustering_size = max_clustering_size, evoFE.threads = threads)

  # Query all initial thread settings BEFORE setting any thread limits
  old_dt <- if (requireNamespace("data.table", quietly = TRUE)) tryCatch(data.table::getDTthreads(), error = function(e) NULL) else NULL
  old_omp <- if (requireNamespace("RhpcBLASctl", quietly = TRUE)) tryCatch(RhpcBLASctl::omp_get_max_threads(), error = function(e) NULL) else NULL
  old_blas <- if (requireNamespace("RhpcBLASctl", quietly = TRUE)) tryCatch(RhpcBLASctl::blas_get_num_procs(), error = function(e) NULL) else NULL
  old_qf <- if (requireNamespace("quitefastmst", quietly = TRUE)) tryCatch(quitefastmst::omp_get_max_threads(), error = function(e) NULL) else NULL

  # Validate seed
  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1 || is.na(seed) ||
      !is.finite(seed))) {
    stop("'seed' must be NULL or a single finite number.")
  }
  seed <- if (!is.null(seed)) as.integer(seed) else NULL
  old_seed <- if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    get(".Random.seed", envir = globalenv(), inherits = FALSE)
  } else {
    NULL
  }
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Apply thread modifications
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    if (!is.null(old_omp) && !is.na(old_omp) && is.numeric(old_omp) && !is.null(threads) && !is.na(threads) && is.numeric(threads)) {
      tryCatch(RhpcBLASctl::omp_set_num_threads(as.integer(threads)), error = function(e) NULL)
    }
    if (!is.null(old_blas) && !is.na(old_blas) && is.numeric(old_blas) && !is.null(threads) && !is.na(threads) && is.numeric(threads)) {
      tryCatch(RhpcBLASctl::blas_set_num_threads(as.integer(threads)), error = function(e) NULL)
    }
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    if (!is.null(threads) && !is.na(threads) && is.numeric(threads)) {
      tryCatch(data.table::setDTthreads(as.integer(threads)), error = function(e) NULL)
    }
  }
  if (requireNamespace("quitefastmst", quietly = TRUE)) {
    if (!is.null(old_qf) && !is.na(old_qf) && is.numeric(old_qf) && !is.null(threads) && !is.na(threads) && is.numeric(threads)) {
      tryCatch(quitefastmst::omp_set_num_threads(as.integer(threads)), error = function(e) NULL)
    }
  }

  list(
    old_max_size = old_max_size,
    old_threads = old_threads,
    old_opt_dt = old_opt_dt,
    old_dt = old_dt,
    old_omp = old_omp,
    old_blas = old_blas,
    old_qf = old_qf,
    old_seed = old_seed,
    seed = seed
  )
}

#' Restore core environment configuration, threads, and RNG seed
#'
#' @param env_state State list produced by setup_core_env
#' @noRd
restore_core_env <- function(env_state) {
  if (is.null(env_state)) return(invisible(NULL))
  options(evoFE.max_clustering_size = env_state$old_max_size)
  options(evoFE.threads = env_state$old_threads)
  if (!is.null(env_state$old_omp) && !is.na(env_state$old_omp) && is.numeric(env_state$old_omp) && requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    tryCatch(RhpcBLASctl::omp_set_num_threads(env_state$old_omp), error = function(e) NULL)
  }
  if (!is.null(env_state$old_qf) && !is.na(env_state$old_qf) && is.numeric(env_state$old_qf) && requireNamespace("quitefastmst", quietly = TRUE)) {
    tryCatch(quitefastmst::omp_set_num_threads(env_state$old_qf), error = function(e) NULL)
  }
  if (!is.null(env_state$old_blas) && !is.na(env_state$old_blas) && is.numeric(env_state$old_blas) && requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    tryCatch(RhpcBLASctl::blas_set_num_threads(env_state$old_blas), error = function(e) NULL)
  }
  if (!is.null(env_state$old_dt) && !is.na(env_state$old_dt) && is.numeric(env_state$old_dt) && requireNamespace("data.table", quietly = TRUE)) {
    tryCatch(data.table::setDTthreads(env_state$old_dt), error = function(e) NULL)
  }
  if (!is.null(env_state$old_opt_dt) && !is.na(env_state$old_opt_dt) && is.numeric(env_state$old_opt_dt)) {
    tryCatch(options(datatable.threads = env_state$old_opt_dt), error = function(e) NULL)
  }
  # Restored LAST because quitefastmst::omp_set_num_threads() alters .Random.seed
  if (!is.null(env_state$old_seed)) {
    assign(".Random.seed", env_state$old_seed, envir = globalenv())
  } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    rm(".Random.seed", envir = globalenv())
  }
  invisible(NULL)
}
