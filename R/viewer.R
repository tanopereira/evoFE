# Package-level environment to store active viewer session
.viewer_env <- new.env(parent = emptyenv())

#' Start the Evolution Live Viewer Server
#'
#' Launches an httpuv web server and returns a controller list to interact with it.
#'
#' @param port Optional port number. If NULL, a random free port is used.
#' @return A list with url, server, send, and stop functions.
#' @keywords internal
start_evolution_viewer <- function(port = NULL) {
  if (!requireNamespace("httpuv", quietly = TRUE)) {
    stop("Package 'httpuv' is required for the evolution viewer. Install it with: install.packages('httpuv')")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required for the evolution viewer. Install it with: install.packages('jsonlite')")
  }

  if (is.null(port)) {
    port <- getOption("evoFE.viewer_port", NULL)
  }
  port <- .validate_viewer_port(port)
  original_port <- port

  # Stop any existing active viewer server in this session
  if (exists("active_viewer", envir = .viewer_env)) {
    tryCatch({
      .viewer_env$active_viewer$stop()
    }, error = function(e) {
      # Ignore
    })
    if (exists("active_viewer", envir = .viewer_env)) {
      tryCatch({
        rm("active_viewer", envir = .viewer_env)
      }, error = function(e) {
        # Ignore
      })
    }
  }

  if (is.null(port)) {
    port <- with_preserved_seed(httpuv::randomPort())
  }

  ws_connections <- list()
  replay_state <- .new_replay_state()
  is_stopped <- FALSE

  is_ws_open <- function(ws) {
    !is.null(ws) && !is.null(ws$handle)
  }

  prune_ws <- function() {
    if (length(ws_connections) > 0L) {
      ws_connections <<- Filter(is_ws_open, ws_connections)
    }
  }

  drop_ws <- function(target_ws) {
    if (length(ws_connections) > 0L) {
      ws_connections <<- Filter(function(conn) !identical(conn, target_ws) && is_ws_open(conn), ws_connections)
    }
  }

  html_content <- enc2utf8(.load_viewer_template(
    missing_fallback = "<html><body><h1>Evolution Viewer Template Missing</h1></body></html>"
  ))

  server <- NULL
  attempts <- 0L
  max_attempts <- 5L
  last_error <- NULL

  while (is.null(server) && attempts < max_attempts) {
    attempts <- attempts + 1L
    if (attempts > 1L || is.null(port)) {
      port <- with_preserved_seed(httpuv::randomPort())
    }
    server <- tryCatch({
      with_preserved_seed(
        httpuv::startServer("127.0.0.1", port,
          list(
            call = function(req) {
              if (identical(req$PATH_INFO, "/favicon.ico")) {
                return(list(status = 204L, headers = list(), body = ""))
              }
              list(
                status = 200L,
                headers = list(
                  "Content-Type" = "text/html; charset=utf-8",
                  "Cache-Control" = "no-store"
                ),
                body = html_content
              )
            },
            onWSOpen = function(ws) {
              prune_ws()
              ws$onClose(function() {
                drop_ws(ws)
              })
              ws_connections[[length(ws_connections) + 1L]] <<- ws
              for (msg in .build_replay_messages(replay_state)) {
                sent <- tryCatch({
                  if (!is_ws_open(ws)) {
                    FALSE
                  } else {
                    ws$send(msg)
                    TRUE
                  }
                }, error = function(e) FALSE)
                if (!sent) {
                  drop_ws(ws)
                  break
                }
              }
            }
          )
        )
      )
    }, error = function(e) {
      last_error <<- e
      NULL
    })
  }

  if (is.null(server)) {
    stop("Failed to start evolution viewer server after 5 attempts: ", conditionMessage(last_error))
  }

  if (!is.null(original_port) && port != original_port) {
    warning(sprintf("Port %d was already in use. Falling back to port %d.", original_port, port))
  }

  url <- sprintf("http://127.0.0.1:%d", port)

  viewer <- list(
    url = url,
    server = server,
    get_connection = function() {
      prune_ws()
      if (length(ws_connections) > 0L) ws_connections[[1L]] else NULL
    },
    send = function(data) {
      tryCatch(
        with_preserved_seed({
          if (is_stopped) {
            return(invisible(NULL))
          }
          json <- viewer_to_json(data, pretty = FALSE)
          replay_state <<- .record_replay_message(replay_state, data, json)

          prune_ws()
          if (length(ws_connections) > 0L) {
            alive_conns <- list()
            for (ws in ws_connections) {
              ok <- tryCatch({
                if (!is_ws_open(ws)) {
                  FALSE
                } else {
                  ws$send(json)
                  TRUE
                }
              }, error = function(e) FALSE)
              if (ok) {
                alive_conns[[length(alive_conns) + 1L]] <- ws
              }
            }
            ws_connections <<- alive_conns
          }

          suppressWarnings(tryCatch(httpuv::service(NA), error = function(e) NULL))
        }),
        error = function(e) NULL
      )
      invisible(NULL)
    },
    stop = function() {
      if (is_stopped) {
        return(invisible(NULL))
      }
      is_stopped <<- TRUE
      if (length(ws_connections) > 0L) {
        for (ws in ws_connections) {
          tryCatch(ws$close(), error = function(e) NULL)
        }
        ws_connections <<- list()
      }
      suppressWarnings(tryCatch(with_preserved_seed(httpuv::stopServer(server)), error = function(e) NULL))
      if (exists("active_viewer", envir = .viewer_env) && identical(.viewer_env$active_viewer$server, server)) {
        tryCatch({
          rm("active_viewer", envir = .viewer_env)
        }, error = function(e) {
          # Ignore
        })
      }
      invisible(NULL)
    }
  )

  attr(viewer, "flush") <- function(timeout_ms = 50L) {
    tryCatch(
      with_preserved_seed({
        if (is_stopped) {
          return(invisible(NULL))
        }
        prune_ws()
        if (length(ws_connections) > 0L) {
          wait_ms <- max(1L, min(as.integer(timeout_ms), 50L))
          suppressWarnings(httpuv::service(wait_ms))
        }
      }),
      error = function(e) NULL
    )
    invisible(NULL)
  }

  .viewer_env$active_viewer <- viewer
  viewer
}

.validate_viewer_port <- function(port) {
  if (is.null(port)) {
    return(NULL)
  }
  if (!is.numeric(port) || length(port) != 1L || is.na(port) || !is.finite(port) ||
        port < 1 || port > 65535 || port != floor(port)) {
    stop("port must be NULL or a single integer between 1 and 65535.")
  }
  as.integer(port)
}

.load_viewer_template <- function(missing_fallback = NULL) {
  html_path <- system.file("viewer", "index.html", package = "evoFE")
  if (html_path == "") {
    html_path <- file.path(getwd(), "inst", "viewer", "index.html")
  }
  if (!file.exists(html_path)) {
    if (!is.null(missing_fallback)) {
      return(missing_fallback)
    }
    stop("Evolution Viewer template not found at inst/viewer/index.html")
  }
  paste(readLines(html_path, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
}

.new_replay_state <- function() {
  list(
    config = NULL,
    baseline = NULL,
    history = list(),
    in_progress_status = list(),
    in_progress_evals = list(),
    post = list()
  )
}

.record_replay_message <- function(state, data, json) {
  msg_type <- if (is.list(data) && !is.null(data$type)) as.character(data$type)[1L] else ""
  payload <- if (is.list(data) && !is.null(data$data)) data$data else NULL
  if (msg_type == "config") {
    state$config <- json
  } else if (msg_type == "baseline") {
    state$baseline <- json
  } else if (msg_type == "status") {
    key <- if (is.list(payload) && !is.null(payload$island)) as.character(payload$island)[1L] else "default"
    state$in_progress_status[[key]] <- json
  } else if (msg_type == "island_evaluated") {
    key <- if (is.list(payload) && !is.null(payload$island)) as.character(payload$island)[1L] else "default"
    state$in_progress_evals[[key]] <- json
  } else if (msg_type == "generation") {
    state$in_progress_status <- list()
    state$in_progress_evals <- list()
    state$history[[length(state$history) + 1L]] <- json
  } else if (msg_type %in% c("tournament", "pooled", "historical", "confirmation")) {
    state$post[[msg_type]] <- json
  } else if (msg_type == "complete") {
    state$in_progress_status <- list()
    state$in_progress_evals <- list()
    state$post[["complete"]] <- json
  } else {
    state$history[[length(state$history) + 1L]] <- json
  }
  state
}

.build_replay_messages <- function(state) {
  c(
    if (!is.null(state$config)) list(state$config) else list(),
    if (!is.null(state$baseline)) list(state$baseline) else list(),
    state$history,
    unname(state$in_progress_status),
    unname(state$in_progress_evals),
    unname(state$post)
  )
}

.as_json_array <- function(x) {
  if (is.null(x) || length(x) == 0L) {
    return(list())
  }
  if (is.list(x)) {
    return(I(unname(x)))
  }
  if (is.factor(x)) {
    x <- as.character(x)
  }
  I(unname(as.vector(x)))
}

.as_json_object <- function(x) {
  if (is.null(x) || length(x) == 0L) {
    return(structure(list(), names = character(0)))
  }
  as.list(x)
}

.prep_sample_obj <- function(smp) {
  if (is.null(smp) || length(smp) == 0L) {
    return(.as_json_object(NULL))
  }
  out <- lapply(smp, .as_json_array)
  names(out) <- names(smp)
  out
}

.prep_config <- function(cfg) {
  if (is.null(cfg)) return(NULL)
  cfg$numeric_cols <- .as_json_array(cfg$numeric_cols)
  cfg$categorical_cols <- .as_json_array(cfg$categorical_cols)
  cfg$datetime_cols <- .as_json_array(cfg$datetime_cols)
  if (!is.null(cfg$adj_list)) {
    cfg$adj_list <- .as_json_array(lapply(cfg$adj_list, .as_json_array))
  }
  cfg
}

.prep_baseline <- function(base) {
  if (is.null(base)) return(NULL)
  base$sample <- .prep_sample_obj(base$sample)
  base$importances <- .as_json_object(base$importances)
  if (!is.null(base$islands)) {
    base$islands <- .as_json_array(base$islands)
  }
  base
}

.prep_island_evaluated <- function(ev) {
  if (is.null(ev)) return(NULL)
  ev$all_fitness <- .as_json_array(ev$all_fitness)
  ev
}

.prep_migration <- function(mig) {
  if (is.null(mig)) return(NULL)
  mig$migrated_genes <- .as_json_array(mig$migrated_genes)
  mig
}

.prep_generation <- function(gen) {
  if (is.null(gen)) return(NULL)
  if (!is.null(gen$islands)) {
    gen$islands <- .as_json_array(lapply(gen$islands, function(isl) {
      isl$population <- .as_json_array(isl$population)
      isl$all_fitness <- .as_json_array(isl$all_fitness)
      isl
    }))
  }
  gen$global_best_importances <- .as_json_object(gen$global_best_importances)
  gen$global_best_genes <- .as_json_array(gen$global_best_genes)
  if (!is.null(gen$migrations)) {
    gen$migrations <- .as_json_array(lapply(gen$migrations, .prep_migration))
  }
  gen$sample <- .prep_sample_obj(gen$sample)
  gen
}

.prep_tournament <- function(tourn) {
  if (is.null(tourn)) return(NULL)
  tourn$candidates <- .as_json_array(tourn$candidates)
  tourn
}

.prep_confirmation <- function(conf) {
  if (is.null(conf)) return(NULL)
  conf$split1_fitness <- .as_json_array(conf$split1_fitness)
  conf$split2_fitness <- .as_json_array(conf$split2_fitness)
  conf$mean_fitness <- .as_json_array(conf$mean_fitness)
  conf
}

.prep_final <- function(fin) {
  if (is.null(fin)) return(NULL)
  fin$global_best_importances <- .as_json_object(fin$global_best_importances)
  fin$global_best_genes <- .as_json_array(fin$global_best_genes)
  fin$sample <- .prep_sample_obj(fin$sample)
  fin
}

.prep_payload_by_type <- function(msg_type, payload) {
  switch(msg_type,
    "config" = .prep_config(payload),
    "baseline" = .prep_baseline(payload),
    "island_evaluated" = .prep_island_evaluated(payload),
    "generation" = .prep_generation(payload),
    "migration" = .prep_migration(payload),
    "tournament" = .prep_tournament(payload),
    "confirmation" = .prep_confirmation(payload),
    "complete" = .prep_final(payload),
    payload
  )
}

.prep_evolution_log <- function(log_obj) {
  if (is.null(log_obj)) return(NULL)
  if (!is.null(log_obj$config)) log_obj$config <- .prep_config(log_obj$config)
  if (!is.null(log_obj$baseline)) log_obj$baseline <- .prep_baseline(log_obj$baseline)
  gens <- if (is.null(log_obj$generations)) list() else Filter(Negate(is.null), log_obj$generations)
  log_obj$generations <- .as_json_array(lapply(gens, .prep_generation))
  if (!is.null(log_obj$tournament)) log_obj$tournament <- .prep_tournament(log_obj$tournament)
  if (!is.null(log_obj$confirmation)) log_obj$confirmation <- .prep_confirmation(log_obj$confirmation)
  if (!is.null(log_obj$final)) log_obj$final <- .prep_final(log_obj$final)
  log_obj
}

viewer_to_json <- function(data, pretty = FALSE, is_full_log = FALSE, escape_script = FALSE) {
  prepared <- if (isTRUE(is_full_log)) {
    .prep_evolution_log(data)
  } else if (is.list(data) && !is.null(data$type) && !is.null(data$data)) {
    data$data <- .prep_payload_by_type(as.character(data$type)[1L], data$data)
    data
  } else {
    data
  }

  json <- as.character(
    jsonlite::toJSON(
      prepared,
      auto_unbox = TRUE,
      na = "null",
      null = "null",
      digits = 6,
      pretty = pretty
    )
  )
  if (isTRUE(escape_script)) {
    json <- gsub("<", "\\u003c", json, fixed = TRUE)
    json <- gsub(">", "\\u003e", json, fixed = TRUE)
  }
  enc2utf8(json)
}

.inject_evolution_json <- function(html, json_data) {
  script_tag <- '<script id="evolution-data" type="application/json">__EVOLUTION_DATA__</script>'
  tag_pos <- regexpr(script_tag, html, fixed = TRUE)
  if (tag_pos > 0L) {
    prefix <- substr(html, 1L, tag_pos - 1L)
    suffix <- substr(html, tag_pos + nchar(script_tag), nchar(html))
    return(paste0(
      prefix,
      '<script id="evolution-data" type="application/json">',
      json_data,
      "</script>",
      suffix
    ))
  }

  matches <- gregexpr("__EVOLUTION_DATA__", html, fixed = TRUE)[[1L]]
  if (length(matches) == 1L && matches[1L] == -1L) {
    stop("Evolution Viewer template is missing the __EVOLUTION_DATA__ placeholder.")
  }
  pos <- matches[length(matches)]
  token_len <- nchar("__EVOLUTION_DATA__")
  paste0(
    substr(html, 1L, pos - 1L),
    json_data,
    substr(html, pos + token_len, nchar(html))
  )
}

viewer_send <- function(viewer, type, data) {
  if (is.null(viewer) || !is.function(viewer$send)) {
    return(invisible(NULL))
  }
  tryCatch(
    with_preserved_seed(viewer$send(list(type = type, data = data))),
    error = function(e) NULL
  )
  invisible(NULL)
}

viewer_flush <- function(viewer, timeout_ms = 50L) {
  flush_fn <- if (!is.null(viewer)) attr(viewer, "flush", exact = TRUE) else NULL
  if (is.function(flush_fn)) {
    tryCatch(with_preserved_seed(flush_fn(timeout_ms)), error = function(e) NULL)
  }
  invisible(NULL)
}

.serialize_genes_for_log <- function(genes, importances = NULL, sort_by_importance = TRUE) {
  if (length(genes) == 0L) {
    return(list())
  }
  serialized_genes <- lapply(genes, function(gene) {
    col <- gene$output_col
    imp_val <- if (!is.null(importances) && col %in% names(importances)) {
      as.numeric(importances[[col]])
    } else {
      0.0
    }
    list(
      formula = gene_to_formula(gene),
      output_col = col,
      importance = imp_val
    )
  })
  if (isTRUE(sort_by_importance) && length(serialized_genes) > 0L) {
    gene_imps <- sapply(serialized_genes, function(x) x$importance)
    serialized_genes <- serialized_genes[order(gene_imps, decreasing = TRUE)]
  }
  serialized_genes
}

.format_sample_dt <- function(best_dt) {
  best_list <- lapply(names(best_dt), function(col) {
    val <- best_dt[[col]]
    if (is.numeric(val)) round(val, 4) else as.character(val)
  })
  names(best_list) <- names(best_dt)
  best_list
}

.build_sample_preview <- function(ind, ref_dt, max_rows = 5L) {
  head_dt <- data.table::copy(data.table::as.data.table(utils::head(ref_dt, max_rows)))
  res_sample <- tryCatch(
    {
      with_preserved_seed(
        apply_individual(ind, head_dt, NULL, NULL, state_cache = NULL, allow_prune = FALSE)
      )
    },
    error = function(e) list(train = head_dt)
  )
  .format_sample_dt(res_sample$train)
}

#' View the evolution history of an evo_recipe
#'
#' Opens an interactive HTML page in the browser to visualize the evolutionary
#' feature engineering process, either in real-time or post-hoc.
#'
#' @param recipe An \code{evo_recipe} object.
#' @param ... Additional arguments (not used).
#' @return Invisible file path string to the generated HTML viewer file.
#' @export
view <- function(recipe, ...) {
  UseMethod("view")
}

#' @rdname view
#' @export
view.evo_recipe <- function(recipe, ...) {
  if (is.null(recipe$evolution_log)) {
    stop("No evolution log found. Run evolve_features() with record = TRUE.")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required to serialize evolution log. Install it with: install.packages('jsonlite')")
  }

  json_data <- viewer_to_json(
    recipe$evolution_log,
    pretty = TRUE,
    is_full_log = TRUE,
    escape_script = TRUE
  )
  html <- .load_viewer_template()
  html <- .inject_evolution_json(html, json_data)

  out_path <- tempfile(fileext = ".html")
  writeBin(charToRaw(paste0(enc2utf8(html), "\n")), out_path)
  if (isTRUE(getOption("evoFE.viewer_browse", interactive()))) {
    tryCatch(
      with_preserved_seed(utils::browseURL(out_path)),
      error = function(e) NULL
    )
  }
  invisible(out_path)
}
