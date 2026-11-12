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
    port <- getOption("evoFE.viewer_port", NULL)
  }
  if (is.null(port)) {
    port <- with_preserved_seed(httpuv::randomPort())
  }

  ws_connections <- list()
  history_buffer <- list()

  html_content <- .load_viewer_template(
    missing_fallback = "<html><body><h1>Evolution Viewer Template Missing</h1></body></html>"
  )

  original_port <- port
  server <- NULL
  attempts <- 0
  max_attempts <- 5
  last_error <- NULL

  while (is.null(server) && attempts < max_attempts) {
    attempts <- attempts + 1
    if (attempts > 1 || is.null(port)) {
      port <- with_preserved_seed(httpuv::randomPort())
    }
    server <- tryCatch({
      httpuv::startServer("127.0.0.1", port,
        list(
          call = function(req) {
            list(
              status = 200L,
              headers = list("Content-Type" = "text/html"),
              body = html_content
            )
          },
          onWSOpen = function(ws) {
            ws_connections[[length(ws_connections) + 1]] <<- ws
            # Instantly replay all buffered history to reconnected client
            for (msg in history_buffer) {
              tryCatch({
                ws$send(msg)
              }, error = function(e) {
                # Ignore transient send error during replay
              })
            }
          }
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
      if (length(ws_connections) > 0) ws_connections[[1]] else NULL
    },
    send = function(data) {
      json <- jsonlite::toJSON(data, auto_unbox = TRUE)
      history_buffer[[length(history_buffer) + 1]] <<- json

      alive_conns <- list()
      for (ws in ws_connections) {
        err <- tryCatch({
          ws$send(json)
          FALSE
        }, error = function(e) TRUE)
        if (!err) {
          alive_conns[[length(alive_conns) + 1]] <- ws
        }
      }
      ws_connections <<- alive_conns

      # Service httpuv event loop (1ms timeout)
      suppressWarnings(httpuv::service(1))
    },
    stop = function() {
      suppressWarnings(httpuv::stopServer(server))
      if (exists("active_viewer", envir = .viewer_env) && identical(.viewer_env$active_viewer$server, server)) {
        tryCatch({
          rm("active_viewer", envir = .viewer_env)
        }, error = function(e) {
          # Ignore
        })
      }
    }
  )

  .viewer_env$active_viewer <- viewer
  viewer
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
  paste(readLines(html_path, warn = FALSE), collapse = "\n")
}

viewer_send <- function(viewer, type, data) {
  if (is.null(viewer)) {
    return(invisible(NULL))
  }
  viewer$send(list(type = type, data = data))
  invisible(NULL)
}

.serialize_genes_for_log <- function(genes, importances = NULL, sort_by_importance = TRUE) {
  if (length(genes) == 0) {
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
  if (isTRUE(sort_by_importance) && length(serialized_genes) > 0) {
    gene_imps <- sapply(serialized_genes, function(x) x$importance)
    serialized_genes <- serialized_genes[order(gene_imps, decreasing = TRUE)]
  }
  serialized_genes
}

.build_sample_preview <- function(ind, ref_dt, max_rows = 5L) {
  head_dt <- utils::head(ref_dt, max_rows)
  res_sample <- tryCatch(
    {
      with_preserved_seed(
        apply_individual(ind, head_dt, NULL, NULL, state_cache = NULL, allow_prune = FALSE)
      )
    },
    error = function(e) list(train = head_dt)
  )
  best_dt <- res_sample$train
  best_list <- lapply(names(best_dt), function(col) {
    val <- best_dt[[col]]
    if (is.numeric(val)) round(val, 4) else as.character(val)
  })
  names(best_list) <- names(best_dt)
  best_list
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

  json_data <- jsonlite::toJSON(recipe$evolution_log, auto_unbox = TRUE, pretty = TRUE)
  html <- .load_viewer_template()
  # Inject the logged JSON data into the template placeholder
  parts <- strsplit(html, "__EVOLUTION_DATA__", fixed = TRUE)[[1]]
  html <- paste0(parts[1], json_data, if (length(parts) > 1) parts[2] else "")

  out_path <- tempfile(fileext = ".html")
  writeLines(html, out_path)
  if (interactive()) {
    utils::browseURL(out_path)
  }
  invisible(out_path)
}
