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
    port <- httpuv::randomPort()
  }

  ws_connections <- list()
  history_buffer <- list()

  html_path <- system.file("viewer", "index.html", package = "evoFE")
  if (html_path == "") {
    # Fallback/development path
    html_path <- file.path(getwd(), "inst", "viewer", "index.html")
  }

  html_content <- if (file.exists(html_path)) {
    paste(readLines(html_path, warn = FALSE), collapse = "\n")
  } else {
    "<html><body><h1>Evolution Viewer Template Missing</h1></body></html>"
  }

  original_port <- port
  server <- NULL
  attempts <- 0
  max_attempts <- 5
  last_error <- NULL

  while (is.null(server) && attempts < max_attempts) {
    attempts <- attempts + 1
    if (attempts > 1 || is.null(port)) {
      port <- httpuv::randomPort()
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

      # Service httpuv event loop
      suppressWarnings(httpuv::service(100))
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

#' View the evolution history of an evo_recipe
#'
#' Opens an interactive HTML page in the browser to visualize the evolutionary
#' feature engineering process, either in real-time or post-hoc.
#'
#' @param recipe An \code{evo_recipe} object.
#' @param ... Additional arguments (not used).
#'
#' @export
view <- function(recipe, ...) {
  UseMethod("view")
}

#' @export
view.evo_recipe <- function(recipe, ...) {
  if (is.null(recipe$evolution_log)) {
    stop("No evolution log found. Run evolve_features() with record = TRUE.")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("Package 'jsonlite' is required to serialize evolution log. Install it with: install.packages('jsonlite')")
  }

  json_data <- jsonlite::toJSON(recipe$evolution_log, auto_unbox = TRUE, pretty = TRUE)
  html_path <- system.file("viewer", "index.html", package = "evoFE")
  if (html_path == "") {
    html_path <- file.path(getwd(), "inst", "viewer", "index.html")
  }

  if (!file.exists(html_path)) {
    stop("Evolution Viewer template not found at inst/viewer/index.html")
  }

  html <- paste(readLines(html_path, warn = FALSE), collapse = "\n")
  # Inject the logged JSON data into the template placeholder
  html <- sub("__EVOLUTION_DATA__", json_data, html, fixed = TRUE)

  out_path <- tempfile(fileext = ".html")
  writeLines(html, out_path)
  utils::browseURL(out_path)
  invisible(out_path)
}
