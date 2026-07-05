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
  if (is.null(port)) {
    port <- httpuv::randomPort()
  }

  ws_connection <- NULL
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

  server <- httpuv::startServer("127.0.0.1", port,
    list(
      call = function(req) {
        list(
          status = 200L,
          headers = list("Content-Type" = "text/html"),
          body = html_content
        )
      },
      onWSOpen = function(ws) {
        ws_connection <<- ws
      }
    )
  )

  url <- sprintf("http://127.0.0.1:%d", port)

  list(
    url = url,
    server = server,
    get_connection = function() {
      ws_connection
    },
    send = function(data) {
      if (!is.null(ws_connection)) {
        json <- jsonlite::toJSON(data, auto_unbox = TRUE)
        ws_connection$send(json)
      }
      # service httpuv loops to process websocket buffer
      httpuv::service(100)
    },
    stop = function() {
      httpuv::stopServer(server)
    }
  )
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
