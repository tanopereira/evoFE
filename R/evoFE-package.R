#' @keywords internal
"_PACKAGE"

#' evoFE: Evolutionary Feature Engineering
#'
#' Automates feature engineering using evolutionary algorithms inspired by
#' genetic programming.
#'
#' @section Package Options:
#'
#' The following options can be set via \code{options()} to control package behavior:
#'
#' \describe{
#'   \item{\code{evoFE.verbose}}{(integer) Verbosity level. 0 for silent, 1 for normal, 2 for detailed.}
#'   \item{\code{evoFE.threads}}{(integer) Default number of threads to use for parallel operations.}
#'   \item{\code{evoFE.max_clustering_size}}{(integer) Maximum number of unique training rows to cluster. Default is 5000.}
#'   \item{\code{evoFE.redundancy_cor_threshold}}{(numeric) Correlation threshold above which one of two correlated features is dropped during pruning. Default is 0.95.}
#'   \item{\code{evoFE.importance_threshold}}{(numeric) Feature importance threshold below which a feature is dropped during pruning. Default is 0.001.}
#' }
#'
#' @name evoFE-package
NULL
