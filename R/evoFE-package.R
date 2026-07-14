#' @keywords internal
"_PACKAGE"

#' evoFE: Evolutionary Feature Engineering
#'
#' \pkg{evoFE} automates tabular feature engineering using a genetic algorithm
#' inspired by genetic programming.  Starting from raw input features the
#' package evolves candidate transformation recipes through selection, crossover,
#' and mutation, evaluating fitness via cross-validation or train/validation
#' splits with gradient-boosted tree models (LightGBM or XGBoost), linear
#' models (\pkg{glmnet}), or deep-learning models (\pkg{keras3}).
#'
#' The main entry point is \code{\link{evolve_features}}, which returns an
#' \code{evo_recipe} S3 object that can be \code{predict}ed on new data.
#'
#' \strong{Key subsystems}
#' \describe{
#'   \item{Transformer library}{42 built-in transformers spanning arithmetic,
#'     supervised encoding, group-by aggregations, dimensionality reduction, and
#'     manifold/graph learning.  Register custom transformers via
#'     \code{\link{register_transformer}}.}
#'   \item{Island model}{Optionally partition evolution into independent
#'     sub-populations (islands) with periodic recipe-level and gene-level
#'     migration.  Configured via the \code{islands}, \code{migration_interval},
#'     \code{migration_rate}, \code{gene_migration_prob},
#'     \code{row_split_islands}, and \code{per_island_validation} parameters of
#'     \code{\link{evolve_features}}.}
#'   \item{Hybrid Active Feature Mask}{Each individual carries an active mask
#'     over the raw input columns so that the genetic search simultaneously
#'     selects which original features to include and what transformations to
#'     apply.  Mask mutations are controlled by \code{raw_toggle_prob},
#'     \code{recalculate_mask_prob}, and \code{mask_temp_factor}.}
#'   \item{Bayesian hyperparameter tuning}{Wrap any registered evaluator in an
#'     \pkg{mlr3mbo} Bayesian optimisation loop via \code{\link{make_tunable}}.}
#'   \item{Live evolution viewer}{Enable real-time visualisation of the
#'     evolutionary search in a browser via \code{record = TRUE} in
#'     \code{\link{evolve_features}}.}
#' }
#'
#' @section Package Options:
#'
#' The following options can be set via \code{options()} to control package
#' behaviour at runtime:
#'
#' \describe{
#'   \item{\code{evoFE.verbose}}{(integer/logical) Verbosity level.
#'     \code{0} or \code{FALSE} for silent; \code{1} or \code{TRUE} for normal;
#'     \code{2} for detailed transformer-level logging.  Default \code{0}.}
#'   \item{\code{evoFE.threads}}{(integer) Default number of threads to use for
#'     parallel operations inside evaluators and clustering transformers.
#'     Default \code{2}.}
#'   \item{\code{evoFE.max_clustering_size}}{(integer) Maximum number of unique
#'     training rows used when fitting clustering/manifold transformers (Genie,
#'     Lumbermark, UMAP, MST, Deadwood).  Excess rows are randomly downsampled
#'     before fitting.  Set to \code{0} or \code{NULL} to disable downsampling.
#'     Default \code{5000}.}
#'   \item{\code{evoFE.redundancy_cor_threshold}}{(numeric in (0, 1]) Pearson
#'     correlation threshold above which a newly generated feature column is
#'     considered a near-duplicate of an existing column and rejected.  Lower
#'     values impose stricter redundancy pruning.  Set to \code{1.0} to
#'     disable.  Default \code{0.95}.}
#'   \item{\code{evoFE.importance_threshold}}{(numeric >= 0) Normalised feature
#'     importance threshold below which a feature's weight is zeroed out before
#'     importance-guided mutation.  Default \code{0.001}.}
#' }
#'
#' @name evoFE-package
NULL
