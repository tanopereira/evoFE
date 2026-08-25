## Test environments
* local macOS 15.4 (arm64), R 4.6.1
* win-builder (devel and release)
* R-hub v2:
  - Linux (ubuntu-latest), R-release and R-devel
  - Windows (windows-latest), R-release and R-devel
  - macOS (macos-13 and macos-15-arm64), R-release and R-devel

## R CMD check results

0 errors | 0 warnings | 0 notes

The only NOTE seen locally ("checking HTML version of manual") is an
artifact of the outdated local `tidy` binary and missing `V8` package;
the HTML manual validates cleanly on win-builder and R-hub.

## Changes in version 1.0.0

Resubmission: long-running island-model integration tests are now skipped
on CRAN (`skip_on_cran()`), bringing overall check time on
r-devel-windows below the 10-minute limit. All tests still run in local
and CI environments.

This is a major release of evoFE with significant new capabilities and robustness improvements:

### Island Model & Multi-Island Evolution
- Added island model architecture with configurable migration topologies (`topology_ring`, `topology_grid`, `topology_hypercube`, `topology_tiered`, `topology_torus`, `topology_complete`, `topology_custom`).
- Added configurable migration policies (`policy_push_uniform`, `policy_gibbs_push`, `policy_gibbs_pull`, `policy_tiered_admission`).
- Added Caruana ensemble selection (`ensemble_islands`) to greedily combine Pareto-diverse models across islands.

### Thread Safety & CRAN Compliance
- Added robust OpenMP and `data.table` thread setting and restoration in `evolve_features()` via top-level `on.exit()`, protecting against macOS `NA` thread reporting.
- Added defensive `requireNamespace()` checks for all optional and suggested dependencies.
- Added `testthat::skip_if_not_installed()` guards across all test suites.
- Added `@return` / `\value` roxygen documentation for all exported S3 methods and functions.

### New Features & Optimizations
- Added hybrid active feature masks with importance-guided sampling and dynamic mask mutation.
- Added Bayesian hyperparameter tuning wrapper (`make_tunable`).
- Added interactive real-time and post-hoc HTML evolution visualizer (`view()`).
- Added TS-refinement and calibrated regression metrics (`cal_rmse`, `cal_mae`).
