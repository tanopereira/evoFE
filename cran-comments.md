## Test environments
* local macOS 15.4 (arm64), R 4.6.0
* win-builder (devel and release)
* R-hub v2:
  - Linux (ubuntu-latest), R-release and R-devel
  - Windows (windows-latest), R-release and R-devel
  - macOS (macos-13 and macos-15-arm64), R-release and R-devel

## R CMD check results

0 errors | 0 warnings | 0 notes

## Changes since 0.1.0

This is an update to the CRAN-published version 0.1.0. Key changes include:

### Bug Fixes
- Fixed genotype corruption, `-Inf` fitness errors, and `NA` handling in quantile binning.
- Fixed `is_logits` flag in XGBoost custom metric evaluation.
- Fixed Laplace smoothing in TS-refinement formulation.
- Clamped infinite values and imputed `NA`/`NaN` in TS-refinement log-likelihood calculations.

### Enhancements
- Added input validation for `split_ids` with automatic strategy switching.
- Optimised clustering deduplication using `duplicated()`.
- Set XGBoost default `min_child_weight=20` to match LightGBM defaults.
- Removed `seed` parameter and `set.seed()` calls inside package functions.

### Documentation
- Reformatted Rd documentation for 80-character line width compliance.
- Added detailed `\value` tags to all exported functions.
- Converted arXiv reference to DOI format.
