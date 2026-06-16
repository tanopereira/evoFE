# evoFE 0.2.0

## Bug Fixes

* Fixed genotype corruption issue by treating failing or constant features as lethal mutations during training.
* Fixed `-Inf` fitness error by passing `allow_prune` to `evaluate_fitness` for pooled features.
* Fixed `NA` handling in quantile binning.
* Removed explicit `[Cache Hit]` printing from `evaluate_pop`.
* Fixed `is_logits` flag in xgboost custom metric evaluation and aligned implementation.
* Fixed `xgb_feval`: XGBoost `custom_metric` always receives raw logits, set `is_logits=TRUE` unconditionally.
* Fixed Laplace smoothing in TS-refinement formulation.

## Enhancements & Refactoring

* Improved TS-refinement evaluation metric and taboo threshold calculations.
* Set XGBoost default `min_child_weight=20` to match LightGBM `min_data_in_leaf` and prevent overconfident predictions.
* Removed `seed` parameter and `set.seed()` calls inside package code for CRAN compliance.
* Added input validation for `split_ids`: length check, label validation, and automatic `evaluation_strategy` switching to `"split"`.
* Display actual split proportions (computed from `split_ids`) in the evolution header.
* Optimised `.cluster_prep_x()` deduplication using `duplicated()` instead of `data.table` grouping for improved performance.
* Clamped infinite values and imputed `NA`/`NaN` in TS-refinement log-likelihood calculations.
* Added `globalVariables()` declarations for `data.table` NSE symbols to eliminate R CMD check NOTEs.

## Documentation

* Added CRAN status badge and updated installation instructions for CRAN release.
* Documented parameters for CRAN compliance.
* Reformatted Rd documentation to stay within 80-character line width.
* Improved `split_ids` documentation with usage examples.
* Added `allowed_transformers` parameter documentation to `mutate()` and `initialize_population()`.

# evoFE 0.1.0

* Initial CRAN submission.
