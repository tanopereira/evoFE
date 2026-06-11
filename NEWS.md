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

## Documentation

* Added CRAN status badge and updated installation instructions for CRAN release.
* Documented parameters for CRAN compliance.

# evoFE 0.1.0

* Initial CRAN submission.
