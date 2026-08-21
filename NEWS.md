# evoFE 1.0.0

## New Features

* **Untouched confirmation holdout (`holdout_frac`)**: With `evaluation_strategy = "cv"`, a stratified fraction of rows is excluded from the entire search and scored once after evolution. The result is reported as `holdout_fitness` and a new `search_gap` diagnostic (holdout minus validation score) on the `evo_recipe` object and in `summary()`, making search-level overfitting visible.
* **Leakage-safe CV fold strategies (`cv_strategy`)**: New options `"time"` (contiguous chronological blocks ordered by `time_col`, so validation always lies in the future of training) and `"group"` (all rows of a `group_col` entity stay in one fold, balanced greedy assignment) alongside the default `"random"`.
* **Multi-fidelity evaluation (`multi_fidelity`, `mf_sample_frac`, `mf_warmup_frac`)**: Optional cost reduction in which warm-up generations screen individuals on row-subsampled folds; the most promising half is re-evaluated at full fidelity before any selection decision, keeping fitness comparisons apples-to-apples.
* **New transformer `date_diff`**: Signed difference in days between two datetime columns.

## Breaking Changes

* The `mbo_infill_opt = "ea"` option is deprecated and ignored with a warning. The `emoa` package is no longer needed.

## Enhancements

* Migrated Bayesian Optimisation backend from `mlrMBO` / `ParamHelpers` / `smoof` / `lhs` to `mlr3mbo` / `paradox` / `bbotk`. The public API (`make_tunable()`, `lightgbm_mbo`, `xgboost_mbo`) remains fully unchanged.
* Removed legacy packages `mlr`, `randomForest`, `DiceKriging`, and `emoa` from Suggested dependencies. The manual Kriging-to-RandomForest surrogate fallback has been replaced by `mlr3mbo`'s transparent internal surrogate management.
* Reduced package dependencies in `Imports` from 12 to 11.
* Added support for a **Hybrid Island Population Model** in `evolve_features()` via parameters `islands`, `migration_interval`, `migration_rate`, and `gene_migration_prob`.
  * **Recipe-Level Migration**: Exchanges complete successful feature recipes between islands in a Ring topology to preserve co-adapted interactions. Guarded against edge cases where dynamic population sizes shrink below `migration_rate`.
  * **Gene-Level Migration (Injection)**: Periodically injects highly successful individual features (genes) from the ring neighbor's best individual into the local mutation pool. Genes diffuse naturally through the ring over successive migrations, creating a gradient of innovation where each island builds hierarchically on its neighbor's features.
  * **Independent Local Stagnation**: Dynamic population sizing and adaptive exploration rates are computed independently per island, allowing stagnant islands to expand and explore while active ones stay compact. Population growth is capped at 5x `pop_size` to prevent runaway expansion.
  * **Verbose Reporting**: Clear logging has been integrated to distinguish exactly which island and individual is being evaluated, including gene pool injection events.
* Added support for a **Hybrid Feature Mask Approach** for raw feature selection alongside evolved feature engineering:
  * **Sigmoid Active Mask Initialization**: Guides active feature masks of new individuals using baseline model importances centered at average feature importance.
  * **Smart Mask Mutation Operators**: Supports multi-column toggles scaled via a dynamic geometric distribution, single raw feature toggles with temperature-scaled importance probability, and complete mask recalculation.
  * **With-Replacement Multivariate Sampling**: Employs with-replacement sampling and deduplication for multivariate transformers to concentrate inputs around active, important features.
  * **Pruning Safety Floor Restoration**: Safeguards evaluations by automatically restoring raw features to meet the `min_active` floor if all genes are pruned during evaluation.
  * **Cache Hit Logging Reduction**: Silenced redundant `(Cache Hit)` prints for centroid distance components to reduce console verbosity.

# evoFE 0.2.0

## Bug Fixes

* Fixed CatBoost evaluator writing to disk by enforcing `allow_writing_files = FALSE` and redirecting any diagnostic files to `tempdir()`.
* Fixed genotype corruption issue by treating failing or constant features as lethal mutations during training.
* Fixed `-Inf` fitness error by passing `allow_prune` to `evaluate_fitness` for pooled features.
* Fixed `NA` handling in quantile binning.
* Removed explicit `[Cache Hit]` printing from `evaluate_pop`.
* Fixed `is_logits` flag in xgboost custom metric evaluation and aligned implementation.
* Fixed `xgb_feval`: XGBoost `custom_metric` always receives raw logits, set `is_logits=TRUE` unconditionally.
* Fixed Laplace smoothing in TS-refinement formulation.
* Fixed mutation bug: the `p` (power transform) and `q` (groupby_quantile) parameters were missing from mutation logic and are now actively mutated.
* Fixed multi-component UMAP genes to sample shared parameters once per transformer addition, ensuring consistency across components.

## Enhancements & Refactoring

* Optimized stateful clustering transformers: vectorized the fallback KNN distance search (3.8x speedup), integrated UMAP downsampling (7x speedup on fitting), and enabled UMAP prediction caching (1310x speedup on cache hits).
* Reduced memory footprint in cross-validation loops by removing redundant `data.table::copy()` calls.
* Hardened `.cluster_prep_x()` and `.cluster_knn_apply()` to safely handle edge-case configurations like vector-valued option inputs and NA distance matrices.
* Improved TS-refinement evaluation metric and taboo threshold calculations.
* Set XGBoost default `min_child_weight=20` to match LightGBM `min_data_in_leaf` and prevent overconfident predictions.
* Removed `seed` parameter and `set.seed()` calls inside package code for CRAN compliance.
* Added input validation for `split_ids`: length check, label validation, and automatic `evaluation_strategy` switching to `"split"`.
* Display actual split proportions (computed from `split_ids`) in the evolution header.
* Optimised `.cluster_prep_x()` deduplication using `duplicated()` instead of `data.table` grouping for improved performance.
* Clamped infinite values and imputed `NA`/`NaN` in TS-refinement log-likelihood calculations.
* Added `globalVariables()` declarations for `data.table` NSE symbols to eliminate R CMD check NOTEs.
* Integrated dynamic `n_neighbors` (Poisson mean 15) and `dens_scale` (uniform [0,1]) parameters into the UMAP transformer.
* Updated `gene_to_state_formula` to include all configuration parameters in the cache key to prevent collision.
* Replaced the flawed taboo threshold formula with a piecewise, hybrid scale-invariant approach for proper scaling across bounded and unbounded metrics.

## Documentation

* Added CRAN status badge and updated installation instructions for CRAN release.
* Documented parameters for CRAN compliance.
* Reformatted Rd documentation to stay within 80-character line width.
* Improved `split_ids` documentation with usage examples.
* Added `allowed_transformers` parameter documentation to `mutate()` and `initialize_population()`.
* Added missing `@param datetime_cols` documentation in `create_individual()` and `initialize_population()`.
* Created central package-level documentation accessible via `?evoFE` outlining all package options (`evoFE.redundancy_cor_threshold`, `evoFE.importance_threshold`, `evoFE.max_clustering_size`, `evoFE.threads`, and `evoFE.verbose`).
* Added `\donttest{}` `@examples` blocks to the internal-but-exported functions: `create_individual()`, `mutate()`, `crossover()`, and `tournament_select()`.

# evoFE 0.1.0

* Initial CRAN submission.
