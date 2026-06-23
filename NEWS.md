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
