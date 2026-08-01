# Context: Genetic Algorithm & Feature Transformers

## Transformer Caching & Digest State `[Confirmed]`

### `xxhash64` + `md5` State Cache Keying
- **Rationale**: Feature transformations (especially computationally intensive ones like `uwot` UMAP, SVD/PCA, and `quitefastmst` MST clustering) can be repeatedly applied across candidate recipes in the population.
- **Implementation**: `apply_gene()` computes a data digest (`xxhash64`) on the target column combined with an `md5` formula string of the gene parameters. If the state exists in `state_cache`, the pre-computed fitted state is reused.

---

## Redundancy & Constant Feature Pruning `[Confirmed]`

### Guarding `skip_filtering` During Inference vs Candidate Mutation
- **Rationale**: During training/evolution, newly mutated candidate features that generate constant values (0 variance) or near-perfect correlation (`r >= 0.95` with existing numeric features) are pruned immediately to keep feature spaces lean and prevent collinearity.
- **Critical Invariant**: During inference (`predict()`, validation evaluation, or final model tournament), `skip_filtering = TRUE` MUST be passed. Correlation/variance thresholds depend on data splits; filtering features during prediction could drop a gene that the trained downstream model relies on, causing catastrophic inference failures.

---

## Transformer Primitive Rationales `[Confirmed]`

### `uwot` (UMAP Dimensionality Reduction)
- **Rationale**: Provides non-linear manifold embeddings for high-dimensional feature spaces. Must store fitted UMAP model in `gene$state` to transform new validation/test sets consistently.

### `quitefastmst` (Graph Minimum Spanning Tree Clustering)
- **Rationale**: Fast C++ backed MST clustering for feature distance graphs. Chosen over traditional `stats::hclust` due to speed and memory efficiency on high-dimensional feature tables.

---

## Failed Approaches Log

### [Failed Approach] Dynamic Gene Pruning During Validation/Predict
- **Attempt**: Applying correlation filtering (`r >= 0.95`) inside `apply_gene()` universally without checking if the call was for candidate generation or prediction.
- **Outcome**: Validation predictions failed because a feature that was valid on training split had low variance or high correlation on validation split, dropping the column and breaking `predict.lightgbm`.
- **Rule**: Never prune features during inference/prediction (`skip_filtering = TRUE`).
