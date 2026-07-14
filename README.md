# evoFE: Evolutionary Feature Engineering in R

[![CRAN status](https://www.r-pkg.org/badges/version/evoFE)](https://CRAN.R-project.org/package=evoFE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R-CMD-check](https://github.com/tanopereira/evoFE/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/tanopereira/evoFE/actions)

**evoFE** (Evolutionary Feature Engineering) is an R package that uses a genetic algorithm to automatically discover, combine, and optimize feature transformations for tabular datasets. Instead of manually engineering interaction terms, ratios, or binning strategies, `evoFE` searches the space of possible feature recipes to maximize the predictive performance of LightGBM, XGBoost, or other ML models.

The final output is a reusable **`evo_recipe`** object that can be easily applied to new data at prediction time.

---

## Features

* **Genetic Algorithm Optimization:** Searches the feature transformation space using selection, crossover, and mutation.
* **Hierarchical Chaining:** Evolved features can build on top of other proven features from previous generations (e.g., `log(ratio(x1, x2))`).
* **Hybrid Active Feature Mask:** The genetic search simultaneously selects which original raw features to include *and* what transformations to apply, guided by feature importances with temperature-scaled sigmoid sampling.
* **42 Built-in Transformers:** Arithmetic, group-by aggregations, supervised encodings, dimensionality reduction (PCA, SVD, UMAP), and manifold/graph learning (Genie, Lumbermark, MST, Deadwood).
* **Stateful Transformers:** Includes PCA, SVD, UMAP, Genie Clustering, Lumbermark Clustering, and Deadwood Anomaly Detection, all fit on training data and cached for efficiency.
* **Performance Caching:** Features are cached using matrix-hashing to avoid redundant computations (like KNN search or UMAP projections) during cross-validation folds.
* **Island Model:** Run independent sub-populations with periodic recipe-level and gene-level migration for broader exploration of the search space.
* **Flexible Evaluation:** Supports both Cross-Validation (`cv`) and stratified Train/Validation/Holdout Split (`split`) strategies.
* **Extensible Custom Registry:** Register user-defined transformers with `register_transformer()` or custom ML backends with `register_evaluator()`.
* **Bayesian Hyperparameter Tuning:** Wrap any registered evaluator in an `mlr3mbo` Bayesian optimization loop via `make_tunable()`.
* **Alternative & Custom Metrics:** Optimize for standard metrics (LogLoss, AUC, F1, MAE, TS-Refinement) or pass a custom fitness function.
* **Rich S3 Interface:** `print()`, `summary()`, and `plot()` to inspect and visualize the evolution.
* **Live Evolution Viewer:** A real-time browser dashboard that streams generation-by-generation progress when `record = TRUE`.

---

## Installation

You can install the released version of **evoFE** from [CRAN](https://CRAN.R-project.org/package=evoFE) with:

```R
install.packages("evoFE")
```

Alternatively, you can install the development version directly from [GitHub](https://github.com/tanopereira/evoFE):

```R
# Install devtools if you haven't already
# install.packages("devtools")

# Install evoFE from GitHub
devtools::install_github("tanopereira/evoFE", build_vignettes = TRUE)
```

### macOS OpenMP Configuration (Recommended)
Several of `evoFE`'s core transformers (like Genie and Lumbermark clustering) are implemented in C++ and parallelized using OpenMP. On macOS, R packages compile single-threaded by default. To enable multi-threading:

1. Install `libomp` via Homebrew:
   ```bash
   brew install libomp
   ```
2. Configure your `~/.R/Makevars` file to use OpenMP:
   ```make
   SHLIB_OPENMP_CFLAGS = -Xpreprocessor -fopenmp
   SHLIB_OPENMP_CXXFLAGS = -Xpreprocessor -fopenmp
   CPPFLAGS += -I/opt/homebrew/opt/libomp/include
   LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
   ```
3. Reinstall `quitefastmst`, `genieclust`, `lumbermark`, and `deadwood` from source:
   ```R
   install.packages(c("quitefastmst", "genieclust", "lumbermark", "deadwood"), type = "source")
   ```

---

## Quick Start

Here is a quick example using the `mtcars` dataset for a binary classification task:

```R
library(evoFE)

data(mtcars)
df <- mtcars
df$am <- as.integer(df$am) # target: 0 = automatic, 1 = manual

# Evolve features
set.seed(42)
recipe <- evolve_features(
  data = df,
  target_col = "am",
  task = "classification",
  evaluator = "xgboost",
  generations = 5,
  pop_size = 8,
  cv_folds = 3,
  verbose = TRUE
)

# View the winning recipe overview and detailed summary
print(recipe)
summary(recipe)

# Plot the evolution fitness curve
plot(recipe, type = "fitness")

# Engineer features on new data
engineered_df <- predict(recipe, df[1:5, ])

# Run predictions using the trained model
predictions <- predict_model(recipe, df[1:5, ])
```

---

## Supported Transformers

evoFE ships with **42 built-in transformers** that the genetic algorithm can select from during evolution.

| Category | Transformers |
| :--- | :--- |
| **Arithmetic** | `log`, `sqrt`, `reciprocal`, `power`, `displaced_log`, `add`, `subtract`, `multiply`, `divide`, `normalized_difference`, `log_ratio` |
| **Rank / Distribution** | `rank_transform` |
| **Group-by Aggregations** | `groupby_mean`, `groupby_sd`, `groupby_max`, `groupby_min`, `groupby_median`, `groupby_quantile`, `groupby_ratio`, `groupby_zscore` |
| **Supervised Encoding** | `target_encode`, `pooled_target_encode`, `target_encode_multiclass`, `woe_encode` |
| **Unsupervised Encoding & Binning** | `frequency_encode`, `one_hot_encode`, `concat`, `quantile_binning`, `quantile_binning_cat`, `log_binning`, `log_binning_cat`, `datetime_extract` |
| **Dimensionality Reduction** | `pca`, `truncated_svd`, `random_projection`, `umap` |
| **Manifold & Graph Learning** | `genie`, `genie_centroid_dist`, `umap_genie`, `lumbermark`, `lumbermark_centroid_dist`, `umap_lumbermark`, `mst_score`, `deadwood` |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
