# evoFE: Evolutionary Feature Engineering in R

[![CRAN status](https://www.r-pkg.org/badges/version/evoFE)](https://CRAN.R-project.org/package=evoFE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R-CMD-check](https://github.com/tanopereira/evoFE/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/tanopereira/evoFE/actions)

**evoFE** (Evolutionary Feature Engineering) is an R package that uses a genetic algorithm to automatically discover, combine, and optimize feature transformations for tabular datasets. Instead of manually engineering interaction terms, ratios, or binning strategies, `evoFE` searches the space of possible feature recipes to maximize the predictive performance of LightGBM or XGBoost models.

The final output is a reusable **`evo_recipe`** object that can be easily applied to new data at prediction time.

---

## Features

* **Genetic Algorithm Optimization:** Searches the feature transformation space using selection, crossover, and mutation.
* **Hierarchical Chaining:** Evolved features can build on top of other proven features from previous generations (e.g., `log(ratio(x1, x2))`).
* **Stateful Transformers:** Includes PCA, SVD, UMAP, Genie Clustering, Lumbermark Clustering, and Deadwood Anomaly Detection.
* **Performance Caching:** Features are cached using matrix-hashing to avoid redundant computations (like $K$-NN search or UMAP projections) during cross-validation folds.
* **Flexible Evaluation:** Supports both Cross-Validation (`cv`) and stratified Train/Validation/Holdout Split (`split`) strategies.

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
recipe <- evolve_features(
  data = df,
  target_col = "am",
  task = "classification",
  evaluator = "xgboost",
  generations = 5,
  pop_size = 8,
  cv_folds = 3,
  seed = 42,
  verbose = TRUE
)

# View the winning recipe
cat("Best Recipe: ", individual_to_recipe_string(recipe$best_individual), "\n")
cat("Best Fitness: ", recipe$best_individual$fitness, "\n")

# Engineer features on new data
engineered_df <- predict(recipe, df[1:5, ])

# Run predictions using the trained model
predictions <- predict_model(recipe, df[1:5, ])
```

---

## Supported Transformers

| Category | Transformers |
| :--- | :--- |
| **Arithmetic** | `log`, `sqrt`, `reciprocal`, `add`, `subtract`, `multiply`, `divide`, `normalized_difference`, `log_ratio` |
| **Group-by Aggregations** | `groupby_mean`, `groupby_sd`, `groupby_max`, `groupby_min`, `groupby_ratio`, `groupby_zscore` |
| **Encoding & Binning** | `target_encode`, `frequency_encode`, `one_hot_encode`, `quantile_binning`, `log_binning`, `quantile_binning_cat`, `log_binning_cat` |
| **Dimensionality Reduction** | `pca`, `truncated_svd`, `random_projection`, `umap` |
| **Graph & Clustering** | `genie`, `lumbermark`, `mst_score`, `deadwood` |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
