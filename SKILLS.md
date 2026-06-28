# Migration Plan: mlrMBO → mlr3mbo

Migrate the Bayesian Optimisation backend from the legacy `mlrMBO` / `ParamHelpers` / `smoof` / `lhs` stack to the actively maintained `mlr3mbo` / `paradox` / `bbotk` stack.

---

## 1. Motivation

| | Legacy (current) | mlr3 (target) |
|:---|:---|:---|
| **Packages** | `mlrMBO`, `ParamHelpers`, `smoof`, `lhs` (4) | `mlr3mbo`, `paradox`, `bbotk` (3) |
| **Status** | Maintenance-only (bug fixes) | Active development (v1.1.1, April 2026) |
| **Parameter types** | Manual `makeNumericParam` / `makeIntegerParam` / `makeDiscreteParam` | Declarative `p_dbl()` / `p_int()` / `p_fct()` |
| **Objective wrapper** | `smoof::makeSingleObjectiveFunction` | `bbotk::ObjectiveRFun` |
| **Surrogate fallback** | Manual Kriging→RF via `mlr::makeLearner` + `randomForest` | Built-in surrogate management |
| **Initial design** | `ParamHelpers::generateDesign` + `lhs::maximinLHS` | `bbotk` generates designs internally |
| **Infill strategy** | `setMBOControlInfill(opt = "focussearch"\|"ea")` | Configurable via `AcqOptimizer` |
| **Optional deps eliminated** | — | `mlr`, `randomForest`, `emoa`, `DiceKriging` |

**Key benefits:**
- Eliminates **4 hard imports** and **4 optional suggests** from the dependency tree.
- Removes the hand-coded Kriging→RF surrogate fallback logic entirely.
- Drops the `emoa` workaround for integer-only parameter spaces.
- Aligns with the ecosystem the mlr team actively maintains and tests.

---

## 2. Scope — Affected Files

The BO code is isolated to exactly **2 source files** plus package metadata. No vignettes, tests, or other source files reference the legacy stack.

### Source files

| File | Role | Changes |
|:---|:---|:---|
| `R/make_tunable.R` (295 lines) | Generic `make_tunable()` wrapper | Rewrite BO internals; keep public API identical |
| `R/tuners.R` (226 lines) | Hard-coded `lightgbm_mbo` evaluator | Rewrite BO internals; keep public API identical |

### Package metadata

| File | Changes |
|:---|:---|
| `DESCRIPTION` | Replace `Imports` entries; update `Suggests` |
| `NAMESPACE` | Auto-updated by roxygen2 after `@importFrom` changes |

### Files explicitly NOT affected

- `R/evolve.R`, `R/evaluate.R`, `R/individual.R`, `R/transformers.R`, `R/model_registry.R`, `R/predict.R`, `R/s3.R`, `R/population.R`, `R/model.R`, `R/evoFE-package.R`, `R/zzz.R`
- `vignettes/evoFE.Rmd` — no MBO references
- `tests/testthat/test-core.R` — no MBO/tuner tests (yet)

---

## 3. Public API — No Breaking Changes

The public interface stays identical. Users will not need to change any code.

### `make_tunable()` signature (unchanged)

```r
make_tunable(base_model_name, param_ranges, tuner_name = paste0(base_model_name, "_mbo"))
```

### `...` control parameters passed through evaluators (unchanged)

| Parameter | Type | Default | Notes |
|:---|:---|:---|:---|
| `mbo_iters` | integer | 5 | BO iterations |
| `mbo_init_design` | integer | 8 | Initial LHS design size |
| `mbo_folds` | integer | 3 | Internal CV folds |
| `mbo_infill_opt` | character | `"focussearch"` | **Deprecate** `"ea"` option (see §4.5) |
| `best_params` | list | `NULL` | Seed previous best |

---

## 4. Implementation Steps

### 4.1 Update `DESCRIPTION`

```diff
 Imports:
     data.table,
     lightgbm,
     xgboost,
     stats,
     digest,
     uwot,
     quitefastmst,
     genieclust,
-    ParamHelpers,
-    lhs,
-    mlrMBO,
-    smoof
+    paradox,
+    bbotk,
+    mlr3mbo
 Suggests:
     glmnet,
     RhpcBLASctl,
     testthat,
     knitr,
     rmarkdown,
     lumbermark,
     deadwood,
-    mlr,
     keras3,
-    DiceKriging,
-    randomForest,
-    emoa
```

> **Decision point:** Alternatively, move `paradox`, `bbotk`, `mlr3mbo` to `Suggests` instead of `Imports`, guarded by `requireNamespace()` in the tuner functions. This is already the pattern used today (lines 139–142 of `make_tunable.R`). Doing so means users who never call `make_tunable()` or `*_mbo` evaluators won't need to install the BO stack at all. The trade-off is losing the `@importFrom` roxygen declarations and using `::` everywhere instead.

### 4.2 Rewrite `R/make_tunable.R` — parameter space construction

**Before** (lines 115–129):
```r
# ParamHelpers parameter set
make_param <- function(name, def) {
  if (def$type == "numeric") {
    ParamHelpers::makeNumericParam(name, lower = def$lower, upper = def$upper)
  } else if (def$type == "integer") {
    ParamHelpers::makeIntegerParam(name, lower = def$lower, upper = def$upper)
  } else if (def$type == "discrete") {
    ParamHelpers::makeDiscreteParam(name, values = def$values)
  }
}
ps <- do.call(ParamHelpers::makeParamSet,
              lapply(names(tunable_defs), function(n) make_param(n, tunable_defs[[n]])))
```

**After:**
```r
# paradox parameter set
make_param <- function(name, def) {
  if (def$type == "numeric") {
    paradox::p_dbl(lower = def$lower, upper = def$upper)
  } else if (def$type == "integer") {
    paradox::p_int(lower = def$lower, upper = def$upper)
  } else if (def$type == "discrete") {
    paradox::p_fct(levels = def$values)
  }
}
param_list <- lapply(names(tunable_defs), function(n) make_param(n, tunable_defs[[n]]))
names(param_list) <- names(tunable_defs)
domain <- do.call(paradox::ps, param_list)
```

### 4.3 Rewrite `R/make_tunable.R` — objective, optimisation, and result extraction

**Before** (lines 210–267):
```r
# smoof objective
obj_fun <- smoof::makeSingleObjectiveFunction(
  name = paste0(base_model_name, "_mbo_tuning"),
  fn = fn, par.set = ps, has.simple.signature = FALSE, minimize = TRUE
)
# mlrMBO control
control <- mlrMBO::makeMBOControl()
control <- mlrMBO::setMBOControlTermination(control, iters = mbo_iters)
control <- mlrMBO::setMBOControlInfill(control, opt = mbo_infill_opt)
# LHS initial design
design <- ParamHelpers::generateDesign(n = mbo_init_design, par.set = ps, fun = lhs::maximinLHS)
# ... best_params seeding ...
# Run with Kriging → RF fallback
mbo_res <- tryCatch({
  suppressWarnings(mlrMBO::mbo(obj_fun, design = design, control = control, show.info = verbose))
}, error = function(e) {
  rf_surrogate <- mlr::makeLearner("regr.randomForest", predict.type = "se")
  suppressWarnings(mlrMBO::mbo(obj_fun, design = design, learner = rf_surrogate,
                               control = control, show.info = verbose))
})
best_hyperparams <- mbo_res$x
```

**After:**
```r
# bbotk objective — fn must accept a data.table of candidates and return a data.table of results
obj_fn_bbotk <- function(xdt) {
  scores <- vapply(seq_len(nrow(xdt)), function(i) {
    fn(as.list(xdt[i, ]))
  }, numeric(1))
  data.table::data.table(y = scores)
}
codomain <- paradox::ps(y = paradox::p_dbl(tags = "minimize"))
objective <- bbotk::ObjectiveRFunDt$new(
  fun = obj_fn_bbotk, domain = domain, codomain = codomain
)

# Optimisation instance with total budget = initial design + BO iterations
instance <- bbotk::OptimInstanceBatchSingleCrit$new(
  objective = objective,
  terminator = bbotk::trm("evals", n_evals = mbo_init_design + mbo_iters)
)

# Seed initial design (including best_params if provided)
init_design <- paradox::generate_design_lhs(domain, n = mbo_init_design)$data
if (!is.null(best_params)) {
  req_params <- names(tunable_defs)
  if (all(req_params %in% names(best_params))) {
    best_dt <- data.table::as.data.table(best_params[req_params])
    init_design <- rbind(best_dt, init_design)
  }
}

# Run MBO — surrogate fallback is handled internally by mlr3mbo
optimizer <- mlr3mbo::opt("mbo")
optimizer$optimize(instance)

best_hyperparams <- as.list(instance$result[, names(domain$params()), with = FALSE])
```

> **Note:** The entire Kriging→RF fallback block (lines 251–265 of `make_tunable.R`, lines 161–175 of `tuners.R`) is **deleted**. `mlr3mbo` handles surrogate selection and error recovery internally.

### 4.4 Rewrite `R/tuners.R` — same pattern

Apply the identical transformation to the hard-coded `lightgbm_mbo` evaluator. The structure mirrors `make_tunable.R`:

1. Replace `ParamHelpers::makeParamSet(...)` → `paradox::ps(...)`
2. Replace `smoof::makeSingleObjectiveFunction(...)` → `bbotk::ObjectiveRFunDt$new(...)`
3. Replace `mlrMBO::mbo(...)` → `optimizer$optimize(instance)`
4. Replace `mbo_res$x` → `instance$result`
5. Delete the Kriging→RF `tryCatch` fallback block
6. Delete the `emoa` integer-space workaround (lines 130–135)

### 4.5 Deprecate `mbo_infill_opt` parameter

The `"ea"` option required the `emoa` package and had a special workaround for integer-only spaces. In `mlr3mbo`, infill optimisation is configured differently (via `AcqOptimizer`).

**Strategy:**
- Keep `mbo_infill_opt` in the function signature for backward compatibility.
- If `"ea"` is passed, emit a deprecation warning and ignore it.
- If `"focussearch"` is passed (or default), proceed silently — it maps to `mlr3mbo`'s default focus search.
- Document the deprecation in `NEWS.md`.

```r
if (!missing(mbo_infill_opt) && mbo_infill_opt == "ea") {
  warning("mbo_infill_opt = 'ea' is deprecated and ignored. ",
          "mlr3mbo handles infill optimisation internally.", call. = FALSE)
}
```

### 4.6 Update roxygen2 tags

In `R/make_tunable.R`, replace:

```diff
-#' @importFrom mlrMBO makeMBOControl setMBOControlTermination setMBOControlInfill mbo
-#' @importFrom ParamHelpers makeParamSet makeNumericParam makeIntegerParam makeDiscreteParam generateDesign
-#' @importFrom smoof makeSingleObjectiveFunction
-#' @importFrom lhs maximinLHS
```

If using `Imports` in DESCRIPTION:
```diff
+#' @importFrom paradox ps p_dbl p_int p_fct generate_design_lhs
+#' @importFrom bbotk ObjectiveRFunDt OptimInstanceBatchSingleCrit trm
+#' @importFrom mlr3mbo opt
```

If using `Suggests` instead (recommended):
- Remove `@importFrom` entirely; use `package::function()` calls throughout.

### 4.7 Update documentation text

In `R/make_tunable.R`, update the `@description`:

```diff
-#' Wraps an existing registered model evaluator in a Bayesian Optimization tuning loop
-#' using the \pkg{mlrMBO} framework.
+#' Wraps an existing registered model evaluator in a Bayesian Optimization tuning loop
+#' using the \pkg{mlr3mbo} framework.
```

Update the `@details` to remove references to Kriging→RF fallback, since `mlr3mbo` handles this transparently.

### 4.8 Rebuild NAMESPACE

```bash
Rscript -e 'roxygen2::roxygenise()'
```

This will auto-generate the updated `NAMESPACE` file from the new `@importFrom` tags.

---

## 5. Verification Plan

### 5.1 R CMD check

```bash
R CMD build . && R CMD check evoFE_*.tar.gz --as-cran
```

Must pass with 0 errors, 0 warnings, 0 notes.

### 5.2 Functional tests

Run the existing test suite:

```bash
Rscript -e 'testthat::test_local()'
```

> **Note:** The existing `test-core.R` has no MBO/tuner tests. This migration should add at least a basic integration test for `make_tunable()`.

### 5.3 Manual smoke test — `make_tunable()`

```r
library(evoFE)

# Register a mock evaluator and make it tunable
register_evaluator(
  "mock_base",
  train_func = function(x_train, y_train, x_val = NULL, y_val = NULL,
                        task = "regression", ...) {
    args <- list(...)
    val_score <- 100 - abs(args$param_a - 4.5)
    list(
      model = list(val_score = val_score),
      predictions = if (!is.null(x_val)) rep(val_score, nrow(x_val)) else NULL
    )
  },
  predict_func = function(model, x_new, task, ...) rep(model$val_score, nrow(x_new))
)

make_tunable("mock_base", list(
  param_a = list(type = "numeric", lower = 1.0, upper = 8.0)
), tuner_name = "mock_tuned")

x <- matrix(rnorm(20), ncol = 2); colnames(x) <- c("x1", "x2")
fit <- train_model(x, rnorm(10), task = "regression", evaluator = "mock_tuned",
                   mbo_iters = 3, mbo_init_design = 4)
stopifnot(!is.null(fit$best_params))
cat("make_tunable() smoke test passed.\n")
```

### 5.4 Manual smoke test — `lightgbm_mbo` end-to-end

```r
library(evoFE)
data(mtcars)
df <- mtcars; df$am <- as.integer(df$am)

recipe <- evolve_features(
  data = df, target_col = "am", task = "classification",
  evaluator = "lightgbm_mbo", generations = 2, pop_size = 4,
  cv_folds = 2, verbose = TRUE, mbo_iters = 2, mbo_init_design = 4
)
stopifnot(inherits(recipe, "evo_recipe"))
cat("lightgbm_mbo end-to-end smoke test passed.\n")
```

### 5.5 Deprecation warning test

```r
# Verify the "ea" deprecation warning fires
tryCatch(
  train_model(x, rnorm(10), task = "regression", evaluator = "mock_tuned",
              mbo_iters = 2, mbo_init_design = 4, mbo_infill_opt = "ea"),
  warning = function(w) {
    stopifnot(grepl("deprecated", w$message))
    cat("Deprecation warning test passed.\n")
  }
)
```

---

## 6. NEWS.md Entry

```markdown
# evoFE 0.3.0

## Breaking Changes

* The `mbo_infill_opt = "ea"` option is deprecated and ignored with a warning.
  The `emoa` package is no longer needed.

## Enhancements

* Migrated Bayesian Optimisation backend from `mlrMBO` / `ParamHelpers` / `smoof`
  to `mlr3mbo` / `paradox` / `bbotk`. The public API (`make_tunable()`,
  `lightgbm_mbo`, `xgboost_mbo`) is unchanged.
* Removed `mlr`, `randomForest`, `DiceKriging`, and `emoa` from `Suggests`.
  The manual Kriging→RF surrogate fallback is no longer needed — `mlr3mbo`
  handles surrogate management internally.
* Reduced hard dependency count from 12 to 11 (net −1 package in `Imports`).
```

---

## 7. Checklist

- [ ] Update `DESCRIPTION` — swap imports/suggests
- [ ] Rewrite `R/make_tunable.R` — paradox params, bbotk objective, mlr3mbo optimizer
- [ ] Rewrite `R/tuners.R` — same transformation
- [ ] Delete Kriging→RF fallback blocks from both files
- [ ] Delete `emoa` integer-space workaround from both files
- [ ] Add `mbo_infill_opt = "ea"` deprecation warning
- [ ] Update roxygen `@importFrom` / `@description` / `@details` tags
- [ ] Run `roxygen2::roxygenise()` to regenerate `NAMESPACE` and `.Rd` files
- [ ] `R CMD check --as-cran` — 0/0/0
- [ ] Smoke test `make_tunable()` with mock evaluator
- [ ] Smoke test `lightgbm_mbo` end-to-end with mtcars
- [ ] Smoke test deprecation warning for `mbo_infill_opt = "ea"`
- [ ] Add basic `make_tunable()` integration test to `tests/testthat/`
- [ ] Write `NEWS.md` entry

---

*Last updated: 2026-06-28*
