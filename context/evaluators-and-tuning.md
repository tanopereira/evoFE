# Context: Evaluators & Tuning

## Model Evaluator Architecture

### Single-Thread Enforcement Inside Evaluators `[Confirmed]`
- **Rationale**: When evaluating populations during evolutionary search, `evoFE` parallelizes evaluations across individuals or CV folds. If underlying model trainers (`lightgbm`, `xgboost`) launch multi-threaded OpenMP training loops simultaneously (`nthreads > 1`), CPU thread contention causes massive slowdowns and thread starvation.
- **Invariant**: Evaluators MUST restrict internal trainer threads (`nthread = 1` or `num_threads = 1`) unless explicit top-level configuration overrides it for single-model evaluations.

### CV Fold Isolation & Target Leakage `[Confirmed]`
- **Rationale**: Model fitness MUST be evaluated using cross-validation or train/val splits. Evaluators must never allow target encoding, scaling, or imputation parameters computed on validation splits to leak into training data.
- **Invariant**: Any stateful operation must be fit on `train_data` only, and subsequently transformed on `val_data`.

---

## Bayesian Optimization & Tuning Architecture

### Migration: `mlrMBO` → `mlr3mbo` `[Confirmed]`
- **Rationale**: `mlrMBO`, `ParamHelpers`, `smoof`, and `lhs` are legacy/maintenance-only packages. `mlr3mbo`, `paradox`, and `bbotk` represent the modern, actively maintained mlr3 ecosystem.
- **Eliminated Dependencies**: Removed `mlr`, `randomForest`, `DiceKriging`, `ParamHelpers`, `smoof`, and `emoa`.
- **Public API Contract**: `make_tunable(base_model_name, param_ranges, tuner_name)` signature remains 100% backward-compatible. Users do not need to change code when tuning models via `lightgbm_mbo` or custom tuners.

### Deprecated Control Parameters `[Confirmed]`
- **`mbo_infill_opt = "ea"`**: In `mlrMBO`, the `"ea"` option required the `emoa` package for integer parameter spaces. In `mlr3mbo`, infill optimization is handled internally via `AcqOptimizer`. Passing `"ea"` triggers a deprecation warning and defaults silently to standard focus search. **Do not restore `emoa` support.**

---

## Failed Approaches Log

### [Failed Approach] Multi-Threaded LightGBM + Parallel Fold Evaluation
- **Attempt**: Setting `num_threads = 4` inside `lightgbm` evaluator while running 4 CV folds in parallel via `future.apply`.
- **Outcome**: OpenMP thread contention caused 3x slower execution compared to single-threaded model evaluation inside parallel fold workers.
- **Rule**: Keep model trainers single-threaded whenever top-level execution is parallelized.
