# Context: CRAN & Dependency Lightweighting

## CRAN Release Requirements `[Confirmed]`

### Clean Check Standard (`0/0/0`)
- **Invariant**: Every release build must pass `R CMD check --as-cran` with:
  - **0 Errors**
  - **0 Warnings**
  - **0 Notes**

### Dependency Lightweighting Strategy `[Confirmed]`
- **Rationale**: Excessive `Imports` increase installation failures, maintenance burden, and CRAN check vulnerabilities (if an imported package is archived or updated with breaking changes).
- **Guidelines**:
  - Prefer moving specialized modeling or tuning packages (`glmnet`, `keras3`, `mlr3mbo`) to `Suggests` guarded by `requireNamespace()` when possible, or keep `Imports` minimal.
  - Avoid importing meta-packages or legacy stacks (`mlr`, `ParamHelpers`, `smoof`, `emoa`).

---

## Migration History

### `mlrMBO` Removal (evoFE 0.4.0) `[Confirmed]`
- **From**: `mlrMBO`, `ParamHelpers`, `smoof`, `lhs` in `Imports`; `mlr`, `DiceKriging`, `randomForest`, `emoa` in `Suggests`.
- **To**: `paradox`, `bbotk`, `mlr3mbo` in `Imports`.
- **Result**: Net reduction of hard imports and optional suggests, removing unmaintained legacy packages while keeping `make_tunable()` public API identical.
