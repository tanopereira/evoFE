# evoFE Context & Rationale Directory (`context/`)

This directory contains repo-native context files created under the **Keep the Why** architecture. It records technical rationale, statistical invariants, performance choices, and failed experiments for the `evoFE` package.

## Topic Index

| File | Domain & Purpose |
| :--- | :--- |
| **[evaluators-and-tuning.md](file:///Users/vero/git/evoFE/context/evaluators-and-tuning.md)** | Model evaluators (`lightgbm`, `xgboost`), Bayesian tuning (`mlr3mbo`), thread boundaries, CV fold isolation. |
| **[genetic-transformers.md](file:///Users/vero/git/evoFE/context/genetic-transformers.md)** | Evolutionary algorithm logic, transformer primitives, caching digests, data leakage prevention, redundancy pruning rules. |
| **[performance-invariants.md](file:///Users/vero/git/evoFE/context/performance-invariants.md)** | R memory allocation, GC pressure management, `data.table` copy vs reference semantics, empirical benchmarking rules. |
| **[cran-and-deps.md](file:///Users/vero/git/evoFE/context/cran-and-deps.md)** | Dependency management, `mlrMBO` -> `mlr3mbo` migration history, CRAN compliance checks (0/0/0). |

## Classification Tags

All technical rationale entries in these files are tagged with one of the following labels:
- **`[Confirmed]`**: Verified by unit tests, empirical benchmarks, or mathematical/statistical guarantees.
- **`[Inferred]`**: Architectural deductions based on codebase structure.
- **`[Failed Approach]`**: An attempt that was tested and discarded. **Do not re-implement.**
- **`[Superseded]`**: An old design that was intentionally replaced by a superior solution.
