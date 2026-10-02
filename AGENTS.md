# evoFE Agent Guidelines

## Performance & Optimization
- **Avoid recalculating operations, allocations, or data transformations inside internal loops if they can be hoisted outside or pre-allocated.**
- **Zero-Copy Input Paths**: Never re-sanitize (e.g. `std::isfinite`), re-standardize, or re-clamp matrices inside per-epoch or feature-ablation loops if they were already processed once outside the loop. Pass pre-standardized matrices directly by const reference or pointer.
- **Single-Pass Fusion**: Fuse sanitization, centering, scaling (via multiplication by precomputed `1.0 / s`), and clamping (`std::clamp`) into a single cache-contiguous loop over columns/elements rather than chaining multiple array allocations or multiple passes over memory.
- **Model Deserialization**: Model parameters and neural network weights are verified numbers; load them via direct memory copies (`Eigen::Map` / `memcpy`) rather than checking each float through scalar lambdas.
- **Workspace Pre-allocation**: Keep inner-loop allocations to zero. Allocate workspaces and buffers before entering iterative loops (epochs, batches, genetic generations, island migrations).

---

## Architectural Boundaries & Design Heuristics
*Constraints, not a checklist: when two conflict, choose the lowest future cost for this repo and state the rationale in the commit message.*

- **Separation of Concerns:**
  - **High-Performance Core (`src/`):** Numerical computation, matrix transformations, inner evaluation loops, and C++/Eigen kernels. Keep C++ routines pure, memory-fused, and free of R API allocation overhead inside tight loops.
  - **R Orchestration & Dispatch (`R/`):** Parameter validation, population management (`population.R`, `individual.R`), genetic search control (`evolve.R`), and S3 method dispatch (`s3.R`).
  - **Learner & Tuner Registry (`R/model_registry.R`, `R/tuners.R`, `R/make_tunable.R`):** Encapsulated backend integration (e.g. LightGBM, mlr3mbo). Callers interact with models and tuners through uniform interfaces, never through backend-specific ad-hoc branching.
  - **Presentation & Output (`R/s3.R` print/summary/plot):** Formats and displays results. Name the single concern of the file you edit.
- **Encapsulation & Dependency Direction:** Domain functions consume and return plain vectors, lists, or matrices. Never introduce UI, graphics, or foreign package dependencies inside core numerical/evolution loops.
- **Cohesion / Coupling:** A single rule or algorithm change should touch one module. If you need ad-hoc workarounds or circular references between R source files, the logic is in the wrong file.
- **DRY:** Grep before writing logic. One home per transformation formula, fitness penalty, threshold, or schema fact. Do not abstract coincidental similarity.
- **KISS / YAGNI:** Simplest working shape; pure functions over complex class abstractions; no speculative hooks, flags, or premature indirection.
- **Single Responsibility:** If you describe a function's purpose with "and", split it. Names communicate intent; comments explain *why* (never *what*).
- **Code Health:** Composition over inheritance · Law of Demeter · Fail fast (validate inputs at API boundaries, never swallow errors silently) · Optimize for deletion · Prefer boring, standard idioms.

---

## Hard Invariants (Never Violate)

### 1. One Owning Module per Domain
- Each domain's logic (evolutionary loop, feature transformations, surrogate tuning, evaluation/fitness scoring, C++ matrix ops) lives in its designated owning file/module with its roxygen documentation. All callers invoke the owner.
- Consolidate a scattered domain before adding to it. Extend existing domain modules; create a new one only when you can articulate why the existing owner cannot own it.
- A new rule or transformation goes into its owning module, not into the first caller or test that happens to need it.

### 2. Never Duplicate Logic
- Second use of existing logic (matrix scaling, fitness penalties, parameter validation, loss calculation):
  1. Move it to a shared helper/module (e.g. `R/utils.R` or `src/utils.h`).
  2. Switch the original caller and verify tests stay green with **zero** behavioral change.
  3. Only then build the new feature or use.
- Grep first for the expression (arithmetic, thresholds, sanitization logic) and identify all copies. The move must switch all copies or the commit must document why an exception was left.
- Creating a new helper next to existing copies is adding yet another duplicate.
- Preserve exact caller semantics (guards, numerical precision, edge-case clamps); any intentional behavior change belongs in its own separate commit.

### 3. No Algorithmic / Domain Logic in Presentation
- `print()`, `summary()`, and `plot()` methods only format and display data.
- **Never compute fitness scores, run transformations, or evaluate models inside S3 presentation methods.** Computing or mutating domain data in presentation code is a bug; move it to the owning evaluation or model module.

---

## How to Work & Verification Workflow

1. **Refactor first, then change:**
   - Always perform a behavior-preserving refactor with all tests passing before introducing functional changes.
   - Never combine refactoring and behavior modifications into a single unverifiable diff.
2. **Gates, not promises:**
   - A prompted "never" does not protect the codebase; rules must be verified with automated gates before completion.
   - Run verification commands:
     - `devtools::test()` or `testthat::test_file(...)` for unit tests.
     - `devtools::load_all()` to verify C++ compilation (`Rcpp` / `Eigen`) and symbol exports.
     - `R CMD check` / `devtools::check()` for package integrity and zero warnings/notes.
