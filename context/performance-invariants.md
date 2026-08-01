# Context: Performance & Memory Invariants

## R Memory & Garbage Collection (GC) Pressure `[Confirmed]`

### Empirical Benchmarking Requirement
- **Rule**: Never declare code "optimized" or "faster" based on theoretical assumptions (e.g., assuming vectorization or matrix conversion is automatically faster).
- **Benchmarking Protocol**: Always run empirical benchmarks comparing `HEAD` against `baseline` on realistic data sizes before committing performance changes.

### Pre-Allocation & Avoiding High-Frequency GC
- **Rationale**: In R, allocating new objects inside tight inner loops (e.g. repeated `as.list()`, `as.data.table()`, or `data.frame()` construction) triggers R's Garbage Collector frequently. GC pauses often account for >50% of total runtime in evolutionary searches.
- **Invariant**: Pre-allocate vectors/lists where size is known. Avoid repacking `data.table` rows inside inner loop iterations.

---

## Early Exits & Low-Level Pointer Operations `[Confirmed]`

### Preserving Algorithmic Early Exits
- **Rationale**: Never replace an early-exiting loop (`break` / `stop` on first condition match, such as first-duplicate detection) with a full matrix or batch operation unless empirical benchmarks prove the matrix operation is faster across all expected input sizes.

### `data.table` Reference Copies vs Object Copying
- **Rationale**: `data.table` provides fast shallow copying and in-place mutation (`:=`). Replacing `data.table::copy()` with full S3 object unpacking and repacking creates massive memory bloat during population evolution.

---

## Environment Cache Lookups `[Confirmed]`

### `get0()` vs `exists()` + `get()` Double-Lookup
- **Rationale**: The pattern `exists(key, envir = e, inherits = FALSE)` followed by `get(key, envir = e, inherits = FALSE)` performs two hash-table probes. `get0(key, envir = e, inherits = FALSE)` performs one probe and returns `NULL` on miss.
- **Benchmark**: `bench::mark()` on real MD5-keyed caches (sizes 20, 100, 500) at 50% hit rate → consistently **0.66–0.68x** (32–34% faster) across all cache sizes.
- **Invariant**: All environment cache lookups (fitness cache in `evaluate_pop()`, state cache in `apply_gene()`) MUST use `get0()` instead of `exists()` + `get()`.

---

## Type-Safe Apply Functions `[Confirmed]`

### `sapply` → `vapply` in Hot Paths
- **Rationale**: `sapply` performs runtime type-detection on the result, adding overhead. `vapply` skips this and is also type-safe.
- **Benchmark**: `bench::mark()` on real `evo_individual`-shaped objects → **0.60–0.89x** (11–40% faster) consistently across pop_size 10, 30, 100. Also reduces allocations (880B → 288B for pop=30).
- **Invariant**: Use `vapply(..., double(1))` or `vapply(..., character(1))` in generation-loop hot paths: `tournament_select()` fitness extraction, population fitness sorting in `evolve_features()`, island fitness sorting.

---

## Verbose Path `[Confirmed]`

### `supports_color()` — Cache Result Outside Inner Loops
- **Rationale**: `supports_color()` calls `Sys.getenv("TERM")`, `Sys.getenv("RSTUDIO")`, `.Platform$OS.type`, and `isatty(stdout())` on every individual evaluation when `verbose = TRUE`. Terminal environment does not change during a run.
- **Benchmark**: `bench::mark()` simulating the 3-call-per-individual verbose loop → **0.02–0.05x** (95–98% reduction) at pop_size 10, 30, 50. At pop=50: 197µs → 3.7µs per generation.
- **Invariant**: Call `supports_color()` exactly once at the start of `evaluate_pop()`, store result, and use pre-computed ANSI strings throughout the loop.

---

## Mutation Weight Computation `[Confirmed]`

### `weighted_sample()` — Vectorized Weight Extraction
- **Rationale**: The scalar `sapply` loop in `weighted_sample()` (called in `mutate()`) performs `c %in% names(importances)` per column inside an interpreted loop. Named-vector lookup is vectorizable.
- **Benchmark**: `bench::mark()` on real importance vectors (Boston 16 cols, diamonds 9 cols) → **0.12–0.17x** (5–8x faster) with zero additional allocation.
- **Invariant**: Replace the scalar `sapply` with vectorized named-vector indexing:
  ```r
  is_active <- !all_cols %in% all_raw | all_cols %in% active_raw
  vals      <- importances[all_cols]
  vals[is.na(vals) | !is.finite(vals)] <- baseline
  vals[!is_active] <- 0.0
  weights   <- exp(vals / temperature)
  weights[is.na(weights) | !is.finite(weights)] <- 0
  if (sum(weights) == 0) weights <- rep(1, length(weights))
  ```


---

## Failed Approaches Log

### [Failed Approach] Vectorizing Early-Exit Feature Deduplication
- **Attempt**: Replacing an early-exiting `for` loop that checks correlation feature-by-feature with a full matrix `stats::cor()` compute across all features at once.
- **Outcome**: For populations with 50+ candidate features, `stats::cor()` computed 1,225 pairwise correlations when the loop usually exited after 3 checks, making population generation 4x slower.
- **Rule**: Keep early-exit scalar loops when early termination probability is high.

### [Failed Approach] Pre-Caching Population Output Signatures for Duplicate Check
- **Attempt**: Pre-compute `get_out()` for all population members once per generation before calling `is_invalid_individual()`, to avoid recomputation per candidate.
- **Outcome**: `bench::mark()` on real `evo_individual` objects. Full-scan (no duplicate): identical performance (~1.00x). Early-exit (duplicate at position 1): **9–27x slower** because `lapply(pop, get_out_fn)` is paid unconditionally even when the first comparison already matches. The early-exit case dominates in practice since `is_invalid_individual()` returns immediately on the first match.
- **Rule**: Never pre-compute all signatures unconditionally. The current recompute-per-candidate with early exit is optimal.

### [Failed Approach] `rowSums(data.matrix(...))` for `add` and `multiply` Transformers
- **Attempt**: Replace `Reduce('+', lapply(input_cols, function(c) as.numeric(data[[c]])))` with `rowSums(data.matrix(data[, input_cols, with = FALSE]))` for the `add` transformer (and equivalent for `multiply`).
- **Outcome**: `bench::mark()` on diamonds (53k rows) and Boston (506 rows) at 2–5 input columns. `rowSums` was **5.9–14.5x slower** across all configurations. `data.matrix()` allocates a full copy of the selected columns as a dense numeric matrix (e.g. 4.17MB for diamonds with 2 cols) before `rowSums` runs, far exceeding the cost of the sequential `as.numeric()` + `Reduce` approach which allocates only one column at a time.
- **Rule**: Keep `Reduce('+', lapply(...))` for `add` and `Reduce('*', lapply(...))` for `multiply`. The column-at-a-time approach avoids the matrix allocation overhead that dominates at all practical input widths.
### [Failed Approach] Pre-Allocating `sorted_genes` in `topological_sort_genes()`
- **Attempt**: Replace `sorted_genes <- c(sorted_genes, list(gene))` with a pre-allocated `vector("list", length(genes))` and an integer index counter, to avoid O(n²) copy overhead.
- **Outcome**: `bench::mark()` on Boston column names at n_genes 2, 5, 10. At n=2 (most common): **1.15x slower**. At n=5: 1.04x (noise). At n=10: 0.97x (noise). The pre-allocation overhead outweighs any copy savings at the small gene counts typical in evoFE (usually 1–5 genes).
- **Rule**: Keep the growing-list approach. Gene counts are small enough that R's list copy overhead is negligible and the index-counter bookkeeping costs more.
