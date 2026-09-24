# evoFE Agent Guidelines

## Performance & Optimization
- **Avoid recalculating operations, allocations, or data transformations inside internal loops if they can be hoisted outside or pre-allocated.**
- **Zero-Copy Input Paths**: Never re-sanitize (e.g. `std::isfinite`), re-standardize, or re-clamp matrices inside per-epoch or feature-ablation loops if they were already processed once outside the loop. Pass pre-standardized matrices directly by const reference or pointer.
- **Single-Pass Fusion**: Fuse sanitization, centering, scaling (via multiplication by precomputed `1.0 / s`), and clamping (`std::clamp`) into a single cache-contiguous loop over columns/elements rather than chaining multiple array allocations or multiple passes over memory.
- **Model Deserialization**: Model parameters and neural network weights are verified numbers; load them via direct memory copies (`Eigen::Map` / `memcpy`) rather than checking each float through scalar lambdas.
- **Workspace Pre-allocation**: Keep inner-loop allocations to zero. Allocate workspaces and buffers before entering iterative loops (epochs, batches, genetic generations, island migrations).
