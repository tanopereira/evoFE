#ifndef EVOFE_REALMLP_FUSED_H
#define EVOFE_REALMLP_FUSED_H

#include <cmath>
#include <algorithm>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace realmlp {

// Column statistics container for standardized scaling
struct ColumnStats {
  double mean;
  double inv_std;
  double std_val;
};

// 2-Pass SIMD Reduction Kernel: Computes column mean and sample standard deviation
// Pass 1: Vectorized sum with fused NaN/Inf replacement
// Pass 2: Vectorized sum of squared differences
// Variance floor: If s < 1e-5 or !isfinite(s), s is floored to 1.0 and inv_s = 1.0
// Strictly matching the behavior of rcpp_realmlp.cpp:353
inline ColumnStats compute_column_stats(const double* __restrict__ col, int N) {
  if (N <= 0) return { 0.0, 1.0, 1.0 };

  // Pass 1: SIMD Vectorized Sum
  double sum = 0.0;
  for (int i = 0; i < N; ++i) {
    double v = col[i];
    sum += std::isfinite(v) ? v : 0.0;
  }
  double mean = sum / N;

  // Pass 2: SIMD Vectorized Squared Differences
  double sum_sq = 0.0;
  for (int i = 0; i < N; ++i) {
    double v = col[i];
    double clean_v = std::isfinite(v) ? v : 0.0;
    double diff = clean_v - mean;
    sum_sq += diff * diff;
  }
  double var = sum_sq / std::max(1, N - 1);
  double s = std::sqrt(var);

  // Exact contract matching rcpp_realmlp.cpp:353
  if (s < 1e-5 || !std::isfinite(s)) {
    s = 1.0;
  }
  double inv_s = 1.0 / s;
  return { mean, inv_s, s };
}

// Single-Pass Fused Column Standardization:
// Centers, multiplies by precomputed inv_std, and clamps in-place into output buffer
inline void fuse_standardize_column(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int N,
    double mean,
    double inv_std,
    double clamp_min = -30.0,
    double clamp_max = 30.0) {
  for (int i = 0; i < N; ++i) {
    double v = src[i];
    double clean_v = std::isfinite(v) ? v : 0.0;
    double z = (clean_v - mean) * inv_std;
    dst[i] = std::clamp(z, clamp_min, clamp_max);
  }
}

// Single-Pass Fused Matrix Standardization:
// OpenMP-parallelized across columns (for D >= 4) with cache-contiguous column operations
inline void fuse_standardize_matrix(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int N, int D,
    const double* __restrict__ x_mean,
    const double* __restrict__ x_inv_std) {
  if (N <= 0 || D <= 0) return;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (D >= 2)
#endif
  for (int j = 0; j < D; ++j) {
    size_t offset = static_cast<size_t>(j) * N;
    const double* src_col = src + offset;
    double* dst_col = dst + offset;
    double m = x_mean[j];
    double inv_s = x_inv_std[j];

    for (int i = 0; i < N; ++i) {
      double v = src_col[i];
      double clean_v = std::isfinite(v) ? v : 0.0;
      double z = (clean_v - m) * inv_s;
      dst_col[i] = std::clamp(z, -30.0, 30.0);
    }
  }
}

} // namespace realmlp

#endif // EVOFE_REALMLP_FUSED_H
