#ifndef EVOFE_TRANSFORMERS_FUSED_H
#define EVOFE_TRANSFORMERS_FUSED_H

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace evofe {

// Row-wise minimum across k columns of length n
inline void fused_row_min_impl(
    const std::vector<const double*>& ptrs,
    int k,
    int n,
    double* out_ptr,
    int threads = 1) {

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double m = std::numeric_limits<double>::infinity();
    bool any_valid = false;
    for (int j = 0; j < k; ++j) {
      double val = ptrs[j][i];
      if (std::isfinite(val)) {
        if (val < m) m = val;
        any_valid = true;
      }
    }
    out_ptr[i] = any_valid ? m : std::numeric_limits<double>::quiet_NaN();
  }
}

// Row-wise maximum across k columns of length n
inline void fused_row_max_impl(
    const std::vector<const double*>& ptrs,
    int k,
    int n,
    double* out_ptr,
    int threads = 1) {

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double m = -std::numeric_limits<double>::infinity();
    bool any_valid = false;
    for (int j = 0; j < k; ++j) {
      double val = ptrs[j][i];
      if (std::isfinite(val)) {
        if (val > m) m = val;
        any_valid = true;
      }
    }
    out_ptr[i] = any_valid ? m : std::numeric_limits<double>::quiet_NaN();
  }
}

// Geometric mean across k columns of length n: exp( (1/k) * sum(log(max(val, eps))) )
inline void fused_geometric_mean_impl(
    const std::vector<const double*>& ptrs,
    int k,
    int n,
    double eps,
    double* out_ptr,
    int threads = 1) {

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double sum_log = 0.0;
    int count = 0;
    for (int j = 0; j < k; ++j) {
      double val = ptrs[j][i];
      if (std::isfinite(val)) {
        double pos_val = std::clamp(val, eps, 1e15);
        sum_log += std::log(pos_val);
        count++;
      }
    }
    out_ptr[i] = (count > 0) ? std::exp(sum_log / count) : std::numeric_limits<double>::quiet_NaN();
  }
}

// Harmonic mean across k columns of length n: k / sum(1 / max(val, eps))
inline void fused_harmonic_mean_impl(
    const std::vector<const double*>& ptrs,
    int k,
    int n,
    double eps,
    double* out_ptr,
    int threads = 1) {

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double sum_recip = 0.0;
    int count = 0;
    for (int j = 0; j < k; ++j) {
      double val = ptrs[j][i];
      if (std::isfinite(val)) {
        double pos_val = std::clamp(val, eps, 1e15);
        sum_recip += 1.0 / pos_val;
        count++;
      }
    }
    out_ptr[i] = (count > 0 && sum_recip > 0.0) ? (static_cast<double>(count) / sum_recip) : std::numeric_limits<double>::quiet_NaN();
  }
}

// Single-pass Pythagorean imbalance across k columns of length n: AM - HM
inline void fused_pythagorean_imbalance_impl(
    const std::vector<const double*>& ptrs,
    int k,
    int n,
    double eps,
    double* out_ptr,
    int threads = 1) {

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double sum_arith = 0.0;
    double sum_recip = 0.0;
    int count = 0;
    for (int j = 0; j < k; ++j) {
      double val = ptrs[j][i];
      if (std::isfinite(val)) {
        double pos_val = std::clamp(val, eps, 1e15);
        sum_arith += pos_val;
        sum_recip += 1.0 / pos_val;
        count++;
      }
    }
    if (count > 0 && sum_recip > 0.0) {
      double am = sum_arith / count;
      double hm = static_cast<double>(count) / sum_recip;
      out_ptr[i] = am - hm;
    } else {
      out_ptr[i] = std::numeric_limits<double>::quiet_NaN();
    }
  }
}

// Relative rating: primary (ptrs[0]) minus mean of other columns (ptrs[1..k-1])
inline void fused_relative_rating_impl(
    const std::vector<const double*>& ptrs,
    int k,
    int n,
    double* out_ptr,
    int threads = 1) {

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double primary = ptrs[0][i];
    if (!std::isfinite(primary)) {
      out_ptr[i] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    double sum_other = 0.0;
    int count_other = 0;
    for (int j = 1; j < k; ++j) {
      double val = ptrs[j][i];
      if (std::isfinite(val)) {
        sum_other += val;
        count_other++;
      }
    }
    out_ptr[i] = (count_other > 0) ? (primary - (sum_other / count_other)) : 0.0;
  }
}

} // namespace evofe

#endif // EVOFE_TRANSFORMERS_FUSED_H
