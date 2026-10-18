#ifndef EVOFE_TRANSFORMERS_FUSED_H
#define EVOFE_TRANSFORMERS_FUSED_H

#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include "utils.h"

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

// Evaluate Yeo-Johnson scalar transform given precomputed log(|x| + 1)
inline double eval_yj_from_log1p(
    double x,
    double log_abs_xp1,
    double lambda,
    bool pos_is_log,
    double inv_lambda,
    double two_m_lam,
    bool neg_is_log,
    double inv_two_m_lam) {
  if (x >= 0.0) {
    return pos_is_log ? log_abs_xp1 : (std::expm1(lambda * log_abs_xp1) * inv_lambda);
  } else {
    return neg_is_log ? (-log_abs_xp1) : (-std::expm1(two_m_lam * log_abs_xp1) * inv_two_m_lam);
  }
}

// Fit Yeo-Johnson power transformation via 1D Brent MLE with hoisted log invariants
inline void fit_yeo_johnson_impl(
    const double* x,
    int n,
    double& out_lambda,
    double& out_mean,
    double& out_sd) {
  out_lambda = 1.0;
  out_mean = 0.0;
  out_sd = 1.0;
  if (n <= 0) return;

  constexpr int kMaxFitSamples = 20000;
  const int stride = (n > kMaxFitSamples) ? (n / kMaxFitSamples) : 1;
  const int reserve_n = (stride > 1) ? (n / stride + 1) : n;

  std::vector<double> sub_x;
  std::vector<double> sub_log_xp1;
  sub_x.reserve(reserve_n);
  sub_log_xp1.reserve(reserve_n);

  double sum_signed_log = 0.0;
  double x_min = std::numeric_limits<double>::infinity();
  double x_max = -std::numeric_limits<double>::infinity();

  for (int i = 0; i < n; i += stride) {
    double val = x[i];
    if (std::isfinite(val)) {
      double cval = std::clamp(val, -1e15, 1e15);
      double lxp1 = std::log1p(std::abs(cval));
      sub_x.push_back(cval);
      sub_log_xp1.push_back(lxp1);
      sum_signed_log += (cval >= 0.0) ? lxp1 : -lxp1;
      if (cval < x_min) x_min = cval;
      if (cval > x_max) x_max = cval;
    }
  }

  const int m = static_cast<int>(sub_x.size());
  if (m < 2 || (x_max - x_min) <= 1e-12) {
    out_mean = (m > 0) ? sub_x[0] : 0.0;
    return;
  }

  const double mean_signed_log = sum_signed_log / static_cast<double>(m);

  auto nll_fn = [&](double lam) -> double {
    const bool pos_is_log = (std::abs(lam) <= 1e-6);
    const double inv_lam = pos_is_log ? 1.0 : (1.0 / lam);
    const double two_m_lam = 2.0 - lam;
    const bool neg_is_log = (std::abs(two_m_lam) <= 1e-6);
    const double inv_two_m_lam = neg_is_log ? 1.0 : (1.0 / two_m_lam);

    double w_mean = 0.0;
    double w_m2 = 0.0;
    for (int i = 0; i < m; ++i) {
      double y = eval_yj_from_log1p(
          sub_x[i], sub_log_xp1[i], lam,
          pos_is_log, inv_lam, two_m_lam, neg_is_log, inv_two_m_lam);
      double delta = y - w_mean;
      w_mean += delta / static_cast<double>(i + 1);
      w_m2 += delta * (y - w_mean);
    }
    double var = w_m2 / static_cast<double>(m);
    if (var <= 1e-30 || !std::isfinite(var)) return 1e30;
    return 0.5 * std::log(var) - (lam - 1.0) * mean_signed_log;
  };

  out_lambda = brent_minimize(nll_fn, -2.0, 2.0, 1e-5, 50);

  const bool pos_is_log = (std::abs(out_lambda) <= 1e-6);
  const double inv_lam = pos_is_log ? 1.0 : (1.0 / out_lambda);
  const double two_m_lam = 2.0 - out_lambda;
  const bool neg_is_log = (std::abs(two_m_lam) <= 1e-6);
  const double inv_two_m_lam = neg_is_log ? 1.0 : (1.0 / two_m_lam);

  double w_mean = 0.0;
  double w_m2 = 0.0;
  for (int i = 0; i < m; ++i) {
    double y = eval_yj_from_log1p(
        sub_x[i], sub_log_xp1[i], out_lambda,
        pos_is_log, inv_lam, two_m_lam, neg_is_log, inv_two_m_lam);
    double delta = y - w_mean;
    w_mean += delta / static_cast<double>(i + 1);
    w_m2 += delta * (y - w_mean);
  }
  out_mean = std::isfinite(w_mean) ? w_mean : 0.0;
  double sd_val = std::sqrt(w_m2 / static_cast<double>(m - 1));
  out_sd = (std::isfinite(sd_val) && sd_val > 1e-12) ? sd_val : 1.0;
}

// Single-pass fused Yeo-Johnson transform + standardization
inline void apply_yeo_johnson_impl(
    const double* x,
    int n,
    double lambda,
    double mean,
    double sd,
    double* out_ptr,
    int threads = 1) {
  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

  const double inv_sd = (std::isfinite(sd) && sd > 1e-12) ? (1.0 / sd) : 1.0;
  const bool pos_is_log = (std::abs(lambda) <= 1e-6);
  const double inv_lam = pos_is_log ? 1.0 : (1.0 / lambda);
  const double two_m_lam = 2.0 - lambda;
  const bool neg_is_log = (std::abs(two_m_lam) <= 1e-6);
  const double inv_two_m_lam = neg_is_log ? 1.0 : (1.0 / two_m_lam);

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double val = x[i];
    if (!std::isfinite(val)) {
      out_ptr[i] = 0.0;
      continue;
    }
    double cval = std::clamp(val, -1e15, 1e15);
    double lxp1 = std::log1p(std::abs(cval));
    double y = eval_yj_from_log1p(
        cval, lxp1, lambda,
        pos_is_log, inv_lam, two_m_lam, neg_is_log, inv_two_m_lam);
    out_ptr[i] = (y - mean) * inv_sd;
  }
}

// Fit shifted Box-Cox power transformation via 1D Brent MLE with hoisted log invariants
inline void fit_box_cox_impl(
    const double* x,
    int n,
    double& out_lambda,
    double& out_shift,
    double& out_mean,
    double& out_sd) {
  out_lambda = 1.0;
  out_shift = 0.0;
  out_mean = 0.0;
  out_sd = 1.0;
  if (n <= 0) return;

  double x_min = std::numeric_limits<double>::infinity();
  double x_max = -std::numeric_limits<double>::infinity();
  int finite_count = 0;
  for (int i = 0; i < n; ++i) {
    double val = x[i];
    if (std::isfinite(val)) {
      double cval = std::clamp(val, -1e15, 1e15);
      if (cval < x_min) x_min = cval;
      if (cval > x_max) x_max = cval;
      finite_count++;
    }
  }

  if (finite_count < 2 || (x_max - x_min) <= 1e-12) {
    out_mean = (finite_count > 0) ? x_min : 0.0;
    return;
  }

  if (x_min <= 0.0) {
    double margin = std::max(1e-4, 0.05 * (x_max - x_min));
    out_shift = -x_min + margin;
  } else {
    out_shift = 0.0;
  }

  constexpr int kMaxFitSamples = 20000;
  const int stride = (n > kMaxFitSamples) ? (n / kMaxFitSamples) : 1;
  const int reserve_n = (stride > 1) ? (n / stride + 1) : n;

  std::vector<double> sub_log_u;
  sub_log_u.reserve(reserve_n);
  double sum_log_u = 0.0;

  for (int i = 0; i < n; i += stride) {
    double val = x[i];
    if (std::isfinite(val)) {
      double cval = std::clamp(val, -1e15, 1e15);
      double u = std::clamp(cval + out_shift, 1e-6, 1e15);
      double lu = std::log(u);
      sub_log_u.push_back(lu);
      sum_log_u += lu;
    }
  }

  const int m = static_cast<int>(sub_log_u.size());
  if (m < 2) {
    out_mean = x_min + out_shift;
    return;
  }

  const double mean_log_u = sum_log_u / static_cast<double>(m);

  auto nll_fn = [&](double lam) -> double {
    const bool is_log = (std::abs(lam) <= 1e-6);
    const double inv_lam = is_log ? 1.0 : (1.0 / lam);

    double w_mean = 0.0;
    double w_m2 = 0.0;
    for (int i = 0; i < m; ++i) {
      double lu = sub_log_u[i];
      double y = is_log ? lu : (std::expm1(lam * lu) * inv_lam);
      double delta = y - w_mean;
      w_mean += delta / static_cast<double>(i + 1);
      w_m2 += delta * (y - w_mean);
    }
    double var = w_m2 / static_cast<double>(m);
    if (var <= 1e-30 || !std::isfinite(var)) return 1e30;
    return 0.5 * std::log(var) - (lam - 1.0) * mean_log_u;
  };

  out_lambda = brent_minimize(nll_fn, -2.0, 2.0, 1e-5, 50);

  const bool is_log = (std::abs(out_lambda) <= 1e-6);
  const double inv_lam = is_log ? 1.0 : (1.0 / out_lambda);
  double w_mean = 0.0;
  double w_m2 = 0.0;
  for (int i = 0; i < m; ++i) {
    double lu = sub_log_u[i];
    double y = is_log ? lu : (std::expm1(out_lambda * lu) * inv_lam);
    double delta = y - w_mean;
    w_mean += delta / static_cast<double>(i + 1);
    w_m2 += delta * (y - w_mean);
  }
  out_mean = std::isfinite(w_mean) ? w_mean : 0.0;
  double sd_val = std::sqrt(w_m2 / static_cast<double>(m - 1));
  out_sd = (std::isfinite(sd_val) && sd_val > 1e-12) ? sd_val : 1.0;
}

// Single-pass fused shifted Box-Cox transform + standardization
inline void apply_box_cox_impl(
    const double* x,
    int n,
    double lambda,
    double shift,
    double mean,
    double sd,
    double* out_ptr,
    int threads = 1) {
  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

  const double inv_sd = (std::isfinite(sd) && sd > 1e-12) ? (1.0 / sd) : 1.0;
  const bool is_log = (std::abs(lambda) <= 1e-6);
  const double inv_lam = is_log ? 1.0 : (1.0 / lambda);

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double val = x[i];
    if (!std::isfinite(val)) {
      out_ptr[i] = 0.0;
      continue;
    }
    double cval = std::clamp(val, -1e15, 1e15);
    double u = std::clamp(cval + shift, 1e-6, 1e15);
    double lu = std::log(u);
    double y = is_log ? lu : (std::expm1(lambda * lu) * inv_lam);
    out_ptr[i] = (y - mean) * inv_sd;
  }
}

} // namespace evofe

#endif // EVOFE_TRANSFORMERS_FUSED_H
