#ifndef EVOFE_METRICS_CALIBRATION_H
#define EVOFE_METRICS_CALIBRATION_H

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace evofe {

static constexpr double MIN_LOG_PROB = -34.538776394910684; // std::log(1e-15)

// 1D Brent minimization on interval [a, b]
template <typename Func>
inline double brent_minimize(Func&& f, double a, double b, double tol = 1e-5, int max_iter = 100) {
  const double golden = 0.5 * (3.0 - std::sqrt(5.0)); // 0.381966011250105
  double x = a + golden * (b - a);
  double w = x;
  double v = w;
  double d = 0.0;
  double e = 0.0;
  double fx = f(x);
  double fw = fx;
  double fv = fw;

  for (int iter = 0; iter < max_iter; ++iter) {
    double m = 0.5 * (a + b);
    double tol1 = tol * std::abs(x) + 1e-10;
    double tol2 = 2.0 * tol1;

    if (std::abs(x - m) <= tol2 - 0.5 * (b - a)) {
      break;
    }

    double p = 0.0, q = 0.0, r = 0.0;
    if (std::abs(e) > tol1) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) p = -p;
      q = std::abs(q);
      double etemp = e;
      e = d;
      if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
        e = (x >= m) ? a - x : b - x;
        d = golden * e;
      } else {
        d = p / q;
        double u = x + d;
        if (u - a < tol2 || b - u < tol2) {
          d = (m - x >= 0.0) ? tol1 : -tol1;
        }
      }
    } else {
      e = (x >= m) ? a - x : b - x;
      d = golden * e;
    }

    double u = (std::abs(d) >= tol1) ? x + d : x + ((d >= 0.0) ? tol1 : -tol1);
    double fu = f(u);

    if (fu <= fx) {
      if (u >= x) a = x; else b = x;
      v = w; fv = fw;
      w = x; fw = fx;
      x = u; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w; fv = fw;
        w = u; fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
      }
    }
  }
  return x;
}

// Binary Temperature-Scaled Refinement
inline double compute_ts_refinement_binary_impl(
    const double* __restrict__ y_true_ptr,
    const double* __restrict__ y_pred_ptr,
    int n,
    double alpha = 1.0,
    bool is_logits = false,
    int threads = 1) {

  if (n <= 0) return 0.0;

  // Outer Calculation 1: Count class frequencies
  int N1 = 0;
  int N0 = 0;
  std::vector<uint8_t> y_bin(n);
  for (int i = 0; i < n; ++i) {
    uint8_t val = (y_true_ptr[i] > 0.5) ? 1 : 0;
    y_bin[i] = val;
    if (val == 1) N1++; else N0++;
  }

  // Outer Calculation 2: Precompute smoothed label scalar weights (only 2 distinct values)
  const double w1 = (static_cast<double>(N1) + alpha) / (static_cast<double>(N1) + 2.0 * alpha);
  const double w1_comp = 1.0 - w1;
  const double w0 = alpha / (static_cast<double>(N0) + 2.0 * alpha);
  const double w0_comp = 1.0 - w0;
  const double inv_n = 1.0 / static_cast<double>(n);

  // Outer Calculation 3: Precompute sanitized logits in contiguous memory once
  std::vector<double> z(n);
  for (int i = 0; i < n; ++i) {
    double val;
    if (is_logits) {
      val = y_pred_ptr[i];
    } else {
      double p = std::clamp(y_pred_ptr[i], 1e-15, 1.0 - 1e-15);
      val = std::log(p / (1.0 - p));
    }
    if (std::isnan(val) || !std::isfinite(val)) val = 0.0;
    z[i] = std::clamp(val, -35.0, 35.0);
  }

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

  // Inner Brent Objective Function: Zero heap allocations, all invariants hoisted
  auto obj_fn = [&](double temp) -> double {
    const double inv_temp = 1.0 / temp;
    double total_loss = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:total_loss) schedule(static) if (n >= 2048)
#endif
    for (int i = 0; i < n; ++i) {
      double p_T = 1.0 / (1.0 + std::exp(-z[i] * inv_temp));
      p_T = std::clamp(p_T, 1e-15, 1.0 - 1e-15);
      double log_p = std::log(p_T);
      double log_1_p = std::log(1.0 - p_T);
      if (y_bin[i] == 1) {
        total_loss -= (w1 * log_p + w1_comp * log_1_p);
      } else {
        total_loss -= (w0 * log_p + w0_comp * log_1_p);
      }
    }
    return total_loss * inv_n;
  };

  double best_temp = brent_minimize(obj_fn, 0.001, 10.0, 1e-5);

  // Un-smoothed log-loss at optimal temperature
  const double inv_best_temp = 1.0 / best_temp;
  double unsmoothed_loss = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:unsmoothed_loss) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double p_T = 1.0 / (1.0 + std::exp(-z[i] * inv_best_temp));
    p_T = std::clamp(p_T, 1e-15, 1.0 - 1e-15);
    if (y_bin[i] == 1) {
      unsmoothed_loss -= std::log(p_T);
    } else {
      unsmoothed_loss -= std::log(1.0 - p_T);
    }
  }

  return unsmoothed_loss * inv_n;
}

// Multiclass Temperature-Scaled Refinement
template <typename LabelType>
inline double compute_ts_refinement_multiclass_impl(
    const LabelType* __restrict__ y_true_0_ptr,
    const double* __restrict__ y_pred_colmajor_ptr,
    int n,
    int c,
    double alpha = 1.0,
    bool is_logits = false,
    int threads = 1) {

  if (n <= 0 || c <= 0) return 0.0;

  // Outer Calculation 1: Count class frequencies
  std::vector<int> N_vec(c, 0);
  for (int i = 0; i < n; ++i) {
    int y = static_cast<int>(y_true_0_ptr[i]);
    if (y >= 0 && y < c) {
      N_vec[y]++;
    }
  }

  // Outer Calculation 2: Precompute compact C x C smoothed target lookup table S (L1 cache friendly)
  // S[k * c + j]: for true class k and target class column j
  std::vector<double> S(c * c, 0.0);
  for (int k = 0; k < c; ++k) {
    int N_k = N_vec[k];
    double true_target = (static_cast<double>(N_k) + alpha) / (static_cast<double>(N_k) + 2.0 * alpha);
    double leftover_mass = alpha / (static_cast<double>(N_k) + 2.0 * alpha);
    int denom = n - N_k;
    double mass_per_item = (denom == 0) ? 0.0 : (leftover_mass / static_cast<double>(denom));

    for (int j = 0; j < c; ++j) {
      if (j == k) {
        S[k * c + j] = true_target;
      } else {
        S[k * c + j] = static_cast<double>(N_vec[j]) * mass_per_item;
      }
    }
  }

  // Outer Calculation 3: Precompute sanitized row-major contiguous logits Z [N x C]
  // Converts R's column-major matrix into row-contiguous memory so row features are cache-local
  std::vector<double> Z(n * c);
  for (int j = 0; j < c; ++j) {
    const double* col_ptr = y_pred_colmajor_ptr + static_cast<size_t>(j) * n;
    for (int i = 0; i < n; ++i) {
      double val;
      if (is_logits) {
        val = col_ptr[i];
      } else {
        double p = std::clamp(col_ptr[i], 1e-15, 1.0 - 1e-15);
        val = std::log(p);
      }
      if (std::isnan(val) || !std::isfinite(val)) val = 0.0;
      Z[i * c + j] = std::clamp(val, -35.0, 35.0);
    }
  }

  const double inv_n = 1.0 / static_cast<double>(n);

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

  // Inner Brent Objective Function: Zero heap allocations, row-contiguous memory
  auto obj_fn = [&](double temp) -> double {
    const double inv_temp = 1.0 / temp;
    double total_loss = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:total_loss) schedule(static) if (n >= 1024)
#endif
    for (int i = 0; i < n; ++i) {
      const double* __restrict__ z_row = &Z[i * c];
      int k = static_cast<int>(y_true_0_ptr[i]);
      if (k < 0) k = 0;
      if (k >= c) k = c - 1;
      const double* __restrict__ s_row = &S[k * c];

      double m = z_row[0] * inv_temp;
      for (int j = 1; j < c; ++j) {
        double v = z_row[j] * inv_temp;
        if (v > m) m = v;
      }

      double sum_exp = 0.0;
      for (int j = 0; j < c; ++j) {
        sum_exp += std::exp(z_row[j] * inv_temp - m);
      }
      double log_sum_exp = m + std::log(sum_exp);

      double row_loss = 0.0;
      for (int j = 0; j < c; ++j) {
        double log_p = std::clamp((z_row[j] * inv_temp) - log_sum_exp, MIN_LOG_PROB, 0.0);
        row_loss -= s_row[j] * log_p;
      }
      total_loss += row_loss;
    }
    return total_loss * inv_n;
  };

  double best_temp = brent_minimize(obj_fn, 0.001, 10.0, 1e-5);

  // Un-smoothed multiclass log-loss at optimal temperature
  // Evaluates only the true class column for each row (zero wasted operations)
  const double inv_best_temp = 1.0 / best_temp;
  double unsmoothed_loss = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:unsmoothed_loss) schedule(static) if (n >= 1024)
#endif
  for (int i = 0; i < n; ++i) {
    const double* __restrict__ z_row = &Z[i * c];
    int k = static_cast<int>(y_true_0_ptr[i]);
    if (k < 0) k = 0;
    if (k >= c) k = c - 1;

    double m = z_row[0] * inv_best_temp;
    for (int j = 1; j < c; ++j) {
      double v = z_row[j] * inv_best_temp;
      if (v > m) m = v;
    }

    double sum_exp = 0.0;
    for (int j = 0; j < c; ++j) {
      sum_exp += std::exp(z_row[j] * inv_best_temp - m);
    }
    double log_sum_exp = m + std::log(sum_exp);

    double log_p_true = std::clamp((z_row[k] * inv_best_temp) - log_sum_exp, MIN_LOG_PROB, 0.0);
    unsmoothed_loss -= log_p_true;
  }

  return unsmoothed_loss * inv_n;
}

} // namespace evofe

#endif // EVOFE_METRICS_CALIBRATION_H
