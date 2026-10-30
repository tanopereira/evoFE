#ifndef EVOFE_METRICS_CALIBRATION_H
#define EVOFE_METRICS_CALIBRATION_H

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include "utils.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace evofe {

static constexpr double MIN_LOG_PROB = -34.538776394910684; // std::log(1e-15)

// Safeguarded 1D Newton-Raphson update on inverse temperature beta = 1 / T in [0.1, 1000.0].
// Returns true when convergence tolerances are satisfied.
inline bool step_newton_beta(double g, double hess_sum, double inv_n,
                             double& beta, double& beta_lo, double& beta_hi) noexcept {
  if (std::abs(g) * inv_n < 1e-7) return true;

  if (g > 0.0) {
    beta_hi = std::min(beta_hi, beta);
  } else {
    beta_lo = std::max(beta_lo, beta);
  }

  double cand;
  if (hess_sum > 1e-12) {
    double step = std::clamp(g / hess_sum, -3.0 * beta, 0.75 * beta);
    cand = beta - step;
  } else {
    cand = -1.0;
  }

  if (cand <= beta_lo || cand >= beta_hi) {
    if (g < 0.0 && beta_hi == 1000.0) {
      cand = std::min(beta_hi, beta * 4.0);
    } else if (g > 0.0 && beta_lo == 0.1) {
      cand = std::max(beta_lo, beta * 0.25);
    } else {
      cand = std::sqrt(beta_lo * beta_hi);
    }
  }

  cand = std::clamp(cand, 0.1, 1000.0);
  if (std::abs(cand - beta) <= 1e-5 * beta || (beta_hi - beta_lo) <= 1e-5 * beta) {
    beta = cand;
    return true;
  }
  beta = cand;
  return false;
}

// Binary Temperature-Scaled Refinement
// Uses safeguarded Newton-Raphson on inverse temperature beta = 1 / T in [0.1, 1000.0],
// exploiting strict convexity of smoothed cross-entropy in beta and hoisting sum(w_i * z_i).
inline double compute_ts_refinement_binary_impl(
    const double* __restrict__ y_true_ptr,
    const double* __restrict__ y_pred_ptr,
    int n,
    double alpha = 1.0,
    bool is_logits = false,
    int threads = 1) {

  if (n <= 0) return 0.0;

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

  // Outer Calculation 1: Count class frequencies
  int N1 = 0;
  std::vector<uint8_t> y_bin(n);
  for (int i = 0; i < n; ++i) {
    uint8_t val = (y_true_ptr[i] > 0.5) ? 1 : 0;
    y_bin[i] = val;
    if (val == 1) N1++;
  }
  int N0 = n - N1;

  // Outer Calculation 2: Precompute smoothed label scalar weights
  const double w1 = (static_cast<double>(N1) + alpha) / (static_cast<double>(N1) + 2.0 * alpha);
  const double w0 = alpha / (static_cast<double>(N0) + 2.0 * alpha);
  const double inv_n = 1.0 / static_cast<double>(n);

  // Outer Calculation 3: Precompute sanitized FP32 logits z[i] and hoisted linear sum sum_i(w_i * z_i)
  std::vector<float> z(n);
  double sum_wz = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_wz) schedule(static) if (n >= 4096)
#endif
  for (int i = 0; i < n; ++i) {
    double val;
    if (is_logits) {
      val = y_pred_ptr[i];
    } else {
      double p = std::clamp(y_pred_ptr[i], 1e-15, 1.0 - 1e-15);
      val = std::log(p / (1.0 - p));
    }
    if (std::isnan(val) || !std::isfinite(val)) val = 0.0;
    val = std::clamp(val, -35.0, 35.0);
    z[i] = static_cast<float>(val);
    sum_wz += ((y_bin[i] == 1) ? w1 : w0) * val;
  }

  // Safeguarded Newton-Raphson on inverse temperature beta = 1 / T in [0.1, 1000.0]
  // Objective: L(beta) = (1/n) * [ sum_i log(1 + exp(beta * z_i)) - beta * sum_wz ]
  // Derivative: g(beta) = sum_i sigma(beta * z_i) * z_i - sum_wz (strictly increasing in beta)
  double beta = 1.0;
  double beta_lo = 0.1;    // corresponds to T_max = 10.0
  double beta_hi = 1000.0; // corresponds to T_min = 0.001

  for (int iter = 0; iter < 12; ++iter) {
    float beta_f = static_cast<float>(beta);
    double grad_sum = 0.0;
    double hess_sum = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:grad_sum, hess_sum) schedule(static) if (n >= 2048)
#endif
    for (int i = 0; i < n; ++i) {
      float zi = z[i];
      float u = std::clamp(beta_f * zi, -35.0f, 35.0f);
      float eu = std::exp(-std::abs(u));
      float denom = 1.0f + eu;
      float p = (u >= 0.0f) ? (1.0f / denom) : (eu / denom);
      float pq = eu / (denom * denom);
      grad_sum += static_cast<double>(p * zi);
      hess_sum += static_cast<double>(pq * zi * zi);
    }

    if (step_newton_beta(grad_sum - sum_wz, hess_sum, inv_n, beta, beta_lo, beta_hi)) {
      break;
    }
  }

  // Un-smoothed log-loss at optimal inverse temperature beta = 1 / best_temp
  const double inv_best_temp = beta;
  double unsmoothed_loss = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:unsmoothed_loss) schedule(static) if (n >= 2048)
#endif
  for (int i = 0; i < n; ++i) {
    double p_T = 1.0 / (1.0 + std::exp(-static_cast<double>(z[i]) * inv_best_temp));
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
// Uses safeguarded Newton-Raphson on inverse temperature beta = 1 / T in [0.1, 1000.0],
// row-centering logits in FP64 once and fusing the hoisted target dot-product sum_i sum_j(S[k_i, j] * Z[i, j]).
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

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

  // Outer Calculation 1: Count class frequencies
  std::vector<int> N_vec(c, 0);
  for (int i = 0; i < n; ++i) {
    int y = static_cast<int>(y_true_0_ptr[i]);
    if (y >= 0 && y < c) {
      N_vec[y]++;
    }
  }

  // Outer Calculation 2: Precompute compact C x C smoothed target lookup table S (L1 cache friendly)
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

  // Outer Calculation 3: Single-pass fused sanitization, FP64 row-max centering into FP32 Z [N x C],
  // and hoisted target dot-product sum_sz = sum_i sum_j S[k_i, j] * Z[i, j].
  // Because max_j Z[i, j] == 0.0f for every row i, exp(beta * z_j) needs no inner max pass and
  // ez2_bar - ez_bar^2 never suffers shift cancellation on near-uniform probabilities.
  std::vector<float> Z(static_cast<size_t>(n) * c);
  double sum_sz = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_sz) schedule(static) if (n >= 1024)
#endif
  for (int i = 0; i < n; ++i) {
    float* __restrict__ z_row = &Z[static_cast<size_t>(i) * c];
    int k = static_cast<int>(y_true_0_ptr[i]);
    if (k < 0) k = 0;
    if (k >= c) k = c - 1;
    const double* __restrict__ s_row = &S[k * c];

    double row_max = -35.0;
    for (int j = 0; j < c; ++j) {
      double raw = y_pred_colmajor_ptr[static_cast<size_t>(j) * n + i];
      double val = is_logits ? raw : std::log(std::clamp(raw, 1e-15, 1.0 - 1e-15));
      if (std::isnan(val) || !std::isfinite(val)) val = 0.0;
      val = std::clamp(val, -35.0, 35.0);
      if (j == 0 || val > row_max) row_max = val;
    }

    double dot = 0.0;
    for (int j = 0; j < c; ++j) {
      double raw = y_pred_colmajor_ptr[static_cast<size_t>(j) * n + i];
      double val = is_logits ? raw : std::log(std::clamp(raw, 1e-15, 1.0 - 1e-15));
      if (std::isnan(val) || !std::isfinite(val)) val = 0.0;
      double z_centered = std::clamp(val, -35.0, 35.0) - row_max;
      z_row[j] = static_cast<float>(z_centered);
      dot += s_row[j] * z_centered;
    }
    sum_sz += dot;
  }

  const double inv_n = 1.0 / static_cast<double>(n);

  // Safeguarded Newton-Raphson on inverse temperature beta = 1 / T in [0.1, 1000.0]
  double beta = 1.0;
  double beta_lo = 0.1;
  double beta_hi = 1000.0;

  for (int iter = 0; iter < 12; ++iter) {
    float beta_f = static_cast<float>(beta);
    double grad_sum = 0.0;
    double hess_sum = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:grad_sum, hess_sum) schedule(static) if (n >= 1024)
#endif
    for (int i = 0; i < n; ++i) {
      const float* __restrict__ z_row = &Z[static_cast<size_t>(i) * c];
      float sum_exp = 0.0f;
      float sum_ez = 0.0f;
      float sum_ez2 = 0.0f;
      for (int j = 0; j < c; ++j) {
        float zj = z_row[j]; // zj <= 0.0f with max_j(zj) == 0.0f
        float e = std::exp(zj * beta_f);
        sum_exp += e;
        sum_ez += e * zj;
        sum_ez2 += e * zj * zj;
      }

      float inv_se = 1.0f / sum_exp;
      double ez_bar = static_cast<double>(sum_ez * inv_se);
      double ez2_bar = static_cast<double>(sum_ez2 * inv_se);
      grad_sum += ez_bar;
      hess_sum += std::max(0.0, ez2_bar - ez_bar * ez_bar);
    }

    if (step_newton_beta(grad_sum - sum_sz, hess_sum, inv_n, beta, beta_lo, beta_hi)) {
      break;
    }
  }

  // Un-smoothed multiclass log-loss at optimal inverse temperature beta = 1 / best_temp
  const double inv_best_temp = beta;
  double unsmoothed_loss = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:unsmoothed_loss) schedule(static) if (n >= 1024)
#endif
  for (int i = 0; i < n; ++i) {
    const float* __restrict__ z_row = &Z[static_cast<size_t>(i) * c];
    int k = static_cast<int>(y_true_0_ptr[i]);
    if (k < 0) k = 0;
    if (k >= c) k = c - 1;

    double sum_exp = 0.0;
    for (int j = 0; j < c; ++j) {
      sum_exp += std::exp(static_cast<double>(z_row[j]) * inv_best_temp);
    }
    double log_sum_exp = std::log(sum_exp);

    double log_p_true = std::clamp((static_cast<double>(z_row[k]) * inv_best_temp) - log_sum_exp, MIN_LOG_PROB, 0.0);
    unsmoothed_loss -= log_p_true;
  }

  return unsmoothed_loss * inv_n;
}

// Calibrated RMSE: RMSE after optimal linear post-calibration (a + b * y_pred)
// Equivalent to sd_true * sqrt(max(0, 1 - r^2)) where r is Pearson correlation
inline double compute_calibrated_rmse_impl(
    const double* __restrict__ y_true_ptr,
    const double* __restrict__ y_pred_ptr,
    int n) {
  if (n <= 1) return 0.0;

  // Single-pass filter and accumulate means
  double sum_yt = 0.0;
  double sum_yp = 0.0;
  int count = 0;
  for (int i = 0; i < n; ++i) {
    double yt = y_true_ptr[i];
    double yp = y_pred_ptr[i];
    if (std::isfinite(yt) && std::isfinite(yp)) {
      sum_yt += yt;
      sum_yp += yp;
      count++;
    }
  }

  if (count <= 1) return 0.0;

  double mean_yt = sum_yt / static_cast<double>(count);
  double mean_yp = sum_yp / static_cast<double>(count);

  double var_yt = 0.0;
  double var_yp = 0.0;
  double cov = 0.0;

  for (int i = 0; i < n; ++i) {
    double yt = y_true_ptr[i];
    double yp = y_pred_ptr[i];
    if (std::isfinite(yt) && std::isfinite(yp)) {
      double dyt = yt - mean_yt;
      double dyp = yp - mean_yp;
      var_yt += dyt * dyt;
      var_yp += dyp * dyp;
      cov += dyt * dyp;
    }
  }

  double sd_true = std::sqrt(var_yt / (count - 1));
  if (sd_true < 1e-9 || !std::isfinite(sd_true)) return 0.0;

  double sd_pred = std::sqrt(var_yp / (count - 1));
  if (sd_pred < 1e-9 || !std::isfinite(sd_pred)) {
    // Constant prediction: sqrt(mean((y_true - mean(y_pred))^2))
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      double yt = y_true_ptr[i];
      double yp = y_pred_ptr[i];
      if (std::isfinite(yt) && std::isfinite(yp)) {
        double d = yt - mean_yp;
        sum_sq += d * d;
      }
    }
    return std::sqrt(sum_sq / count);
  }

  double denom = std::sqrt(var_yt * var_yp);
  if (denom < 1e-12 || !std::isfinite(denom)) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
      double yt = y_true_ptr[i];
      double yp = y_pred_ptr[i];
      if (std::isfinite(yt) && std::isfinite(yp)) {
        double d = yt - yp;
        sum_sq += d * d;
      }
    }
    return std::sqrt(sum_sq / count);
  }

  double r = cov / denom;
  double r2 = std::clamp(r * r, 0.0, 1.0);
  return sd_true * std::sqrt(1.0 - r2);
}

// Calibrated MAE: MAE after optimal L1 linear calibration a + b * y_pred
// For any slope b, optimal a is median(y_true - b * y_pred)
// Minimized over b via 1D Brent's method
inline double compute_calibrated_mae_impl(
    const double* __restrict__ y_true_ptr,
    const double* __restrict__ y_pred_ptr,
    int n) {
  if (n <= 1) return 0.0;

  std::vector<double> yt;
  std::vector<double> yp;
  yt.reserve(n);
  yp.reserve(n);

  for (int i = 0; i < n; ++i) {
    double t_val = y_true_ptr[i];
    double p_val = y_pred_ptr[i];
    if (std::isfinite(t_val) && std::isfinite(p_val)) {
      yt.push_back(t_val);
      yp.push_back(p_val);
    }
  }

  int valid_n = static_cast<int>(yt.size());
  if (valid_n <= 1) return 0.0;

  double sum_yt = 0.0;
  double sum_yp = 0.0;
  for (int i = 0; i < valid_n; ++i) {
    sum_yt += yt[i];
    sum_yp += yp[i];
  }
  double mean_yt = sum_yt / valid_n;
  double mean_yp = sum_yp / valid_n;

  double var_yp = 0.0;
  double cov = 0.0;
  for (int i = 0; i < valid_n; ++i) {
    double dyt = yt[i] - mean_yt;
    double dyp = yp[i] - mean_yp;
    var_yp += dyp * dyp;
    cov += dyt * dyp;
  }

  // Constant prediction case
  if (var_yp < 1e-9 || !std::isfinite(var_yp)) {
    std::vector<double> tmp = yt;
    int mid = valid_n / 2;
    std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
    double a_star = tmp[mid];
    if (valid_n % 2 == 0) {
      auto max_lower = std::max_element(tmp.begin(), tmp.begin() + mid);
      a_star = 0.5 * (*max_lower + tmp[mid]);
    }
    double sum_ae = 0.0;
    for (int i = 0; i < valid_n; ++i) {
      sum_ae += std::abs(yt[i] - a_star);
    }
    return sum_ae / valid_n;
  }

  double b_ols = cov / var_yp;
  if (!std::isfinite(b_ols)) b_ols = 1.0;

  std::vector<double> res(valid_n);

  auto eval_b = [&](double b) -> double {
    for (int i = 0; i < valid_n; ++i) {
      res[i] = yt[i] - b * yp[i];
    }
    int mid = valid_n / 2;
    std::nth_element(res.begin(), res.begin() + mid, res.end());
    double a_star = res[mid];
    if (valid_n % 2 == 0) {
      auto max_lower = std::max_element(res.begin(), res.begin() + mid);
      a_star = 0.5 * (*max_lower + res[mid]);
    }
    double sum_ae = 0.0;
    for (int i = 0; i < valid_n; ++i) {
      sum_ae += std::abs(res[i] - a_star);
    }
    return sum_ae / valid_n;
  };

  double span = 4.0 * std::max(1.0, std::abs(b_ols));
  double b_low = b_ols - span;
  double b_high = b_ols + span;

  double best_b = brent_minimize(eval_b, b_low, b_high, 1e-5);
  double min_mae = eval_b(best_b);

  // Raw uncalibrated MAE fallback safety
  double raw_ae = 0.0;
  for (int i = 0; i < valid_n; ++i) {
    raw_ae += std::abs(yt[i] - yp[i]);
  }
  double raw_mae = raw_ae / valid_n;

  return std::min(min_mae, raw_mae);
}

} // namespace evofe

#endif // EVOFE_METRICS_CALIBRATION_H
