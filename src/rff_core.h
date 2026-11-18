#ifndef EVOFE_RFF_CORE_H
#define EVOFE_RFF_CORE_H

#include "nn_common.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace evofe {
namespace rff {

constexpr int RFF_CHUNK_SIZE = 4096;
constexpr size_t RFF_MAX_CACHE_FLOATS = 16777216u; // 64 MB FP32 budget for cached Z_tr / Z_val

// Single-pass fused standardization into ColMajor FP32:
// Non-finite values are imputed to training mean (which maps to 0.0f after centering).
// Optional ARD weights scale each column after [-30, 30] z-score clamping.
inline void standardize_impute_mean_f32(
    const double* __restrict__ src, int src_lda,
    float* __restrict__ dst, int dst_lda,
    int N, int D,
    const double* __restrict__ x_mean,
    const double* __restrict__ x_inv_std,
    const double* __restrict__ ard_weights = nullptr) {
  if (N <= 0 || D <= 0) return;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (D >= 2 && N >= 64)
#endif
  for (int j = 0; j < D; ++j) {
    const double* src_col = src + static_cast<size_t>(j) * src_lda;
    float* dst_col = dst + static_cast<size_t>(j) * dst_lda;
    const double m = x_mean[j];
    const double inv_s = x_inv_std[j];
    const float ard = (ard_weights != nullptr) ? static_cast<float>(ard_weights[j]) : 1.0f;
    for (int i = 0; i < N; ++i) {
      const double v = src_col[i];
      if (std::isfinite(v)) {
        const double z = (v - m) * inv_s;
        dst_col[i] = static_cast<float>(std::clamp(z, -30.0, 30.0)) * ard;
      } else {
        dst_col[i] = 0.0f;
      }
    }
  }
}

struct RFFWorkspace {
  int chunk_size = RFF_CHUNK_SIZE;
  Eigen::MatrixXf X_c;
  Eigen::MatrixXf Z_c;
  Eigen::MatrixXf Z_w_c;
  Eigen::MatrixXf XW_c;
  Eigen::MatrixXf K_cm_c;
  Eigen::VectorXf x_norm2_c;
  Eigen::MatrixXf eta_c;
  Eigen::MatrixXf prob_c;
  Eigen::MatrixXf res_c;
  Eigen::MatrixXf G_f32;
  Eigen::MatrixXf G0_f32;
  Eigen::MatrixXf grad_f32;
  Eigen::MatrixXd G_f64;
  Eigen::MatrixXd rhs_f64;
  Eigen::LDLT<Eigen::MatrixXd> ldlt;
  std::vector<std::pair<double, int>> auc_pairs_buf;

  void init(int d_in, int p_cols, int k_freq, int m_landmarks, int out_dim, bool for_train) {
    chunk_size = RFF_CHUNK_SIZE;
    if (d_in > 0) {
      X_c.resize(chunk_size, d_in);
    }
    Z_c.resize(chunk_size, p_cols);
    if (k_freq > 0) {
      XW_c.resize(chunk_size, k_freq);
    }
    if (m_landmarks > 0) {
      K_cm_c.resize(chunk_size, m_landmarks);
      x_norm2_c.resize(chunk_size);
    }
    eta_c.resize(chunk_size, out_dim);
    prob_c.resize(chunk_size, out_dim);
    if (for_train) {
      Z_w_c.resize(chunk_size, p_cols);
      res_c.resize(chunk_size, out_dim);
      G_f32.resize(p_cols, p_cols);
      G0_f32.resize(p_cols, p_cols);
      grad_f32.resize(p_cols, out_dim);
      G_f64.resize(p_cols, p_cols);
      rhs_f64.resize(p_cols, out_dim);
    }
  }
};

struct ValidationContext {
  const double* x_ptr = nullptr;
  int n_val = 0;
  int d_val = 0;
  const double* offset_ptr = nullptr;
  std::function<double(const Eigen::MatrixXd&, bool& higher_is_better)> score_fn;
  bool active = false;
  Eigen::MatrixXd best_preds;
  double best_score = std::numeric_limits<double>::quiet_NaN();
};

struct RFFModel {
  std::string task = "regression";
  std::string kernel = "rbf";
  std::string approx_method = "rff";
  int d_in = 0;
  int n_features = 512;
  int p_cols = 1;
  int out_dim = 1;
  double sigma = 1.0;
  double med_sigma_ref = 1.0;
  double lambda = 1e-2;
  float lin_scale = 1.0f;
  static constexpr double kLinSkipPen = 0.05;

  struct PairScreen {
    int d1 = 0;
    int d2 = 0;
    double score = 0.0;
  };

  Eigen::VectorXd x_mean;
  Eigen::VectorXd x_sd;
  Eigen::VectorXd x_inv_std;
  Eigen::VectorXd ard_weights;
  Eigen::VectorXd ard_w1d;
  std::vector<PairScreen> ard_pairs;

  // RFF parameters
  int k_freq = 0;
  float rff_scale = 1.0f;
  Eigen::MatrixXf W_rff; // d_in x k_freq

  // Nystrom parameters
  int m_landmarks = 0;
  Eigen::MatrixXf landmarks;      // m_landmarks x d_in
  Eigen::VectorXf landmark_norm2; // m_landmarks
  Eigen::MatrixXf P_nys;          // m_landmarks x m_landmarks

  // Linear head weights (last row is unpenalized intercept)
  Eigen::MatrixXf W_out; // p_cols x out_dim

  // Two-scale Mercer mixture kernel shared between Nystrom K_mm and K_cm
  static inline double eval_kernel_from_dist2(double d2, bool is_rbf, double sig) {
    if (is_rbf) {
      const double s1 = 0.45 * sig;
      const double s2 = 1.15 * sig;
      return 0.5 * std::exp(-0.5 * d2 / (s1 * s1)) +
             0.5 * std::exp(-0.5 * d2 / (s2 * s2));
    }
    const double dist = std::sqrt(d2);
    return 0.5 * std::exp(-dist / (0.50 * sig)) +
           0.5 * std::exp(-dist / (1.25 * sig));
  }

  // Compute features into Z_dst(0..B-1, 0..p_cols-1) with stride ld_z
  void compute_features_to(const float* X_c_ptr, int lda_x, int B,
                           float* Z_dst, int ld_z, RFFWorkspace& ws) const {
    if (B <= 0) return;

    if (d_in <= 0 || (approx_method == "rff" && k_freq <= 0) ||
        (approx_method == "nystrom" && m_landmarks <= 0)) {
      float* intercept_col = Z_dst + static_cast<size_t>(p_cols - 1) * ld_z;
      std::fill(intercept_col, intercept_col + B, 1.0f);
      return;
    }

    // 1. Linear skip columns: 0 .. d_in - 1
    const float l_scale = lin_scale;
    for (int d = 0; d < d_in; ++d) {
      const float* x_col = X_c_ptr + static_cast<size_t>(d) * lda_x;
      float* z_col = Z_dst + static_cast<size_t>(d) * ld_z;
      for (int i = 0; i < B; ++i) {
        z_col[i] = x_col[i] * l_scale;
      }
    }

    float* Z_nl = Z_dst + static_cast<size_t>(d_in) * ld_z;

    if (approx_method == "rff") {
      const int ld_xw = static_cast<int>(ws.XW_c.outerStride());
      evofe::nn::gemm_f32_raw(false, false, B, k_freq, d_in,
                              X_c_ptr, lda_x,
                              W_rff.data(), d_in,
                              ws.XW_c.data(), ld_xw,
                              1.0f, 0.0f);
      const float scale = rff_scale;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (k_freq >= 4 && B >= 64)
#endif
      for (int k = 0; k < k_freq; ++k) {
        const float* in_col = ws.XW_c.data() + static_cast<size_t>(k) * ld_xw;
        float* cos_col = Z_nl + static_cast<size_t>(k) * ld_z;
        float* sin_col = Z_nl + static_cast<size_t>(k + k_freq) * ld_z;
        evofe::nn::vcos_f32(B, in_col, cos_col);
        evofe::nn::vsin_f32(B, in_col, sin_col);
        for (int i = 0; i < B; ++i) {
          cos_col[i] *= scale;
          sin_col[i] *= scale;
        }
      }
    } else {
      // Nystrom
      std::fill(ws.x_norm2_c.data(), ws.x_norm2_c.data() + B, 0.0f);
      for (int d = 0; d < d_in; ++d) {
        const float* x_col = X_c_ptr + static_cast<size_t>(d) * lda_x;
        float* n2 = ws.x_norm2_c.data();
        for (int i = 0; i < B; ++i) {
          n2[i] += x_col[i] * x_col[i];
        }
      }

      const int ld_k = static_cast<int>(ws.K_cm_c.outerStride());
      evofe::nn::gemm_f32_raw(false, true, B, m_landmarks, d_in,
                              X_c_ptr, lda_x,
                              landmarks.data(), m_landmarks,
                              ws.K_cm_c.data(), ld_k,
                              -2.0f, 0.0f);

      const bool is_rbf = (kernel == "rbf");
      const double sig_use = sigma;
      const float* x_n2 = ws.x_norm2_c.data();
      const float* l_n2 = landmark_norm2.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (m_landmarks >= 4 && B >= 64)
#endif
      for (int j = 0; j < m_landmarks; ++j) {
        float* k_col = ws.K_cm_c.data() + static_cast<size_t>(j) * ld_k;
        const float ln2 = l_n2[j];
        for (int i = 0; i < B; ++i) {
          const double d2 = std::max(0.0, static_cast<double>(x_n2[i] + ln2 + k_col[i]));
          k_col[i] = static_cast<float>(eval_kernel_from_dist2(d2, is_rbf, sig_use));
        }
      }

      evofe::nn::gemm_f32_raw(false, false, B, m_landmarks, m_landmarks,
                              ws.K_cm_c.data(), ld_k,
                              P_nys.data(), m_landmarks,
                              Z_nl, ld_z,
                              1.0f, 0.0f);
    }

    float* intercept_col = Z_dst + static_cast<size_t>(p_cols - 1) * ld_z;
    std::fill(intercept_col, intercept_col + B, 1.0f);
  }

  void compute_features_chunk(const float* X_c_ptr, int lda_x, int B, RFFWorkspace& ws) const {
    compute_features_to(X_c_ptr, lda_x, B, ws.Z_c.data(), static_cast<int>(ws.Z_c.outerStride()), ws);
  }

  static void eval_logits_from_Z(int B, const float* Z_ptr, int ld_z,
                                 int p_cols_in, int out_dim_in,
                                 const Eigen::MatrixXf& W,
                                 const double* offset_ptr, int offset_lda, int row_start,
                                 float* eta_ptr, int ld_eta) {
    evofe::nn::gemm_f32_raw(false, false, B, out_dim_in, p_cols_in,
                            Z_ptr, ld_z,
                            W.data(), p_cols_in,
                            eta_ptr, ld_eta,
                            1.0f, 0.0f);
    if (offset_ptr != nullptr) {
      for (int c = 0; c < out_dim_in; ++c) {
        const double* off_col = offset_ptr + static_cast<size_t>(c) * offset_lda + row_start;
        float* eta_col = eta_ptr + static_cast<size_t>(c) * ld_eta;
        for (int i = 0; i < B; ++i) {
          const double ov = off_col[i];
          if (std::isfinite(ov)) {
            const float base_e = std::clamp(eta_col[i], -20.0f, 20.0f);
            eta_col[i] = base_e + static_cast<float>(ov);
          }
        }
      }
    }
  }

  void eval_logits_chunk(int B, const Eigen::MatrixXf& W,
                         const double* offset_ptr, int offset_lda, int row_start,
                         RFFWorkspace& ws) const {
    eval_logits_from_Z(B, ws.Z_c.data(), static_cast<int>(ws.Z_c.outerStride()),
                       p_cols, out_dim, W, offset_ptr, offset_lda, row_start,
                       ws.eta_c.data(), static_cast<int>(ws.eta_c.outerStride()));
  }

  static void logits_to_preds(const float* eta_ptr, int ld_eta, int B, int row_offset,
                              const std::string& task_str, int out_dim_in,
                              Eigen::MatrixXd& out_preds) {
    if (task_str == "regression") {
      for (int i = 0; i < B; ++i) {
        out_preds(row_offset + i, 0) = static_cast<double>(eta_ptr[i]);
      }
    } else if (task_str == "classification") {
      for (int i = 0; i < B; ++i) {
        const double e = std::clamp(static_cast<double>(eta_ptr[i]), -34.0, 34.0);
        out_preds(row_offset + i, 0) = 1.0 / (1.0 + std::exp(-e));
      }
    } else {
      for (int i = 0; i < B; ++i) {
        float max_e = eta_ptr[i];
        for (int c = 1; c < out_dim_in; ++c) {
          const float ec = eta_ptr[static_cast<size_t>(c) * ld_eta + i];
          if (ec > max_e) max_e = ec;
        }
        double sum_exp = 0.0;
        for (int c = 0; c < out_dim_in; ++c) {
          const float ec = eta_ptr[static_cast<size_t>(c) * ld_eta + i];
          const float diff = std::clamp(ec - max_e, -80.0f, 0.0f);
          const double ev = static_cast<double>(evofe::nn::fast_expf(diff));
          out_preds(row_offset + i, c) = ev;
          sum_exp += ev;
        }
        const double inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : (1.0 / out_dim_in);
        for (int c = 0; c < out_dim_in; ++c) {
          out_preds(row_offset + i, c) *= inv_sum;
        }
      }
    }
  }

  void standardize_chunk_to_X_c(const double* X_ptr, int N_new, int D_use,
                                int start, int B, RFFWorkspace& ws) const {
    const double* ard_ptr = (ard_weights.size() == d_in) ? ard_weights.data() : nullptr;
    if (D_use > 0) {
      standardize_impute_mean_f32(
          X_ptr + start, N_new,
          ws.X_c.data(), ws.chunk_size,
          B, D_use,
          x_mean.data(), x_inv_std.data(), ard_ptr);
    }
    if (D_use < d_in) {
      for (int j = D_use; j < d_in; ++j) {
        float* dst_col = ws.X_c.data() + static_cast<size_t>(j) * ws.chunk_size;
        std::fill(dst_col, dst_col + B, 0.0f);
      }
    }
  }

  // Predict on raw (unstandardized) ColMajor double matrix X_raw(N_new, d_in)
  Eigen::MatrixXd predict_raw(const double* X_ptr, int N_new, int D_new,
                              const double* offset_ptr, RFFWorkspace& ws,
                              const Eigen::MatrixXf* W_override = nullptr) const {
    Eigen::MatrixXd preds(N_new, out_dim);
    if (N_new <= 0) return preds;

    const Eigen::MatrixXf& W_use = (W_override != nullptr) ? *W_override : W_out;
    const int D_use = std::min(d_in, D_new);
    const int ld_eta = static_cast<int>(ws.eta_c.outerStride());

    for (int start = 0; start < N_new; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N_new - start);
      standardize_chunk_to_X_c(X_ptr, N_new, D_use, start, B, ws);
      compute_features_chunk(ws.X_c.data(), ws.chunk_size, B, ws);
      eval_logits_chunk(B, W_use, offset_ptr, N_new, start, ws);
      logits_to_preds(ws.eta_c.data(), ld_eta, B, start, task, out_dim, preds);
    }
    return preds;
  }

  // Precompute standardized + features matrix for validation if within budget
  bool build_cached_Z_raw(const double* X_ptr, int N_new, int D_new,
                          Eigen::MatrixXf& Z_out, RFFWorkspace& ws) const {
    if (N_new <= 0 || static_cast<size_t>(N_new) * static_cast<size_t>(p_cols) > RFF_MAX_CACHE_FLOATS) {
      return false;
    }
    Z_out.resize(N_new, p_cols);
    const int D_use = std::min(d_in, D_new);
    for (int start = 0; start < N_new; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N_new - start);
      standardize_chunk_to_X_c(X_ptr, N_new, D_use, start, B, ws);
      compute_features_to(ws.X_c.data(), ws.chunk_size, B, Z_out.data() + start, N_new, ws);
    }
    return true;
  }

  Eigen::MatrixXd predict_from_Z(const Eigen::MatrixXf& Z_val, int N_new,
                                 const double* offset_ptr, const Eigen::MatrixXf& W_use,
                                 RFFWorkspace& ws) const {
    Eigen::MatrixXd preds(N_new, out_dim);
    if (N_new <= 0) return preds;
    const int ld_eta = static_cast<int>(ws.eta_c.outerStride());
    for (int start = 0; start < N_new; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N_new - start);
      eval_logits_from_Z(B, Z_val.data() + start, N_new, p_cols, out_dim, W_use,
                         offset_ptr, N_new, start, ws.eta_c.data(), ld_eta);
      logits_to_preds(ws.eta_c.data(), ld_eta, B, start, task, out_dim, preds);
    }
    return preds;
  }

  double evaluate_candidate_val(const Eigen::MatrixXf& W_cand, const ValidationContext& val_ctx,
                                bool has_Z_val, const Eigen::MatrixXf& Z_val,
                                RFFWorkspace& ws, Eigen::MatrixXd& cand_preds,
                                bool& higher_is_better) const {
    if (has_Z_val) {
      cand_preds = predict_from_Z(Z_val, val_ctx.n_val, val_ctx.offset_ptr, W_cand, ws);
    } else {
      cand_preds = predict_raw(val_ctx.x_ptr, val_ctx.n_val, val_ctx.d_val, val_ctx.offset_ptr, ws, &W_cand);
    }
    return val_ctx.score_fn(cand_preds, higher_is_better);
  }

  static bool is_better_score(double cand, double best, bool higher_is_better) {
    if (!std::isfinite(cand)) return false;
    if (!std::isfinite(best)) return true;
    return higher_is_better ? (cand > best) : (cand < best);
  }

  // Supervised linear-residual + 1D + SAVE + 2D ARD relevance screening (O(N * D))
  // Expects X_tr to be unit-variance standardized. Multiplies X_tr columns in-place by ard_weights.
  void compute_ard_weights(Eigen::MatrixXf& X_tr, const std::vector<double>& y_clean,
                           const std::vector<int>& y_cls, const double* offset_ptr, int N) {
    ard_weights = Eigen::VectorXd::Ones(std::max(0, d_in));
    ard_w1d = Eigen::VectorXd::Ones(std::max(0, d_in));
    ard_pairs.clear();
    if (d_in < 2 || N < 16) return;

    const int N_s = std::min(N, 2048);
    const double inv_Ns = 1.0 / static_cast<double>(N_s);
    std::vector<int> idx_s(N_s);
    for (int i = 0; i < N_s; ++i) {
      idx_s[i] = static_cast<int>((static_cast<int64_t>(i) * N) / N_s);
    }

    const int K_t = (task == "multiclass") ? std::min(out_dim, 32) : 1;
    Eigen::MatrixXd Y_s = Eigen::MatrixXd::Zero(N_s, K_t);
    std::vector<int> active_cols;

    if (task == "regression") {
      double sum_y = 0.0;
      for (int i = 0; i < N_s; ++i) {
        const int r = idx_s[i];
        const double off = (offset_ptr != nullptr && std::isfinite(offset_ptr[r])) ? offset_ptr[r] : 0.0;
        const double v = y_clean[r] - off;
        Y_s(i, 0) = v;
        sum_y += v;
      }
      const double mean_y = sum_y * inv_Ns;
      double ss = 0.0;
      for (int i = 0; i < N_s; ++i) {
        Y_s(i, 0) -= mean_y;
        ss += Y_s(i, 0) * Y_s(i, 0);
      }
      const double sd_y = std::sqrt(ss * inv_Ns);
      if (!std::isfinite(sd_y) || sd_y < 1e-12) return;
      Y_s.col(0) /= sd_y;
      active_cols.push_back(0);
    } else if (task == "classification") {
      double sum_y = 0.0;
      for (int i = 0; i < N_s; ++i) {
        const int r = idx_s[i];
        double base_p = 0.0;
        if (offset_ptr != nullptr && std::isfinite(offset_ptr[r])) {
          const double e = std::clamp(offset_ptr[r], -30.0, 30.0);
          base_p = 1.0 / (1.0 + std::exp(-e));
        }
        const double v = y_clean[r] - base_p;
        Y_s(i, 0) = v;
        sum_y += v;
      }
      const double mean_y = sum_y * inv_Ns;
      double ss = 0.0;
      for (int i = 0; i < N_s; ++i) {
        Y_s(i, 0) -= mean_y;
        ss += Y_s(i, 0) * Y_s(i, 0);
      }
      const double sd_y = std::sqrt(ss * inv_Ns);
      if (!std::isfinite(sd_y) || sd_y < 1e-12) return;
      Y_s.col(0) /= sd_y;
      active_cols.push_back(0);
    } else {
      for (int i = 0; i < N_s; ++i) {
        const int r = idx_s[i];
        const int c = y_cls[r];
        if (c < K_t) {
          Y_s(i, c) = 1.0;
        }
      }
      for (int c = 0; c < K_t; ++c) {
        const double mc = Y_s.col(c).mean();
        Y_s.col(c).array() -= mc;
        const double sdc = std::sqrt(Y_s.col(c).squaredNorm() * inv_Ns);
        if (std::isfinite(sdc) && sdc >= 1e-12) {
          Y_s.col(c) /= sdc;
          active_cols.push_back(c);
        }
      }
      if (active_cols.empty()) return;
    }

    const int n_act = static_cast<int>(active_cols.size());
    Eigen::MatrixXd Y_act(N_s, n_act);
    for (int ci = 0; ci < n_act; ++ci) {
      Y_act.col(ci) = Y_s.col(active_cols[ci]);
    }

    Eigen::MatrixXd X_s(N_s, d_in);
    for (int d = 0; d < d_in; ++d) {
      const float* src_col = X_tr.data() + static_cast<size_t>(d) * N;
      double* dst_col = X_s.data() + static_cast<size_t>(d) * N_s;
      for (int i = 0; i < N_s; ++i) {
        dst_col[i] = static_cast<double>(src_col[idx_s[i]]);
      }
    }

    // 1. Marginal linear squared correlations & linear ridge projection to form nonlinear residual R_s
    Eigen::MatrixXd corr_mat = (X_s.transpose() * Y_act) * inv_Ns;
    Eigen::VectorXd score_lin = Eigen::VectorXd::Zero(d_in);
    const double corr_floor = 1.5 * inv_Ns;
    for (int d = 0; d < d_in; ++d) {
      double sl = 0.0;
      for (int ci = 0; ci < n_act; ++ci) {
        const double r = corr_mat(d, ci);
        sl += std::max(0.0, r * r - corr_floor);
      }
      score_lin(d) = sl / static_cast<double>(n_act);
    }

    const int D_lin = std::min(d_in, 64);
    std::vector<int> lin_order(d_in);
    std::iota(lin_order.begin(), lin_order.end(), 0);
    std::partial_sort(lin_order.begin(), lin_order.begin() + D_lin, lin_order.end(),
                      [&](int a, int b) {
                        return (score_lin(a) != score_lin(b)) ? (score_lin(a) > score_lin(b)) : (a < b);
                      });

    Eigen::MatrixXd X_lin(N_s, D_lin);
    for (int u = 0; u < D_lin; ++u) {
      X_lin.col(u) = X_s.col(lin_order[u]);
    }
    Eigen::MatrixXd G_lin = (X_lin.transpose() * X_lin) * inv_Ns;
    for (int u = 0; u < D_lin; ++u) G_lin(u, u) += 0.05;
    Eigen::MatrixXd rhs_lin = (X_lin.transpose() * Y_act) * inv_Ns;
    Eigen::MatrixXd R_s = Y_act;
    Eigen::LDLT<Eigen::MatrixXd> llt_lin(G_lin);
    if (llt_lin.info() == Eigen::Success) {
      for (int ci = 0; ci < n_act; ++ci) {
        Eigen::VectorXd rhs_c = rhs_lin.col(ci);
        Eigen::VectorXd beta_c = llt_lin.solve(rhs_c);
        Eigen::VectorXd res_c = Y_act.col(ci) - X_lin * beta_c;
        res_c.array() -= res_c.mean();
        const double sdr = std::sqrt(res_c.squaredNorm() * inv_Ns);
        if (std::isfinite(sdr) && sdr >= 1e-4) {
          R_s.col(ci) = res_c / sdr;
        } else {
          R_s.col(ci) = Y_act.col(ci);
        }
      }
    }

    // SAVE second-moment residual target for regression
    Eigen::VectorXd y_sq(N_s);
    bool has_ysq = false;
    if (task == "regression") {
      double sum_y2 = 0.0;
      for (int i = 0; i < N_s; ++i) {
        const double r0 = R_s(i, 0);
        y_sq(i) = r0 * r0;
        sum_y2 += y_sq(i);
      }
      const double mean_y2 = sum_y2 * inv_Ns;
      double ss2 = 0.0;
      for (int i = 0; i < N_s; ++i) {
        y_sq(i) -= mean_y2;
        ss2 += y_sq(i) * y_sq(i);
      }
      const double sd_y2 = std::sqrt(ss2 * inv_Ns);
      if (std::isfinite(sd_y2) && sd_y2 >= 1e-12) {
        y_sq /= sd_y2;
        has_ysq = true;
      }
    }

    // 2. 1D nonlinear 5-basis screening against residual R_s (and SAVE second-moment y_sq)
    Eigen::VectorXd score_1d = Eigen::VectorXd::Zero(d_in);
    const double reg = 6.0;
    const double inv_sqrt2 = 0.7071067811865475;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (d_in >= 4 && N_s >= 256)
#endif
    for (int d = 0; d < d_in; ++d) {
      const double* col_ptr = X_s.col(d).data();
      double sum_phi[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
      double G_raw[5][5];
      for (int a = 0; a < 5; ++a) {
        for (int b = 0; b < 5; ++b) G_raw[a][b] = 0.0;
      }

      Eigen::Matrix<double, 5, Eigen::Dynamic> rhs_mat =
          Eigen::Matrix<double, 5, Eigen::Dynamic>::Zero(5, n_act);
      double rhs2_arr[5] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

      for (int i = 0; i < N_s; ++i) {
        const double x = col_ptr[i];
        const double x15 = 1.5 * x;
        const double x30 = 3.0 * x;
        const double phi[5] = {
          (x * x - 1.0) * inv_sqrt2,
          std::cos(x15),
          std::sin(x15),
          std::cos(x30),
          std::sin(x30)
        };

        for (int a = 0; a < 5; ++a) {
          sum_phi[a] += phi[a];
          for (int b = a; b < 5; ++b) {
            G_raw[a][b] += phi[a] * phi[b];
          }
        }
        for (int ci = 0; ci < n_act; ++ci) {
          const double rc = R_s(i, ci);
          for (int a = 0; a < 5; ++a) {
            rhs_mat(a, ci) += phi[a] * rc;
          }
        }
        if (has_ysq) {
          const double y2 = y_sq(i);
          for (int a = 0; a < 5; ++a) {
            rhs2_arr[a] += phi[a] * y2;
          }
        }
      }

      Eigen::Matrix<double, 5, 5> G_d;
      for (int a = 0; a < 5; ++a) {
        const double ma = sum_phi[a] * inv_Ns;
        for (int b = a; b < 5; ++b) {
          const double mb = sum_phi[b] * inv_Ns;
          const double cov = G_raw[a][b] - static_cast<double>(N_s) * ma * mb;
          G_d(a, b) = cov;
          G_d(b, a) = cov;
        }
        G_d(a, a) += reg;
      }

      Eigen::LLT<Eigen::Matrix<double, 5, 5>> llt_d(G_d);
      if (llt_d.info() != Eigen::Success) {
        score_1d(d) = 0.0;
        continue;
      }

      // Compute trace(G_d^{-1}) via 5 vector solves (avoids _dtrsm_)
      double g_inv_trace = 0.0;
      for (int k = 0; k < 5; ++k) {
        Eigen::Matrix<double, 5, 1> e_k = Eigen::Matrix<double, 5, 1>::Zero();
        e_k(k) = 1.0;
        Eigen::Matrix<double, 5, 1> x_k = llt_d.solve(e_k);
        g_inv_trace += x_k(k);
      }
      const double eff_df = std::max(0.0, 5.0 - reg * g_inv_trace);
      const double null_thr = 1.5 * eff_df * inv_Ns;

      double r2_1_sum = 0.0;
      for (int ci = 0; ci < n_act; ++ci) {
        Eigen::Matrix<double, 5, 1> rhs_c = rhs_mat.col(ci);
        Eigen::Matrix<double, 5, 1> w_c = llt_d.solve(rhs_c);
        r2_1_sum += w_c.dot(rhs_c) * inv_Ns;
      }
      const double s1 = std::max(0.0, (r2_1_sum / static_cast<double>(n_act)) - null_thr);

      double s2 = 0.0;
      if (has_ysq) {
        Eigen::Matrix<double, 5, 1> rhs2_v;
        for (int a = 0; a < 5; ++a) rhs2_v(a) = rhs2_arr[a];
        Eigen::Matrix<double, 5, 1> w2 = llt_d.solve(rhs2_v);
        s2 = std::max(0.0, w2.dot(rhs2_v) * inv_Ns - null_thr);
      }
      score_1d(d) = s1 + 0.35 * s2;
    }

    // 3. Pairwise 2D interaction screening M = X_cand^T diag(R_s) X_cand / N_s
    const int D_cand = std::min(d_in, 48);
    std::vector<int> cand_idx(d_in);
    std::iota(cand_idx.begin(), cand_idx.end(), 0);
    std::partial_sort(cand_idx.begin(), cand_idx.begin() + D_cand, cand_idx.end(),
                      [&](int a, int b) {
                        const double sa = score_1d(a) + score_lin(a);
                        const double sb = score_1d(b) + score_lin(b);
                        return (sa != sb) ? (sa > sb) : (a < b);
                      });

    Eigen::MatrixXd X_cand(N_s, D_cand);
    for (int u = 0; u < D_cand; ++u) {
      X_cand.col(u) = X_s.col(cand_idx[u]);
    }

    Eigen::MatrixXd pair_r2 = Eigen::MatrixXd::Zero(D_cand, D_cand);
    Eigen::MatrixXd XR_buf(N_s, D_cand);
    const double pair_floor = 3.5 * inv_Ns;
    for (int ci = 0; ci < n_act; ++ci) {
      XR_buf.array() = X_cand.array().colwise() * R_s.col(ci).array();
      Eigen::MatrixXd M = (X_cand.transpose() * XR_buf) * inv_Ns;
      for (int u = 0; u < D_cand; ++u) {
        for (int v = u + 1; v < D_cand; ++v) {
          const double mv = M(u, v);
          const double pr = std::max(0.0, mv * mv - pair_floor) / static_cast<double>(n_act);
          pair_r2(u, v) += pr;
          pair_r2(v, u) += pr;
        }
      }
    }

    Eigen::VectorXd score_2d = Eigen::VectorXd::Zero(d_in);
    for (int u = 0; u < D_cand; ++u) {
      score_2d(cand_idx[u]) = pair_r2.row(u).sum();
    }

    // 4. Combine ARD weights and apply to X_tr in-place
    Eigen::VectorXd tot_sc(d_in);
    double max_tot = 0.0;
    for (int d = 0; d < d_in; ++d) {
      const double ts = score_1d(d) + score_2d(d) + 0.20 * score_lin(d);
      tot_sc(d) = ts;
      if (ts > max_tot) max_tot = ts;
    }

    if (max_tot > 1e-12) {
      double sum_a2 = 0.0;
      for (int d = 0; d < d_in; ++d) {
        const double ad = std::sqrt(tot_sc(d) / max_tot + 0.01);
        ard_weights(d) = ad;
        sum_a2 += ad * ad;
      }
      const double inv_rms = std::sqrt(static_cast<double>(d_in) / std::max(1e-12, sum_a2));
      for (int d = 0; d < d_in; ++d) {
        ard_weights(d) *= inv_rms;
        const float af = static_cast<float>(ard_weights(d));
        float* col_ptr = X_tr.data() + static_cast<size_t>(d) * N;
        for (int i = 0; i < N; ++i) {
          col_ptr[i] *= af;
        }
      }
    }

    double max_w1d = 0.0;
    for (int d = 0; d < d_in; ++d) {
      ard_w1d(d) = score_1d(d) + 0.03 * (score_lin(d) + score_2d(d));
      if (ard_w1d(d) > max_w1d) max_w1d = ard_w1d(d);
    }
    const double w1d_floor = 1e-4 * std::max(1.0, max_w1d);
    double sum_w1d = 0.0;
    for (int d = 0; d < d_in; ++d) {
      ard_w1d(d) += w1d_floor;
      sum_w1d += ard_w1d(d);
    }
    if (sum_w1d > 0.0) ard_w1d /= sum_w1d;

    ard_pairs.reserve(static_cast<size_t>(D_cand) * (D_cand - 1) / 2);
    for (int u = 0; u < D_cand; ++u) {
      for (int v = u + 1; v < D_cand; ++v) {
        const int d1 = cand_idx[u];
        const int d2 = cand_idx[v];
        const double sc = pair_r2(u, v) + 0.01 * std::max(1e-8, ard_w1d(d1) * ard_w1d(d2));
        ard_pairs.push_back(PairScreen{ std::min(d1, d2), std::max(d1, d2), sc });
      }
    }
    std::sort(ard_pairs.begin(), ard_pairs.end(),
              [](const PairScreen& a, const PairScreen& b) {
                if (a.score != b.score) return a.score > b.score;
                return (a.d1 != b.d1) ? (a.d1 < b.d1) : (a.d2 < b.d2);
              });
  }

  // Deterministic largest-remainder integer quota allocation
  static std::vector<int> largest_remainder_quotas(const std::vector<double>& weights, int total_budget) {
    const int n = static_cast<int>(weights.size());
    std::vector<int> counts(n, 0);
    if (n <= 0 || total_budget <= 0) return counts;
    double wsum = 0.0;
    for (double w : weights) wsum += std::max(0.0, w);
    std::vector<std::pair<double, int>> rem(n);
    int used = 0;
    for (int i = 0; i < n; ++i) {
      const double prob = (wsum > 0.0) ? (std::max(0.0, weights[i]) / wsum) : (1.0 / n);
      const double exact = static_cast<double>(total_budget) * prob;
      const int fl = static_cast<int>(std::floor(exact));
      counts[i] = fl;
      used += fl;
      rem[i] = { exact - fl, i };
    }
    const int left = total_budget - used;
    if (left > 0) {
      std::partial_sort(rem.begin(), rem.begin() + std::min(left, n), rem.end(),
                        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                          return (a.first != b.first) ? (a.first > b.first) : (a.second < b.second);
                        });
      for (int r = 0; r < left; ++r) {
        counts[rem[r % n].second]++;
      }
    }
    return counts;
  }

  // Compute median heuristic sigma on standardized X_tr_f32 (up to 2048 pairs)
  static double estimate_median_sigma(const Eigen::MatrixXf& X_tr, int N, int D, int seed) {
    if (N < 2 || D <= 0) {
      return std::sqrt(static_cast<double>(std::max(1, D)));
    }
    const int max_pairs = 2048;
    std::mt19937_64 rng(static_cast<uint64_t>(seed) ^ 0x9e3779b97f4a7c15ULL);
    std::uniform_int_distribution<int> dist_idx(0, N - 1);
    std::vector<double> dists;
    dists.reserve(max_pairs);

    for (int p = 0; p < max_pairs; ++p) {
      int i = dist_idx(rng);
      int j = dist_idx(rng);
      if (i == j) j = (i + 1) % N;
      double d2 = 0.0;
      for (int d = 0; d < D; ++d) {
        const double diff = static_cast<double>(X_tr(i, d)) - static_cast<double>(X_tr(j, d));
        d2 += diff * diff;
      }
      if (d2 > 0.0 && std::isfinite(d2)) {
        dists.push_back(std::sqrt(d2));
      }
    }
    if (dists.empty()) {
      return std::clamp(std::sqrt(static_cast<double>(D)), 1e-3, 1e3);
    }
    const size_t mid = dists.size() / 2;
    std::nth_element(dists.begin(), dists.begin() + mid, dists.end());
    const double med = dists[mid];
    return std::clamp(med, 1e-3, 1e3);
  }

  void init_basis(const Eigen::MatrixXf& X_tr, int N, int seed) {
    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    if (d_in <= 0 || N <= 0) {
      k_freq = 0;
      m_landmarks = 0;
      p_cols = 1;
      lin_scale = 0.0f;
      W_out = Eigen::MatrixXf::Zero(p_cols, out_dim);
      return;
    }

    lin_scale = 1.0f / std::sqrt(static_cast<float>(d_in));

    if (approx_method == "rff") {
      k_freq = std::max(1, n_features / 2);
      p_cols = d_in + 2 * k_freq + 1;
      rff_scale = 1.0f / std::sqrt(static_cast<float>(k_freq));
      W_rff.setZero(d_in, k_freq);

      int k_1d = 0, n_pairs = 0, k_2d = 0, k_glob = 0;
      if (d_in == 1) {
        k_1d = k_freq;
      } else if (k_freq < 6) {
        k_1d = k_freq / 2;
        k_glob = k_freq - k_1d;
      } else {
        k_1d = static_cast<int>(std::floor(0.40 * k_freq));
        n_pairs = static_cast<int>(std::floor(0.22 * k_freq));
        k_2d = 2 * n_pairs;
        k_glob = k_freq - k_1d - k_2d;
      }

      const double mults[3] = { 0.5, 1.0, 2.0 };
      const double sigma_ratio = std::clamp(sigma / std::max(0.5, med_sigma_ref), 0.25, 4.0);
      std::normal_distribution<double> norm_dist(0.0, 1.0);
      std::uniform_real_distribution<double> unif_dist(1e-6, 1.0 - 1e-6);
      const bool is_rbf = (kernel == "rbf");

      auto draw_freq = [&]() -> double {
        if (is_rbf) return norm_dist(rng);
        const double u = unif_dist(rng);
        return std::clamp(std::tan(M_PI * (u - 0.5)), -6.0, 6.0) * 0.65;
      };

      int col = 0;
      // 1. Deterministic quota-allocated, stratified positive 1D frequencies
      if (k_1d > 0) {
        std::vector<double> w1d(d_in, 1.0 / static_cast<double>(d_in));
        if (ard_w1d.size() == d_in) {
          for (int d = 0; d < d_in; ++d) w1d[d] = std::max(0.0, ard_w1d(d));
        }
        const std::vector<int> counts_1d = largest_remainder_quotas(w1d, k_1d);
        for (int d = 0; d < d_in; ++d) {
          const int m_d = counts_1d[d];
          if (m_d <= 0) continue;
          const double a_d = (ard_weights.size() == d_in) ? std::max(1e-3, ard_weights(d)) : 1.0;
          const double denom = sigma_ratio * a_d;
          for (int r = 0; r < m_d && col < k_freq; ++r, ++col) {
            const double u = (static_cast<double>(r) + 0.5) / static_cast<double>(m_d);
            const double f_r = (0.25 + 2.75 * u) / denom;
            W_rff(d, col) = static_cast<float>(f_r);
          }
        }
      }

      // 2. Deterministic quota-allocated 2D conjugate-pair Halton/golden-ratio frequencies
      if (n_pairs > 0 && d_in >= 2) {
        std::vector<PairScreen> pairs_use = ard_pairs;
        if (pairs_use.empty()) {
          const int m_fb = std::min(d_in, 48);
          for (int u = 0; u < m_fb; ++u) {
            pairs_use.push_back(PairScreen{ u, (u + 1) % d_in, 1.0 });
          }
        }
        std::vector<double> pair_w(pairs_use.size());
        for (size_t i = 0; i < pairs_use.size(); ++i) pair_w[i] = std::max(0.0, pairs_use[i].score);
        const std::vector<int> counts_2d = largest_remainder_quotas(pair_w, n_pairs);

        constexpr double phi_gold = 0.6180339887498949;
        for (size_t pi = 0; pi < pairs_use.size(); ++pi) {
          const int m_p = counts_2d[pi];
          if (m_p <= 0) continue;
          const int d1 = pairs_use[pi].d1;
          const int d2 = pairs_use[pi].d2;
          const double a1 = (ard_weights.size() == d_in) ? std::max(1e-3, ard_weights(d1)) : 1.0;
          const double a2 = (ard_weights.size() == d_in) ? std::max(1e-3, ard_weights(d2)) : 1.0;
          const double den1 = sigma_ratio * a1;
          const double den2 = sigma_ratio * a2;
          for (int r = 0; r < m_p && col + 1 < k_freq; ++r) {
            const double u1 = (static_cast<double>(r) + 0.5) / static_cast<double>(m_p);
            const double prod = static_cast<double>(r + 1) * phi_gold;
            const double u2 = prod - std::floor(prod);
            const float f1 = static_cast<float>((0.18 + 0.85 * u1) / den1);
            const float f2 = static_cast<float>((0.18 + 0.85 * u2) / den2);
            W_rff(d1, col) = f1;
            W_rff(d2, col) = f2;
            ++col;
            W_rff(d1, col) = f1;
            W_rff(d2, col) = -f2;
            ++col;
          }
        }
      }

      // 3. Remaining global D-dim ARD frequencies across multi-scale bandwidths
      const double sig_base = std::max(1e-4, sigma);
      while (col < k_freq) {
        const double inv_sc = 1.0 / (mults[col % 3] * sig_base);
        for (int d = 0; d < d_in; ++d) {
          W_rff(d, col) = static_cast<float>(draw_freq() * inv_sc);
        }
        ++col;
      }
    } else {
      // Nystrom
      m_landmarks = std::min(std::max(1, n_features), N);
      p_cols = d_in + m_landmarks + 1;

      std::vector<int> idx(N);
      std::iota(idx.begin(), idx.end(), 0);
      for (int i = 0; i < m_landmarks; ++i) {
        std::uniform_int_distribution<int> d_u(i, N - 1);
        std::swap(idx[i], idx[d_u(rng)]);
      }

      landmarks.resize(m_landmarks, d_in);
      landmark_norm2.resize(m_landmarks);
      landmark_norm2.setZero();
      for (int i = 0; i < m_landmarks; ++i) {
        const int row = idx[i];
        float n2 = 0.0f;
        for (int d = 0; d < d_in; ++d) {
          const float v = X_tr(row, d);
          landmarks(i, d) = v;
          n2 += v * v;
        }
        landmark_norm2(i) = n2;
      }

      Eigen::MatrixXf K_mm_f32 = Eigen::MatrixXf::Zero(m_landmarks, m_landmarks);
      evofe::nn::syrk_lower_f32(false, m_landmarks, d_in,
                                landmarks.data(), m_landmarks,
                                K_mm_f32.data(), m_landmarks,
                                -2.0f, 0.0f);

      Eigen::MatrixXd K_mm_f64(m_landmarks, m_landmarks);
      const bool is_rbf = (kernel == "rbf");

      for (int j = 0; j < m_landmarks; ++j) {
        for (int i = j; i < m_landmarks; ++i) {
          const double d2 = std::max(0.0,
              static_cast<double>(landmark_norm2(i)) +
              static_cast<double>(landmark_norm2(j)) +
              static_cast<double>(K_mm_f32(i, j)));
          double val = eval_kernel_from_dist2(d2, is_rbf, sigma);
          if (i == j) val += 1e-4;
          K_mm_f64(i, j) = val;
          K_mm_f64(j, i) = val;
        }
      }

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(K_mm_f64);
      P_nys.resize(m_landmarks, m_landmarks);
      if (eig.info() != Eigen::Success) {
        P_nys.setIdentity();
      } else {
        const Eigen::VectorXd& evals = eig.eigenvalues();
        const Eigen::MatrixXd& evecs = eig.eigenvectors();
        const double max_ev = std::max(1e-8, evals.maxCoeff());
        const double floor_ev = std::max(1e-5, 1e-6 * max_ev);

        Eigen::MatrixXf V_scaled(m_landmarks, m_landmarks);
        for (int k = 0; k < m_landmarks; ++k) {
          const double inv_q = 1.0 / std::sqrt(std::sqrt(std::max(evals(k), floor_ev)));
          for (int i = 0; i < m_landmarks; ++i) {
            V_scaled(i, k) = static_cast<float>(evecs(i, k) * inv_q);
          }
        }
        evofe::nn::syrk_lower_f32(false, m_landmarks, m_landmarks,
                                  V_scaled.data(), m_landmarks,
                                  P_nys.data(), m_landmarks,
                                  1.0f, 0.0f);
        for (int j = 0; j < m_landmarks; ++j) {
          for (int i = j + 1; i < m_landmarks; ++i) {
            P_nys(j, i) = P_nys(i, j);
          }
        }
      }
    }
    W_out = Eigen::MatrixXf::Zero(p_cols, out_dim);
  }

  Eigen::VectorXd make_pen_vector(double reg_lambda) const {
    Eigen::VectorXd pen(p_cols);
    const double lin_pen = kLinSkipPen * reg_lambda;
    for (int j = 0; j < p_cols; ++j) {
      pen(j) = (j < d_in) ? lin_pen : ((j < p_cols - 1) ? reg_lambda : 1e-8);
    }
    return pen;
  }

  // Symmetrize lower-triangular FP32 Gram matrix (optionally scaled by scale_g) into FP64 G_f64 + differential ridge
  void prepare_gram_f64(const Eigen::MatrixXf& G_src, RFFWorkspace& ws,
                        double reg_lambda, double scale_g = 1.0) const {
    const double lin_pen = kLinSkipPen * reg_lambda;
    for (int j = 0; j < p_cols; ++j) {
      for (int i = j; i < p_cols; ++i) {
        const double v = static_cast<double>(G_src(i, j)) * scale_g;
        ws.G_f64(i, j) = v;
        ws.G_f64(j, i) = v;
      }
      const double diag_reg = (j < d_in) ? lin_pen : ((j < p_cols - 1) ? reg_lambda : 1e-8);
      ws.G_f64(j, j) += diag_reg;
    }
  }

  // Compute tr(diag(pen) * A^{-1}) from already-factored LDLT without _dtrsm_
  static double compute_ldlt_pen_trace(const Eigen::LDLT<Eigen::MatrixXd>& ldlt,
                                       const Eigen::VectorXd& pen) {
    const int p = static_cast<int>(pen.size());
    if (p <= 0) return 0.0;
    const Eigen::VectorXd& vecD = ldlt.vectorD();
    Eigen::VectorXd invD(p);
    for (int i = 0; i < p; ++i) {
      const double d = vecD(i);
      invD(i) = (std::isfinite(d) && d > 1e-15) ? (1.0 / d) : 0.0;
    }

    if (p <= 768) {
      const Eigen::MatrixXd& Lmat = ldlt.matrixLDLT();
      const Eigen::VectorXd pen_perm = ldlt.transpositionsP() * pen;
      double trace_val = 0.0;
      Eigen::VectorXd col(p);
      for (int j = 0; j < p; ++j) {
        const double pj = pen_perm(j);
        if (pj <= 0.0) continue;
        col.setZero();
        col(j) = 1.0;
        for (int k = j; k < p - 1; ++k) {
          const double ck = col(k);
          if (ck != 0.0) {
            const int len = p - 1 - k;
            col.segment(k + 1, len).noalias() -= ck * Lmat.col(k).segment(k + 1, len);
          }
        }
        double qnorm = 0.0;
        for (int i = j; i < p; ++i) {
          qnorm += invD(i) * col(i) * col(i);
        }
        trace_val += pj * qnorm;
      }
      return trace_val;
    }

    // 64 deterministic Rademacher probes (vector-only ldlt.solve avoids _dtrsm_)
    constexpr int n_probes = 64;
    std::mt19937_64 rng(0x517cc1b727220a95ULL);
    Eigen::VectorXd z(p), x(p);
    double sum_tr = 0.0;
    for (int r = 0; r < n_probes; ++r) {
      uint64_t bits = 0;
      for (int j = 0; j < p; ++j) {
        if ((j & 63) == 0) bits = rng();
        z(j) = ((bits >> (j & 63)) & 1ULL) ? 1.0 : -1.0;
      }
      x = ldlt.solve(z);
      for (int j = 0; j < p; ++j) {
        sum_tr += pen(j) * z(j) * x(j);
      }
    }
    return std::max(0.0, sum_tr / static_cast<double>(n_probes));
  }

  // Accumulate unweighted Gram G0_f32 and optional Z_tr cache
  bool prepare_G0_and_cache(const Eigen::MatrixXf& X_tr, int N,
                            Eigen::MatrixXf& Z_tr_cache, RFFWorkspace& ws) const {
    ws.G0_f32.setZero();
    const bool cache_z = (static_cast<size_t>(N) * static_cast<size_t>(p_cols) <= RFF_MAX_CACHE_FLOATS);
    if (cache_z) {
      Z_tr_cache.resize(N, p_cols);
      for (int start = 0; start < N; start += ws.chunk_size) {
        const int B = std::min(ws.chunk_size, N - start);
        compute_features_to(X_tr.data() + start, N, B, Z_tr_cache.data() + start, N, ws);
      }
      evofe::nn::syrk_lower_f32(true, p_cols, N,
                                Z_tr_cache.data(), N,
                                ws.G0_f32.data(), p_cols,
                                1.0f, 0.0f);
    } else {
      const int ld_z = static_cast<int>(ws.Z_c.outerStride());
      for (int start = 0; start < N; start += ws.chunk_size) {
        const int B = std::min(ws.chunk_size, N - start);
        compute_features_chunk(X_tr.data() + start, N, B, ws);
        evofe::nn::syrk_lower_f32(true, p_cols, B,
                                  ws.Z_c.data(), ld_z,
                                  ws.G0_f32.data(), p_cols,
                                  1.0f, 1.0f);
      }
    }
    return cache_z;
  }

  void fit_regression(const Eigen::MatrixXf& X_tr, const double* y_ptr,
                      const double* offset_ptr, int N, double lambda_req,
                      ValidationContext& val_ctx, RFFWorkspace& ws) {
    if (N <= 0) return;
    double sum_y_res = 0.0;
    for (int i = 0; i < N; ++i) {
      const double off = (offset_ptr != nullptr && std::isfinite(offset_ptr[i])) ? offset_ptr[i] : 0.0;
      sum_y_res += (y_ptr[i] - off);
    }
    const double y_res_mean = sum_y_res / static_cast<double>(N);

    ws.G0_f32.setZero();
    ws.grad_f32.setZero();
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());
    double y_ss = 0.0;

    for (int start = 0; start < N; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N - start);
      compute_features_chunk(X_tr.data() + start, N, B, ws);
      evofe::nn::syrk_lower_f32(true, p_cols, B,
                                ws.Z_c.data(), ld_z,
                                ws.G0_f32.data(), p_cols,
                                1.0f, 1.0f);
      for (int i = 0; i < B; ++i) {
        const double off = (offset_ptr != nullptr && std::isfinite(offset_ptr[start + i]))
                               ? offset_ptr[start + i] : 0.0;
        const double yc = (y_ptr[start + i] - off) - y_res_mean;
        y_ss += yc * yc;
        ws.res_c(i, 0) = static_cast<float>(yc);
      }
      evofe::nn::gemm_f32_raw(true, false, p_cols, 1, B,
                              ws.Z_c.data(), ld_z,
                              ws.res_c.data(), ws.chunk_size,
                              ws.grad_f32.data(), p_cols,
                              1.0f, 1.0f);
    }

    const Eigen::VectorXd rhs0 = ws.grad_f32.col(0).cast<double>();

    if (std::isfinite(lambda_req) && lambda_req > 0.0) {
      lambda = lambda_req;
      prepare_gram_f64(ws.G0_f32, ws, lambda);
      ws.ldlt.compute(ws.G_f64);
      const Eigen::VectorXd w_sol = ws.ldlt.solve(rhs0);
      W_out.col(0) = w_sol.cast<float>();
      W_out(p_cols - 1, 0) += static_cast<float>(y_res_mean);
      return;
    }

    // Auto-lambda path selection
    const double n_scale = std::max(0.25, static_cast<double>(N) / 500.0);
    const double mults[9] = { 25.6, 6.4, 1.6, 0.4, 0.1, 0.025, 0.005, 0.001, 0.0002 };

    Eigen::MatrixXf Z_val;
    bool has_Z_val = false;
    if (val_ctx.active && val_ctx.n_val > 0) {
      has_Z_val = build_cached_Z_raw(val_ctx.x_ptr, val_ctx.n_val, val_ctx.d_val, Z_val, ws);
    }

    Eigen::MatrixXf W_cand(p_cols, 1);
    double best_crit = std::numeric_limits<double>::quiet_NaN();
    double best_lam = 1e-2;
    Eigen::MatrixXf best_W = Eigen::MatrixXf::Zero(p_cols, 1);
    Eigen::MatrixXd cand_preds;

    for (int gi = 0; gi < 9; ++gi) {
      const double lam = mults[gi] * n_scale;
      prepare_gram_f64(ws.G0_f32, ws, lam);
      ws.ldlt.compute(ws.G_f64);
      const Eigen::VectorXd w_sol = ws.ldlt.solve(rhs0);
      W_cand.col(0) = w_sol.cast<float>();
      W_cand(p_cols - 1, 0) += static_cast<float>(y_res_mean);

      if (val_ctx.active && val_ctx.n_val > 0) {
        bool higher_is_better = false;
        const double vscore = evaluate_candidate_val(W_cand, val_ctx, has_Z_val, Z_val,
                                                     ws, cand_preds, higher_is_better);
        if (is_better_score(vscore, best_crit, higher_is_better)) {
          best_crit = vscore;
          best_lam = lam;
          best_W = W_cand;
          val_ctx.best_preds = std::move(cand_preds);
          val_ctx.best_score = vscore;
        }
      } else {
        const Eigen::VectorXd pen = make_pen_vector(lam);
        const double pen_norm2 = w_sol.dot(pen.cwiseProduct(w_sol));
        const double sum_resid2 = std::max(1e-12 * static_cast<double>(N),
                                           y_ss - w_sol.dot(rhs0) - pen_norm2);
        const double mse_tr = sum_resid2 / static_cast<double>(N);
        const double df = std::max(0.0, static_cast<double>(p_cols) - compute_ldlt_pen_trace(ws.ldlt, pen));
        const double denom = 1.0 - std::min(0.95, df / static_cast<double>(N));
        const double gcv = mse_tr / (denom * denom);
        if (!std::isfinite(best_crit) || gcv < best_crit) {
          best_crit = gcv;
          best_lam = lam;
          best_W = W_cand;
        }
      }
    }

    lambda = best_lam;
    W_out = best_W;
  }

  // Binary loss from precomputed logits eta(0..N-1)
  static double compute_binary_nll_from_eta(const float* eta_ptr, const double* y_ptr, int N) {
    double loss = 0.0;
    for (int i = 0; i < N; ++i) {
      const double e = static_cast<double>(eta_ptr[i]);
      const double softplus = (e > 0.0) ? (e + std::log1p(std::exp(-e))) : std::log1p(std::exp(e));
      loss += softplus - y_ptr[i] * e;
    }
    return loss;
  }

  double compute_reg_penalty(const Eigen::MatrixXf& W, double reg_lambda) const {
    double reg = 0.0;
    const int cols = static_cast<int>(W.cols());
    for (int c = 0; c < cols; ++c) {
      for (int j = 0; j < p_cols - 1; ++j) {
        const double w = static_cast<double>(W(j, c));
        const double scale = (j < d_in) ? kLinSkipPen : 1.0;
        reg += scale * w * w;
      }
    }
    return 0.5 * reg_lambda * reg;
  }

  double eval_binary_loss_streamed(const Eigen::MatrixXf& X_tr, const double* y_ptr,
                                   const double* offset_ptr, int N,
                                   const Eigen::MatrixXf& W, double reg_lambda,
                                   RFFWorkspace& ws, double* data_nll_out = nullptr) const {
    double data_loss = 0.0;
    for (int start = 0; start < N; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N - start);
      compute_features_chunk(X_tr.data() + start, N, B, ws);
      eval_logits_chunk(B, W, offset_ptr, N, start, ws);
      data_loss += compute_binary_nll_from_eta(ws.eta_c.data(), y_ptr + start, B);
    }
    if (data_nll_out != nullptr) *data_nll_out = data_loss;
    return data_loss + compute_reg_penalty(W, reg_lambda);
  }

  struct BinarySolveScratch {
    Eigen::VectorXf eta, d_eta, eta_cand, res, sqrt_w;
    Eigen::MatrixXf Z_w, W_cand;
    void init(int N, int p_cols, bool alloc_zw) {
      eta.resize(N);
      d_eta.resize(N);
      eta_cand.resize(N);
      res.resize(N);
      sqrt_w.resize(N);
      W_cand.resize(p_cols, 1);
      if (alloc_zw) {
        Z_w.resize(N, p_cols);
      }
    }
  };

  struct MulticlassSolveScratch {
    Eigen::MatrixXf eta, d_eta, eta_cand, res, W_cand, delta_f32;
    Eigen::VectorXd grad_col, delta_col;
    void init(int N, int p_cols, int out_dim) {
      eta.resize(N, out_dim);
      d_eta.resize(N, out_dim);
      eta_cand.resize(N, out_dim);
      res.resize(N, out_dim);
      W_cand.resize(p_cols, out_dim);
      delta_f32.resize(p_cols, out_dim);
      grad_col.resize(p_cols);
      delta_col.resize(p_cols);
    }
  };

  // Solve binary logistic for a single reg_lambda on cached Z_tr using Bohning warm-start + exact IRLS polish.
  // Leaves exact weighted IRLS Hessian factored in ws.ldlt when compute_exact_hessian is true.
  double solve_binary_cached(const Eigen::MatrixXf& Z_tr, const double* y_ptr,
                             const double* offset_ptr, int N, double reg_lambda,
                             int max_iter, double tol, bool compute_exact_hessian,
                             Eigen::MatrixXf& W_cur, BinarySolveScratch& sc,
                             RFFWorkspace& ws) const {
    const Eigen::VectorXd pen = make_pen_vector(reg_lambda);
    Eigen::VectorXf& eta = sc.eta;
    Eigen::VectorXf& d_eta = sc.d_eta;
    Eigen::VectorXf& eta_cand = sc.eta_cand;
    Eigen::VectorXf& res = sc.res;
    Eigen::VectorXf& sqrt_w = sc.sqrt_w;
    Eigen::MatrixXf& Z_w = sc.Z_w;
    Eigen::MatrixXf& W_cand = sc.W_cand;

    auto compute_eta = [&](const Eigen::MatrixXf& W_mat, Eigen::VectorXf& out_eta) {
      evofe::nn::gemm_f32_raw(false, false, N, 1, p_cols,
                              Z_tr.data(), N,
                              W_mat.data(), p_cols,
                              out_eta.data(), N,
                              1.0f, 0.0f);
      if (offset_ptr != nullptr) {
        for (int i = 0; i < N; ++i) {
          const double ov = offset_ptr[i];
          if (std::isfinite(ov)) out_eta(i) += static_cast<float>(ov);
        }
      }
    };

    // Phase A: Bohning majorization with H0 = 0.25 * G0 + Pen factored ONCE
    prepare_gram_f64(ws.G0_f32, ws, reg_lambda, 0.25);
    ws.ldlt.compute(ws.G_f64);

    compute_eta(W_cur, eta);
    double data_nll = compute_binary_nll_from_eta(eta.data(), y_ptr, N);
    double cur_loss = data_nll + compute_reg_penalty(W_cur, reg_lambda);

    const int bohning_iters = std::min(15, std::max(8, max_iter));
    for (int iter = 0; iter < bohning_iters; ++iter) {
      for (int i = 0; i < N; ++i) {
        const float e = std::clamp(eta(i), -40.0f, 40.0f);
        const float p = 1.0f / (1.0f + evofe::nn::fast_expf(-e));
        res(i) = p - static_cast<float>(y_ptr[i]);
      }
      evofe::nn::gemm_f32_raw(true, false, p_cols, 1, N,
                              Z_tr.data(), N,
                              res.data(), N,
                              ws.grad_f32.data(), p_cols,
                              1.0f, 0.0f);
      for (int j = 0; j < p_cols; ++j) {
        ws.rhs_f64(j, 0) = -(static_cast<double>(ws.grad_f32(j, 0)) + pen(j) * static_cast<double>(W_cur(j, 0)));
      }
      const Eigen::VectorXd delta = ws.ldlt.solve(ws.rhs_f64.col(0));
      if (delta.lpNorm<Eigen::Infinity>() < std::max(1e-4, tol * 10.0)) {
        break;
      }

      Eigen::MatrixXf delta_f32 = delta.cast<float>();
      evofe::nn::gemm_f32_raw(false, false, N, 1, p_cols,
                              Z_tr.data(), N,
                              delta_f32.data(), p_cols,
                              d_eta.data(), N,
                              1.0f, 0.0f);

      const double steps[5] = { 1.6, 1.0, 0.5, 0.25, 0.125 };
      bool accepted = false;
      for (int ls = 0; ls < 5; ++ls) {
        const float st = static_cast<float>(steps[ls]);
        for (int j = 0; j < p_cols; ++j) {
          W_cand(j, 0) = W_cur(j, 0) + st * delta_f32(j, 0);
        }
        for (int i = 0; i < N; ++i) {
          eta_cand(i) = eta(i) + st * d_eta(i);
        }
        const double cand_nll = compute_binary_nll_from_eta(eta_cand.data(), y_ptr, N);
        const double cand_loss = cand_nll + compute_reg_penalty(W_cand, reg_lambda);
        if (cand_loss <= cur_loss + 1e-12) {
          W_cur = W_cand;
          eta.swap(eta_cand);
          const double rel_imp = std::abs(cur_loss - cand_loss) / std::max(1.0, std::abs(cur_loss));
          data_nll = cand_nll;
          cur_loss = cand_loss;
          accepted = true;
          if (rel_imp < std::max(1e-6, tol)) {
            iter = bohning_iters; // jump to polish
          }
          break;
        }
      }
      if (!accepted) break;
    }

    // Phase B: Exact weighted IRLS polish (2-4 Newton steps)
    const int polish_iters = std::min(4, std::max(2, max_iter));
    for (int iter = 0; iter < polish_iters; ++iter) {
      for (int i = 0; i < N; ++i) {
        const float e = std::clamp(eta(i), -40.0f, 40.0f);
        const float p = 1.0f / (1.0f + evofe::nn::fast_expf(-e));
        res(i) = p - static_cast<float>(y_ptr[i]);
        sqrt_w(i) = std::sqrt(std::max(p * (1.0f - p), 1e-6f));
      }
      for (int j = 0; j < p_cols; ++j) {
        const float* z_col = Z_tr.data() + static_cast<size_t>(j) * N;
        float* zw_col = Z_w.data() + static_cast<size_t>(j) * N;
        const float* sw = sqrt_w.data();
        for (int i = 0; i < N; ++i) {
          zw_col[i] = z_col[i] * sw[i];
        }
      }
      evofe::nn::syrk_lower_f32(true, p_cols, N, Z_w.data(), N, ws.G_f32.data(), p_cols, 1.0f, 0.0f);
      prepare_gram_f64(ws.G_f32, ws, reg_lambda);
      ws.ldlt.compute(ws.G_f64);

      evofe::nn::gemm_f32_raw(true, false, p_cols, 1, N,
                              Z_tr.data(), N,
                              res.data(), N,
                              ws.grad_f32.data(), p_cols,
                              1.0f, 0.0f);
      for (int j = 0; j < p_cols; ++j) {
        ws.rhs_f64(j, 0) = -(static_cast<double>(ws.grad_f32(j, 0)) + pen(j) * static_cast<double>(W_cur(j, 0)));
      }
      const Eigen::VectorXd delta = ws.ldlt.solve(ws.rhs_f64.col(0));
      if (delta.lpNorm<Eigen::Infinity>() < tol) {
        break;
      }

      Eigen::MatrixXf delta_f32 = delta.cast<float>();
      evofe::nn::gemm_f32_raw(false, false, N, 1, p_cols,
                              Z_tr.data(), N,
                              delta_f32.data(), p_cols,
                              d_eta.data(), N,
                              1.0f, 0.0f);

      double alpha = 1.0;
      bool accepted = false;
      for (int ls = 0; ls < 6; ++ls) {
        const float st = static_cast<float>(alpha);
        for (int j = 0; j < p_cols; ++j) {
          W_cand(j, 0) = W_cur(j, 0) + st * delta_f32(j, 0);
        }
        for (int i = 0; i < N; ++i) {
          eta_cand(i) = eta(i) + st * d_eta(i);
        }
        const double cand_nll = compute_binary_nll_from_eta(eta_cand.data(), y_ptr, N);
        const double cand_loss = cand_nll + compute_reg_penalty(W_cand, reg_lambda);
        if (cand_loss <= cur_loss + 1e-12) {
          W_cur = W_cand;
          eta.swap(eta_cand);
          const double rel_imp = std::abs(cur_loss - cand_loss) / std::max(1.0, std::abs(cur_loss));
          data_nll = cand_nll;
          cur_loss = cand_loss;
          accepted = true;
          if (rel_imp < tol * 0.1) {
            iter = polish_iters;
          }
          break;
        }
        alpha *= 0.5;
      }
      if (!accepted) break;
    }

    if (compute_exact_hessian) {
      // Ensure ws.ldlt holds exact weighted Hessian at final W_cur
      for (int i = 0; i < N; ++i) {
        const float e = std::clamp(eta(i), -40.0f, 40.0f);
        const float p = 1.0f / (1.0f + evofe::nn::fast_expf(-e));
        sqrt_w(i) = std::sqrt(std::max(p * (1.0f - p), 1e-6f));
      }
      for (int j = 0; j < p_cols; ++j) {
        const float* z_col = Z_tr.data() + static_cast<size_t>(j) * N;
        float* zw_col = Z_w.data() + static_cast<size_t>(j) * N;
        const float* sw = sqrt_w.data();
        for (int i = 0; i < N; ++i) {
          zw_col[i] = z_col[i] * sw[i];
        }
      }
      evofe::nn::syrk_lower_f32(true, p_cols, N, Z_w.data(), N, ws.G_f32.data(), p_cols, 1.0f, 0.0f);
      prepare_gram_f64(ws.G_f32, ws, reg_lambda);
      ws.ldlt.compute(ws.G_f64);
    }

    return data_nll;
  }

  double solve_binary_streamed(const Eigen::MatrixXf& X_tr, const double* y_ptr,
                               const double* offset_ptr, int N, double reg_lambda,
                               int max_iter, double tol,
                               Eigen::MatrixXf& W_cur, RFFWorkspace& ws) const {
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());
    const int ld_zw = static_cast<int>(ws.Z_w_c.outerStride());
    const Eigen::VectorXd pen = make_pen_vector(reg_lambda);
    Eigen::MatrixXf W_cand(p_cols, 1);
    double data_nll = 0.0;
    double cur_loss = eval_binary_loss_streamed(X_tr, y_ptr, offset_ptr, N, W_cur, reg_lambda, ws, &data_nll);

    for (int iter = 0; iter < std::max(1, max_iter); ++iter) {
      ws.G_f32.setZero();
      ws.grad_f32.setZero();

      for (int start = 0; start < N; start += ws.chunk_size) {
        const int B = std::min(ws.chunk_size, N - start);
        compute_features_chunk(X_tr.data() + start, N, B, ws);
        eval_logits_chunk(B, W_cur, offset_ptr, N, start, ws);

        const float* eta = ws.eta_c.data();
        for (int i = 0; i < B; ++i) {
          const float e = std::clamp(eta[i], -40.0f, 40.0f);
          const float p = 1.0f / (1.0f + evofe::nn::fast_expf(-e));
          const float y = static_cast<float>(y_ptr[start + i]);
          ws.res_c(i, 0) = p - y;
          ws.prob_c(i, 0) = std::sqrt(std::max(p * (1.0f - p), 1e-6f));
        }

        const float* sw = ws.prob_c.data();
        for (int j = 0; j < p_cols; ++j) {
          const float* z_col = ws.Z_c.data() + static_cast<size_t>(j) * ld_z;
          float* zw_col = ws.Z_w_c.data() + static_cast<size_t>(j) * ld_zw;
          for (int i = 0; i < B; ++i) {
            zw_col[i] = z_col[i] * sw[i];
          }
        }

        evofe::nn::syrk_lower_f32(true, p_cols, B,
                                  ws.Z_w_c.data(), ld_zw,
                                  ws.G_f32.data(), p_cols,
                                  1.0f, 1.0f);
        evofe::nn::gemm_f32_raw(true, false, p_cols, 1, B,
                                ws.Z_c.data(), ld_z,
                                ws.res_c.data(), ws.chunk_size,
                                ws.grad_f32.data(), p_cols,
                                1.0f, 1.0f);
      }

      prepare_gram_f64(ws.G_f32, ws, reg_lambda);
      for (int j = 0; j < p_cols; ++j) {
        ws.rhs_f64(j, 0) = -(static_cast<double>(ws.grad_f32(j, 0)) + pen(j) * static_cast<double>(W_cur(j, 0)));
      }

      ws.ldlt.compute(ws.G_f64);
      const Eigen::VectorXd delta = ws.ldlt.solve(ws.rhs_f64.col(0));
      if (delta.lpNorm<Eigen::Infinity>() < tol) {
        break;
      }

      double alpha = 1.0;
      bool accepted = false;
      for (int ls = 0; ls < 8; ++ls) {
        for (int j = 0; j < p_cols; ++j) {
          W_cand(j, 0) = W_cur(j, 0) + static_cast<float>(alpha * delta(j));
        }
        double cand_nll = 0.0;
        const double cand_loss = eval_binary_loss_streamed(X_tr, y_ptr, offset_ptr, N, W_cand, reg_lambda, ws, &cand_nll);
        if (cand_loss <= cur_loss + 1e-12) {
          W_cur = W_cand;
          const double rel_imp = std::abs(cur_loss - cand_loss) / std::max(1.0, std::abs(cur_loss));
          data_nll = cand_nll;
          cur_loss = cand_loss;
          accepted = true;
          if (rel_imp < tol * 0.1) {
            return data_nll;
          }
          break;
        }
        alpha *= 0.5;
      }
      if (!accepted) break;
    }
    return data_nll;
  }

  void fit_binary_irls(const Eigen::MatrixXf& X_tr, const double* y_ptr,
                       const double* offset_ptr, int N, double lambda_req,
                       int max_iter, double tol,
                       ValidationContext& val_ctx, RFFWorkspace& ws) {
    if (N <= 0) return;
    W_out.setZero();

    Eigen::MatrixXf Z_tr_cache;
    const bool has_cached_z = prepare_G0_and_cache(X_tr, N, Z_tr_cache, ws);

    if (std::isfinite(lambda_req) && lambda_req > 0.0) {
      lambda = lambda_req;
      if (has_cached_z) {
        BinarySolveScratch sc;
        sc.init(N, p_cols, true);
        solve_binary_cached(Z_tr_cache, y_ptr, offset_ptr, N, lambda, max_iter, tol, false, W_out, sc, ws);
      } else {
        solve_binary_streamed(X_tr, y_ptr, offset_ptr, N, lambda, max_iter, tol, W_out, ws);
      }
      return;
    }

    // Auto-lambda descending warm-start path
    const double n_scale = std::max(0.25, static_cast<double>(N) / 500.0);
    const double mults[9] = { 64.0, 24.0, 8.0, 3.0, 1.0, 0.35, 0.12, 0.04, 0.015 };

    Eigen::MatrixXf Z_val;
    bool has_Z_val = false;
    if (val_ctx.active && val_ctx.n_val > 0) {
      has_Z_val = build_cached_Z_raw(val_ctx.x_ptr, val_ctx.n_val, val_ctx.d_val, Z_val, ws);
    }

    BinarySolveScratch sc;
    if (has_cached_z) {
      sc.init(N, p_cols, true);
    }

    Eigen::MatrixXf W_cur = Eigen::MatrixXf::Zero(p_cols, 1);
    Eigen::MatrixXf best_W = W_cur;
    double best_crit = std::numeric_limits<double>::quiet_NaN();
    double best_lam = 0.25 * n_scale;
    Eigen::MatrixXd cand_preds;
    int worse_streak = 0;

    for (int gi = 0; gi < 9; ++gi) {
      const double lam = mults[gi] * n_scale;
      const bool need_hess = !(val_ctx.active && val_ctx.n_val > 0);
      const double data_nll = has_cached_z
          ? solve_binary_cached(Z_tr_cache, y_ptr, offset_ptr, N, lam, max_iter, tol, need_hess, W_cur, sc, ws)
          : solve_binary_streamed(X_tr, y_ptr, offset_ptr, N, lam, max_iter, tol, W_cur, ws);

      if (val_ctx.active && val_ctx.n_val > 0) {
        bool higher_is_better = false;
        const double vscore = evaluate_candidate_val(W_cur, val_ctx, has_Z_val, Z_val,
                                                     ws, cand_preds, higher_is_better);
        if (is_better_score(vscore, best_crit, higher_is_better)) {
          best_crit = vscore;
          best_lam = lam;
          best_W = W_cur;
          val_ctx.best_preds = std::move(cand_preds);
          val_ctx.best_score = vscore;
          worse_streak = 0;
        } else if (gi >= 3) {
          worse_streak++;
          if (worse_streak >= 3) break;
        }
      } else {
        const Eigen::VectorXd pen = make_pen_vector(lam);
        const double df = std::max(0.0, static_cast<double>(p_cols) - compute_ldlt_pen_trace(ws.ldlt, pen));
        const double aic = (data_nll + df) / static_cast<double>(N);
        if (!std::isfinite(best_crit) || aic < best_crit) {
          best_crit = aic;
          best_lam = lam;
          best_W = W_cur;
          worse_streak = 0;
        } else if (gi >= 3) {
          worse_streak++;
          if (worse_streak >= 3) break;
        }
      }
    }

    lambda = best_lam;
    W_out = best_W;
  }

  // Compute multiclass softmax probabilities, data NLL, and optionally residuals & mean class weights
  static double multiclass_probs_and_nll(const float* eta_ptr, int ld_eta, const int* y_cls,
                                         int N, int out_dim_in,
                                         float* res_ptr, int ld_res,
                                         bool write_res,
                                         double* w_bar_out) {
    double data_nll = 0.0;
    if (w_bar_out != nullptr) {
      std::fill(w_bar_out, w_bar_out + out_dim_in, 0.0);
    }
    for (int i = 0; i < N; ++i) {
      float max_e = eta_ptr[i];
      for (int c = 1; c < out_dim_in; ++c) {
        const float ec = eta_ptr[static_cast<size_t>(c) * ld_eta + i];
        if (ec > max_e) max_e = ec;
      }
      double sum_exp = 0.0;
      float probs_buf[64];
      std::vector<float> probs_vec;
      float* p_row = probs_buf;
      if (out_dim_in > 64) {
        probs_vec.resize(out_dim_in);
        p_row = probs_vec.data();
      }
      for (int c = 0; c < out_dim_in; ++c) {
        const float ec = eta_ptr[static_cast<size_t>(c) * ld_eta + i];
        const float diff = std::clamp(ec - max_e, -80.0f, 0.0f);
        const float ev = evofe::nn::fast_expf(diff);
        p_row[c] = ev;
        sum_exp += static_cast<double>(ev);
      }
      const float inv_sum = (sum_exp > 0.0) ? static_cast<float>(1.0 / sum_exp)
                                            : (1.0f / static_cast<float>(out_dim_in));
      const int tc = y_cls[i];
      const float p_tc = std::max(p_row[tc] * inv_sum, 1e-15f);
      data_nll -= std::log(static_cast<double>(p_tc));
      if (write_res) {
        for (int c = 0; c < out_dim_in; ++c) {
          const float pc = p_row[c] * inv_sum;
          res_ptr[static_cast<size_t>(c) * ld_res + i] = pc - ((c == tc) ? 1.0f : 0.0f);
          if (w_bar_out != nullptr) {
            w_bar_out[c] += static_cast<double>(pc * (1.0f - pc));
          }
        }
      } else if (w_bar_out != nullptr) {
        for (int c = 0; c < out_dim_in; ++c) {
          const float pc = p_row[c] * inv_sum;
          w_bar_out[c] += static_cast<double>(pc * (1.0f - pc));
        }
      }
    }
    if (w_bar_out != nullptr && N > 0) {
      const double inv_N = 1.0 / static_cast<double>(N);
      for (int c = 0; c < out_dim_in; ++c) {
        w_bar_out[c] *= inv_N;
      }
    }
    return data_nll;
  }

  double solve_multiclass_cached(const Eigen::MatrixXf& Z_tr, const int* y_cls,
                                 const double* offset_ptr, int N, double reg_lambda,
                                 int max_iter, double tol,
                                 Eigen::MatrixXf& W_cur, std::vector<double>& w_bar_out,
                                 MulticlassSolveScratch& sc, RFFWorkspace& ws) const {
    const Eigen::VectorXd pen = make_pen_vector(reg_lambda);
    // Exact Bohning upper bound for softmax Hessian: H <= 0.5 * G0 + Pen (factored ONCE)
    prepare_gram_f64(ws.G0_f32, ws, reg_lambda, 0.5);
    ws.ldlt.compute(ws.G_f64);

    Eigen::MatrixXf& eta = sc.eta;
    Eigen::MatrixXf& d_eta = sc.d_eta;
    Eigen::MatrixXf& eta_cand = sc.eta_cand;
    Eigen::MatrixXf& res = sc.res;
    Eigen::MatrixXf& W_cand = sc.W_cand;
    Eigen::MatrixXf& delta_f32 = sc.delta_f32;
    Eigen::VectorXd& grad_col = sc.grad_col;
    Eigen::VectorXd& delta_col = sc.delta_col;
    w_bar_out.assign(out_dim, 0.0);

    eval_logits_from_Z(N, Z_tr.data(), N, p_cols, out_dim, W_cur, offset_ptr, N, 0, eta.data(), N);
    double data_nll = multiclass_probs_and_nll(eta.data(), N, y_cls, N, out_dim,
                                               res.data(), N, true, w_bar_out.data());
    double cur_loss = data_nll + compute_reg_penalty(W_cur, reg_lambda);

    const int n_iters = std::max(25, max_iter);
    const double steps[6] = { 1.8, 1.25, 1.0, 0.5, 0.25, 0.125 };

    for (int iter = 0; iter < n_iters; ++iter) {
      evofe::nn::gemm_f32_raw(true, false, p_cols, out_dim, N,
                              Z_tr.data(), N,
                              res.data(), N,
                              ws.grad_f32.data(), p_cols,
                              1.0f, 0.0f);

      double max_grad = 0.0;
      double slope = 0.0;
      for (int c = 0; c < out_dim; ++c) {
        for (int j = 0; j < p_cols; ++j) {
          const double g = static_cast<double>(ws.grad_f32(j, c)) + pen(j) * static_cast<double>(W_cur(j, c));
          grad_col(j) = g;
          const double ag = std::abs(g);
          if (ag > max_grad) max_grad = ag;
        }
        // Solve column-by-column with VectorXd so Eigen never dispatches to _dtrsm_
        delta_col = ws.ldlt.solve(-grad_col);
        slope += grad_col.dot(delta_col);
        for (int j = 0; j < p_cols; ++j) {
          delta_f32(j, c) = static_cast<float>(delta_col(j));
        }
      }

      if (max_grad / static_cast<double>(std::max(1, N)) < tol) {
        break;
      }

      evofe::nn::gemm_f32_raw(false, false, N, out_dim, p_cols,
                              Z_tr.data(), N,
                              delta_f32.data(), p_cols,
                              d_eta.data(), N,
                              1.0f, 0.0f);

      bool accepted = false;
      const size_t n_elems = static_cast<size_t>(N) * static_cast<size_t>(out_dim);
      const float* eta_raw = eta.data();
      const float* d_eta_raw = d_eta.data();
      float* eta_cand_raw = eta_cand.data();

      for (int ls = 0; ls < 6; ++ls) {
        const float st = static_cast<float>(steps[ls]);
        const size_t w_elems = static_cast<size_t>(p_cols) * static_cast<size_t>(out_dim);
        for (size_t idx = 0; idx < w_elems; ++idx) {
          W_cand.data()[idx] = W_cur.data()[idx] + st * delta_f32.data()[idx];
        }
        for (size_t idx = 0; idx < n_elems; ++idx) {
          eta_cand_raw[idx] = eta_raw[idx] + st * d_eta_raw[idx];
        }
        const double cand_nll = multiclass_probs_and_nll(eta_cand_raw, N, y_cls, N, out_dim,
                                                         nullptr, N, false, nullptr);
        const double cand_loss = cand_nll + compute_reg_penalty(W_cand, reg_lambda);
        if (cand_loss <= cur_loss + 1e-4 * steps[ls] * slope + 1e-12) {
          W_cur = W_cand;
          eta.swap(eta_cand);
          const double rel_imp = std::abs(cur_loss - cand_loss) / std::max(1.0, std::abs(cur_loss));
          data_nll = multiclass_probs_and_nll(eta.data(), N, y_cls, N, out_dim,
                                              res.data(), N, true, w_bar_out.data());
          cur_loss = cand_loss;
          accepted = true;
          if (rel_imp < tol * 0.1) {
            return data_nll;
          }
          break;
        }
      }
      if (!accepted) break;
    }

    return data_nll;
  }

  double eval_multiclass_streamed(const Eigen::MatrixXf& X_tr, const int* y_cls,
                                  const double* offset_ptr, int N,
                                  const Eigen::MatrixXf& W, double reg_lambda,
                                  bool compute_grad, RFFWorkspace& ws,
                                  double* data_nll_out = nullptr,
                                  std::vector<double>* w_bar_out = nullptr) const {
    double data_nll = 0.0;
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());
    const int ld_eta = static_cast<int>(ws.eta_c.outerStride());
    const int ld_res = static_cast<int>(ws.res_c.outerStride());
    if (compute_grad) ws.grad_f32.setZero();
    if (w_bar_out != nullptr) w_bar_out->assign(out_dim, 0.0);

    std::vector<double> chunk_w(out_dim, 0.0);
    for (int start = 0; start < N; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N - start);
      compute_features_chunk(X_tr.data() + start, N, B, ws);
      eval_logits_chunk(B, W, offset_ptr, N, start, ws);
      data_nll += multiclass_probs_and_nll(
          ws.eta_c.data(), ld_eta, y_cls + start, B, out_dim,
          ws.res_c.data(), ld_res, compute_grad,
          (w_bar_out != nullptr) ? chunk_w.data() : nullptr);
      if (w_bar_out != nullptr) {
        for (int c = 0; c < out_dim; ++c) {
          (*w_bar_out)[c] += chunk_w[c] * static_cast<double>(B);
        }
      }
      if (compute_grad) {
        evofe::nn::gemm_f32_raw(true, false, p_cols, out_dim, B,
                                ws.Z_c.data(), ld_z,
                                ws.res_c.data(), ld_res,
                                ws.grad_f32.data(), p_cols,
                                1.0f, 1.0f);
      }
    }
    if (w_bar_out != nullptr && N > 0) {
      const double inv_N = 1.0 / static_cast<double>(N);
      for (int c = 0; c < out_dim; ++c) (*w_bar_out)[c] *= inv_N;
    }
    if (data_nll_out != nullptr) *data_nll_out = data_nll;
    return data_nll + compute_reg_penalty(W, reg_lambda);
  }

  double solve_multiclass_streamed(const Eigen::MatrixXf& X_tr, const int* y_cls,
                                   const double* offset_ptr, int N, double reg_lambda,
                                   int max_iter, double tol,
                                   Eigen::MatrixXf& W_cur, std::vector<double>& w_bar_out,
                                   RFFWorkspace& ws) const {
    const Eigen::VectorXd pen = make_pen_vector(reg_lambda);
    prepare_gram_f64(ws.G0_f32, ws, reg_lambda, 0.5);
    ws.ldlt.compute(ws.G_f64);

    Eigen::MatrixXf W_cand(p_cols, out_dim), delta_f32(p_cols, out_dim);
    Eigen::VectorXd grad_col(p_cols), delta_col(p_cols);
    double data_nll = 0.0;
    double cur_loss = eval_multiclass_streamed(X_tr, y_cls, offset_ptr, N, W_cur,
                                               reg_lambda, true, ws, &data_nll, &w_bar_out);

    const int n_iters = std::max(25, max_iter);
    const double steps[6] = { 1.8, 1.25, 1.0, 0.5, 0.25, 0.125 };

    for (int iter = 0; iter < n_iters; ++iter) {
      double max_grad = 0.0;
      double slope = 0.0;
      for (int c = 0; c < out_dim; ++c) {
        for (int j = 0; j < p_cols; ++j) {
          const double g = static_cast<double>(ws.grad_f32(j, c)) + pen(j) * static_cast<double>(W_cur(j, c));
          grad_col(j) = g;
          const double ag = std::abs(g);
          if (ag > max_grad) max_grad = ag;
        }
        delta_col = ws.ldlt.solve(-grad_col);
        slope += grad_col.dot(delta_col);
        for (int j = 0; j < p_cols; ++j) {
          delta_f32(j, c) = static_cast<float>(delta_col(j));
        }
      }
      if (max_grad / static_cast<double>(std::max(1, N)) < tol) {
        break;
      }

      bool accepted = false;
      const size_t w_elems = static_cast<size_t>(p_cols) * static_cast<size_t>(out_dim);
      for (int ls = 0; ls < 6; ++ls) {
        const float st = static_cast<float>(steps[ls]);
        for (size_t idx = 0; idx < w_elems; ++idx) {
          W_cand.data()[idx] = W_cur.data()[idx] + st * delta_f32.data()[idx];
        }
        double cand_nll = 0.0;
        const double cand_loss = eval_multiclass_streamed(X_tr, y_cls, offset_ptr, N, W_cand,
                                                          reg_lambda, false, ws, &cand_nll, nullptr);
        if (cand_loss <= cur_loss + 1e-4 * steps[ls] * slope + 1e-12) {
          W_cur = W_cand;
          const double rel_imp = std::abs(cur_loss - cand_loss) / std::max(1.0, std::abs(cur_loss));
          cur_loss = eval_multiclass_streamed(X_tr, y_cls, offset_ptr, N, W_cur,
                                              reg_lambda, true, ws, &data_nll, &w_bar_out);
          accepted = true;
          if (rel_imp < tol * 0.1) {
            return data_nll;
          }
          break;
        }
      }
      if (!accepted) break;
    }
    return data_nll;
  }

  double estimate_multiclass_df(const std::vector<double>& w_bar, double reg_lambda, RFFWorkspace& ws) const {
    const Eigen::VectorXd pen = make_pen_vector(reg_lambda);
    if (out_dim <= 6) {
      double df_tot = 0.0;
      for (int c = 0; c < out_dim; ++c) {
        const double wc = std::max(1e-6, w_bar[c]);
        prepare_gram_f64(ws.G0_f32, ws, reg_lambda, wc);
        ws.ldlt.compute(ws.G_f64);
        df_tot += std::max(0.0, static_cast<double>(p_cols) - compute_ldlt_pen_trace(ws.ldlt, pen));
      }
      return df_tot;
    }
    double sum_w = 0.0;
    for (int c = 0; c < out_dim; ++c) sum_w += w_bar[c];
    const double wc = std::max(1e-6, sum_w / static_cast<double>(out_dim));
    prepare_gram_f64(ws.G0_f32, ws, reg_lambda, wc);
    ws.ldlt.compute(ws.G_f64);
    const double df_one = std::max(0.0, static_cast<double>(p_cols) - compute_ldlt_pen_trace(ws.ldlt, pen));
    return df_one * static_cast<double>(out_dim);
  }

  void fit_multiclass_lbfgs(const Eigen::MatrixXf& X_tr, const int* y_cls,
                            const double* offset_ptr, int N, double lambda_req,
                            int max_iter, double tol,
                            ValidationContext& val_ctx, RFFWorkspace& ws) {
    if (N <= 0) return;
    W_out.setZero();

    Eigen::MatrixXf Z_tr_cache;
    const bool has_cached_z = prepare_G0_and_cache(X_tr, N, Z_tr_cache, ws);
    std::vector<double> w_bar(out_dim, 0.0);

    if (std::isfinite(lambda_req) && lambda_req > 0.0) {
      lambda = lambda_req;
      if (has_cached_z) {
        MulticlassSolveScratch sc;
        sc.init(N, p_cols, out_dim);
        solve_multiclass_cached(Z_tr_cache, y_cls, offset_ptr, N, lambda, max_iter, tol, W_out, w_bar, sc, ws);
      } else {
        solve_multiclass_streamed(X_tr, y_cls, offset_ptr, N, lambda, max_iter, tol, W_out, w_bar, ws);
      }
      return;
    }

    // Auto-lambda descending warm-start path
    const double n_scale = std::max(0.25, static_cast<double>(N) / 500.0);
    const double mults[9] = { 64.0, 24.0, 8.0, 3.0, 1.0, 0.35, 0.12, 0.04, 0.015 };

    Eigen::MatrixXf Z_val;
    bool has_Z_val = false;
    if (val_ctx.active && val_ctx.n_val > 0) {
      has_Z_val = build_cached_Z_raw(val_ctx.x_ptr, val_ctx.n_val, val_ctx.d_val, Z_val, ws);
    }

    MulticlassSolveScratch sc;
    if (has_cached_z) {
      sc.init(N, p_cols, out_dim);
    }

    Eigen::MatrixXf W_cur = Eigen::MatrixXf::Zero(p_cols, out_dim);
    Eigen::MatrixXf best_W = W_cur;
    double best_crit = std::numeric_limits<double>::quiet_NaN();
    double best_lam = 1.0 * n_scale;
    Eigen::MatrixXd cand_preds;
    int worse_streak = 0;

    for (int gi = 0; gi < 9; ++gi) {
      const double lam = mults[gi] * n_scale;
      const double data_nll = has_cached_z
          ? solve_multiclass_cached(Z_tr_cache, y_cls, offset_ptr, N, lam, max_iter, tol, W_cur, w_bar, sc, ws)
          : solve_multiclass_streamed(X_tr, y_cls, offset_ptr, N, lam, max_iter, tol, W_cur, w_bar, ws);

      if (val_ctx.active && val_ctx.n_val > 0) {
        bool higher_is_better = false;
        const double vscore = evaluate_candidate_val(W_cur, val_ctx, has_Z_val, Z_val,
                                                     ws, cand_preds, higher_is_better);
        if (is_better_score(vscore, best_crit, higher_is_better)) {
          best_crit = vscore;
          best_lam = lam;
          best_W = W_cur;
          val_ctx.best_preds = std::move(cand_preds);
          val_ctx.best_score = vscore;
          worse_streak = 0;
        } else if (gi >= 3) {
          worse_streak++;
          if (worse_streak >= 3) break;
        }
      } else {
        const double df_tot = estimate_multiclass_df(w_bar, lam, ws);
        const double aic = (data_nll + df_tot) / static_cast<double>(N);
        if (!std::isfinite(best_crit) || aic < best_crit) {
          best_crit = aic;
          best_lam = lam;
          best_W = W_cur;
          worse_streak = 0;
        } else if (gi >= 3) {
          worse_streak++;
          if (worse_streak >= 3) break;
        }
      }
    }

    lambda = best_lam;
    W_out = best_W;
  }

  // Fast permutation-free mean occlusion ablation on up to 1000 rows
  std::vector<double> compute_importances(const Eigen::MatrixXf& X_tr, int N,
                                          const double* offset_ptr, RFFWorkspace& ws) const {
    std::vector<double> imps(std::max(0, d_in), 0.0);
    if (N <= 0 || d_in <= 0) return imps;

    const int N_imp = std::min(N, 1000);
    Eigen::MatrixXf X_sub(N_imp, d_in);
    for (int d = 0; d < d_in; ++d) {
      std::memcpy(X_sub.data() + static_cast<size_t>(d) * N_imp,
                  X_tr.data() + static_cast<size_t>(d) * N,
                  static_cast<size_t>(N_imp) * sizeof(float));
    }

    const int ld_eta = static_cast<int>(ws.eta_c.outerStride());
    auto eval_preds = [&](Eigen::MatrixXd& out_preds) {
      compute_features_chunk(X_sub.data(), N_imp, N_imp, ws);
      eval_logits_chunk(N_imp, W_out, offset_ptr, N, 0, ws);
      out_preds.resize(N_imp, out_dim);
      logits_to_preds(ws.eta_c.data(), ld_eta, N_imp, 0, task, out_dim, out_preds);
    };

    Eigen::MatrixXd base_preds, abl_preds;
    eval_preds(base_preds);

    std::vector<float> saved_col(N_imp);
    for (int d = 0; d < d_in; ++d) {
      float* col_ptr = X_sub.data() + static_cast<size_t>(d) * N_imp;
      std::memcpy(saved_col.data(), col_ptr, static_cast<size_t>(N_imp) * sizeof(float));
      std::fill(col_ptr, col_ptr + N_imp, 0.0f);

      eval_preds(abl_preds);
      double diff_sum = 0.0;
      for (int c = 0; c < out_dim; ++c) {
        for (int i = 0; i < N_imp; ++i) {
          diff_sum += std::abs(base_preds(i, c) - abl_preds(i, c));
        }
      }
      imps[d] = diff_sum / static_cast<double>(N_imp * out_dim);
      std::memcpy(col_ptr, saved_col.data(), static_cast<size_t>(N_imp) * sizeof(float));
    }
    return imps;
  }
};

} // namespace rff
} // namespace evofe

#endif // EVOFE_RFF_CORE_H
