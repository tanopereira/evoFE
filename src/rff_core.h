#ifndef EVOFE_RFF_CORE_H
#define EVOFE_RFF_CORE_H

#include "nn_common.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace evofe {
namespace rff {

constexpr int RFF_CHUNK_SIZE = 4096;

// Single-pass fused standardization into ColMajor FP32:
// Non-finite values are imputed to training mean (which maps to 0.0f after centering).
inline void standardize_impute_mean_f32(
    const double* __restrict__ src, int src_lda,
    float* __restrict__ dst, int dst_lda,
    int N, int D,
    const double* __restrict__ x_mean,
    const double* __restrict__ x_inv_std) {
  if (N <= 0 || D <= 0) return;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (D >= 2 && N >= 64)
#endif
  for (int j = 0; j < D; ++j) {
    const double* src_col = src + static_cast<size_t>(j) * src_lda;
    float* dst_col = dst + static_cast<size_t>(j) * dst_lda;
    const double m = x_mean[j];
    const double inv_s = x_inv_std[j];
    for (int i = 0; i < N; ++i) {
      const double v = src_col[i];
      if (std::isfinite(v)) {
        const double z = (v - m) * inv_s;
        dst_col[i] = static_cast<float>(std::clamp(z, -30.0, 30.0));
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
  Eigen::MatrixXf grad_f32;
  Eigen::MatrixXd G_f64;
  Eigen::MatrixXd rhs_f64;
  Eigen::LDLT<Eigen::MatrixXd> ldlt;

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
      grad_f32.resize(p_cols, out_dim);
      G_f64.resize(p_cols, p_cols);
      rhs_f64.resize(p_cols, out_dim);
    }
  }
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
  double lambda = 1e-2;

  Eigen::VectorXd x_mean;
  Eigen::VectorXd x_sd;
  Eigen::VectorXd x_inv_std;

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

  // Compute Z_c(0..B-1, 0..p_cols-1) from standardized X_c(0..B-1, 0..d_in-1)
  void compute_features_chunk(const float* X_c_ptr, int lda_x, int B, RFFWorkspace& ws) const {
    if (B <= 0) return;
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());

    if (d_in <= 0 || (approx_method == "rff" && k_freq <= 0) ||
        (approx_method == "nystrom" && m_landmarks <= 0)) {
      float* intercept_col = ws.Z_c.data() + static_cast<size_t>(p_cols - 1) * ld_z;
      std::fill(intercept_col, intercept_col + B, 1.0f);
      return;
    }

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
        float* cos_col = ws.Z_c.data() + static_cast<size_t>(k) * ld_z;
        float* sin_col = ws.Z_c.data() + static_cast<size_t>(k + k_freq) * ld_z;
        evofe::nn::vcos_f32(B, in_col, cos_col);
        evofe::nn::vsin_f32(B, in_col, sin_col);
        for (int i = 0; i < B; ++i) {
          cos_col[i] *= scale;
          sin_col[i] *= scale;
        }
      }
      float* intercept_col = ws.Z_c.data() + static_cast<size_t>(p_cols - 1) * ld_z;
      std::fill(intercept_col, intercept_col + B, 1.0f);
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
      const float inv_2sig2 = -0.5f / static_cast<float>(sigma * sigma);
      const float inv_sig = -1.0f / static_cast<float>(sigma);
      const float* x_n2 = ws.x_norm2_c.data();
      const float* l_n2 = landmark_norm2.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (m_landmarks >= 4 && B >= 64)
#endif
      for (int j = 0; j < m_landmarks; ++j) {
        float* k_col = ws.K_cm_c.data() + static_cast<size_t>(j) * ld_k;
        const float ln2 = l_n2[j];
        if (is_rbf) {
          for (int i = 0; i < B; ++i) {
            const float d2 = std::max(0.0f, x_n2[i] + ln2 + k_col[i]);
            k_col[i] = d2 * inv_2sig2;
          }
        } else {
          for (int i = 0; i < B; ++i) {
            const float d2 = std::max(0.0f, x_n2[i] + ln2 + k_col[i]);
            k_col[i] = std::sqrt(d2) * inv_sig;
          }
        }
        evofe::nn::vexp_f32(B, k_col, k_col);
      }

      evofe::nn::gemm_f32_raw(false, false, B, m_landmarks, m_landmarks,
                              ws.K_cm_c.data(), ld_k,
                              P_nys.data(), m_landmarks,
                              ws.Z_c.data(), ld_z,
                              1.0f, 0.0f);
      float* intercept_col = ws.Z_c.data() + static_cast<size_t>(p_cols - 1) * ld_z;
      std::fill(intercept_col, intercept_col + B, 1.0f);
    }
  }

  // Compute linear logits eta_c = Z_c * W + offset_c for chunk [start, start + B)
  void eval_logits_chunk(int B, const Eigen::MatrixXf& W,
                         const double* offset_ptr, int offset_lda, int row_start,
                         RFFWorkspace& ws) const {
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());
    const int ld_eta = static_cast<int>(ws.eta_c.outerStride());
    evofe::nn::gemm_f32_raw(false, false, B, out_dim, p_cols,
                            ws.Z_c.data(), ld_z,
                            W.data(), p_cols,
                            ws.eta_c.data(), ld_eta,
                            1.0f, 0.0f);
    if (offset_ptr != nullptr) {
      for (int c = 0; c < out_dim; ++c) {
        const double* off_col = offset_ptr + static_cast<size_t>(c) * offset_lda + row_start;
        float* eta_col = ws.eta_c.data() + static_cast<size_t>(c) * ld_eta;
        for (int i = 0; i < B; ++i) {
          const double ov = off_col[i];
          if (std::isfinite(ov)) {
            eta_col[i] += static_cast<float>(ov);
          }
        }
      }
    }
  }

  // Predict on raw (unstandardized) ColMajor double matrix X_raw(N_new, d_in)
  Eigen::MatrixXd predict_raw(const double* X_ptr, int N_new, int D_new,
                              const double* offset_ptr, RFFWorkspace& ws) const {
    Eigen::MatrixXd preds(N_new, out_dim);
    if (N_new <= 0) return preds;

    const int D_use = std::min(d_in, D_new);
    const int ld_eta = static_cast<int>(ws.eta_c.outerStride());

    for (int start = 0; start < N_new; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N_new - start);
      if (D_use > 0) {
        standardize_impute_mean_f32(
            X_ptr + start, N_new,
            ws.X_c.data(), ws.chunk_size,
            B, D_use,
            x_mean.data(), x_inv_std.data());
        if (D_use < d_in) {
          for (int j = D_use; j < d_in; ++j) {
            float* dst_col = ws.X_c.data() + static_cast<size_t>(j) * ws.chunk_size;
            std::fill(dst_col, dst_col + B, 0.0f);
          }
        }
      }
      compute_features_chunk(ws.X_c.data(), ws.chunk_size, B, ws);
      eval_logits_chunk(B, W_out, offset_ptr, N_new, start, ws);

      if (task == "regression") {
        for (int i = 0; i < B; ++i) {
          preds(start + i, 0) = static_cast<double>(ws.eta_c(i, 0));
        }
      } else if (task == "classification") {
        const float* eta_col = ws.eta_c.data();
        for (int i = 0; i < B; ++i) {
          const float e = std::clamp(eta_col[i], -40.0f, 40.0f);
          preds(start + i, 0) = static_cast<double>(1.0f / (1.0f + evofe::nn::fast_expf(-e)));
        }
      } else {
        // multiclass softmax
        for (int i = 0; i < B; ++i) {
          float max_e = ws.eta_c(i, 0);
          for (int c = 1; c < out_dim; ++c) {
            if (ws.eta_c(i, c) > max_e) max_e = ws.eta_c(i, c);
          }
          double sum_exp = 0.0;
          for (int c = 0; c < out_dim; ++c) {
            const float diff = std::clamp(ws.eta_c(i, c) - max_e, -80.0f, 0.0f);
            const double ev = static_cast<double>(evofe::nn::fast_expf(diff));
            preds(start + i, c) = ev;
            sum_exp += ev;
          }
          const double inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : (1.0 / out_dim);
          for (int c = 0; c < out_dim; ++c) {
            preds(start + i, c) *= inv_sum;
          }
        }
      }
    }
    return preds;
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
      W_out = Eigen::MatrixXf::Zero(p_cols, out_dim);
      return;
    }

    if (approx_method == "rff") {
      k_freq = std::max(1, n_features / 2);
      p_cols = 2 * k_freq + 1;
      rff_scale = 1.0f / std::sqrt(static_cast<float>(k_freq));
      W_rff.resize(d_in, k_freq);

      // Multi-scale: 0.5 * sigma, 1.0 * sigma, 2.0 * sigma
      int k0 = k_freq / 3;
      int k1 = k_freq / 3;
      int k2 = k_freq - k0 - k1;
      if (k_freq < 3) {
        k0 = 0;
        k1 = k_freq;
        k2 = 0;
      }
      const double scales[3] = { 0.5 * sigma, 1.0 * sigma, 2.0 * sigma };
      const int counts[3] = { k0, k1, k2 };

      std::normal_distribution<double> norm_dist(0.0, 1.0);
      std::uniform_real_distribution<double> unif_dist(1e-6, 1.0 - 1e-6);
      const bool is_rbf = (kernel == "rbf");

      int col = 0;
      for (int s_idx = 0; s_idx < 3; ++s_idx) {
        const double inv_s = 1.0 / std::max(1e-6, scales[s_idx]);
        for (int c = 0; c < counts[s_idx]; ++c) {
          for (int d = 0; d < d_in; ++d) {
            if (is_rbf) {
              W_rff(d, col) = static_cast<float>(norm_dist(rng) * inv_s);
            } else {
              const double u = unif_dist(rng);
              W_rff(d, col) = static_cast<float>(std::tan(M_PI * (u - 0.5)) * inv_s);
            }
          }
          col++;
        }
      }
    } else {
      // Nystrom
      m_landmarks = std::min(std::max(1, n_features), N);
      p_cols = m_landmarks + 1;

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

      // Compute K_mm = exp(-...) in FP32 via syrk_lower_f32
      Eigen::MatrixXf K_mm_f32 = Eigen::MatrixXf::Zero(m_landmarks, m_landmarks);
      evofe::nn::syrk_lower_f32(false, m_landmarks, d_in,
                                landmarks.data(), m_landmarks,
                                K_mm_f32.data(), m_landmarks,
                                -2.0f, 0.0f);

      Eigen::MatrixXd K_mm_f64(m_landmarks, m_landmarks);
      const bool is_rbf = (kernel == "rbf");
      const double inv_2sig2 = -0.5 / (sigma * sigma);
      const double inv_sig = -1.0 / sigma;

      for (int j = 0; j < m_landmarks; ++j) {
        for (int i = j; i < m_landmarks; ++i) {
          const double d2 = std::max(0.0,
              static_cast<double>(landmark_norm2(i)) +
              static_cast<double>(landmark_norm2(j)) +
              static_cast<double>(K_mm_f32(i, j)));
          double val = is_rbf ? std::exp(d2 * inv_2sig2) : std::exp(std::sqrt(d2) * inv_sig);
          if (i == j) val += 1e-4; // diagonal jitter
          K_mm_f64(i, j) = val;
          K_mm_f64(j, i) = val;
        }
      }

      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(K_mm_f64);
      const Eigen::VectorXd& evals = eig.eigenvalues();
      const Eigen::MatrixXd& evecs = eig.eigenvectors();
      const double max_ev = std::max(1e-8, evals.maxCoeff());
      const double floor_ev = std::max(1e-5, 1e-6 * max_ev);

      Eigen::VectorXd inv_sqrt_ev(m_landmarks);
      for (int i = 0; i < m_landmarks; ++i) {
        inv_sqrt_ev(i) = 1.0 / std::sqrt(std::max(evals(i), floor_ev));
      }
      Eigen::MatrixXd P_f64 = evecs * inv_sqrt_ev.asDiagonal() * evecs.transpose();
      P_nys = P_f64.cast<float>();
    }
    W_out = Eigen::MatrixXf::Zero(p_cols, out_dim);
  }

  // Symmetrize lower-triangular FP32 Gram matrix into FP64 G_f64 and add ridge
  void prepare_gram_f64(RFFWorkspace& ws, double reg_lambda) const {
    for (int j = 0; j < p_cols; ++j) {
      for (int i = j; i < p_cols; ++i) {
        const double v = static_cast<double>(ws.G_f32(i, j));
        ws.G_f64(i, j) = v;
        ws.G_f64(j, i) = v;
      }
      const double diag_reg = (j < p_cols - 1) ? reg_lambda : 1e-8;
      ws.G_f64(j, j) += diag_reg;
    }
  }

  void fit_regression(const Eigen::MatrixXf& X_tr, const double* y_ptr,
                      const double* offset_ptr, int N, RFFWorkspace& ws) {
    if (N <= 0) return;
    ws.G_f32.setZero();
    ws.grad_f32.setZero();
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());

    for (int start = 0; start < N; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N - start);
      compute_features_chunk(X_tr.data() + start, N, B, ws);
      evofe::nn::syrk_lower_f32(true, p_cols, B,
                                ws.Z_c.data(), ld_z,
                                ws.G_f32.data(), p_cols,
                                1.0f, 1.0f);
      for (int i = 0; i < B; ++i) {
        const double off = (offset_ptr != nullptr && std::isfinite(offset_ptr[start + i]))
                               ? offset_ptr[start + i] : 0.0;
        ws.res_c(i, 0) = static_cast<float>(y_ptr[start + i] - off);
      }
      evofe::nn::gemm_f32_raw(true, false, p_cols, 1, B,
                              ws.Z_c.data(), ld_z,
                              ws.res_c.data(), ws.chunk_size,
                              ws.grad_f32.data(), p_cols,
                              1.0f, 1.0f);
    }

    prepare_gram_f64(ws, lambda);
    ws.rhs_f64.col(0) = ws.grad_f32.col(0).cast<double>();
    ws.ldlt.compute(ws.G_f64);
    Eigen::VectorXd w_sol = ws.ldlt.solve(ws.rhs_f64.col(0));
    W_out.col(0) = w_sol.cast<float>();
  }

  double eval_binary_loss(const Eigen::MatrixXf& X_tr, const double* y_ptr,
                          const double* offset_ptr, int N,
                          const Eigen::MatrixXf& W, RFFWorkspace& ws) const {
    double loss = 0.0;
    for (int start = 0; start < N; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N - start);
      compute_features_chunk(X_tr.data() + start, N, B, ws);
      eval_logits_chunk(B, W, offset_ptr, N, start, ws);
      const float* eta = ws.eta_c.data();
      for (int i = 0; i < B; ++i) {
        const double e = static_cast<double>(eta[i]);
        const double y = y_ptr[start + i];
        // log(1 + exp(e)) - y * e, numerically stable
        const double softplus = (e > 0.0) ? (e + std::log1p(std::exp(-e))) : std::log1p(std::exp(e));
        loss += softplus - y * e;
      }
    }
    double reg = 0.0;
    for (int j = 0; j < p_cols - 1; ++j) {
      const double wj = static_cast<double>(W(j, 0));
      reg += wj * wj;
    }
    return loss + 0.5 * lambda * reg;
  }

  void fit_binary_irls(const Eigen::MatrixXf& X_tr, const double* y_ptr,
                       const double* offset_ptr, int N,
                       int max_iter, double tol, RFFWorkspace& ws) {
    if (N <= 0) return;
    W_out.setZero();
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());
    const int ld_zw = static_cast<int>(ws.Z_w_c.outerStride());

    Eigen::MatrixXf W_cand(p_cols, 1);
    double cur_loss = eval_binary_loss(X_tr, y_ptr, offset_ptr, N, W_out, ws);

    for (int iter = 0; iter < std::max(1, max_iter); ++iter) {
      ws.G_f32.setZero();
      ws.grad_f32.setZero();

      for (int start = 0; start < N; start += ws.chunk_size) {
        const int B = std::min(ws.chunk_size, N - start);
        compute_features_chunk(X_tr.data() + start, N, B, ws);
        eval_logits_chunk(B, W_out, offset_ptr, N, start, ws);

        const float* eta = ws.eta_c.data();
        for (int i = 0; i < B; ++i) {
          const float e = std::clamp(eta[i], -40.0f, 40.0f);
          const float p = 1.0f / (1.0f + evofe::nn::fast_expf(-e));
          const float y = static_cast<float>(y_ptr[start + i]);
          ws.res_c(i, 0) = p - y;
          ws.prob_c(i, 0) = std::sqrt(std::max(p * (1.0f - p), 1e-6f));
        }

        // Weighted Z_w_c = diag(sqrt(w)) * Z_c
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

      prepare_gram_f64(ws, lambda);
      for (int j = 0; j < p_cols; ++j) {
        const double pen = (j < p_cols - 1) ? (lambda * static_cast<double>(W_out(j, 0))) : 0.0;
        ws.rhs_f64(j, 0) = -(static_cast<double>(ws.grad_f32(j, 0)) + pen);
      }

      ws.ldlt.compute(ws.G_f64);
      Eigen::VectorXd delta = ws.ldlt.solve(ws.rhs_f64.col(0));
      if (delta.lpNorm<Eigen::Infinity>() < tol) {
        break;
      }

      // Damped Armijo backtracking line search
      double alpha = 1.0;
      bool accepted = false;
      for (int ls = 0; ls < 8; ++ls) {
        for (int j = 0; j < p_cols; ++j) {
          W_cand(j, 0) = W_out(j, 0) + static_cast<float>(alpha * delta(j));
        }
        const double cand_loss = eval_binary_loss(X_tr, y_ptr, offset_ptr, N, W_cand, ws);
        if (cand_loss <= cur_loss + 1e-12) {
          W_out = W_cand;
          const double rel_imp = std::abs(cur_loss - cand_loss) / std::max(1.0, std::abs(cur_loss));
          cur_loss = cand_loss;
          accepted = true;
          if (rel_imp < tol * 0.1) {
            return;
          }
          break;
        }
        alpha *= 0.5;
      }
      if (!accepted) break;
    }
  }

  // Evaluate multiclass cross-entropy loss + L2 penalty and optionally accumulate gradient
  double eval_multiclass_loss_grad(const Eigen::MatrixXf& X_tr, const int* y_cls,
                                   const double* offset_ptr, int N,
                                   const Eigen::MatrixXf& W,
                                   Eigen::MatrixXd* grad_out,
                                   RFFWorkspace& ws) const {
    double loss = 0.0;
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());
    const int ld_eta = static_cast<int>(ws.eta_c.outerStride());
    const int ld_res = static_cast<int>(ws.res_c.outerStride());
    if (grad_out != nullptr) {
      ws.grad_f32.setZero();
    }

    for (int start = 0; start < N; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N - start);
      compute_features_chunk(X_tr.data() + start, N, B, ws);
      eval_logits_chunk(B, W, offset_ptr, N, start, ws);

      for (int i = 0; i < B; ++i) {
        float max_e = ws.eta_c(i, 0);
        for (int c = 1; c < out_dim; ++c) {
          if (ws.eta_c(i, c) > max_e) max_e = ws.eta_c(i, c);
        }
        double sum_exp = 0.0;
        for (int c = 0; c < out_dim; ++c) {
          const float diff = std::clamp(ws.eta_c(i, c) - max_e, -80.0f, 0.0f);
          const float ev = evofe::nn::fast_expf(diff);
          ws.res_c(i, c) = ev;
          sum_exp += static_cast<double>(ev);
        }
        const float inv_sum = (sum_exp > 0.0) ? static_cast<float>(1.0 / sum_exp)
                                              : (1.0f / static_cast<float>(out_dim));
        const int tc = y_cls[start + i];
        const float p_tc = std::max(ws.res_c(i, tc) * inv_sum, 1e-15f);
        loss -= std::log(static_cast<double>(p_tc));
        if (grad_out != nullptr) {
          for (int c = 0; c < out_dim; ++c) {
            ws.res_c(i, c) = ws.res_c(i, c) * inv_sum - ((c == tc) ? 1.0f : 0.0f);
          }
        }
      }

      if (grad_out != nullptr) {
        evofe::nn::gemm_f32_raw(true, false, p_cols, out_dim, B,
                                ws.Z_c.data(), ld_z,
                                ws.res_c.data(), ld_res,
                                ws.grad_f32.data(), p_cols,
                                1.0f, 1.0f);
      }
    }

    double reg = 0.0;
    for (int c = 0; c < out_dim; ++c) {
      for (int j = 0; j < p_cols - 1; ++j) {
        const double w = static_cast<double>(W(j, c));
        reg += w * w;
      }
    }
    if (grad_out != nullptr) {
      for (int c = 0; c < out_dim; ++c) {
        for (int j = 0; j < p_cols; ++j) {
          const double pen = (j < p_cols - 1) ? (lambda * static_cast<double>(W(j, c))) : 0.0;
          (*grad_out)(j, c) = static_cast<double>(ws.grad_f32(j, c)) + pen;
        }
      }
    }
    return loss + 0.5 * lambda * reg;
  }

  void fit_multiclass_lbfgs(const Eigen::MatrixXf& X_tr, const int* y_cls,
                            const double* offset_ptr, int N,
                            int max_iter, double tol, RFFWorkspace& ws) {
    if (N <= 0) return;
    // Ridge warm start (one-vs-rest)
    ws.G_f32.setZero();
    ws.grad_f32.setZero();
    const int ld_z = static_cast<int>(ws.Z_c.outerStride());
    const int ld_res = static_cast<int>(ws.res_c.outerStride());
    for (int start = 0; start < N; start += ws.chunk_size) {
      const int B = std::min(ws.chunk_size, N - start);
      compute_features_chunk(X_tr.data() + start, N, B, ws);
      evofe::nn::syrk_lower_f32(true, p_cols, B,
                                ws.Z_c.data(), ld_z,
                                ws.G_f32.data(), p_cols,
                                1.0f, 1.0f);
      for (int c = 0; c < out_dim; ++c) {
        float* r_col = ws.res_c.data() + static_cast<size_t>(c) * ld_res;
        for (int i = 0; i < B; ++i) {
          r_col[i] = (y_cls[start + i] == c) ? 1.0f : -1.0f;
        }
      }
      evofe::nn::gemm_f32_raw(true, false, p_cols, out_dim, B,
                              ws.Z_c.data(), ld_z,
                              ws.res_c.data(), ld_res,
                              ws.grad_f32.data(), p_cols,
                              1.0f, 1.0f);
    }
    prepare_gram_f64(ws, std::max(lambda, 1.0));
    ws.rhs_f64 = ws.grad_f32.cast<double>();
    ws.ldlt.compute(ws.G_f64);
    Eigen::MatrixXd W_f64 = ws.ldlt.solve(ws.rhs_f64);
    W_out = W_f64.cast<float>();

    // L-BFGS (m = 10)
    const int dim = p_cols * out_dim;
    const int m_hist = 10;
    std::vector<Eigen::VectorXd> s_hist, y_hist;
    std::vector<double> rho_hist;
    s_hist.reserve(m_hist);
    y_hist.reserve(m_hist);
    rho_hist.reserve(m_hist);

    Eigen::MatrixXd grad_mat(p_cols, out_dim);
    double cur_loss = eval_multiclass_loss_grad(X_tr, y_cls, offset_ptr, N, W_out, &grad_mat, ws);
    Eigen::VectorXd g_vec = Eigen::Map<const Eigen::VectorXd>(grad_mat.data(), dim);
    Eigen::VectorXd w_vec = Eigen::Map<const Eigen::VectorXd>(W_f64.data(), dim);
    Eigen::MatrixXf W_cand(p_cols, out_dim);

    const int n_iters = std::max(1, max_iter);
    for (int iter = 0; iter < n_iters; ++iter) {
      if (g_vec.lpNorm<Eigen::Infinity>() / std::max(1, N) < tol) {
        break;
      }

      // Two-loop recursion
      Eigen::VectorXd q = g_vec;
      const int k_cur = static_cast<int>(s_hist.size());
      std::vector<double> alpha_lbfgs(k_cur, 0.0);
      for (int i = k_cur - 1; i >= 0; --i) {
        alpha_lbfgs[i] = rho_hist[i] * s_hist[i].dot(q);
        q.noalias() -= alpha_lbfgs[i] * y_hist[i];
      }
      double gamma = 1.0 / std::max(1.0, static_cast<double>(N));
      if (k_cur > 0) {
        const double yy = y_hist.back().squaredNorm();
        if (yy > 1e-16) {
          gamma = s_hist.back().dot(y_hist.back()) / yy;
        }
      }
      Eigen::VectorXd r = gamma * q;
      for (int i = 0; i < k_cur; ++i) {
        const double beta = rho_hist[i] * y_hist[i].dot(r);
        r.noalias() += (alpha_lbfgs[i] - beta) * s_hist[i];
      }
      Eigen::VectorXd dir = -r;
      double slope = g_vec.dot(dir);
      if (slope >= 0.0) {
        dir = -g_vec;
        slope = -g_vec.squaredNorm();
      }

      double step = 1.0;
      bool accepted = false;
      Eigen::MatrixXd new_grad_mat(p_cols, out_dim);
      double new_loss = cur_loss;
      Eigen::VectorXd w_new = w_vec;

      for (int ls = 0; ls < 10; ++ls) {
        w_new = w_vec + step * dir;
        for (int j = 0; j < dim; ++j) {
          W_cand.data()[j] = static_cast<float>(w_new(j));
        }
        new_loss = eval_multiclass_loss_grad(X_tr, y_cls, offset_ptr, N, W_cand, &new_grad_mat, ws);
        if (new_loss <= cur_loss + 1e-4 * step * slope) {
          accepted = true;
          break;
        }
        step *= 0.5;
      }

      if (!accepted) break;

      Eigen::VectorXd g_new = Eigen::Map<const Eigen::VectorXd>(new_grad_mat.data(), dim);
      Eigen::VectorXd s_k = w_new - w_vec;
      Eigen::VectorXd y_k = g_new - g_vec;
      const double sy = s_k.dot(y_k);
      if (sy > 1e-16) {
        if (static_cast<int>(s_hist.size()) >= m_hist) {
          s_hist.erase(s_hist.begin());
          y_hist.erase(y_hist.begin());
          rho_hist.erase(rho_hist.begin());
        }
        s_hist.push_back(std::move(s_k));
        y_hist.push_back(std::move(y_k));
        rho_hist.push_back(1.0 / sy);
      }

      const double rel_imp = std::abs(cur_loss - new_loss) / std::max(1.0, std::abs(cur_loss));
      w_vec = std::move(w_new);
      g_vec = std::move(g_new);
      W_out = W_cand;
      cur_loss = new_loss;
      if (rel_imp < tol * 0.1) break;
    }
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

    auto eval_preds = [&](Eigen::MatrixXd& out_preds) {
      compute_features_chunk(X_sub.data(), N_imp, N_imp, ws);
      eval_logits_chunk(N_imp, W_out, offset_ptr, N, 0, ws);
      out_preds.resize(N_imp, out_dim);
      if (task == "regression") {
        for (int i = 0; i < N_imp; ++i) {
          out_preds(i, 0) = static_cast<double>(ws.eta_c(i, 0));
        }
      } else if (task == "classification") {
        for (int i = 0; i < N_imp; ++i) {
          const float e = std::clamp(ws.eta_c(i, 0), -40.0f, 40.0f);
          out_preds(i, 0) = static_cast<double>(1.0f / (1.0f + evofe::nn::fast_expf(-e)));
        }
      } else {
        for (int i = 0; i < N_imp; ++i) {
          float max_e = ws.eta_c(i, 0);
          for (int c = 1; c < out_dim; ++c) {
            if (ws.eta_c(i, c) > max_e) max_e = ws.eta_c(i, c);
          }
          double sum_exp = 0.0;
          for (int c = 0; c < out_dim; ++c) {
            const float diff = std::clamp(ws.eta_c(i, c) - max_e, -80.0f, 0.0f);
            const double ev = static_cast<double>(evofe::nn::fast_expf(diff));
            out_preds(i, c) = ev;
            sum_exp += ev;
          }
          const double inv_sum = (sum_exp > 0.0) ? (1.0 / sum_exp) : (1.0 / out_dim);
          for (int c = 0; c < out_dim; ++c) {
            out_preds(i, c) *= inv_sum;
          }
        }
      }
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
