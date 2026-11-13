#ifndef EVOFE_REALMLP_CORE_H
#define EVOFE_REALMLP_CORE_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "nn_common.h"
#include "realmlp_fused.h"
#include "realmlp_workspace.h"

namespace realmlp {

using evofe::nn::fast_expf;
using evofe::nn::gemm_f32;
using evofe::nn::coslog_schedule;
using evofe::nn::coslog4_schedule;
using evofe::nn::AdamParam;

inline void apply_activation(const Eigen::Ref<const Eigen::MatrixXf>& in, Eigen::Ref<Eigen::MatrixXf> out, bool is_cls) {
  evofe::nn::apply_activation(in, out, is_cls ? evofe::nn::ActType::SELU : evofe::nn::ActType::Mish);
}

inline void apply_activation_grad_inplace(const Eigen::Ref<const Eigen::MatrixXf>& act_in, Eigen::Ref<Eigen::MatrixXf> delta, bool is_cls) {
  evofe::nn::apply_activation_grad_inplace(act_in, delta, is_cls ? evofe::nn::ActType::SELU : evofe::nn::ActType::Mish);
}

inline void apply_activation_with_bias(const Eigen::Ref<const Eigen::MatrixXf>& in,
                                       const Eigen::Ref<const Eigen::MatrixXf>& bias,
                                       Eigen::Ref<Eigen::MatrixXf> out, bool is_cls) {
  evofe::nn::apply_activation_with_bias(in, bias, out, is_cls ? evofe::nn::ActType::SELU : evofe::nn::ActType::Mish);
}

inline void apply_activation_grad_and_accum_bias(const Eigen::Ref<const Eigen::MatrixXf>& act_in,
                                                 const Eigen::Ref<const Eigen::MatrixXf>& bias,
                                                 Eigen::Ref<Eigen::MatrixXf> delta,
                                                 Eigen::MatrixXf& grad_b,
                                                 bool is_cls,
                                                 const Eigen::Ref<const Eigen::MatrixXf>* act_out = nullptr) {
  evofe::nn::apply_activation_grad_and_accum_bias(
      act_in, bias, delta, grad_b, is_cls ? evofe::nn::ActType::SELU : evofe::nn::ActType::Mish, act_out);
}

// ScalingLayer abstraction (stride-aware for row-sliced Eigen::Block batch views)
struct ScalingLayer {
  Eigen::VectorXf scale;

  void forward(const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& X,
               Eigen::Ref<Eigen::MatrixXf> out) const {
    out.noalias() = (X.array().rowwise() * scale.transpose().array()).matrix();
  }

  void backward(const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& X,
                const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& dY,
                Eigen::Ref<Eigen::MatrixXf> dX,
                Eigen::Ref<Eigen::VectorXf> dscale) const {
    dscale.noalias() = (X.cwiseProduct(dY)).colwise().sum();
    dX.noalias() = (dY.array().rowwise() * scale.transpose().array()).matrix();
  }
};

// NTPLinear abstraction (stride-aware for row-sliced Eigen::Block batch views)
struct NTPLinear {
  Eigen::MatrixXf W;
  Eigen::VectorXf b;

  void forward(const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& X,
               Eigen::Ref<Eigen::MatrixXf> out) const {
    out.noalias() = X * W;
    out.rowwise() += b.transpose();
  }

  void backward(const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& X,
                const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& dY,
                Eigen::Ref<Eigen::MatrixXf> dX,
                Eigen::Ref<Eigen::MatrixXf> dW,
                Eigen::Ref<Eigen::VectorXf> db) const {
    dW.noalias() = X.transpose() * dY;
    db.noalias() = dY.colwise().sum();
    dX.noalias() = dY * W.transpose();
  }
};

// PBLD (Periodic Bias Linear DenseNet) feature embedder
class PBLDEmbedder {
public:
  int n_features;
  int k_freq; // default 16
  int d_proj; // default 4
  double sigma; // default 0.1

  AdamParam omega;
  AdamParam b_phase;
  std::vector<AdamParam> W_proj;
  AdamParam beta_proj;

  PBLDEmbedder() : n_features(0), k_freq(16), d_proj(4), sigma(0.1) {}

  void init(int n_feat, std::mt19937& rng, int k = 16, int d = 4, double sig = 0.1) {
    n_features = n_feat;
    k_freq = k;
    d_proj = d;
    sigma = sig;

    omega.init(n_features, k_freq);
    b_phase.init(n_features, k_freq);
    W_proj.resize(n_features);
    beta_proj.init(n_features, d_proj);

    std::normal_distribution<double> dist_omega(0.0, sigma);
    std::uniform_real_distribution<double> dist_phase(-M_PI, M_PI);
    double w_bound = 1.0 / std::sqrt(static_cast<double>(k_freq));
    std::uniform_real_distribution<double> dist_w(-w_bound, w_bound);

    for (int j = 0; j < n_features; ++j) {
      for (int f = 0; f < k_freq; ++f) {
        omega.val(j, f) = static_cast<float>(dist_omega(rng));
        b_phase.val(j, f) = static_cast<float>(dist_phase(rng));
      }
      W_proj[j].init(k_freq, d_proj);
      for (int r = 0; r < k_freq; ++r) {
        for (int c = 0; c < d_proj; ++c) {
          W_proj[j].val(r, c) = static_cast<float>(dist_w(rng));
        }
      }
      for (int c = 0; c < d_proj; ++c) {
        beta_proj.val(j, c) = 0.0f;
      }
    }
  }

  int total_out_dim() const {
    return n_features * (1 + d_proj);
  }

  // Forward pass through PBLD — writes into pre-allocated E, cache_Z, cache_Theta
  // Fused single-pass SIMD vector loops with Apple Accelerate vForce (vvcosf)
  void forward(const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& x, Eigen::MatrixXf& E,
               std::vector<Eigen::MatrixXf>& cache_Z,
               std::vector<Eigen::MatrixXf>& cache_Theta,
               bool save_cache) const {
    int B = static_cast<int>(x.rows());
    constexpr float TWO_PI_F = static_cast<float>(2.0 * M_PI);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_features >= 2)
#endif
    for (int j = 0; j < n_features; ++j) {
      int out_col_start = j * (1 + d_proj);
      const float* x_col = x.col(j).data();
      std::memcpy(E.col(out_col_start).data(), x_col, B * sizeof(float));

      // 1. Theta = 2*pi * x_j * omega_j + b_j and Z = cos(Theta)
      for (int k = 0; k < k_freq; ++k) {
        float two_pi_om = TWO_PI_F * omega.val(j, k);
        float b_ph = b_phase.val(j, k);
        float* th_col = cache_Theta[j].col(k).data();
        float* z_col = cache_Z[j].col(k).data();
        for (int i = 0; i < B; ++i) {
          th_col[i] = two_pi_om * x_col[i] + b_ph;
        }
        evofe::nn::vcos_f32(B, th_col, z_col);
      }

      // 2. Fused U = Z * W_j + beta_j (zero sgemm, zero OpenMP lock contention)
      for (int c = 0; c < d_proj; ++c) {
        float b_val = beta_proj.val(j, c);
        float* out_col = E.col(out_col_start + 1 + c).data();
        for (int i = 0; i < B; ++i) {
          out_col[i] = b_val;
        }
        for (int k = 0; k < k_freq; ++k) {
          float w_val = W_proj[j].val(k, c);
          const float* z_col = cache_Z[j].col(k).data();
          for (int i = 0; i < B; ++i) {
            out_col[i] += w_val * z_col[i];
          }
        }
      }
    }
  }

  // Forward pass without caching (writing into pre-allocated E and thread_Z_buf)
  void forward_nocache(const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& x,
                       Eigen::MatrixXf& E,
                       std::vector<Eigen::MatrixXf>& thread_Z_buf) const {
    int B = static_cast<int>(x.rows());
    int out_dim = total_out_dim();
    if (E.rows() < B || E.cols() < out_dim) {
      E.resize(B, out_dim);
    }
    int max_t = 1;
#if defined(_OPENMP)
    max_t = std::max(1, omp_get_max_threads());
#endif
    if (static_cast<int>(thread_Z_buf.size()) < max_t) {
      thread_Z_buf.resize(max_t);
    }
    for (int t = 0; t < max_t; ++t) {
      if (thread_Z_buf[t].rows() < B || thread_Z_buf[t].cols() < k_freq) {
        thread_Z_buf[t].resize(B, std::max(1, k_freq));
      }
    }

    constexpr float TWO_PI_F = static_cast<float>(2.0 * M_PI);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_features >= 2)
#endif
    for (int j = 0; j < n_features; ++j) {
#if defined(_OPENMP)
      int tid = omp_get_thread_num();
#else
      int tid = 0;
#endif
      int out_col_start = j * (1 + d_proj);
      const float* x_col = x.col(j).data();
      std::memcpy(E.col(out_col_start).data(), x_col, B * sizeof(float));

      auto& Z_buf = thread_Z_buf[tid];
      for (int k = 0; k < k_freq; ++k) {
        float two_pi_om = TWO_PI_F * omega.val(j, k);
        float b_ph = b_phase.val(j, k);
        float* z_col = Z_buf.col(k).data();
        for (int i = 0; i < B; ++i) {
          z_col[i] = two_pi_om * x_col[i] + b_ph;
        }
        evofe::nn::vcos_f32(B, z_col, z_col);
      }

      // Fused U = Z * W_j + beta_j
      for (int c = 0; c < d_proj; ++c) {
        float b_val = beta_proj.val(j, c);
        float* out_col = E.col(out_col_start + 1 + c).data();
        for (int i = 0; i < B; ++i) {
          out_col[i] = b_val;
        }
        for (int k = 0; k < k_freq; ++k) {
          float w_val = W_proj[j].val(k, c);
          const float* z_col = Z_buf.col(k).data();
          for (int i = 0; i < B; ++i) {
            out_col[i] += w_val * z_col[i];
          }
        }
      }
    }
  }

  // Forward pass without caching (allocates temporary buffer when workspace not provided)
  Eigen::MatrixXf forward_nocache(const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& x) const {
    int B = static_cast<int>(x.rows());
    int out_dim = total_out_dim();
    Eigen::MatrixXf E(B, out_dim);
    std::vector<Eigen::MatrixXf> local_Z_buf;
    forward_nocache(x, E, local_Z_buf);
    return E;
  }

  // Backward pass through PBLD and Adam parameter update (zero inner-loop heap allocations)
  void backward_and_update(const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& x,
                           const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& grad_E,
                           const std::vector<Eigen::MatrixXf>& cache_Z,
                           const std::vector<Eigen::MatrixXf>& cache_Theta,
                           Eigen::MatrixXf& grad_omega_buf,
                           Eigen::MatrixXf& grad_b_buf,
                           Eigen::MatrixXf& grad_beta_buf,
                           std::vector<Eigen::MatrixXf>& grad_W_tls,
                           std::vector<Eigen::MatrixXf>& grad_Z_tls,
                           std::vector<Eigen::MatrixXf>& grad_Th_tls,
                           double lr, double beta1, double beta2, double eps,
                           double b1_corr, double b2_corr) {
    int B = static_cast<int>(x.rows());
    grad_omega_buf.setZero();
    grad_b_buf.setZero();
    grad_beta_buf.setZero();

    float alpha = static_cast<float>(lr * std::sqrt(b2_corr) / b1_corr);
    float eps_scaled = static_cast<float>(eps * std::sqrt(b2_corr));
    float b1_f = static_cast<float>(beta1);
    float one_m_b1 = static_cast<float>(1.0 - beta1);
    float b2_f = static_cast<float>(beta2);
    float one_m_b2 = static_cast<float>(1.0 - beta2);
    constexpr float TWO_PI_F = static_cast<float>(2.0 * M_PI);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_features >= 2)
#endif
    for (int j = 0; j < n_features; ++j) {
#if defined(_OPENMP)
      int tid = omp_get_thread_num();
#else
      int tid = 0;
#endif
      auto& grad_W = grad_W_tls[tid];
      auto& grad_Z = grad_Z_tls[tid];
      auto& grad_Th = grad_Th_tls[tid];

      int out_col_start = j * (1 + d_proj);

      // 1. grad_beta = sum(grad_U) and update beta_proj in parallel
      for (int c = 0; c < d_proj; ++c) {
        const float* u_col = grad_E.col(out_col_start + 1 + c).data();
        float sum = 0.0f;
        for (int i = 0; i < B; ++i) sum += u_col[i];
        grad_beta_buf(j, c) = sum;
        evofe::nn::adam_scalar_step(
            beta_proj.val(j, c), beta_proj.m(j, c), beta_proj.v(j, c),
            sum, alpha, eps_scaled, b1_f, one_m_b1, b2_f, one_m_b2);
      }

      // 2. grad_W = cache_Z[j]^T * grad_U (zero sgemm, cache-contiguous dot products)
      for (int c = 0; c < d_proj; ++c) {
        const float* u_col = grad_E.col(out_col_start + 1 + c).data();
        for (int k = 0; k < k_freq; ++k) {
          const float* z_col = cache_Z[j].col(k).data();
          float dot = 0.0f;
          for (int i = 0; i < B; ++i) {
            dot += z_col[i] * u_col[i];
          }
          grad_W(k, c) = dot;
        }
      }
      W_proj[j].update_serial(grad_W, lr, beta1, beta2, eps, b1_corr, b2_corr);

      // 3. grad_Z = grad_U * W_j^T (zero sgemm)
      for (int k = 0; k < k_freq; ++k) {
        float* gz_col = grad_Z.col(k).data();
        for (int i = 0; i < B; ++i) gz_col[i] = 0.0f;
        for (int c = 0; c < d_proj; ++c) {
          float w_val = W_proj[j].val(k, c);
          const float* u_col = grad_E.col(out_col_start + 1 + c).data();
          for (int i = 0; i < B; ++i) {
            gz_col[i] += w_val * u_col[i];
          }
        }
      }

      // 4. grad_Th = -grad_Z * sin(cache_Theta), and accumulate/update grad_b
      for (int k = 0; k < k_freq; ++k) {
        const float* gz_col = grad_Z.col(k).data();
        const float* th_col = cache_Theta[j].col(k).data();
        float* gth_col = grad_Th.col(k).data();
        float sum_b = 0.0f;
        evofe::nn::vsin_f32(B, th_col, gth_col);
        for (int i = 0; i < B; ++i) {
          float gth = -gz_col[i] * gth_col[i];
          gth_col[i] = gth;
          sum_b += gth;
        }
        grad_b_buf(j, k) = sum_b;
        evofe::nn::adam_scalar_step(
            b_phase.val(j, k), b_phase.m(j, k), b_phase.v(j, k),
            sum_b, alpha, eps_scaled, b1_f, one_m_b1, b2_f, one_m_b2);
      }

      // 5. grad_omega = 2*pi * x.col(j)^T * grad_Th and in-parallel update
      const float* x_col = x.col(j).data();
      for (int k = 0; k < k_freq; ++k) {
        const float* gth_col = grad_Th.col(k).data();
        float dot = 0.0f;
        for (int i = 0; i < B; ++i) {
          dot += x_col[i] * gth_col[i];
        }
        float g_om = TWO_PI_F * dot;
        grad_omega_buf(j, k) = g_om;
        evofe::nn::adam_scalar_step(
            omega.val(j, k), omega.m(j, k), omega.v(j, k),
            g_om, alpha, eps_scaled, b1_f, one_m_b1, b2_f, one_m_b2);
      }
    }
  }
};

// Pre-allocated workspace for training (FP32, eliminates inner-loop allocations)
struct TrainWorkspace {
  // PBLD caches
  std::vector<Eigen::MatrixXf> cache_Z;
  std::vector<Eigen::MatrixXf> cache_Theta;
  Eigen::MatrixXf grad_omega, grad_b, grad_beta;

  // Thread-local PBLD backward buffers to eliminate inner-loop heap allocations
  std::vector<Eigen::MatrixXf> grad_W_tls;
  std::vector<Eigen::MatrixXf> grad_Z_tls;
  std::vector<Eigen::MatrixXf> grad_Th_tls;

  // Batch data (pre-allocated to max batch size)
  Eigen::MatrixXf X_batch, Y_batch;
  // PBLD output
  Eigen::MatrixXf E;
  // MLP forward caches
  Eigen::MatrixXf H0, A1, H1, A2, H2, A3, H3, Out;
  // MLP backward
  Eigen::MatrixXf grad_out, P;
  Eigen::MatrixXf grad_W4, grad_b4, delta3;
  Eigen::MatrixXf grad_W3, grad_b3, delta2;
  Eigen::MatrixXf grad_W2, grad_b2, delta1;
  Eigen::MatrixXf grad_W1, grad_b1, delta0;
  Eigen::MatrixXf grad_scale, delta_E;

  // Shuffled dataset (avoid per-batch row copies)
  Eigen::MatrixXf X_shuf, Y_shuf;

  void allocate(int max_B, int D, int out_dim, int hidden_dim, int embed_dim, int n_features, int k_freq, int d_proj, int N, int max_threads = 1) {
    // PBLD caches
    cache_Z.resize(n_features);
    cache_Theta.resize(n_features);
    for (int j = 0; j < n_features; ++j) {
      cache_Z[j].resize(max_B, k_freq);
      cache_Theta[j].resize(max_B, k_freq);
    }
    grad_omega.resize(n_features, k_freq);
    grad_b.resize(n_features, k_freq);
    grad_beta.resize(n_features, d_proj);

    // Thread-local backward buffers
    grad_W_tls.resize(max_threads);
    grad_Z_tls.resize(max_threads);
    grad_Th_tls.resize(max_threads);
    for (int t = 0; t < max_threads; ++t) {
      grad_W_tls[t].resize(k_freq, d_proj);
      grad_Z_tls[t].resize(max_B, k_freq);
      grad_Th_tls[t].resize(max_B, k_freq);
    }

    // Batch data — not needed when using shuffled dataset views
    E.resize(max_B, embed_dim);

    // MLP forward
    H0.resize(max_B, embed_dim);
    A1.resize(max_B, hidden_dim);
    H1.resize(max_B, hidden_dim);
    A2.resize(max_B, hidden_dim);
    H2.resize(max_B, hidden_dim);
    A3.resize(max_B, hidden_dim);
    H3.resize(max_B, hidden_dim);
    Out.resize(max_B, out_dim);

    // Loss/gradient
    grad_out.resize(max_B, out_dim);
    P.resize(max_B, out_dim);

    // MLP backward
    grad_W4.resize(hidden_dim, out_dim);
    grad_b4.resize(1, out_dim);
    delta3.resize(max_B, hidden_dim);
    grad_W3.resize(hidden_dim, hidden_dim);
    grad_b3.resize(1, hidden_dim);
    delta2.resize(max_B, hidden_dim);
    grad_W2.resize(hidden_dim, hidden_dim);
    grad_b2.resize(1, hidden_dim);
    delta1.resize(max_B, hidden_dim);
    grad_W1.resize(embed_dim, hidden_dim);
    grad_b1.resize(1, hidden_dim);
    delta0.resize(max_B, embed_dim);
    grad_scale.resize(1, embed_dim);
    delta_E.resize(max_B, embed_dim);

    // Shuffled dataset
    X_shuf.resize(N, D);
    Y_shuf.resize(N, out_dim);
  }
};

// Complete RealMLP Model (FP32 GEMM/Activation Core + FP64 Standardization/Output)
class RealMLPModel {
public:
  std::string task;
  int n_features;
  int output_dim;
  int hidden_dim;
  bool is_classification;

  double y_mean;
  double y_std;

  Eigen::VectorXd x_mean;
  Eigen::VectorXd x_std;

  PBLDEmbedder embedder;
  AdamParam front_scale;
  AdamParam W1, b1;
  AdamParam W2, b2;
  AdamParam W3, b3;
  AdamParam W4, b4;

  RealMLPModel() : task("regression"), n_features(0), output_dim(1), hidden_dim(256),
                   is_classification(false), y_mean(0.0), y_std(1.0) {}

  void init(int n_feat, int out_dim, const std::string& t_name, int seed = 42, int h_dim = 256) {
    task = t_name;
    is_classification = (task == "classification" || task == "multiclass");
    n_features = n_feat;
    output_dim = out_dim;
    hidden_dim = h_dim;

    x_mean = Eigen::VectorXd::Zero(n_features);
    x_std = Eigen::VectorXd::Ones(n_features);

    std::mt19937 rng(static_cast<unsigned int>(seed));
    embedder.init(n_features, rng, 16, 4, 0.1);

    int in_dim = embedder.total_out_dim();

    front_scale.init(1, in_dim);
    front_scale.val.setOnes();

    auto init_linear = [&](AdamParam& W, AdamParam& b, int fan_in, int fan_out, bool zero_init = false) {
      W.init(fan_in, fan_out);
      b.init(1, fan_out);
      if (zero_init) {
        W.val.setZero();
        b.val.setZero();
      } else {
        std::normal_distribution<double> dist(0.0, 1.0);
        for (int r = 0; r < fan_in; ++r) {
          for (int c = 0; c < fan_out; ++c) {
            W.val(r, c) = static_cast<float>(dist(rng));
          }
        }
        b.val.setZero();
      }
    };

    init_linear(W1, b1, in_dim, hidden_dim, false);
    init_linear(W2, b2, hidden_dim, hidden_dim, false);
    init_linear(W3, b3, hidden_dim, hidden_dim, false);
    init_linear(W4, b4, hidden_dim, output_dim, true);
  }

  // Forward pass through MLP — writes into pre-allocated FP32 workspace (dispatches sgemm to Apple AMX)
  void forward_mlp(const Eigen::MatrixXf& E, int cur_B,
                   Eigen::MatrixXf& H0, Eigen::MatrixXf& A1, Eigen::MatrixXf& H1,
                   Eigen::MatrixXf& A2, Eigen::MatrixXf& H2,
                   Eigen::MatrixXf& A3, Eigen::MatrixXf& H3,
                   Eigen::MatrixXf& Out) const {
    int embed_d = static_cast<int>(E.cols());
    float inv_sqrt_w1 = 1.0f / std::sqrt(static_cast<float>(std::max(1, embed_d)));
    float inv_sqrt_hidden = 1.0f / std::sqrt(static_cast<float>(std::max(1, hidden_dim)));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (embed_d >= 4)
#endif
    for (int c = 0; c < embed_d; ++c) {
      float sc = front_scale.val(0, c);
      const float* e_col = E.col(c).data();
      float* h0_col = H0.col(c).data();
      for (int r = 0; r < cur_B; ++r) {
        h0_col[r] = e_col[r] * sc;
      }
    }

    gemm_f32(false, false, cur_B, hidden_dim, embed_d, H0, W1.val, A1, inv_sqrt_w1);
    apply_activation_with_bias(A1.topRows(cur_B), b1.val, H1.topRows(cur_B), is_classification);

    gemm_f32(false, false, cur_B, hidden_dim, hidden_dim, H1, W2.val, A2, inv_sqrt_hidden);
    apply_activation_with_bias(A2.topRows(cur_B), b2.val, H2.topRows(cur_B), is_classification);

    gemm_f32(false, false, cur_B, hidden_dim, hidden_dim, H2, W3.val, A3, inv_sqrt_hidden);
    apply_activation_with_bias(A3.topRows(cur_B), b3.val, H3.topRows(cur_B), is_classification);

    gemm_f32(false, false, cur_B, output_dim, hidden_dim, H3, W4.val, Out, inv_sqrt_hidden);
    Out.topRows(cur_B).rowwise() += b4.val.row(0);
  }

  // Evaluate forward prediction in-place into workspace buffers and out_preds without dynamic heap allocation
  template <typename WorkspaceType>
  void forward_predict_inplace(
      const Eigen::MatrixXf& E_in,
      int N,
      const std::string& task_name,
      int out_dim,
      double ym_val,
      double ys_val,
      WorkspaceType& ws,
      Eigen::MatrixXd& out_preds) const {
    forward_mlp(E_in, N, ws.H0, ws.A1, ws.H1, ws.A2, ws.H2, ws.A3, ws.H3, ws.Out);

    if (task_name == "regression") {
      double ys = (std::isfinite(ys_val) && ys_val > 1e-8) ? ys_val : 1.0;
      double ym = std::isfinite(ym_val) ? ym_val : 0.0;
      for (int i = 0; i < N; ++i) {
        double z = std::clamp(static_cast<double>(ws.Out(i, 0)), -50.0, 50.0);
        out_preds(i, 0) = z * ys + ym;
      }
    } else if (task_name == "classification") {
      for (int i = 0; i < N; ++i) {
        double z = std::clamp(static_cast<double>(ws.Out(i, 0)), -30.0, 30.0);
        out_preds(i, 0) = 1.0 / (1.0 + std::exp(-z));
      }
    } else {
      // Multiclass softmax
      for (int i = 0; i < N; ++i) {
        double max_val = static_cast<double>(ws.Out(i, 0));
        for (int c = 1; c < out_dim; ++c) {
          double v = static_cast<double>(ws.Out(i, c));
          if (v > max_val) max_val = v;
        }
        double sum_exp = 0.0;
        for (int c = 0; c < out_dim; ++c) {
          double ep = std::exp(static_cast<double>(ws.Out(i, c)) - max_val);
          out_preds(i, c) = ep;
          sum_exp += ep;
        }
        if (sum_exp > 0.0) {
          double inv_sum = 1.0 / sum_exp;
          for (int c = 0; c < out_dim; ++c) {
            out_preds(i, c) *= inv_sum;
          }
        } else {
          for (int c = 0; c < out_dim; ++c) {
            out_preds(i, c) = 0.0;
          }
        }
      }
    }
  }

  template <typename WorkspaceType>
  void forward_predict_inplace(
      const Eigen::MatrixXf& E_in,
      int N,
      WorkspaceType& ws,
      Eigen::MatrixXd& out_preds) const {
    forward_predict_inplace(E_in, N, task, output_dim, y_mean, y_std, ws, out_preds);
  }

  // Standardized prediction helper (zero-copy when input is already standardized FP32)
  Eigen::MatrixXd predict_standardized(const Eigen::Ref<const Eigen::MatrixXf>& X_norm) const {
    int B = static_cast<int>(X_norm.rows());
    Eigen::MatrixXf E = embedder.forward_nocache(X_norm);
    Eigen::MatrixXf H0, A1, H1, A2, H2, A3, H3, Out;
    H0.resize(B, E.cols());
    A1.resize(B, hidden_dim);
    H1.resize(B, hidden_dim);
    A2.resize(B, hidden_dim);
    H2.resize(B, hidden_dim);
    A3.resize(B, hidden_dim);
    H3.resize(B, hidden_dim);
    Out.resize(B, output_dim);
    forward_mlp(E, B, H0, A1, H1, A2, H2, A3, H3, Out);

    if (task == "regression") {
      double ys = (std::isfinite(y_std) && y_std > 1e-8) ? y_std : 1.0;
      double ym = std::isfinite(y_mean) ? y_mean : 0.0;
      Eigen::MatrixXd result(B, 1);
      for (int i = 0; i < B; ++i) {
        double z = std::clamp(static_cast<double>(Out(i, 0)), -50.0, 50.0);
        result(i, 0) = z * ys + ym;
      }
      return result;
    } else if (task == "classification") {
      Eigen::MatrixXd result(B, 1);
      for (int i = 0; i < B; ++i) {
        double z = std::clamp(static_cast<double>(Out(i, 0)), -30.0, 30.0);
        result(i, 0) = 1.0 / (1.0 + std::exp(-z));
      }
      return result;
    } else {
      Eigen::MatrixXd probs(B, output_dim);
      for (int i = 0; i < B; ++i) {
        double max_val = static_cast<double>(Out(i, 0));
        for (int c = 1; c < output_dim; ++c) {
          double v = static_cast<double>(Out(i, c));
          if (v > max_val) max_val = v;
        }
        double sum_exp = 0.0;
        for (int c = 0; c < output_dim; ++c) {
          double ep = std::exp(static_cast<double>(Out(i, c)) - max_val);
          probs(i, c) = ep;
          sum_exp += ep;
        }
        if (sum_exp > 0.0) {
          double inv_sum = 1.0 / sum_exp;
          for (int c = 0; c < output_dim; ++c) {
            probs(i, c) *= inv_sum;
          }
        } else {
          for (int c = 0; c < output_dim; ++c) {
            probs(i, c) = 0.0;
          }
        }
      }
      return probs;
    }
  }

  // Full prediction on test data
  Eigen::MatrixXd predict(const Eigen::Ref<const Eigen::MatrixXd>& X_raw, bool normalize_input = true) const {
    if (n_features > 0 && X_raw.cols() != n_features) {
      Rcpp::stop("Input matrix column count does not match model n_features.");
    }
    int D = static_cast<int>(X_raw.cols());

    if (normalize_input && x_mean.size() == D && x_std.size() == D) {
      Eigen::MatrixXf X_norm = fuse_standardize_matrix_f32(X_raw, x_mean, x_std);
      return predict_standardized(X_norm);
    } else {
      Eigen::MatrixXf X_f = X_raw.cast<float>();
      return predict_standardized(X_f);
    }
  }

  // Compute feature importances via Mean Occlusion in PBLD Embedding Space using pre-allocated workspace
  std::vector<double> compute_importances(
      const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& X_eval,
      const std::vector<double>& y_eval,
      FeatureImportanceWorkspace& ws) const {

    int D = n_features;
    std::vector<double> imp(D, 1.0 / std::max(1, D));
    int N_eval = static_cast<int>(X_eval.rows());
    if (D <= 0 || N_eval <= 0 || static_cast<int>(y_eval.size()) < N_eval) {
      return imp;
    }

    int embed_dim = embedder.total_out_dim();
    int d_proj = embedder.d_proj;

    if (ws.E_base.rows() < N_eval || ws.E_base.cols() < embed_dim ||
        ws.E_zero.rows() < D || ws.E_zero.cols() < (1 + d_proj) ||
        ws.base_preds.rows() < N_eval || ws.base_preds.cols() < output_dim) {
      ws.allocate(N_eval, D, output_dim, hidden_dim, embed_dim, d_proj, embedder.k_freq, 1);
    }

    auto compute_loss = [&](const Eigen::MatrixXd& preds) -> double {
      if (task == "regression") {
        double sum_sq = 0.0;
        for (int i = 0; i < N_eval; ++i) {
          double diff = preds(i, 0) - y_eval[i];
          sum_sq += diff * diff;
        }
        return sum_sq / N_eval;
      } else if (task == "classification") {
        double ll = 0.0;
        for (int i = 0; i < N_eval; ++i) {
          double p = std::clamp(preds(i, 0), 1e-15, 1.0 - 1e-15);
          double y = (y_eval[i] > 0.0) ? 1.0 : 0.0;
          ll -= (y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        }
        return ll / N_eval;
      } else {
        double ll = 0.0;
        for (int i = 0; i < N_eval; ++i) {
          int c = static_cast<int>(y_eval[i]);
          double p = 1e-15;
          if (c >= 0 && c < output_dim) {
            p = std::clamp(preds(i, c), 1e-15, 1.0);
          }
          ll -= std::log(p);
        }
        return ll / N_eval;
      }
    };

    // 1. Precompute base embedding into ws.E_base
    embedder.forward_nocache(X_eval, ws.E_base, ws.thread_Z_buf);

    // Evaluate baseline predictions directly into ws.base_preds
    forward_predict_inplace(ws.E_base, N_eval, ws, ws.base_preds);
    double base_loss = compute_loss(ws.base_preds);

    // 2. Precompute zero-feature embedding for all features into contiguous ws.E_zero
    for (int j = 0; j < D; ++j) {
      ws.E_zero(j, 0) = 0.0f;
      ws.E_zero.row(j).tail(d_proj).noalias() =
          embedder.b_phase.val.row(j).array().cos().matrix() * embedder.W_proj[j].val;
      ws.E_zero.row(j).tail(d_proj) += embedder.beta_proj.val.row(j);
    }

    // 3. Occlusion in embedding space: copy E_base once into E_occ
    ws.E_occ.topRows(N_eval) = ws.E_base.topRows(N_eval);

    int block_width = 1 + d_proj;
    for (int j = 0; j < D; ++j) {
      int out_col_start = j * block_width;

      // In-place row broadcasting of zero-feature embedding
      ws.E_occ.block(0, out_col_start, N_eval, block_width).rowwise() = ws.E_zero.row(j);

      // Evaluate occluded predictions in-place into ws.preds
      forward_predict_inplace(ws.E_occ, N_eval, ws, ws.preds);
      double loss_occ = compute_loss(ws.preds);
      double pred_shift = (ws.preds.topRows(N_eval) - ws.base_preds.topRows(N_eval)).cwiseAbs().mean();

      // ZERO-ALLOCATION RESTORE: Restore feature j's slice directly from ws.E_base!
      ws.E_occ.block(0, out_col_start, N_eval, block_width) =
          ws.E_base.block(0, out_col_start, N_eval, block_width);

      double delta_loss = std::max(0.0, loss_occ - base_loss) + 0.1 * pred_shift;
      imp[j] = delta_loss;
    }

    // 4. Normalize importances to sum to 1.0
    double sum_imp = 0.0;
    for (double v : imp) sum_imp += v;
    if (sum_imp > 1e-12 && std::isfinite(sum_imp)) {
      for (int j = 0; j < D; ++j) imp[j] /= sum_imp;
    } else {
      std::fill(imp.begin(), imp.end(), 1.0 / D);
    }

    return imp;
  }

  std::vector<double> compute_importances(
      const Eigen::Ref<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>& X_eval,
      const std::vector<double>& y_eval) const {
    FeatureImportanceWorkspace ws;
    return compute_importances(X_eval, y_eval, ws);
  }

  // Fallback overload if called without arguments
  std::vector<double> compute_importances() const {
    return std::vector<double>(n_features, 1.0 / std::max(1, n_features));
  }

  struct StateSnapshot {
    Eigen::VectorXf scale;
    Eigen::MatrixXf W1;
    Eigen::VectorXf b1;
    Eigen::VectorXf act1_param;
    Eigen::MatrixXf W2;
    Eigen::VectorXf b2;
    Eigen::VectorXf act2_param;
    Eigen::MatrixXf W3;
    Eigen::VectorXf b3;
    Eigen::VectorXf act3_param;
    Eigen::MatrixXf W4;
    Eigen::VectorXf b4;
    // Embedder weights and normalization parameters
    Eigen::MatrixXf embedder_omega;
    Eigen::MatrixXf embedder_b_phase;
    std::vector<Eigen::MatrixXf> embedder_W_proj;
    Eigen::MatrixXf embedder_beta_proj;
    Eigen::MatrixXf front_scale;
    Eigen::VectorXd x_mean;
    Eigen::VectorXd x_std;

    void capture_from(const RealMLPModel& m) {
      embedder_omega = m.embedder.omega.val;
      embedder_b_phase = m.embedder.b_phase.val;
      embedder_W_proj.resize(m.embedder.W_proj.size());
      for (size_t i = 0; i < m.embedder.W_proj.size(); ++i) {
        embedder_W_proj[i] = m.embedder.W_proj[i].val;
      }
      embedder_beta_proj = m.embedder.beta_proj.val;
      front_scale = m.front_scale.val;
      if (m.front_scale.val.rows() > 0) {
        scale = m.front_scale.val.row(0);
      }
      W1 = m.W1.val;
      if (m.b1.val.rows() > 0) b1 = m.b1.val.row(0);
      W2 = m.W2.val;
      if (m.b2.val.rows() > 0) b2 = m.b2.val.row(0);
      W3 = m.W3.val;
      if (m.b3.val.rows() > 0) b3 = m.b3.val.row(0);
      W4 = m.W4.val;
      if (m.b4.val.rows() > 0) b4 = m.b4.val.row(0);
      x_mean = m.x_mean;
      x_std = m.x_std;
    }

    void restore_to(RealMLPModel& m) const {
      m.embedder.omega.val = embedder_omega;
      m.embedder.b_phase.val = embedder_b_phase;
      for (size_t i = 0; i < embedder_W_proj.size(); ++i) {
        m.embedder.W_proj[i].val = embedder_W_proj[i];
      }
      m.embedder.beta_proj.val = embedder_beta_proj;
      m.front_scale.val = front_scale;
      m.W1.val = W1;
      if (b1.size() > 0) m.b1.val.row(0) = b1;
      m.W2.val = W2;
      if (b2.size() > 0) m.b2.val.row(0) = b2;
      m.W3.val = W3;
      if (b3.size() > 0) m.b3.val.row(0) = b3;
      m.W4.val = W4;
      if (b4.size() > 0) m.b4.val.row(0) = b4;
      m.x_mean = x_mean;
      m.x_std = x_std;
    }
  };

  StateSnapshot get_snapshot() const {
    StateSnapshot s;
    s.capture_from(*this);
    return s;
  }

  void restore_snapshot(const StateSnapshot& s) {
    s.restore_to(*this);
  }
};

} // namespace realmlp

#endif // EVOFE_REALMLP_CORE_H
