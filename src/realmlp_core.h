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
#include "realmlp_fused.h"
#include "realmlp_workspace.h"

// M_PI is POSIX, not C++17 standard — provide fallback for strict compilers
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace realmlp {

// Vectorized activation: Mish(x) = x * tanh(softplus(x)) or SELU (stride-aware for blocks)
inline void apply_activation(const Eigen::Ref<const Eigen::MatrixXd>& in, Eigen::Ref<Eigen::MatrixXd> out, bool is_cls) {
  int R = static_cast<int>(in.rows());
  int C = static_cast<int>(in.cols());
  int in_stride = static_cast<int>(in.outerStride());
  int out_stride = static_cast<int>(out.outerStride());
  const double* in_data = in.data();
  double* out_data = out.data();

  if (is_cls) {
    // SELU: lambda * (x if x > 0 else alpha * expm1(x))
    constexpr double LAMBDA = 1.0507009873554804934193349852946;
    constexpr double ALPHA  = 1.6732632423543772848170429916717;
    constexpr double LAMBDA_ALPHA = LAMBDA * ALPHA;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (C >= 4)
#endif
    for (int c = 0; c < C; ++c) {
      const double* in_col = in_data + c * in_stride;
      double* out_col = out_data + c * out_stride;
      for (int r = 0; r < R; ++r) {
        double z = in_col[r];
        out_col[r] = (z > 0.0) ? (LAMBDA * z) : (LAMBDA_ALPHA * std::expm1(z));
      }
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (C >= 2)
#endif
    for (int c = 0; c < C; ++c) {
      const double* in_col = in_data + c * in_stride;
      double* out_col = out_data + c * out_stride;
      for (int r = 0; r < R; ++r) {
        double z = in_col[r];
        if (z > 20.0) {
          out_col[r] = z;
        } else if (z < -20.0) {
          out_col[r] = z * std::exp(z);
        } else {
          double ez = std::exp(z);
          double w = 1.0 + ez;
          double tsp = 1.0 - 2.0 / (w * w + 1.0);
          out_col[r] = z * tsp;
        }
      }
    }
  }
}

inline void apply_activation_grad_inplace(const Eigen::Ref<const Eigen::MatrixXd>& act_in, Eigen::Ref<Eigen::MatrixXd> delta, bool is_cls) {
  int R = static_cast<int>(delta.rows());
  int C = static_cast<int>(delta.cols());
  int a_stride = static_cast<int>(act_in.outerStride());
  int d_stride = static_cast<int>(delta.outerStride());
  const double* a_data = act_in.data();
  double* d_data = delta.data();

  if (is_cls) {
    constexpr double LAMBDA = 1.0507009873554804934193349852946;
    constexpr double ALPHA  = 1.6732632423543772848170429916717;
    constexpr double LAMBDA_ALPHA = LAMBDA * ALPHA;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (C >= 2)
#endif
    for (int c = 0; c < C; ++c) {
      const double* a_col = a_data + c * a_stride;
      double* d_col = d_data + c * d_stride;
      for (int r = 0; r < R; ++r) {
        double a = a_col[r];
        d_col[r] *= (a > 0.0) ? LAMBDA : (LAMBDA_ALPHA * std::exp(a));
      }
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (C >= 2)
#endif
    for (int c = 0; c < C; ++c) {
      const double* a_col = a_data + c * a_stride;
      double* d_col = d_data + c * d_stride;
      for (int r = 0; r < R; ++r) {
        double a = a_col[r];
        if (a > 20.0) {
          // derivative is 1.0, no-op
        } else if (a < -20.0) {
          double ea = std::exp(a);
          d_col[r] *= ea * (1.0 + a);
        } else {
          double ea = std::exp(a);
          double w = 1.0 + ea;
          double tsp = 1.0 - 2.0 / (w * w + 1.0);
          double sig = ea / w;
          d_col[r] *= (tsp + a * sig * (1.0 - tsp * tsp));
        }
      }
    }
  }
}

// Coslog4 learning rate schedule: smoothly anneals from 1.0 at t=0 to 0.0 at t=1
inline double coslog4_schedule(double t) {
  t = std::clamp(t, 0.0, 1.0);
  double tau = std::log2(1.0 + 15.0 * t) / 4.0;
  return 0.5 * (1.0 + std::cos(M_PI * tau));
}

// Zero-allocation, vectorized in-place Adam optimizer state
struct AdamParam {
  Eigen::MatrixXd val;
  Eigen::MatrixXd m;
  Eigen::MatrixXd v;

  void init(int rows, int cols) {
    val = Eigen::MatrixXd::Zero(rows, cols);
    m = Eigen::MatrixXd::Zero(rows, cols);
    v = Eigen::MatrixXd::Zero(rows, cols);
  }

  void update(const Eigen::MatrixXd& grad, double lr, double beta1, double beta2, double eps,
              double b1_corr, double b2_corr) {
    double alpha = lr * std::sqrt(b2_corr) / b1_corr;
    double eps_scaled = eps * std::sqrt(b2_corr);
    int sz = static_cast<int>(val.size());
    double* val_ptr = val.data();
    double* m_ptr = m.data();
    double* v_ptr = v.data();
    const double* g_ptr = grad.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (sz >= 1024)
#endif
    for (int i = 0; i < sz; ++i) {
      double g = g_ptr[i];
      double m_val = beta1 * m_ptr[i] + (1.0 - beta1) * g;
      double v_val = beta2 * v_ptr[i] + (1.0 - beta2) * g * g;
      m_ptr[i] = m_val;
      v_ptr[i] = v_val;
      val_ptr[i] -= alpha * m_val / (std::sqrt(std::max(0.0, v_val)) + eps_scaled);
    }
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
        omega.val(j, f) = dist_omega(rng);
        b_phase.val(j, f) = dist_phase(rng);
      }
      W_proj[j].init(k_freq, d_proj);
      for (int r = 0; r < k_freq; ++r) {
        for (int c = 0; c < d_proj; ++c) {
          W_proj[j].val(r, c) = dist_w(rng);
        }
      }
      for (int c = 0; c < d_proj; ++c) {
        beta_proj.val(j, c) = 0.0;
      }
    }
  }

  int total_out_dim() const {
    return n_features * (1 + d_proj);
  }

  // Forward pass through PBLD — writes into pre-allocated E, cache_Z, cache_Theta
  void forward(const Eigen::MatrixXd& x, Eigen::MatrixXd& E,
               std::vector<Eigen::MatrixXd>& cache_Z,
               std::vector<Eigen::MatrixXd>& cache_Theta,
               bool save_cache) const {
    int B = x.rows();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n_features; ++j) {
      int out_col_start = j * (1 + d_proj);
      E.block(0, out_col_start, B, 1) = x.col(j);

      // Theta = 2*pi * x_j * omega_j + b_j  (write into cache directly)
      cache_Theta[j].topRows(B).noalias() = 2.0 * M_PI * x.col(j) * omega.val.row(j);
      cache_Theta[j].topRows(B).rowwise() += b_phase.val.row(j);
      cache_Z[j].topRows(B).array() = cache_Theta[j].topRows(B).array().cos();

      // U = Z * W_j + beta_j
      E.block(0, out_col_start + 1, B, d_proj).noalias() = cache_Z[j].topRows(B) * W_proj[j].val;
      E.block(0, out_col_start + 1, B, d_proj).rowwise() += beta_proj.val.row(j);
    }
  }

  // Forward pass without caching (writing into pre-allocated E and thread_Z_buf)
  void forward_nocache(const Eigen::Ref<const Eigen::MatrixXd>& x,
                       Eigen::MatrixXd& E,
                       std::vector<Eigen::MatrixXd>& thread_Z_buf) const {
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

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n_features; ++j) {
#if defined(_OPENMP)
      int tid = omp_get_thread_num();
#else
      int tid = 0;
#endif
      int out_col_start = j * (1 + d_proj);
      E.block(0, out_col_start, B, 1) = x.col(j);
      auto& Z_buf = thread_Z_buf[tid];
      Z_buf.topRows(B).noalias() = 2.0 * M_PI * x.col(j) * omega.val.row(j);
      Z_buf.topRows(B).rowwise() += b_phase.val.row(j);
      Z_buf.topRows(B).array() = Z_buf.topRows(B).array().cos();
      E.block(0, out_col_start + 1, B, d_proj).noalias() = Z_buf.topRows(B) * W_proj[j].val;
      E.block(0, out_col_start + 1, B, d_proj).rowwise() += beta_proj.val.row(j);
    }
  }

  // Forward pass without caching (allocates temporary buffer when workspace not provided)
  Eigen::MatrixXd forward_nocache(const Eigen::Ref<const Eigen::MatrixXd>& x) const {
    int B = static_cast<int>(x.rows());
    int out_dim = total_out_dim();
    Eigen::MatrixXd E(B, out_dim);
    std::vector<Eigen::MatrixXd> local_Z_buf;
    forward_nocache(x, E, local_Z_buf);
    return E;
  }

  // Backward pass through PBLD and Adam parameter update (zero inner-loop heap allocations)
  void backward_and_update(const Eigen::MatrixXd& x,
                           const Eigen::MatrixXd& grad_E,
                           const std::vector<Eigen::MatrixXd>& cache_Z,
                           const std::vector<Eigen::MatrixXd>& cache_Theta,
                           Eigen::MatrixXd& grad_omega_buf,
                           Eigen::MatrixXd& grad_b_buf,
                           Eigen::MatrixXd& grad_beta_buf,
                           std::vector<Eigen::MatrixXd>& grad_W_tls,
                           std::vector<Eigen::MatrixXd>& grad_Z_tls,
                           std::vector<Eigen::MatrixXd>& grad_Th_tls,
                           double lr, double beta1, double beta2, double eps,
                           double b1_corr, double b2_corr) {
    int B = x.rows();
    grad_omega_buf.setZero();
    grad_b_buf.setZero();
    grad_beta_buf.setZero();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
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
      auto grad_U = grad_E.block(0, out_col_start + 1, B, d_proj);
      grad_beta_buf.row(j) = grad_U.colwise().sum();

      grad_W.noalias() = cache_Z[j].topRows(B).transpose() * grad_U;
      W_proj[j].update(grad_W, lr, beta1, beta2, eps, b1_corr, b2_corr);

      grad_Z.topRows(B).noalias() = grad_U * W_proj[j].val.transpose();
      grad_Th.topRows(B).array() = -grad_Z.topRows(B).array() * cache_Theta[j].topRows(B).array().sin();
      grad_b_buf.row(j) = grad_Th.topRows(B).colwise().sum();
      grad_omega_buf.row(j) = 2.0 * M_PI * (x.col(j).transpose() * grad_Th.topRows(B));
    }

    omega.update(grad_omega_buf, lr, beta1, beta2, eps, b1_corr, b2_corr);
    b_phase.update(grad_b_buf, lr, beta1, beta2, eps, b1_corr, b2_corr);
    beta_proj.update(grad_beta_buf, lr, beta1, beta2, eps, b1_corr, b2_corr);
  }
};

// Pre-allocated workspace for training (eliminates inner-loop allocations)
struct TrainWorkspace {
  // PBLD caches
  std::vector<Eigen::MatrixXd> cache_Z;
  std::vector<Eigen::MatrixXd> cache_Theta;
  Eigen::MatrixXd grad_omega, grad_b, grad_beta;

  // Thread-local PBLD backward buffers to eliminate inner-loop heap allocations
  std::vector<Eigen::MatrixXd> grad_W_tls;
  std::vector<Eigen::MatrixXd> grad_Z_tls;
  std::vector<Eigen::MatrixXd> grad_Th_tls;

  // Batch data (pre-allocated to max batch size)
  Eigen::MatrixXd X_batch, Y_batch;
  // PBLD output
  Eigen::MatrixXd E;
  // MLP forward caches
  Eigen::MatrixXd H0, A1, H1, A2, H2, A3, H3, Out;
  // MLP backward
  Eigen::MatrixXd grad_out, P;
  Eigen::MatrixXd grad_W4, grad_b4, delta3;
  Eigen::MatrixXd grad_W3, grad_b3, delta2;
  Eigen::MatrixXd grad_W2, grad_b2, delta1;
  Eigen::MatrixXd grad_W1, grad_b1, delta0;
  Eigen::MatrixXd grad_scale, delta_E;

  // Shuffled dataset (avoid per-batch row copies)
  Eigen::MatrixXd X_shuf, Y_shuf;

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

// Complete RealMLP Model
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
        double stddev = 1.0 / std::sqrt(static_cast<double>(fan_in));
        std::normal_distribution<double> dist(0.0, stddev);
        for (int r = 0; r < fan_in; ++r) {
          for (int c = 0; c < fan_out; ++c) {
            W.val(r, c) = dist(rng);
          }
        }
        b.val.setZero();
      }
    };

    init_linear(W1, b1, in_dim, hidden_dim, false);
    init_linear(W2, b2, hidden_dim, hidden_dim, false);
    init_linear(W3, b3, hidden_dim, hidden_dim, false);
    init_linear(W4, b4, hidden_dim, output_dim, false);
  }

  // Forward pass through MLP — writes into pre-allocated workspace
  void forward_mlp(const Eigen::MatrixXd& E, int cur_B,
                   Eigen::MatrixXd& H0, Eigen::MatrixXd& A1, Eigen::MatrixXd& H1,
                   Eigen::MatrixXd& A2, Eigen::MatrixXd& H2,
                   Eigen::MatrixXd& A3, Eigen::MatrixXd& H3,
                   Eigen::MatrixXd& Out) const {
    auto E_block = E.topRows(cur_B);
    H0.topRows(cur_B).noalias() = (E_block.array().rowwise() * front_scale.val.row(0).array()).matrix();

    A1.topRows(cur_B).noalias() = H0.topRows(cur_B) * W1.val;
    A1.topRows(cur_B).rowwise() += b1.val.row(0);
    apply_activation(A1.topRows(cur_B), H1.topRows(cur_B), is_classification);

    A2.topRows(cur_B).noalias() = H1.topRows(cur_B) * W2.val;
    A2.topRows(cur_B).rowwise() += b2.val.row(0);
    apply_activation(A2.topRows(cur_B), H2.topRows(cur_B), is_classification);

    A3.topRows(cur_B).noalias() = H2.topRows(cur_B) * W3.val;
    A3.topRows(cur_B).rowwise() += b3.val.row(0);
    apply_activation(A3.topRows(cur_B), H3.topRows(cur_B), is_classification);

    Out.topRows(cur_B).noalias() = H3.topRows(cur_B) * W4.val;
    Out.topRows(cur_B).rowwise() += b4.val.row(0);
  }

  // Evaluate forward prediction in-place into workspace buffers and out_preds without dynamic heap allocation
  template <typename WorkspaceType>
  void forward_predict_inplace(
      const Eigen::MatrixXd& E_in,
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
        double z = std::clamp(ws.Out(i, 0), -50.0, 50.0);
        out_preds(i, 0) = z * ys + ym;
      }
    } else if (task_name == "classification") {
      for (int i = 0; i < N; ++i) {
        double z = std::clamp(ws.Out(i, 0), -30.0, 30.0);
        out_preds(i, 0) = 1.0 / (1.0 + std::exp(-z));
      }
    } else {
      // Multiclass softmax
      for (int i = 0; i < N; ++i) {
        double max_val = ws.Out(i, 0);
        for (int c = 1; c < out_dim; ++c) {
          if (ws.Out(i, c) > max_val) max_val = ws.Out(i, c);
        }
        double sum_exp = 0.0;
        for (int c = 0; c < out_dim; ++c) {
          double ep = std::exp(ws.Out(i, c) - max_val);
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
      const Eigen::MatrixXd& E_in,
      int N,
      WorkspaceType& ws,
      Eigen::MatrixXd& out_preds) const {
    forward_predict_inplace(E_in, N, task, output_dim, y_mean, y_std, ws, out_preds);
  }

  // Standardized prediction helper (zero-copy when input is already standardized)
  Eigen::MatrixXd predict_standardized(const Eigen::Ref<const Eigen::MatrixXd>& X_norm) const {
    int B = static_cast<int>(X_norm.rows());
    Eigen::MatrixXd E = embedder.forward_nocache(X_norm);
    Eigen::MatrixXd H0, A1, H1, A2, H2, A3, H3, Out;
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
      Eigen::MatrixXd result = Out.topRows(B);
      for (int i = 0; i < B; ++i) {
        double z = std::clamp(result(i, 0), -50.0, 50.0);
        result(i, 0) = z * ys + ym;
      }
      return result;
    } else if (task == "classification") {
      return Out.topRows(B).unaryExpr([](double z) {
        return 1.0 / (1.0 + std::exp(-std::clamp(z, -30.0, 30.0)));
      });
    } else {
      Eigen::MatrixXd probs(B, output_dim);
      for (int i = 0; i < B; ++i) {
        double max_val = Out(i, 0);
        for (int c = 1; c < output_dim; ++c) if (Out(i, c) > max_val) max_val = Out(i, c);
        Eigen::RowVectorXd exp_row = (Out.row(i).array() - max_val).exp();
        double sum_exp = exp_row.sum();
        if (sum_exp > 0.0) {
          probs.row(i) = exp_row / sum_exp;
        } else {
          probs.row(i).setZero();
        }
      }
      return probs;
    }
  }

  // Full prediction on test data (zero-copy when already standardized)
  Eigen::MatrixXd predict(const Eigen::Ref<const Eigen::MatrixXd>& X, bool normalize_input = true) const {
    int B = static_cast<int>(X.rows());
    int D = static_cast<int>(X.cols());

    if (normalize_input && x_mean.size() == D && x_std.size() == D) {
      Eigen::MatrixXd X_norm(B, D);
      Eigen::VectorXd x_inv_std(D);
      for (int j = 0; j < D; ++j) {
        double s = x_std(j);
        if (s < 1e-5 || !std::isfinite(s)) {
          x_inv_std(j) = 1.0;
        } else {
          x_inv_std(j) = 1.0 / s;
        }
      }
      if (X.innerStride() == 1 && X.outerStride() == B) {
        fuse_standardize_matrix(X.data(), X_norm.data(), B, D, x_mean.data(), x_inv_std.data());
      } else {
        for (int j = 0; j < D; ++j) {
          double m = x_mean(j);
          double inv_s = x_inv_std(j);
          for (int i = 0; i < B; ++i) {
            double v = X(i, j);
            double clean_v = std::isfinite(v) ? v : 0.0;
            double z = (clean_v - m) * inv_s;
            X_norm(i, j) = std::clamp(z, -30.0, 30.0);
          }
        }
      }
      return predict_standardized(X_norm);
    } else {
      // Direct zero-copy pass of X!
      return predict_standardized(X);
    }
  }

  // Compute feature importances via Mean Occlusion in PBLD Embedding Space using pre-allocated workspace
  std::vector<double> compute_importances(
      const Eigen::Ref<const Eigen::MatrixXd>& X_eval,
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
        ws.E_zero.rows() < D || ws.E_zero.cols() < (1 + d_proj)) {
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

    // Evaluate baseline predictions directly into ws.preds
    forward_predict_inplace(ws.E_base, N_eval, ws, ws.preds);
    double base_loss = compute_loss(ws.preds);

    // 2. Precompute zero-feature embedding for all features into contiguous ws.E_zero
    for (int j = 0; j < D; ++j) {
      ws.E_zero(j, 0) = 0.0;
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

      // ZERO-ALLOCATION RESTORE: Restore feature j's slice directly from ws.E_base!
      ws.E_occ.block(0, out_col_start, N_eval, block_width) =
          ws.E_base.block(0, out_col_start, N_eval, block_width);

      double delta_loss = std::max(0.0, loss_occ - base_loss);
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
      const Eigen::Ref<const Eigen::MatrixXd>& X_eval,
      const std::vector<double>& y_eval) const {
    FeatureImportanceWorkspace ws;
    return compute_importances(X_eval, y_eval, ws);
  }

  // Fallback overload if called without arguments
  std::vector<double> compute_importances() const {
    return std::vector<double>(n_features, 1.0 / std::max(1, n_features));
  }

  struct StateSnapshot {
    PBLDEmbedder embedder;
    AdamParam front_scale, W1, b1, W2, b2, W3, b3, W4, b4;
    Eigen::VectorXd x_mean, x_std;
  };

  StateSnapshot get_snapshot() const {
    StateSnapshot s;
    s.embedder = embedder;
    s.front_scale = front_scale;
    s.W1 = W1; s.b1 = b1;
    s.W2 = W2; s.b2 = b2;
    s.W3 = W3; s.b3 = b3;
    s.W4 = W4; s.b4 = b4;
    s.x_mean = x_mean;
    s.x_std = x_std;
    return s;
  }

  void restore_snapshot(const StateSnapshot& s) {
    embedder = s.embedder;
    front_scale = s.front_scale;
    W1 = s.W1; b1 = s.b1;
    W2 = s.W2; b2 = s.b2;
    W3 = s.W3; b3 = s.b3;
    W4 = s.W4; b4 = s.b4;
    x_mean = s.x_mean;
    x_std = s.x_std;
  }
};

} // namespace realmlp

#endif // EVOFE_REALMLP_CORE_H
