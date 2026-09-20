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

namespace realmlp {

// Helper: safe softplus for Mish activation
inline double softplus(double x) {
  if (x > 20.0) return x;
  if (x < -20.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

// Mish activation and derivative: f(x) = x * tanh(softplus(x))
inline double mish(double x) {
  return x * std::tanh(softplus(x));
}

inline double mish_grad(double x) {
  double sp = softplus(x);
  double tsp = std::tanh(sp);
  double sig = 1.0 / (1.0 + std::exp(-std::clamp(x, -30.0, 30.0)));
  return tsp + x * sig * (1.0 - tsp * tsp);
}

// SELU constants
constexpr double SELU_LAMBDA = 1.0507009873554804934193349852946;
constexpr double SELU_ALPHA  = 1.6732632423543772848170429916717;

inline double selu(double x) {
  return (x > 0.0) ? (SELU_LAMBDA * x) : (SELU_LAMBDA * SELU_ALPHA * std::expm1(x));
}

inline double selu_grad(double x) {
  return (x > 0.0) ? SELU_LAMBDA : (SELU_LAMBDA * SELU_ALPHA * std::exp(x));
}

inline void apply_activation(const Eigen::MatrixXd& in, Eigen::MatrixXd& out, bool is_cls) {
  int sz = static_cast<int>(in.size());
  out.resize(in.rows(), in.cols());
  const double* src = in.data();
  double* dst = out.data();
  if (is_cls) {
    for (int i = 0; i < sz; ++i) {
      double x = src[i];
      dst[i] = (x > 0.0) ? (SELU_LAMBDA * x) : (SELU_LAMBDA * SELU_ALPHA * std::expm1(x));
    }
  } else {
    for (int i = 0; i < sz; ++i) {
      dst[i] = mish(src[i]);
    }
  }
}

inline void apply_activation_grad(const Eigen::MatrixXd& act_in, Eigen::MatrixXd& delta, bool is_cls) {
  int sz = static_cast<int>(delta.size());
  const double* a_ptr = act_in.data();
  double* d_ptr = delta.data();
  if (is_cls) {
    for (int i = 0; i < sz; ++i) {
      double x = a_ptr[i];
      d_ptr[i] *= (x > 0.0) ? SELU_LAMBDA : (SELU_LAMBDA * SELU_ALPHA * std::exp(x));
    }
  } else {
    for (int i = 0; i < sz; ++i) {
      d_ptr[i] *= mish_grad(a_ptr[i]);
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

    for (int i = 0; i < sz; ++i) {
      double g = g_ptr[i];
      double m_val = beta1 * m_ptr[i] + (1.0 - beta1) * g;
      double v_val = beta2 * v_ptr[i] + (1.0 - beta2) * g * g;
      m_ptr[i] = m_val;
      v_ptr[i] = v_val;
      val_ptr[i] -= alpha * m_val / (std::sqrt(v_val) + eps_scaled);
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

  // Forward pass through PBLD
  Eigen::MatrixXd forward(const Eigen::MatrixXd& x,
                          std::vector<Eigen::MatrixXd>* cache_Z = nullptr,
                          std::vector<Eigen::MatrixXd>* cache_Theta = nullptr) const {
    int B = x.rows();
    int out_dim = total_out_dim();
    Eigen::MatrixXd E(B, out_dim);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n_features; ++j) {
      int out_col_start = j * (1 + d_proj);
      // 1. DenseNet connection: raw feature column
      E.col(out_col_start) = x.col(j);

      // 2. Periodic + phase: Theta = 2*pi * x_j * omega_j + b_j
      Eigen::VectorXd col_j = x.col(j);
      Eigen::RowVectorXd om_j = omega.val.row(j);
      Eigen::RowVectorXd b_j = b_phase.val.row(j);

      Eigen::MatrixXd Theta = (2.0 * M_PI * col_j * om_j).rowwise() + b_j;
      Eigen::MatrixXd Z = Theta.array().cos();

      // 3. Linear projection: U = Z * W_j + beta_j
      Eigen::RowVectorXd bp_j = beta_proj.val.row(j);
      Eigen::MatrixXd U = (Z * W_proj[j].val).rowwise() + bp_j;

      // Fill in embedded channels
      E.block(0, out_col_start + 1, B, d_proj) = U;

      if (cache_Z && cache_Theta) {
        (*cache_Z)[j] = Z;
        (*cache_Theta)[j] = Theta;
      }
    }

    return E;
  }

  // Backward pass through PBLD and Adam parameter update
  void backward_and_update(const Eigen::MatrixXd& x,
                           const Eigen::MatrixXd& grad_E,
                           const std::vector<Eigen::MatrixXd>& cache_Z,
                           const std::vector<Eigen::MatrixXd>& cache_Theta,
                           double lr, double beta1, double beta2, double eps,
                           double b1_corr, double b2_corr) {
    int B = x.rows();
    Eigen::MatrixXd grad_omega = Eigen::MatrixXd::Zero(n_features, k_freq);
    Eigen::MatrixXd grad_b = Eigen::MatrixXd::Zero(n_features, k_freq);
    Eigen::MatrixXd grad_beta = Eigen::MatrixXd::Zero(n_features, d_proj);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < n_features; ++j) {
      int out_col_start = j * (1 + d_proj);
      Eigen::MatrixXd grad_U = grad_E.block(0, out_col_start + 1, B, d_proj);
      grad_beta.row(j) = grad_U.colwise().sum();

      Eigen::MatrixXd grad_W = cache_Z[j].transpose() * grad_U;
      W_proj[j].update(grad_W, lr, beta1, beta2, eps, b1_corr, b2_corr);

      Eigen::MatrixXd grad_Z = grad_U * W_proj[j].val.transpose();
      Eigen::MatrixXd grad_Theta = -grad_Z.cwiseProduct(cache_Theta[j].array().sin().matrix());
      grad_b.row(j) = grad_Theta.colwise().sum();

      Eigen::VectorXd col_j = x.col(j);
      grad_omega.row(j) = (2.0 * M_PI * col_j.transpose() * grad_Theta);
    }

    omega.update(grad_omega, lr, beta1, beta2, eps, b1_corr, b2_corr);
    b_phase.update(grad_b, lr, beta1, beta2, eps, b1_corr, b2_corr);
    beta_proj.update(grad_beta, lr, beta1, beta2, eps, b1_corr, b2_corr);
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

  // Fast forward pass through MLP
  Eigen::MatrixXd forward_mlp(const Eigen::MatrixXd& E,
                              Eigen::MatrixXd* cache_H0 = nullptr,
                              Eigen::MatrixXd* cache_A1 = nullptr,
                              Eigen::MatrixXd* cache_H1 = nullptr,
                              Eigen::MatrixXd* cache_A2 = nullptr,
                              Eigen::MatrixXd* cache_H2 = nullptr,
                              Eigen::MatrixXd* cache_A3 = nullptr,
                              Eigen::MatrixXd* cache_H3 = nullptr) const {
    int B = E.rows();

    Eigen::MatrixXd H0 = E.cwiseProduct(front_scale.val.replicate(B, 1));

    Eigen::MatrixXd A1 = (H0 * W1.val).rowwise() + b1.val.row(0);
    Eigen::MatrixXd H1;
    apply_activation(A1, H1, is_classification);

    Eigen::MatrixXd A2 = (H1 * W2.val).rowwise() + b2.val.row(0);
    Eigen::MatrixXd H2;
    apply_activation(A2, H2, is_classification);

    Eigen::MatrixXd A3 = (H2 * W3.val).rowwise() + b3.val.row(0);
    Eigen::MatrixXd H3;
    apply_activation(A3, H3, is_classification);

    Eigen::MatrixXd Out = (H3 * W4.val).rowwise() + b4.val.row(0);

    if (cache_H0) *cache_H0 = std::move(H0);
    if (cache_A1) *cache_A1 = std::move(A1);
    if (cache_H1) *cache_H1 = std::move(H1);
    if (cache_A2) *cache_A2 = std::move(A2);
    if (cache_H2) *cache_H2 = std::move(H2);
    if (cache_A3) *cache_A3 = std::move(A3);
    if (cache_H3) *cache_H3 = std::move(H3);

    return Out;
  }

  // Full prediction on test data
  Eigen::MatrixXd predict(const Eigen::MatrixXd& X, bool normalize_input = true) const {
    Eigen::MatrixXd X_in = X;
    if (normalize_input && x_mean.size() == X.cols() && x_std.size() == X.cols()) {
      for (int j = 0; j < X.cols(); ++j) {
        double s = x_std(j);
        if (s > 1e-12) {
          X_in.col(j) = (X.col(j).array() - x_mean(j)) / s;
        }
      }
    }
    Eigen::MatrixXd E = embedder.forward(X_in);
    Eigen::MatrixXd logits = forward_mlp(E);

    if (task == "regression") {
      return (logits.array() * y_std + y_mean).matrix();
    } else if (task == "classification") {
      Eigen::MatrixXd probs = logits.unaryExpr([](double z) {
        return 1.0 / (1.0 + std::exp(-std::clamp(z, -30.0, 30.0)));
      });
      return probs;
    } else {
      int B = logits.rows();
      Eigen::MatrixXd probs(B, output_dim);
      for (int i = 0; i < B; ++i) {
        double max_val = logits.row(i).maxCoeff();
        Eigen::RowVectorXd exp_row = (logits.row(i).array() - max_val).exp();
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

  // Compute feature importances
  std::vector<double> compute_importances() const {
    std::vector<double> imp(n_features, 0.0);
    int d_proj = embedder.d_proj;

    for (int j = 0; j < n_features; ++j) {
      int start_col = j * (1 + d_proj);
      double feat_norm = 0.0;
      for (int c = start_col; c <= start_col + d_proj; ++c) {
        double sc = std::abs(front_scale.val(0, c));
        double row_norm = W1.val.row(c).norm();
        feat_norm += sc * row_norm;
      }
      imp[j] = feat_norm;
    }

    double sum_imp = 0.0;
    for (double v : imp) sum_imp += v;
    if (sum_imp > 0.0 && std::isfinite(sum_imp)) {
      for (int j = 0; j < n_features; ++j) imp[j] /= sum_imp;
    }

    return imp;
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
