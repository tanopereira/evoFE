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

// M_PI is POSIX, not C++17 standard — provide fallback for strict compilers
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace realmlp {

// Vectorized activation: Mish(x) = x * tanh(softplus(x))
inline void apply_activation(const Eigen::Ref<const Eigen::MatrixXd>& in, Eigen::Ref<Eigen::MatrixXd> out, bool is_cls) {
  const auto& arr = in.array();
  if (is_cls) {
    // SELU: lambda * (x if x > 0 else alpha * expm1(x))
    constexpr double LAMBDA = 1.0507009873554804934193349852946;
    constexpr double ALPHA  = 1.6732632423543772848170429916717;
    out.array() = (arr > 0.0).select(LAMBDA * arr, LAMBDA * ALPHA * arr.exp() - LAMBDA * ALPHA);
  } else {
    // Mish: x * tanh(softplus(x))  — softplus clamped for stability
    auto sp = (arr > 20.0).select(arr, (arr < -20.0).select(arr.exp(), (1.0 + arr.exp()).log()));
    out.array() = arr * sp.tanh();
  }
}

inline void apply_activation_grad_inplace(const Eigen::Ref<const Eigen::MatrixXd>& act_in, Eigen::Ref<Eigen::MatrixXd> delta, bool is_cls) {
  const auto& a = act_in.array();
  if (is_cls) {
    constexpr double LAMBDA = 1.0507009873554804934193349852946;
    constexpr double ALPHA  = 1.6732632423543772848170429916717;
    delta.array() *= (a > 0.0).select(
      Eigen::ArrayXXd::Constant(delta.rows(), delta.cols(), LAMBDA),
      LAMBDA * ALPHA * a.exp()
    );
  } else {
    // Mish grad: tanh(sp) + x * sigmoid(x) * (1 - tanh²(sp))
    auto sp = (a > 20.0).select(a, (a < -20.0).select(a.exp(), (1.0 + a.exp()).log()));
    auto tsp = sp.tanh();
    auto sig = 1.0 / (1.0 + (-a.max(-30.0).min(30.0)).exp());
    delta.array() *= tsp + a * sig * (1.0 - tsp * tsp);
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
      E.col(out_col_start) = x.col(j);

      // Theta = 2*pi * x_j * omega_j + b_j  (write into cache directly)
      cache_Theta[j].noalias() = 2.0 * M_PI * x.col(j) * omega.val.row(j);
      cache_Theta[j].rowwise() += b_phase.val.row(j);
      cache_Z[j].array() = cache_Theta[j].array().cos();

      // U = Z * W_j + beta_j
      E.block(0, out_col_start + 1, B, d_proj).noalias() = cache_Z[j] * W_proj[j].val;
      E.block(0, out_col_start + 1, B, d_proj).rowwise() += beta_proj.val.row(j);
    }
  }

  // Forward pass without caching (for prediction)
  Eigen::MatrixXd forward_nocache(const Eigen::MatrixXd& x) const {
    int B = x.rows();
    int out_dim = total_out_dim();
    Eigen::MatrixXd E(B, out_dim);

    for (int j = 0; j < n_features; ++j) {
      int out_col_start = j * (1 + d_proj);
      E.col(out_col_start) = x.col(j);
      Eigen::MatrixXd Theta = (2.0 * M_PI * x.col(j) * omega.val.row(j)).rowwise() + b_phase.val.row(j);
      Eigen::MatrixXd Z = Theta.array().cos().matrix();
      E.block(0, out_col_start + 1, B, d_proj).noalias() = Z * W_proj[j].val;
      E.block(0, out_col_start + 1, B, d_proj).rowwise() += beta_proj.val.row(j);
    }
    return E;
  }

  // Backward pass through PBLD and Adam parameter update
  void backward_and_update(const Eigen::MatrixXd& x,
                           const Eigen::MatrixXd& grad_E,
                           const std::vector<Eigen::MatrixXd>& cache_Z,
                           const std::vector<Eigen::MatrixXd>& cache_Theta,
                           Eigen::MatrixXd& grad_omega_buf,
                           Eigen::MatrixXd& grad_b_buf,
                           Eigen::MatrixXd& grad_beta_buf,
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
      int out_col_start = j * (1 + d_proj);
      auto grad_U = grad_E.block(0, out_col_start + 1, B, d_proj);
      grad_beta_buf.row(j) = grad_U.colwise().sum();

      Eigen::MatrixXd grad_W = cache_Z[j].transpose() * grad_U;
      W_proj[j].update(grad_W, lr, beta1, beta2, eps, b1_corr, b2_corr);

      Eigen::MatrixXd grad_Z = grad_U * W_proj[j].val.transpose();
      Eigen::MatrixXd grad_Th = -grad_Z.cwiseProduct(cache_Theta[j].array().sin().matrix());
      grad_b_buf.row(j) = grad_Th.colwise().sum();
      grad_omega_buf.row(j) = (2.0 * M_PI * x.col(j).transpose() * grad_Th);
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

  void allocate(int max_B, int D, int out_dim, int hidden_dim, int embed_dim, int n_features, int k_freq, int d_proj, int N) {
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
    H0.topRows(cur_B).noalias() = E_block.cwiseProduct(front_scale.val.replicate(cur_B, 1));

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

  // Full prediction on test data (allocates freely — not in hot loop)
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
    Eigen::MatrixXd E = embedder.forward_nocache(X_in);
    int B = E.rows();
    Eigen::MatrixXd H0, A1, H1, A2, H2, A3, H3, Out;
    H0.resize(B, E.cols()); A1.resize(B, hidden_dim); H1.resize(B, hidden_dim);
    A2.resize(B, hidden_dim); H2.resize(B, hidden_dim);
    A3.resize(B, hidden_dim); H3.resize(B, hidden_dim);
    Out.resize(B, output_dim);
    forward_mlp(E, B, H0, A1, H1, A2, H2, A3, H3, Out);

    if (task == "regression") {
      return (Out.topRows(B).array() * y_std + y_mean).matrix();
    } else if (task == "classification") {
      return Out.topRows(B).unaryExpr([](double z) {
        return 1.0 / (1.0 + std::exp(-std::clamp(z, -30.0, 30.0)));
      });
    } else {
      Eigen::MatrixXd probs(B, output_dim);
      for (int i = 0; i < B; ++i) {
        double max_val = Out(i, 0);
        for (int c = 1; c < output_dim; ++c) if (Out(i,c) > max_val) max_val = Out(i,c);
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

  // Compute feature importances via Mean Occlusion (Zero-Out Ablation)
  std::vector<double> compute_importances(
      const Eigen::MatrixXd& X_eval,
      const std::vector<double>& y_eval) const {

    int D = n_features;
    std::vector<double> imp(D, 1.0 / std::max(1, D));
    int N_eval = static_cast<int>(X_eval.rows());
    if (D <= 0 || N_eval <= 0 || static_cast<int>(y_eval.size()) < N_eval) {
      return imp;
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
        // multiclass
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

    // 1. Baseline loss (normalize_input = false because X_eval is already standardized)
    Eigen::MatrixXd p_base = predict(X_eval, false);
    double base_loss = compute_loss(p_base);

    // 2. Mean occlusion: since X_eval is standardized, mean of each feature is 0.0
    Eigen::MatrixXd X_occ = X_eval;
    for (int j = 0; j < D; ++j) {
      Eigen::VectorXd orig_col = X_occ.col(j);
      X_occ.col(j).setZero();

      Eigen::MatrixXd p_occ = predict(X_occ, false);
      double loss_occ = compute_loss(p_occ);

      X_occ.col(j) = orig_col;

      double delta_loss = std::max(0.0, loss_occ - base_loss);
      imp[j] = delta_loss;
    }

    // 3. Normalize importances to sum to 1.0
    double sum_imp = 0.0;
    for (double v : imp) sum_imp += v;
    if (sum_imp > 1e-12 && std::isfinite(sum_imp)) {
      for (int j = 0; j < D; ++j) imp[j] /= sum_imp;
    } else {
      std::fill(imp.begin(), imp.end(), 1.0 / D);
    }

    return imp;
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
