#ifndef EVOFE_REALMLP_WORKSPACE_H
#define EVOFE_REALMLP_WORKSPACE_H

#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <algorithm>

namespace realmlp {

// Dedicated pre-allocated workspace for per-epoch validation evaluation.
// Eliminates all dynamic heap allocations across training epoch validation iterations.
struct ValidationWorkspace {
  Eigen::MatrixXd E;
  Eigen::MatrixXd H0, A1, H1, A2, H2, A3, H3, Out;
  Eigen::MatrixXd preds;
  std::vector<std::pair<double, int>> auc_pairs;
  std::vector<Eigen::MatrixXd> thread_Z_buf;

  void allocate(int N_val, int D, int out_dim, int hidden_dim, int embed_dim, int k_freq, int max_threads = 1) {
    if (N_val <= 0 || D <= 0) return;
    E.resize(N_val, embed_dim);
    H0.resize(N_val, embed_dim);
    A1.resize(N_val, hidden_dim);
    H1.resize(N_val, hidden_dim);
    A2.resize(N_val, hidden_dim);
    H2.resize(N_val, hidden_dim);
    A3.resize(N_val, hidden_dim);
    H3.resize(N_val, hidden_dim);
    Out.resize(N_val, out_dim);
    preds.resize(N_val, out_dim);
    auc_pairs.resize(N_val);

    int actual_threads = std::max(1, max_threads);
    thread_Z_buf.resize(actual_threads);
    for (int t = 0; t < actual_threads; ++t) {
      thread_Z_buf[t].resize(N_val, std::max(1, k_freq));
    }
  }
};

// Dedicated pre-allocated workspace for feature importance occlusion ablation.
// Unifies E_zero into a contiguous D x (1 + d_proj) matrix and provides reusable
// thread_Z_buf, eliminating all heap allocations inside the feature ablation loop.
struct FeatureImportanceWorkspace {
  Eigen::MatrixXd E_base;
  Eigen::MatrixXd E_occ;
  Eigen::MatrixXd H0, A1, H1, A2, H2, A3, H3, Out;
  Eigen::MatrixXd preds;
  Eigen::MatrixXd E_zero; // Contiguous D x (1 + d_proj) buffer
  std::vector<Eigen::MatrixXd> thread_Z_buf;

  void allocate(int N_eval, int D, int out_dim, int hidden_dim, int embed_dim, int d_proj, int k_freq = 0, int max_threads = 1) {
    if (N_eval <= 0 || D <= 0) return;
    E_base.resize(N_eval, embed_dim);
    E_occ.resize(N_eval, embed_dim);
    H0.resize(N_eval, embed_dim);
    A1.resize(N_eval, hidden_dim);
    H1.resize(N_eval, hidden_dim);
    A2.resize(N_eval, hidden_dim);
    H2.resize(N_eval, hidden_dim);
    A3.resize(N_eval, hidden_dim);
    H3.resize(N_eval, hidden_dim);
    Out.resize(N_eval, out_dim);
    preds.resize(N_eval, out_dim);
    E_zero.resize(D, 1 + d_proj);

    int actual_threads = std::max(1, max_threads);
    thread_Z_buf.resize(actual_threads);
    for (int t = 0; t < actual_threads; ++t) {
      thread_Z_buf[t].resize(N_eval, std::max(1, k_freq));
    }
  }
};

} // namespace realmlp

#endif // EVOFE_REALMLP_WORKSPACE_H
