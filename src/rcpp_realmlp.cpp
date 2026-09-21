#include <Rcpp.h>
#include <RcppEigen.h>
#include "realmlp_core.h"
#include <chrono>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace realmlp;

// Helper to convert Rcpp NumericMatrix to Eigen::MatrixXd
inline Eigen::MatrixXd to_eigen(const NumericMatrix& mat) {
  int r = mat.nrow();
  int c = mat.ncol();
  Eigen::MatrixXd emat(r, c);
  for (int j = 0; j < c; ++j) {
    for (int i = 0; i < r; ++i) {
      double v = mat(i, j);
      emat(i, j) = std::isnan(v) ? 0.0 : v;
    }
  }
  return emat;
}

// Helper to convert Eigen::MatrixXd to Rcpp NumericMatrix
inline NumericMatrix to_rcpp(const Eigen::MatrixXd& emat) {
  int r = static_cast<int>(emat.rows());
  int c = static_cast<int>(emat.cols());
  NumericMatrix mat(r, c);
  for (int j = 0; j < c; ++j) {
    for (int i = 0; i < r; ++i) {
      mat(i, j) = emat(i, j);
    }
  }
  return mat;
}

inline NumericVector to_rcpp_vec(const Eigen::VectorXd& evec) {
  int n = static_cast<int>(evec.size());
  NumericVector vec(n);
  for (int i = 0; i < n; ++i) vec[i] = evec[i];
  return vec;
}

inline Eigen::VectorXd to_eigen_vec(const NumericVector& vec) {
  int n = vec.size();
  Eigen::VectorXd evec(n);
  for (int i = 0; i < n; ++i) evec[i] = vec[i];
  return evec;
}

// Convert model state to an Rcpp::List with explicit names
List model_to_list(const RealMLPModel& m) {
  List W_proj_list(m.embedder.n_features);
  for (int j = 0; j < m.embedder.n_features; ++j) {
    W_proj_list[j] = to_rcpp(m.embedder.W_proj[j].val);
  }

  List p_embed = List::create(
    Named("omega") = to_rcpp(m.embedder.omega.val),
    Named("b_phase") = to_rcpp(m.embedder.b_phase.val),
    Named("beta_proj") = to_rcpp(m.embedder.beta_proj.val),
    Named("W_proj") = W_proj_list,
    Named("k_freq") = m.embedder.k_freq,
    Named("d_proj") = m.embedder.d_proj,
    Named("sigma") = m.embedder.sigma
  );

  List res = List::create(
    Named("task") = m.task,
    Named("n_features") = m.n_features,
    Named("output_dim") = m.output_dim,
    Named("hidden_dim") = m.hidden_dim,
    Named("y_mean") = m.y_mean,
    Named("y_std") = m.y_std,
    Named("x_mean") = to_rcpp_vec(m.x_mean),
    Named("x_std") = to_rcpp_vec(m.x_std),
    Named("embedder") = p_embed,
    Named("front_scale") = to_rcpp(m.front_scale.val),
    Named("W1") = to_rcpp(m.W1.val),
    Named("b1") = to_rcpp(m.b1.val),
    Named("W2") = to_rcpp(m.W2.val),
    Named("b2") = to_rcpp(m.b2.val),
    Named("W3") = to_rcpp(m.W3.val),
    Named("b3") = to_rcpp(m.b3.val),
    Named("W4") = to_rcpp(m.W4.val),
    Named("b4") = to_rcpp(m.b4.val)
  );

  return res;
}

// Reconstruct RealMLPModel from Rcpp::List for prediction
RealMLPModel list_to_model(const List& lst) {
  RealMLPModel m;
  m.task = as<std::string>(lst["task"]);
  m.is_classification = (m.task == "classification" || m.task == "multiclass");
  m.n_features = as<int>(lst["n_features"]);
  m.output_dim = as<int>(lst["output_dim"]);
  m.hidden_dim = as<int>(lst["hidden_dim"]);
  m.y_mean = as<double>(lst["y_mean"]);
  m.y_std = as<double>(lst["y_std"]);

  if (lst.containsElementNamed("x_mean") && lst.containsElementNamed("x_std")) {
    m.x_mean = to_eigen_vec(as<NumericVector>(lst["x_mean"]));
    m.x_std = to_eigen_vec(as<NumericVector>(lst["x_std"]));
  }

  List p_embed = lst["embedder"];
  m.embedder.n_features = m.n_features;
  m.embedder.k_freq = as<int>(p_embed["k_freq"]);
  m.embedder.d_proj = as<int>(p_embed["d_proj"]);
  m.embedder.sigma = as<double>(p_embed["sigma"]);

  m.embedder.omega.val = to_eigen(as<NumericMatrix>(p_embed["omega"]));
  m.embedder.b_phase.val = to_eigen(as<NumericMatrix>(p_embed["b_phase"]));
  m.embedder.beta_proj.val = to_eigen(as<NumericMatrix>(p_embed["beta_proj"]));

  List W_proj_list = p_embed["W_proj"];
  m.embedder.W_proj.resize(m.n_features);
  for (int j = 0; j < m.n_features; ++j) {
    m.embedder.W_proj[j].val = to_eigen(as<NumericMatrix>(W_proj_list[j]));
  }

  m.front_scale.val = to_eigen(as<NumericMatrix>(lst["front_scale"]));
  m.W1.val = to_eigen(as<NumericMatrix>(lst["W1"]));
  m.b1.val = to_eigen(as<NumericMatrix>(lst["b1"]));
  m.W2.val = to_eigen(as<NumericMatrix>(lst["W2"]));
  m.b2.val = to_eigen(as<NumericMatrix>(lst["b2"]));
  m.W3.val = to_eigen(as<NumericMatrix>(lst["W3"]));
  m.b3.val = to_eigen(as<NumericMatrix>(lst["b3"]));
  m.W4.val = to_eigen(as<NumericMatrix>(lst["W4"]));
  m.b4.val = to_eigen(as<NumericMatrix>(lst["b4"]));

  return m;
}

struct MetricResult {
  double score;
  std::string name;
  bool higher_is_better;
};

inline MetricResult compute_val_metric(
    const Eigen::MatrixXd& val_preds,
    const NumericVector& y_v_vec,
    const std::string& task,
    const std::string& metric_req,
    int out_dim) {

  int N_val = static_cast<int>(val_preds.rows());
  std::string m = metric_req;
  std::transform(m.begin(), m.end(), m.begin(), ::tolower);

  if (task == "regression") {
    if (m == "mae") {
      double sum_ae = 0.0;
      for (int i = 0; i < N_val; ++i) {
        sum_ae += std::abs(val_preds(i, 0) - y_v_vec[i]);
      }
      return { sum_ae / N_val, "Val MAE", false };
    }
    double sum_se = 0.0;
    for (int i = 0; i < N_val; ++i) {
      double diff = val_preds(i, 0) - y_v_vec[i];
      sum_se += diff * diff;
    }
    return { std::sqrt(sum_se / N_val), "Val RMSE", false };
  }

  if (task == "classification") {
    if (m == "accuracy" || m == "acc") {
      int correct = 0;
      for (int i = 0; i < N_val; ++i) {
        int pred_class = (val_preds(i, 0) >= 0.5) ? 1 : 0;
        int true_class = (y_v_vec[i] > 0.0) ? 1 : 0;
        if (pred_class == true_class) correct++;
      }
      return { static_cast<double>(correct) / N_val, "Val accuracy", true };
    }
    if (m == "auc") {
      std::vector<std::pair<double, int>> pairs(N_val);
      int n_pos = 0;
      for (int i = 0; i < N_val; ++i) {
        int y = (y_v_vec[i] > 0.0) ? 1 : 0;
        if (y == 1) n_pos++;
        pairs[i] = { val_preds(i, 0), y };
      }
      int n_neg = N_val - n_pos;
      if (n_pos == 0 || n_neg == 0) return { 0.5, "Val AUC", true };
      std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
      });
      double rank_sum_pos = 0.0;
      for (int i = 0; i < N_val; ++i) {
        if (pairs[i].second == 1) {
          rank_sum_pos += (i + 1);
        }
      }
      double u = rank_sum_pos - (static_cast<double>(n_pos) * (n_pos + 1)) / 2.0;
      double auc = u / (static_cast<double>(n_pos) * n_neg);
      return { auc, "Val AUC", true };
    }
    if (m == "error" || m == "err") {
      int errors = 0;
      for (int i = 0; i < N_val; ++i) {
        int pred_class = (val_preds(i, 0) >= 0.5) ? 1 : 0;
        int true_class = (y_v_vec[i] > 0.0) ? 1 : 0;
        if (pred_class != true_class) errors++;
      }
      return { static_cast<double>(errors) / N_val, "Val error", false };
    }
    // Default for binary classification: logloss
    double ll_sum = 0.0;
    for (int i = 0; i < N_val; ++i) {
      double p = std::clamp(val_preds(i, 0), 1e-15, 1.0 - 1e-15);
      double y = (y_v_vec[i] > 0.0) ? 1.0 : 0.0;
      ll_sum -= (y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
    }
    return { ll_sum / N_val, "Val logloss", false };
  }

  // task == "multiclass"
  if (m == "accuracy" || m == "acc") {
    int correct = 0;
    for (int i = 0; i < N_val; ++i) {
      int best_c = 0;
      double max_p = val_preds(i, 0);
      for (int c = 1; c < out_dim; ++c) {
        if (val_preds(i, c) > max_p) {
          max_p = val_preds(i, c);
          best_c = c;
        }
      }
      if (best_c == static_cast<int>(y_v_vec[i])) correct++;
    }
    return { static_cast<double>(correct) / N_val, "Val accuracy", true };
  }
  if (m == "error" || m == "err") {
    int errors = 0;
    for (int i = 0; i < N_val; ++i) {
      int best_c = 0;
      double max_p = val_preds(i, 0);
      for (int c = 1; c < out_dim; ++c) {
        if (val_preds(i, c) > max_p) {
          max_p = val_preds(i, c);
          best_c = c;
        }
      }
      if (best_c != static_cast<int>(y_v_vec[i])) errors++;
    }
    return { static_cast<double>(errors) / N_val, "Val error", false };
  }
  // Default for multiclass: multi-logloss
  double ll_sum = 0.0;
  for (int i = 0; i < N_val; ++i) {
    int true_c = static_cast<int>(y_v_vec[i]);
    double p = 1e-15;
    if (true_c >= 0 && true_c < out_dim) {
      p = std::clamp(val_preds(i, true_c), 1e-15, 1.0);
    }
    ll_sum -= std::log(p);
  }
  return { ll_sum / N_val, "Val logloss", false };
}

//' Train RealMLP Model in C++
//'
//' @param x_train Numeric matrix of training features.
//' @param y_train Numeric vector of training targets.
//' @param x_val Optional numeric matrix of validation features.
//' @param y_val Optional numeric vector of validation targets.
//' @param task String: "regression", "classification", or "multiclass".
//' @param n_epochs Integer: number of training epochs (default 256).
//' @param batch_size Integer: batch size (default 256).
//' @param lr Numeric: base learning rate (<0 for tuned defaults).
//' @param early_stopping_rounds Integer: patience for early stopping (0 = disabled).
//' @param seed Integer: random seed.
//' @param verbose Integer: verbosity level (0 = silent, 1 = normal, 2 = detailed).
//' @param num_classes Integer: number of classes for multiclass task.
//' @param threads Integer: number of threads for parallel computation.
//' @param metric String: validation metric to optimize and report ("default", "logloss", "accuracy", "auc", "rmse", "mae", "error").
//' @return A List containing model weights, training statistics, and feature importances.
//' @export
// [[Rcpp::export]]
List rcpp_realmlp_train(NumericMatrix x_train,
                        NumericVector y_train,
                        Nullable<NumericMatrix> x_val = R_NilValue,
                        Nullable<NumericVector> y_val = R_NilValue,
                        std::string task = "regression",
                        int n_epochs = 256,
                        int batch_size = 256,
                        double lr = -1.0,
                        int early_stopping_rounds = 0,
                        int seed = 42,
                        int verbose = 0,
                        int num_classes = 0,
                        int threads = 1,
                        std::string metric = "default") {

  int N = x_train.nrow();
  int D = x_train.ncol();

  if (N <= 0 || D <= 0) {
    stop("Training data must be non-empty.");
  }

  bool is_cls = (task == "classification");
  bool is_multi = (task == "multiclass");

  int out_dim = 1;
  if (is_multi) {
    if (num_classes <= 1) {
      stop("num_classes must be >= 2 for multiclass task.");
    }
    out_dim = num_classes;
  }

  // Tuned defaults for learning rate
  double base_lr = lr;
  if (base_lr <= 0.0) {
    base_lr = (is_cls || is_multi) ? 0.01 : 0.005;
  }

  Eigen::MatrixXd X_tr = to_eigen(x_train);

  // Standardize training features
  Eigen::VectorXd x_mean(D);
  Eigen::VectorXd x_std(D);
  for (int j = 0; j < D; ++j) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += X_tr(i, j);
    double mean_j = sum / N;
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
      double diff = X_tr(i, j) - mean_j;
      sum_sq += diff * diff;
    }
    double std_j = std::sqrt(sum_sq / std::max(1, N - 1));
    if (std_j < 1e-8 || !std::isfinite(std_j)) std_j = 1.0;
    x_mean(j) = mean_j;
    x_std(j) = std_j;
    for (int i = 0; i < N; ++i) {
      X_tr(i, j) = (X_tr(i, j) - mean_j) / std_j;
    }
  }

  // Targets processing
  Eigen::MatrixXd Y_tr(N, out_dim);
  double y_mean = 0.0;
  double y_std = 1.0;

  if (task == "regression") {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += y_train[i];
    y_mean = sum / N;

    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
      double diff = y_train[i] - y_mean;
      sum_sq += diff * diff;
    }
    y_std = std::sqrt(sum_sq / std::max(1, N - 1));
    if (y_std < 1e-12) y_std = 1.0;

    for (int i = 0; i < N; ++i) {
      Y_tr(i, 0) = (y_train[i] - y_mean) / y_std;
    }
  } else if (is_cls) {
    for (int i = 0; i < N; ++i) {
      double y_val_raw = (y_train[i] > 0.0) ? 1.0 : 0.0;
      Y_tr(i, 0) = y_val_raw * 0.9 + 0.05;
    }
  } else {
    double eps = 0.1;
    double smooth_base = eps / out_dim;
    Y_tr.fill(smooth_base);
    for (int i = 0; i < N; ++i) {
      int c = static_cast<int>(y_train[i]);
      if (c >= 0 && c < out_dim) {
        Y_tr(i, c) += (1.0 - eps);
      }
    }
  }

  // Validation data
  bool has_val = (x_val.isNotNull() && y_val.isNotNull());
  Eigen::MatrixXd X_v;
  NumericVector y_v_vec;
  int N_val = 0;

  if (has_val) {
    NumericMatrix xv(x_val);
    X_v = to_eigen(xv);
    y_v_vec = NumericVector(y_val);
    N_val = static_cast<int>(X_v.rows());
    for (int j = 0; j < D; ++j) {
      double mean_j = x_mean(j);
      double std_j = x_std(j);
      for (int i = 0; i < N_val; ++i) {
        X_v(i, j) = (X_v(i, j) - mean_j) / std_j;
      }
    }
  }

  // Initialize model
  RealMLPModel model;
  model.init(D, out_dim, task, seed, 256);
  model.y_mean = y_mean;
  model.y_std = y_std;
  model.x_mean = x_mean;
  model.x_std = x_std;

  // Training parameters
  int actual_batch_size = std::max(1, std::min(batch_size, N));
  if (actual_batch_size == N && N > 32) {
    actual_batch_size = std::min(64, std::max(16, N / 4));
  }
  int n_batches = std::max(1, N / actual_batch_size);
  int total_steps = n_epochs * n_batches;

  double beta1 = 0.9;
  double beta2 = 0.95;
  double eps = 1e-8;

  std::mt19937 shuffle_rng(static_cast<unsigned int>(seed + 100));
  std::vector<int> perm(N);
  for (int i = 0; i < N; ++i) perm[i] = i;

  // Save previous thread state for CRAN-compliant restoration
#ifdef _OPENMP
  int old_omp_threads = omp_get_max_threads();
  if (threads > 0) {
    omp_set_num_threads(threads);
  }
#endif
  int old_eigen_threads = Eigen::nbThreads();
  Eigen::setNbThreads(threads > 0 ? threads : 1);

  double best_val_score = std::numeric_limits<double>::quiet_NaN();
  std::string val_metric_nm = "Val metric";
  RealMLPModel::StateSnapshot best_snapshot;
  bool has_best_snapshot = false;
  int no_improve_epochs = 0;
  bool stopped_early = false;
  int stopped_epoch = n_epochs;

  int log_interval = (verbose >= 2) ? 32 : 64;

  if (verbose >= 1) {
    std::string es_info = (early_stopping_rounds > 0 && has_val) ?
      (" (early stopping patience=" + std::to_string(early_stopping_rounds) + ")") : "";
    Rprintf("    [RealMLP C++ %s] Starting training: %d rows (%d features) (epochs: %d, batch: %d%s)...\n",
            task.c_str(), N, D, n_epochs, actual_batch_size, es_info.c_str());
  }

  std::vector<Eigen::MatrixXd> cache_Z(D);
  std::vector<Eigen::MatrixXd> cache_Theta(D);

  int step = 0;

  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    std::shuffle(perm.begin(), perm.end(), shuffle_rng);
    double epoch_loss_sum = 0.0;

    for (int b = 0; b < n_batches; ++b) {
      step++;
      int start = b * actual_batch_size;
      int end = (b == n_batches - 1) ? N : std::min(start + actual_batch_size, N);
      int cur_B = end - start;
      if (cur_B <= 0) break;

      Eigen::MatrixXd X_batch(cur_B, D);
      Eigen::MatrixXd Y_batch(cur_B, out_dim);
      for (int i = 0; i < cur_B; ++i) {
        int idx = perm[start + i];
        X_batch.row(i) = X_tr.row(idx);
        Y_batch.row(i) = Y_tr.row(idx);
      }

      double t_norm = static_cast<double>(step - 1) / static_cast<double>(std::max(1, total_steps - 1));
      double cur_lr = base_lr * coslog4_schedule(t_norm);
      double b1_corr = 1.0 - std::pow(beta1, step);
      double b2_corr = 1.0 - std::pow(beta2, step);

      // 1. Forward PBLD
      Eigen::MatrixXd E = model.embedder.forward(X_batch, &cache_Z, &cache_Theta);

      // 2. Forward MLP
      Eigen::MatrixXd H0, A1, H1, A2, H2, A3, H3;
      Eigen::MatrixXd Out = model.forward_mlp(E, &H0, &A1, &H1, &A2, &H2, &A3, &H3);

      // 3. Loss & output gradient
      Eigen::MatrixXd grad_out(cur_B, out_dim);
      double batch_loss = 0.0;

      if (task == "regression") {
        Eigen::MatrixXd diff = Out - Y_batch;
        grad_out = (2.0 / cur_B) * diff;
        batch_loss = diff.squaredNorm() / cur_B;
      } else if (is_cls) {
        Eigen::MatrixXd P = Out.unaryExpr([](double z) {
          return 1.0 / (1.0 + std::exp(-std::clamp(z, -30.0, 30.0)));
        });
        grad_out = (1.0 / cur_B) * (P - Y_batch);
        for (int i = 0; i < cur_B; ++i) {
          double p = std::clamp(P(i, 0), 1e-15, 1.0 - 1e-15);
          double y = Y_batch(i, 0);
          batch_loss -= (y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        }
        batch_loss /= cur_B;
      } else {
        Eigen::MatrixXd P(cur_B, out_dim);
        for (int i = 0; i < cur_B; ++i) {
          double max_val = Out.row(i).maxCoeff();
          Eigen::RowVectorXd exp_row = (Out.row(i).array() - max_val).exp();
          double sum_exp = exp_row.sum();
          if (sum_exp > 0.0) {
            P.row(i) = exp_row / sum_exp;
          } else {
            P.row(i).fill(1.0 / out_dim);
          }
          for (int c = 0; c < out_dim; ++c) {
            batch_loss -= Y_batch(i, c) * std::log(std::max(1e-15, P(i, c)));
          }
        }
        batch_loss /= cur_B;
        grad_out = (1.0 / cur_B) * (P - Y_batch);
      }
      double max_g = grad_out.array().abs().maxCoeff();
      if (max_g > 10.0) {
        grad_out *= (10.0 / max_g);
      }

      epoch_loss_sum += batch_loss;

      // 4. Backprop MLP
      Eigen::MatrixXd grad_W4 = H3.transpose() * grad_out;
      Eigen::MatrixXd grad_b4 = grad_out.colwise().sum();
      Eigen::MatrixXd delta3 = grad_out * model.W4.val.transpose();
      apply_activation_grad(A3, delta3, model.is_classification);

      Eigen::MatrixXd grad_W3 = H2.transpose() * delta3;
      Eigen::MatrixXd grad_b3 = delta3.colwise().sum();
      Eigen::MatrixXd delta2 = delta3 * model.W3.val.transpose();
      apply_activation_grad(A2, delta2, model.is_classification);

      Eigen::MatrixXd grad_W2 = H1.transpose() * delta2;
      Eigen::MatrixXd grad_b2 = delta2.colwise().sum();
      Eigen::MatrixXd delta1 = delta2 * model.W2.val.transpose();
      apply_activation_grad(A1, delta1, model.is_classification);

      Eigen::MatrixXd grad_W1 = H0.transpose() * delta1;
      Eigen::MatrixXd grad_b1 = delta1.colwise().sum();
      Eigen::MatrixXd delta0 = delta1 * model.W1.val.transpose();

      Eigen::MatrixXd grad_scale = (E.cwiseProduct(delta0)).colwise().sum();
      Eigen::MatrixXd delta_E = delta0.cwiseProduct(model.front_scale.val.replicate(cur_B, 1));

      // 5. Backprop PBLD Embeddings and Adam update
      model.embedder.backward_and_update(X_batch, delta_E, cache_Z, cache_Theta,
                                         cur_lr, beta1, beta2, eps, b1_corr, b2_corr);

      // 6. Adam update MLP
      model.front_scale.update(grad_scale, cur_lr * 6.0, beta1, beta2, eps, b1_corr, b2_corr);
      model.W1.update(grad_W1, cur_lr * 1.0, beta1, beta2, eps, b1_corr, b2_corr);
      model.b1.update(grad_b1, cur_lr * 0.1, beta1, beta2, eps, b1_corr, b2_corr);
      model.W2.update(grad_W2, cur_lr * 1.0, beta1, beta2, eps, b1_corr, b2_corr);
      model.b2.update(grad_b2, cur_lr * 0.1, beta1, beta2, eps, b1_corr, b2_corr);
      model.W3.update(grad_W3, cur_lr * 1.0, beta1, beta2, eps, b1_corr, b2_corr);
      model.b3.update(grad_b3, cur_lr * 0.1, beta1, beta2, eps, b1_corr, b2_corr);
      model.W4.update(grad_W4, cur_lr * 1.0, beta1, beta2, eps, b1_corr, b2_corr);
      model.b4.update(grad_b4, cur_lr * 0.1, beta1, beta2, eps, b1_corr, b2_corr);
    }

    double avg_train_loss = epoch_loss_sum / n_batches;
    double val_score = std::numeric_limits<double>::quiet_NaN();

    if (has_val && N_val > 0) {
      Eigen::MatrixXd val_preds = model.predict(X_v, false);
      MetricResult m_res = compute_val_metric(val_preds, y_v_vec, task, metric, out_dim);
      val_score = m_res.score;
      val_metric_nm = m_res.name;

      if (std::isnan(best_val_score)) {
        best_val_score = m_res.higher_is_better ? -std::numeric_limits<double>::infinity()
                                                : std::numeric_limits<double>::infinity();
      }

      bool is_better = m_res.higher_is_better ? (val_score > best_val_score + 1e-6)
                                              : (val_score < best_val_score - 1e-6);

      if (is_better) {
        best_val_score = val_score;
        best_snapshot = model.get_snapshot();
        has_best_snapshot = true;
        no_improve_epochs = 0;
      } else {
        no_improve_epochs++;
      }
    }

    int cur_epoch = epoch + 1;
    bool is_best = (no_improve_epochs == 0 && has_val);
    bool should_log = (verbose >= 1) && (
      cur_epoch % log_interval == 0 ||
      cur_epoch == 1 ||
      cur_epoch == n_epochs ||
      (verbose >= 2 && is_best)
    );

    if (should_log) {
      if (has_val) {
        std::string star = is_best ? "*" : "";
        Rprintf("    [RealMLP C++ %s] Epoch %3d/%d | Train loss: %.4f | %s: %.4f (best: %.4f%s)\n",
                task.c_str(), cur_epoch, n_epochs, avg_train_loss, val_metric_nm.c_str(),
                val_score, best_val_score, star.c_str());
      } else {
        Rprintf("    [RealMLP C++ %s] Epoch %3d/%d | Train loss: %.4f\n",
                task.c_str(), cur_epoch, n_epochs, avg_train_loss);
      }
    }

    if (early_stopping_rounds > 0 && has_val && no_improve_epochs >= early_stopping_rounds) {
      stopped_early = true;
      stopped_epoch = cur_epoch;
      if (verbose >= 1) {
        Rprintf("    [RealMLP C++ %s] Early stopping triggered at epoch %d (patience = %d, best val: %.4f)\n",
                task.c_str(), cur_epoch, early_stopping_rounds, best_val_score);
      }
      break;
    }

    if (epoch % 16 == 0) {
      R_CheckUserInterrupt();
    }
  }

  if (has_val && has_best_snapshot && early_stopping_rounds > 0) {
    model.restore_snapshot(best_snapshot);
  }

  std::vector<double> importances = model.compute_importances();

  // Restore previous thread state (CRAN requirement)
#ifdef _OPENMP
  omp_set_num_threads(old_omp_threads);
#endif
  Eigen::setNbThreads(old_eigen_threads);

  List res = List::create(
    Named("model_state") = model_to_list(model),
    Named("importances") = wrap(importances),
    Named("stopped_early") = stopped_early,
    Named("stopped_epoch") = stopped_epoch,
    Named("best_val_score") = best_val_score
  );

  return res;
}

//' Predict with RealMLP Model in C++
//'
//' @param model_state List returned by `rcpp_realmlp_train`.
//' @param x_new Numeric matrix of new input features.
//' @return A NumericMatrix of predictions.
//' @export
// [[Rcpp::export]]
NumericMatrix rcpp_realmlp_predict(List model_state, NumericMatrix x_new) {
  RealMLPModel model = list_to_model(model_state);
  Eigen::MatrixXd X_n = to_eigen(x_new);
  Eigen::MatrixXd preds = model.predict(X_n);
  return to_rcpp(preds);
}
