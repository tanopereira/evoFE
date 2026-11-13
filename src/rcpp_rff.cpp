#include "rcpp_nn_bridge.h"
#include "rff_core.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace Rcpp;
using evofe::nn::ThreadGuard;
using evofe::nn::compute_finite_column_stats;
using evofe::nn::compute_val_metric;
using evofe::nn::copy_eigen_fast;
using evofe::nn::to_eigen_vec;
using evofe::nn::to_rcpp;
using evofe::nn::to_rcpp_vec;
using evofe::rff::RFFModel;
using evofe::rff::RFFWorkspace;

namespace {

inline std::string to_lower_str(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

inline const double* resolve_offset_ptr(const Nullable<NumericVector>& off_opt,
                                        NumericVector& holder,
                                        int N, int out_dim,
                                        const char* arg_name) {
  if (off_opt.isNull()) return nullptr;
  holder = NumericVector(off_opt.get());
  const R_xlen_t len = holder.size();
  if (len == 0) return nullptr;
  const R_xlen_t expected = static_cast<R_xlen_t>(N) * static_cast<R_xlen_t>(out_dim);
  if (len != expected) {
    stop("'%s' length (%td) does not match expected N * out_dim (%td).",
         arg_name, static_cast<ptrdiff_t>(len), static_cast<ptrdiff_t>(expected));
  }
  return holder.begin();
}

inline List rff_model_to_list(const RFFModel& m) {
  return List::create(
    Named("task") = m.task,
    Named("kernel") = m.kernel,
    Named("approx_method") = m.approx_method,
    Named("d_in") = m.d_in,
    Named("n_features") = m.n_features,
    Named("p_cols") = m.p_cols,
    Named("out_dim") = m.out_dim,
    Named("sigma") = m.sigma,
    Named("lambda") = m.lambda,
    Named("x_mean") = to_rcpp_vec(m.x_mean),
    Named("x_sd") = to_rcpp_vec(m.x_sd),
    Named("x_inv_std") = to_rcpp_vec(m.x_inv_std),
    Named("k_freq") = m.k_freq,
    Named("rff_scale") = static_cast<double>(m.rff_scale),
    Named("W_rff") = to_rcpp(m.W_rff),
    Named("m_landmarks") = m.m_landmarks,
    Named("landmarks") = to_rcpp(m.landmarks),
    Named("landmark_norm2") = to_rcpp_vec(m.landmark_norm2.cast<double>()),
    Named("P_nys") = to_rcpp(m.P_nys),
    Named("W_out") = to_rcpp(m.W_out)
  );
}

inline RFFModel list_to_rff_model(const List& lst) {
  RFFModel m;
  m.task = as<std::string>(lst["task"]);
  m.kernel = as<std::string>(lst["kernel"]);
  m.approx_method = as<std::string>(lst["approx_method"]);
  m.d_in = as<int>(lst["d_in"]);
  m.n_features = as<int>(lst["n_features"]);
  m.p_cols = as<int>(lst["p_cols"]);
  m.out_dim = as<int>(lst["out_dim"]);
  m.sigma = as<double>(lst["sigma"]);
  m.lambda = as<double>(lst["lambda"]);
  m.x_mean = to_eigen_vec(as<NumericVector>(lst["x_mean"]));
  m.x_sd = to_eigen_vec(as<NumericVector>(lst["x_sd"]));
  m.x_inv_std = to_eigen_vec(as<NumericVector>(lst["x_inv_std"]));
  m.k_freq = as<int>(lst["k_freq"]);
  m.rff_scale = static_cast<float>(as<double>(lst["rff_scale"]));
  m.W_rff = copy_eigen_fast(as<NumericMatrix>(lst["W_rff"]));
  m.m_landmarks = as<int>(lst["m_landmarks"]);
  m.landmarks = copy_eigen_fast(as<NumericMatrix>(lst["landmarks"]));
  m.landmark_norm2 = to_eigen_vec(as<NumericVector>(lst["landmark_norm2"])).cast<float>();
  m.P_nys = copy_eigen_fast(as<NumericMatrix>(lst["P_nys"]));
  m.W_out = copy_eigen_fast(as<NumericMatrix>(lst["W_out"]));
  return m;
}

} // namespace

//' Train RFF or Nystrom Kernel Ridge / Logistic Model in C++
//'
//' @param x_train Numeric matrix of training features.
//' @param y_train Numeric vector of training targets.
//' @param x_val Optional validation feature matrix.
//' @param y_val Optional validation target vector.
//' @param task Task type: "regression", "classification", or "multiclass".
//' @param sigma Kernel bandwidth (> 0); if <= 0, auto median heuristic is used.
//' @param lambda L2 ridge regularization penalty.
//' @param n_features Number of random features or Nystrom landmarks.
//' @param kernel Kernel family: "rbf" or "laplacian".
//' @param approx_method Approximation method: "rff" or "nystrom".
//' @param threads Number of threads for BLAS/OpenMP.
//' @param seed Random seed.
//' @param num_classes Number of classes for multiclass (0 = infer from y_train).
//' @param metric Validation metric name.
//' @param offset_train Optional training offset vector (N or N * K column-major).
//' @param offset_val Optional validation offset vector (N_val or N_val * K column-major).
//' @param max_irls_iter Maximum Newton-IRLS / L-BFGS iterations for classification.
//' @param irls_tol Convergence tolerance for classification solver.
//' @return A List containing model_state, val_preds, val_score, importances, and sigma.
// [[Rcpp::export]]
List rcpp_rff_train(
    NumericMatrix x_train,
    NumericVector y_train,
    Nullable<NumericMatrix> x_val = R_NilValue,
    Nullable<NumericVector> y_val = R_NilValue,
    std::string task = "regression",
    double sigma = -1.0,
    double lambda = 1e-2,
    int n_features = 512,
    std::string kernel = "rbf",
    std::string approx_method = "rff",
    int threads = 1,
    int seed = 42,
    int num_classes = 0,
    std::string metric = "default",
    Nullable<NumericVector> offset_train = R_NilValue,
    Nullable<NumericVector> offset_val = R_NilValue,
    int max_irls_iter = 25,
    double irls_tol = 1e-6) {

  ThreadGuard guard(threads);

  const int N = x_train.nrow();
  const int D = x_train.ncol();
  if (y_train.size() != N) {
    stop("Length of 'y_train' (%d) does not match nrow(x_train) (%d).",
         static_cast<int>(y_train.size()), N);
  }

  RFFModel model;
  model.task = to_lower_str(task);
  if (model.task != "regression" && model.task != "classification" && model.task != "multiclass") {
    stop("Invalid 'task': '%s'. Must be 'regression', 'classification', or 'multiclass'.", task.c_str());
  }
  model.kernel = to_lower_str(kernel);
  if (model.kernel != "rbf" && model.kernel != "laplacian") {
    stop("Invalid 'kernel': '%s'. Must be 'rbf' or 'laplacian'.", kernel.c_str());
  }
  model.approx_method = to_lower_str(approx_method);
  if (model.approx_method != "rff" && model.approx_method != "nystrom") {
    stop("Invalid 'approx_method': '%s'. Must be 'rff' or 'nystrom'.", approx_method.c_str());
  }

  model.d_in = D;
  model.n_features = std::max(1, n_features);
  model.lambda = (std::isfinite(lambda) && lambda > 0.0) ? lambda : 1e-2;

  // Determine out_dim and sanitize y_train
  std::vector<double> y_clean(N, 0.0);
  std::vector<int> y_cls(N, 0);

  if (model.task == "regression") {
    model.out_dim = 1;
    double y_sum = 0.0;
    int y_fin = 0;
    for (int i = 0; i < N; ++i) {
      if (std::isfinite(y_train[i])) {
        y_sum += y_train[i];
        y_fin++;
      }
    }
    const double y_mean = (y_fin > 0) ? (y_sum / y_fin) : 0.0;
    for (int i = 0; i < N; ++i) {
      y_clean[i] = std::isfinite(y_train[i]) ? y_train[i] : y_mean;
    }
  } else if (model.task == "classification") {
    model.out_dim = 1;
    for (int i = 0; i < N; ++i) {
      const double v = y_train[i];
      y_clean[i] = (std::isfinite(v) && v > 0.0) ? 1.0 : 0.0;
    }
  } else {
    int max_c = 0;
    for (int i = 0; i < N; ++i) {
      if (std::isfinite(y_train[i])) {
        const int c = static_cast<int>(y_train[i]);
        if (c > max_c) max_c = c;
      }
    }
    model.out_dim = (num_classes >= 2) ? num_classes : std::max(2, max_c + 1);
    for (int i = 0; i < N; ++i) {
      int c = std::isfinite(y_train[i]) ? static_cast<int>(y_train[i]) : 0;
      y_cls[i] = std::clamp(c, 0, model.out_dim - 1);
    }
  }

  NumericVector off_tr_holder;
  const double* off_tr_ptr = resolve_offset_ptr(
      offset_train, off_tr_holder, N, model.out_dim, "offset_train");

  // Compute training column stats (mean over finite values, sd 0 -> 1)
  model.x_mean.resize(D);
  model.x_sd.resize(D);
  model.x_inv_std.resize(D);
  const double* x_tr_raw = x_train.begin();

  for (int j = 0; j < D; ++j) {
    const auto st = compute_finite_column_stats(x_tr_raw + static_cast<size_t>(j) * N, N);
    model.x_mean(j) = st.mean;
    model.x_sd(j) = st.sd;
    model.x_inv_std(j) = st.inv_std;
  }

  // Standardize X_train once into ColMajor FP32
  Eigen::MatrixXf X_tr_f32(N, D);
  if (N > 0 && D > 0) {
    evofe::rff::standardize_impute_mean_f32(
        x_tr_raw, N, X_tr_f32.data(), N, N, D,
        model.x_mean.data(), model.x_inv_std.data());
  }

  // Bandwidth sigma (auto median heuristic if <= 0)
  if (!std::isfinite(sigma) || sigma <= 0.0) {
    model.sigma = RFFModel::estimate_median_sigma(X_tr_f32, N, D, seed);
  } else {
    model.sigma = std::clamp(sigma, 1e-6, 1e6);
  }

  model.init_basis(X_tr_f32, N, seed);

  RFFWorkspace ws;
  ws.init(model.d_in, model.p_cols, model.k_freq, model.m_landmarks, model.out_dim, true);

  if (N > 0) {
    if (model.task == "regression") {
      model.fit_regression(X_tr_f32, y_clean.data(), off_tr_ptr, N, ws);
    } else if (model.task == "classification") {
      model.fit_binary_irls(X_tr_f32, y_clean.data(), off_tr_ptr, N, max_irls_iter, irls_tol, ws);
    } else {
      model.fit_multiclass_lbfgs(X_tr_f32, y_cls.data(), off_tr_ptr, N, max_irls_iter, irls_tol, ws);
    }
  }

  // Validation evaluation
  SEXP val_preds_sexp = R_NilValue;
  double val_score = std::numeric_limits<double>::quiet_NaN();
  if (x_val.isNotNull() && y_val.isNotNull()) {
    NumericMatrix xv(x_val.get());
    NumericVector yv(y_val.get());
    const int N_val = xv.nrow();
    if (yv.size() != N_val) {
      stop("Length of 'y_val' (%d) does not match nrow(x_val) (%d).",
           static_cast<int>(yv.size()), N_val);
    }
    NumericVector off_val_holder;
    const double* off_val_ptr = resolve_offset_ptr(
        offset_val, off_val_holder, N_val, model.out_dim, "offset_val");
    Eigen::MatrixXd val_preds = model.predict_raw(xv.begin(), N_val, xv.ncol(), off_val_ptr, ws);
    val_preds_sexp = to_rcpp(val_preds);
    if (N_val > 0) {
      auto m_res = compute_val_metric(val_preds, yv, model.task, metric, model.out_dim, nullptr, threads);
      val_score = m_res.score;
    }
  }

  // Fast permutation-free importances on up to 1000 rows
  std::vector<double> importances = model.compute_importances(X_tr_f32, N, off_tr_ptr, ws);

  return List::create(
    Named("model_state") = rff_model_to_list(model),
    Named("val_preds") = val_preds_sexp,
    Named("val_score") = val_score,
    Named("best_val_score") = val_score,
    Named("importances") = wrap(importances),
    Named("sigma") = model.sigma
  );
}

//' Predict with RFF or Nystrom Kernel Model in C++
//'
//' @param model_state List returned inside `rcpp_rff_train(...)$model_state`.
//' @param x_new Numeric matrix of new input features.
//' @param offset_new Optional offset vector (N_new or N_new * K column-major).
//' @param threads Integer: number of threads (0 = unchanged).
//' @return NumericMatrix of predictions.
// [[Rcpp::export]]
NumericMatrix rcpp_rff_predict(
    List model_state,
    NumericMatrix x_new,
    Nullable<NumericVector> offset_new = R_NilValue,
    int threads = 0) {

  ThreadGuard guard(threads);
  RFFModel model = list_to_rff_model(model_state);

  const int N_new = x_new.nrow();
  NumericVector off_holder;
  const double* off_ptr = resolve_offset_ptr(
      offset_new, off_holder, N_new, model.out_dim, "offset_new");

  RFFWorkspace ws;
  ws.init(model.d_in, model.p_cols, model.k_freq, model.m_landmarks, model.out_dim, false);

  Eigen::MatrixXd preds = model.predict_raw(x_new.begin(), N_new, x_new.ncol(), off_ptr, ws);
  return to_rcpp(preds);
}
