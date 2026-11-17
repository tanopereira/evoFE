#ifndef EVOFE_RCPP_NN_BRIDGE_H
#define EVOFE_RCPP_NN_BRIDGE_H

#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#include "nn_common.h"
#include "metrics_calibration.h"

namespace evofe {
namespace nn {

// Dynamic OpenBLAS / Apple Accelerate thread control
static inline int get_openblas_threads() {
#if !defined(_WIN32)
  typedef int (*openblas_get_threads_fn)();
  openblas_get_threads_fn fn = (openblas_get_threads_fn)dlsym(RTLD_DEFAULT, "openblas_get_num_threads");
  if (fn) {
    return fn();
  }
#endif
  return -1;
}

static inline void set_openblas_threads(int num_threads) {
#if !defined(_WIN32)
  if (num_threads <= 0) return;
  typedef void (*openblas_set_threads_fn)(int);
  openblas_set_threads_fn fn = (openblas_set_threads_fn)dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
  if (fn) {
    fn(num_threads);
  }
#endif
}

static inline int get_accelerate_threading() {
#if defined(__APPLE__)
  typedef int (*blas_get_threading_fn)();
  blas_get_threading_fn fn = (blas_get_threading_fn)dlsym(RTLD_DEFAULT, "BLASGetThreading");
  if (fn) {
    return fn();
  }
#endif
  return -1;
}

static inline void set_accelerate_threading(int mode) {
#if defined(__APPLE__)
  if (mode < 0) return;
  typedef int (*blas_set_threading_fn)(int);
  blas_set_threading_fn fn = (blas_set_threading_fn)dlsym(RTLD_DEFAULT, "BLASSetThreading");
  if (fn) {
    fn(mode);
  }
#endif
}

// RAII guard for BLAS thread counts (OpenBLAS and Apple Accelerate).
// blas_threads <= 0 leaves BLAS threading unchanged; 1 forces single-threaded BLAS;
// > 1 enables multi-threaded BLAS.
struct BlasThreadGuard {
  int old_openblas = -1;
  int old_accelerate = -1;

  explicit BlasThreadGuard(int blas_threads) {
    if (blas_threads <= 0) return;
    old_openblas = get_openblas_threads();
    set_openblas_threads(blas_threads);

    old_accelerate = get_accelerate_threading();
    // BLAS_THREADING_MULTI_THREADED = 0, BLAS_THREADING_SINGLE_THREADED = 1
    set_accelerate_threading((blas_threads == 1) ? 1 : 0);
  }

  ~BlasThreadGuard() {
    if (old_openblas > 0) {
      set_openblas_threads(old_openblas);
    }
    if (old_accelerate >= 0) {
      set_accelerate_threading(old_accelerate);
    }
  }

  BlasThreadGuard(const BlasThreadGuard&) = delete;
  BlasThreadGuard& operator=(const BlasThreadGuard&) = delete;
};

// Scoped RAII guard to force single-threaded BLAS (OpenBLAS & Apple Accelerate)
// while leaving OpenMP / Eigen thread settings untouched.
struct BlasSingleThreadGuard : BlasThreadGuard {
  explicit BlasSingleThreadGuard(bool active = true)
    : BlasThreadGuard(active ? 1 : 0) {}
};

// RAII ThreadGuard to ensure CRAN-compliant thread restoration across OpenMP, Eigen, OpenBLAS, and Apple Accelerate
struct ThreadGuard {
  int old_omp = -1;
  int old_eigen = -1;
  BlasThreadGuard blas_guard;

  explicit ThreadGuard(int threads, bool blas_single_thread = false, bool eigen_single_thread = false)
    : blas_guard(blas_single_thread ? 1 : threads) {
    if (eigen_single_thread || threads > 0) {
      old_eigen = Eigen::nbThreads();
      Eigen::setNbThreads(eigen_single_thread ? 1 : threads);
    }
    if (threads <= 0) return;
#ifdef _OPENMP
    old_omp = omp_get_max_threads();
    omp_set_num_threads(threads);
#endif
  }

  ~ThreadGuard() {
#ifdef _OPENMP
    if (old_omp > 0) {
      omp_set_num_threads(old_omp);
    }
#endif
    if (old_eigen > 0) {
      Eigen::setNbThreads(old_eigen);
    }
  }

  ThreadGuard(const ThreadGuard&) = delete;
  ThreadGuard& operator=(const ThreadGuard&) = delete;
};

// Zero-copy Eigen::Map from R NumericMatrix (column-major compatible)
inline Eigen::Map<Eigen::MatrixXd> map_eigen(Rcpp::NumericMatrix& mat) {
  return Eigen::Map<Eigen::MatrixXd>(mat.begin(), mat.nrow(), mat.ncol());
}

// Direct copy without sanitization (for model weights — casts R double to FP32)
inline Eigen::MatrixXf copy_eigen_fast(const Rcpp::NumericMatrix& mat) {
  Eigen::Map<const Eigen::MatrixXd> mapped(mat.begin(), mat.nrow(), mat.ncol());
  return mapped.cast<float>();
}

// Helpers to convert Eigen::MatrixXd / Eigen::MatrixXf / Eigen::VectorXd to/from Rcpp types
inline Rcpp::NumericMatrix to_rcpp(const Eigen::MatrixXd& emat) {
  int r = static_cast<int>(emat.rows());
  int c = static_cast<int>(emat.cols());
  Rcpp::NumericMatrix mat(Rcpp::no_init(r, c));
  if (r > 0 && c > 0) {
    Eigen::Map<Eigen::MatrixXd>(mat.begin(), r, c) = emat;
  }
  return mat;
}

inline Rcpp::NumericMatrix to_rcpp(const Eigen::MatrixXf& emat) {
  int r = static_cast<int>(emat.rows());
  int c = static_cast<int>(emat.cols());
  Rcpp::NumericMatrix mat(Rcpp::no_init(r, c));
  if (r > 0 && c > 0) {
    Eigen::Map<Eigen::MatrixXd>(mat.begin(), r, c) = emat.cast<double>();
  }
  return mat;
}

inline Rcpp::NumericVector to_rcpp_vec(const Eigen::VectorXd& evec) {
  int n = static_cast<int>(evec.size());
  Rcpp::NumericVector vec(Rcpp::no_init(n));
  if (n > 0) {
    std::memcpy(vec.begin(), evec.data(), static_cast<size_t>(n) * sizeof(double));
  }
  return vec;
}

inline Eigen::VectorXd to_eigen_vec(const Rcpp::NumericVector& vec) {
  int n = vec.size();
  Eigen::VectorXd evec(n);
  if (n > 0) {
    std::memcpy(evec.data(), vec.begin(), static_cast<size_t>(n) * sizeof(double));
  }
  return evec;
}

struct MetricResult {
  double score;
  std::string name;
  bool higher_is_better;
};

inline double compute_qwk_concordance_cpp(
    double sum_err2, double sum_y, double sum_p,
    double sum_y2, double sum_p2, int N) {
  if (N <= 0) return 0.0;
  double invN = 1.0 / static_cast<double>(N);
  double num = sum_err2 * invN;
  double my = sum_y * invN;
  double mp = sum_p * invN;
  double den = sum_y2 * invN + sum_p2 * invN - 2.0 * my * mp;
  if (!std::isfinite(den) || den <= 1e-15) {
    return (std::isfinite(num) && num <= 1e-15) ? 1.0 : 0.0;
  }
  return 1.0 - num / den;
}

namespace detail {

inline int fill_binary_auc_pairs(
    const double* __restrict__ p0,
    const double* __restrict__ y_v_ptr,
    int N_val,
    std::vector<std::pair<double, int>>& pairs,
    int n_threads) {
  if (static_cast<int>(pairs.size()) < N_val) {
    pairs.resize(N_val);
  }
  int n_pos = 0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:n_pos) schedule(static) if (n_threads > 1 && N_val >= 65536)
#endif
  for (int i = 0; i < N_val; ++i) {
    int y = (y_v_ptr[i] > 0.0) ? 1 : 0;
    n_pos += y;
    double pv = p0[i];
    double p = std::isfinite(pv) ? pv : -1e300;
    pairs[i] = { p, y };
  }
  return n_pos;
}

inline long long count_multiclass_correct(
    const double* __restrict__ preds_ptr,
    size_t stride,
    const double* __restrict__ y_v_ptr,
    int N_val,
    int out_dim,
    int n_threads) {
  long long correct = 0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:correct) schedule(static) if (n_threads > 1 && N_val >= 65536)
#endif
  for (int i = 0; i < N_val; ++i) {
    int best_c = 0;
    double max_p = preds_ptr[i];
    for (int c = 1; c < out_dim; ++c) {
      double pc = preds_ptr[static_cast<size_t>(c) * stride + i];
      if (pc > max_p) {
        max_p = pc;
        best_c = c;
      }
    }
    if (best_c == static_cast<int>(y_v_ptr[i])) correct++;
  }
  return correct;
}

} // namespace detail

inline MetricResult compute_val_metric(
    const Eigen::MatrixXd& val_preds,
    const double* __restrict__ y_v_ptr,
    const std::string& task,
    const std::string& metric_req,
    int out_dim,
    std::vector<std::pair<double, int>>* auc_pairs_buf = nullptr,
    int threads = 1) {

  int N_val = static_cast<int>(val_preds.rows());
  if (N_val <= 0) return { 0.0, "Val empty", false };
  std::string m = metric_req;
  std::transform(m.begin(), m.end(), m.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif
  const double* __restrict__ p0 = val_preds.data();

  if (task == "regression") {
    if (m == "cal_rmse" || m == "cal-rmse") {
      double score = evofe::compute_calibrated_rmse_impl(y_v_ptr, p0, N_val);
      return { score, "Val Cal-RMSE", false };
    }
    if (m == "cal_mae" || m == "cal-mae") {
      double score = evofe::compute_calibrated_mae_impl(y_v_ptr, p0, N_val);
      return { score, "Val Cal-MAE", false };
    }
    if (m == "mae") {
      double sum_ae = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_ae) schedule(static) if (n_threads > 1 && N_val >= 131072)
#endif
      for (int i = 0; i < N_val; ++i) {
        sum_ae += std::abs(p0[i] - y_v_ptr[i]);
      }
      return { sum_ae / N_val, "Val MAE", false };
    }
    if (m == "rmsle" || m == "root_mean_squared_log_error") {
      double sum_sle = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_sle) schedule(static) if (n_threads > 1 && N_val >= 32768)
#endif
      for (int i = 0; i < N_val; ++i) {
        double yt = std::max(0.0, y_v_ptr[i]);
        double yp_raw = p0[i];
        double yp = std::isfinite(yp_raw) ? std::max(0.0, yp_raw) : 0.0;
        double d = std::log1p(yt) - std::log1p(yp);
        sum_sle += d * d;
      }
      return { std::sqrt(sum_sle / N_val), "Val RMSLE", false };
    }
    if (m == "r2" || m == "rsq" || m == "r_squared" || m == "r-squared") {
      double mean_y = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:mean_y) schedule(static) if (n_threads > 1 && N_val >= 131072)
#endif
      for (int i = 0; i < N_val; ++i) mean_y += y_v_ptr[i];
      mean_y /= N_val;
      double ss_res = 0.0, ss_tot = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:ss_tot, ss_res) schedule(static) if (n_threads > 1 && N_val >= 131072)
#endif
      for (int i = 0; i < N_val; ++i) {
        double dy = y_v_ptr[i] - mean_y;
        double dr = y_v_ptr[i] - p0[i];
        ss_tot += dy * dy;
        ss_res += dr * dr;
      }
      double r2 = (ss_tot > 1e-15) ? (1.0 - ss_res / ss_tot) : ((ss_res <= 1e-15) ? 1.0 : 0.0);
      return { r2, "Val R2", true };
    }
    if (m == "mape" || m == "mean_absolute_percentage_error") {
      double sum_ape = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_ape) schedule(static) if (n_threads > 1 && N_val >= 65536)
#endif
      for (int i = 0; i < N_val; ++i) {
        double denom = std::max(std::abs(y_v_ptr[i]), 1e-8);
        sum_ape += std::abs(y_v_ptr[i] - p0[i]) / denom;
      }
      return { sum_ape / N_val, "Val MAPE", false };
    }
    if (m == "smape" || m == "symmetric_mape") {
      double sum_smape = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_smape) schedule(static) if (n_threads > 1 && N_val >= 65536)
#endif
      for (int i = 0; i < N_val; ++i) {
        double yt = y_v_ptr[i];
        double yp = p0[i];
        double den = std::abs(yt) + std::abs(yp);
        if (den > 1e-12) {
          sum_smape += (2.0 * std::abs(yt - yp)) / den;
        }
      }
      return { sum_smape / N_val, "Val SMAPE", false };
    }
    if (m == "qwk" || m == "quadratic_weighted_kappa" || m == "kappa") {
      double min_y = std::round(y_v_ptr[0]);
      double max_y = min_y;
#if defined(_OPENMP) && _OPENMP >= 201107
#pragma omp parallel for num_threads(n_threads) reduction(min:min_y) reduction(max:max_y) schedule(static) if (n_threads > 1 && N_val >= 65536)
#endif
      for (int i = 1; i < N_val; ++i) {
        double yr = std::round(y_v_ptr[i]);
        if (yr < min_y) min_y = yr;
        if (yr > max_y) max_y = yr;
      }
      double sum_err2 = 0.0, sum_y = 0.0, sum_p = 0.0, sum_y2 = 0.0, sum_p2 = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_err2, sum_y, sum_p, sum_y2, sum_p2) schedule(static) if (n_threads > 1 && N_val >= 65536)
#endif
      for (int i = 0; i < N_val; ++i) {
        double yi = std::round(y_v_ptr[i]);
        double raw_p = std::isfinite(p0[i]) ? p0[i] : min_y;
        double pi = std::clamp(std::round(raw_p), min_y, max_y);
        double d = yi - pi;
        sum_err2 += d * d;
        sum_y += yi;
        sum_p += pi;
        sum_y2 += yi * yi;
        sum_p2 += pi * pi;
      }
      double qwk = compute_qwk_concordance_cpp(sum_err2, sum_y, sum_p, sum_y2, sum_p2, N_val);
      return { qwk, "Val QWK", true };
    }
    double sum_se = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_se) schedule(static) if (n_threads > 1 && N_val >= 131072)
#endif
    for (int i = 0; i < N_val; ++i) {
      double diff = p0[i] - y_v_ptr[i];
      sum_se += diff * diff;
    }
    return { std::sqrt(sum_se / N_val), "Val RMSE", false };
  }

  if (task == "classification") {
    if (m == "accuracy" || m == "acc" || m == "error" || m == "err") {
      long long correct = 0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:correct) schedule(static) if (n_threads > 1 && N_val >= 131072)
#endif
      for (int i = 0; i < N_val; ++i) {
        int pred_class = (p0[i] >= 0.5) ? 1 : 0;
        int true_class = (y_v_ptr[i] > 0.0) ? 1 : 0;
        correct += (pred_class == true_class);
      }
      if (m == "error" || m == "err") {
        return { static_cast<double>(N_val - correct) / N_val, "Val error", false };
      }
      return { static_cast<double>(correct) / N_val, "Val accuracy", true };
    }
    if (m == "f1" || m == "mcc" || m == "balanced_accuracy" || m == "balanced-accuracy" ||
        m == "bal_acc" || m == "bal-acc" || m == "qwk" || m == "quadratic_weighted_kappa" || m == "kappa") {
      long long tp_i = 0, tn_i = 0, fp_i = 0, fn_i = 0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:tp_i, tn_i, fp_i, fn_i) schedule(static) if (n_threads > 1 && N_val >= 131072)
#endif
      for (int i = 0; i < N_val; ++i) {
        int p = (p0[i] >= 0.5) ? 1 : 0;
        int y = (y_v_ptr[i] > 0.0) ? 1 : 0;
        tp_i += (y & p);
        tn_i += ((1 ^ y) & (1 ^ p));
        fp_i += ((1 ^ y) & p);
        fn_i += (y & (1 ^ p));
      }
      double tp = static_cast<double>(tp_i);
      double tn = static_cast<double>(tn_i);
      double fp = static_cast<double>(fp_i);
      double fn = static_cast<double>(fn_i);
      if (m == "f1") {
        double den = 2.0 * tp + fp + fn;
        return { (den > 0.0) ? (2.0 * tp / den) : 0.0, "Val F1", true };
      }
      if (m == "mcc") {
        double den = std::sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
        return { (std::isfinite(den) && den > 0.0) ? ((tp * tn - fp * fn) / den) : 0.0, "Val MCC", true };
      }
      if (m == "qwk" || m == "quadratic_weighted_kappa" || m == "kappa") {
        double sum_err2 = fp + fn;
        double sum_y = tp + fn;
        double sum_p = tp + fp;
        double qwk = compute_qwk_concordance_cpp(sum_err2, sum_y, sum_p, sum_y, sum_p, N_val);
        return { qwk, "Val QWK", true };
      }
      double tpr = (tp + fn > 0.0) ? (tp / (tp + fn)) : 0.0;
      double tnr = (tn + fp > 0.0) ? (tn / (tn + fp)) : 0.0;
      return { 0.5 * (tpr + tnr), "Val BalAcc", true };
    }
    if (m == "aucpr" || m == "pr_auc" || m == "pr-auc" || m == "prauc" || m == "average_precision" || m == "ap") {
      std::vector<std::pair<double, int>> local_pairs;
      std::vector<std::pair<double, int>>& pairs =
          (auc_pairs_buf != nullptr) ? *auc_pairs_buf : local_pairs;
      int n_pos = detail::fill_binary_auc_pairs(p0, y_v_ptr, N_val, pairs, n_threads);
      if (n_pos == 0) return { 0.0, "Val PR-AUC", true };
      if (n_pos == N_val) return { 1.0, "Val PR-AUC", true };
      std::sort(pairs.begin(), pairs.begin() + N_val, [](const auto& a, const auto& b) {
        return a.first > b.first;
      });
      double ap_sum = 0.0;
      double cum_tp = 0.0;
      size_t i = 0;
      while (i < static_cast<size_t>(N_val)) {
        size_t j = i + 1;
        while (j < static_cast<size_t>(N_val) && std::abs(pairs[j].first - pairs[i].first) <= 1e-12) {
          j++;
        }
        double delta_tp = 0.0;
        for (size_t k = i; k < j; ++k) {
          if (pairs[k].second == 1) delta_tp += 1.0;
        }
        cum_tp += delta_tp;
        if (delta_tp > 0.0) {
          ap_sum += delta_tp * (cum_tp / static_cast<double>(j));
        }
        i = j;
      }
      return { ap_sum / static_cast<double>(n_pos), "Val PR-AUC", true };
    }
    if (m == "auc") {
      std::vector<std::pair<double, int>> local_pairs;
      std::vector<std::pair<double, int>>& pairs =
          (auc_pairs_buf != nullptr) ? *auc_pairs_buf : local_pairs;
      int n_pos = detail::fill_binary_auc_pairs(p0, y_v_ptr, N_val, pairs, n_threads);
      int n_neg = N_val - n_pos;
      if (n_pos == 0 || n_neg == 0) return { 0.5, "Val AUC", true };
      std::sort(pairs.begin(), pairs.begin() + N_val, [](const auto& a, const auto& b) {
        return a.first < b.first;
      });
      double pos_rank_sum = 0.0;
      size_t i = 0;
      while (i < static_cast<size_t>(N_val)) {
        size_t j = i + 1;
        while (j < static_cast<size_t>(N_val) && std::abs(pairs[j].first - pairs[i].first) <= 1e-12) {
          j++;
        }
        double avg_rank = 0.5 * ((i + 1) + j);
        int pos_in_group = 0;
        for (size_t k = i; k < j; ++k) {
          if (pairs[k].second == 1) pos_in_group++;
        }
        pos_rank_sum += avg_rank * pos_in_group;
        i = j;
      }
      double u = pos_rank_sum - (static_cast<double>(n_pos) * (n_pos + 1)) / 2.0;
      double auc = u / (static_cast<double>(n_pos) * n_neg);
      return { auc, "Val AUC", true };
    }
    if (m == "ts_refinement" || m == "eval-ts-refinement" || m == "ts-refinement" || m == "eval_ts_refinement") {
      double score = evofe::compute_ts_refinement_binary_impl(y_v_ptr, p0, N_val, 1.0, false, threads);
      return { score, "Val TS-Refinement", false };
    }
    // Default for binary classification: logloss (1 log call per row)
    double ll_sum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:ll_sum) schedule(static) if (n_threads > 1 && N_val >= 32768)
#endif
    for (int i = 0; i < N_val; ++i) {
      double p = std::clamp(p0[i], 1e-15, 1.0 - 1e-15);
      double p_true = (y_v_ptr[i] > 0.0) ? p : (1.0 - p);
      ll_sum -= std::log(p_true);
    }
    return { ll_sum / N_val, "Val logloss", false };
  }

  // task == "multiclass"
  const size_t stride = static_cast<size_t>(val_preds.outerStride());
  if (m == "accuracy" || m == "acc") {
    long long correct = detail::count_multiclass_correct(p0, stride, y_v_ptr, N_val, out_dim, n_threads);
    return { static_cast<double>(correct) / N_val, "Val accuracy", true };
  }
  if (m == "error" || m == "err") {
    long long correct = detail::count_multiclass_correct(p0, stride, y_v_ptr, N_val, out_dim, n_threads);
    return { static_cast<double>(N_val - correct) / N_val, "Val error", false };
  }
  if (m == "macro_f1" || m == "f1_macro" || m == "macro-f1" || m == "f1-macro") {
    int tp_s[64], fp_s[64], fn_s[64];
    std::vector<int> tp_v, fp_v, fn_v;
    int* tp_arr = tp_s;
    int* fp_arr = fp_s;
    int* fn_arr = fn_s;
    if (out_dim <= 64) {
      std::memset(tp_s, 0, sizeof(int) * out_dim);
      std::memset(fp_s, 0, sizeof(int) * out_dim);
      std::memset(fn_s, 0, sizeof(int) * out_dim);
    } else {
      tp_v.assign(out_dim, 0);
      fp_v.assign(out_dim, 0);
      fn_v.assign(out_dim, 0);
      tp_arr = tp_v.data();
      fp_arr = fp_v.data();
      fn_arr = fn_v.data();
    }
    if (n_threads > 1 && N_val >= 65536) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(n_threads)
      {
        std::vector<int> l_tp(out_dim, 0), l_fp(out_dim, 0), l_fn(out_dim, 0);
#pragma omp for schedule(static) nowait
        for (int i = 0; i < N_val; ++i) {
          int best_c = 0;
          double max_p = p0[i];
          for (int c = 1; c < out_dim; ++c) {
            double pc = p0[static_cast<size_t>(c) * stride + i];
            if (pc > max_p) {
              max_p = pc;
              best_c = c;
            }
          }
          int tc = std::clamp(static_cast<int>(y_v_ptr[i]), 0, out_dim - 1);
          if (best_c == tc) {
            l_tp[tc]++;
          } else {
            l_fp[best_c]++;
            l_fn[tc]++;
          }
        }
#pragma omp critical
        {
          for (int c = 0; c < out_dim; ++c) {
            tp_arr[c] += l_tp[c];
            fp_arr[c] += l_fp[c];
            fn_arr[c] += l_fn[c];
          }
        }
      }
#endif
    } else {
      for (int i = 0; i < N_val; ++i) {
        int best_c = 0;
        double max_p = p0[i];
        for (int c = 1; c < out_dim; ++c) {
          double pc = p0[static_cast<size_t>(c) * stride + i];
          if (pc > max_p) {
            max_p = pc;
            best_c = c;
          }
        }
        int tc = std::clamp(static_cast<int>(y_v_ptr[i]), 0, out_dim - 1);
        if (best_c == tc) {
          tp_arr[tc]++;
        } else {
          fp_arr[best_c]++;
          fn_arr[tc]++;
        }
      }
    }
    double f1_sum = 0.0;
    int active_c = 0;
    for (int c = 0; c < out_dim; ++c) {
      int den = 2 * tp_arr[c] + fp_arr[c] + fn_arr[c];
      if (den > 0) {
        f1_sum += (2.0 * tp_arr[c]) / static_cast<double>(den);
        active_c++;
      }
    }
    return { (active_c > 0) ? (f1_sum / active_c) : 1.0, "Val Macro-F1", true };
  }
  if (m == "qwk" || m == "quadratic_weighted_kappa" || m == "kappa") {
    double sum_err2 = 0.0, sum_y = 0.0, sum_p = 0.0, sum_y2 = 0.0, sum_p2 = 0.0;
    double max_c = static_cast<double>(std::max(1, out_dim - 1));
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:sum_err2, sum_y, sum_p, sum_y2, sum_p2) schedule(static) if (n_threads > 1 && N_val >= 32768)
#endif
    for (int i = 0; i < N_val; ++i) {
      double yi = std::clamp(std::round(y_v_ptr[i]), 0.0, max_c);
      double p_sum = 0.0, exp_c = 0.0;
      for (int c = 0; c < out_dim; ++c) {
        double raw_pc = p0[static_cast<size_t>(c) * stride + i];
        double pc = std::isfinite(raw_pc) ? std::max(0.0, raw_pc) : 0.0;
        p_sum += pc;
        exp_c += static_cast<double>(c) * pc;
      }
      double pi = (p_sum > 1e-12) ? std::clamp(std::round(exp_c / p_sum), 0.0, max_c) : 0.0;
      double d = yi - pi;
      sum_err2 += d * d;
      sum_y += yi;
      sum_p += pi;
      sum_y2 += yi * yi;
      sum_p2 += pi * pi;
    }
    double qwk = compute_qwk_concordance_cpp(sum_err2, sum_y, sum_p, sum_y2, sum_p2, N_val);
    return { qwk, "Val QWK", true };
  }
  if (m == "ts_refinement" || m == "eval-ts-refinement" || m == "ts-refinement" || m == "eval_ts_refinement") {
    double score = evofe::compute_ts_refinement_multiclass_impl(y_v_ptr, p0, N_val, out_dim, 1.0, false, threads);
    return { score, "Val TS-Refinement", false };
  }
  // Default for multiclass: multi-logloss
  double ll_sum = 0.0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) reduction(+:ll_sum) schedule(static) if (n_threads > 1 && N_val >= 32768)
#endif
  for (int i = 0; i < N_val; ++i) {
    int true_c = static_cast<int>(y_v_ptr[i]);
    double p = 1e-15;
    if (true_c >= 0 && true_c < out_dim) {
      p = std::clamp(p0[static_cast<size_t>(true_c) * stride + i], 1e-15, 1.0);
    }
    ll_sum -= std::log(p);
  }
  return { ll_sum / N_val, "Val logloss", false };
}

inline MetricResult compute_val_metric(
    const Eigen::MatrixXd& val_preds,
    const Rcpp::NumericVector& y_v_vec,
    const std::string& task,
    const std::string& metric_req,
    int out_dim,
    std::vector<std::pair<double, int>>* auc_pairs_buf = nullptr,
    int threads = 1) {
  return compute_val_metric(val_preds, y_v_vec.begin(), task, metric_req, out_dim, auc_pairs_buf, threads);
}

} // namespace nn
} // namespace evofe

#endif // EVOFE_RCPP_NN_BRIDGE_H
