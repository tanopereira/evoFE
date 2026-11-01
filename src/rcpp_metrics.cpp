#include <Rcpp.h>
#include "metrics_calibration.h"

using namespace Rcpp;

//' Fast C++ Temperature-Scaled Refinement for Binary Classification
//' @keywords internal
// [[Rcpp::export]]
double rcpp_compute_ts_refinement_binary(NumericVector y_true, NumericVector y_pred,
                                         double alpha = 1.0, bool is_logits = false,
                                         int threads = 1) {
  if (y_pred.size() != y_true.size()) Rcpp::stop("y_pred and y_true must have the same length.");
  int n = y_true.size();
  if (n == 0) return 0.0;
  return evofe::compute_ts_refinement_binary_impl(
      y_true.begin(), y_pred.begin(), n, alpha, is_logits, threads);
}

//' Fast C++ Temperature-Scaled Refinement for Multiclass Classification
//' @keywords internal
// [[Rcpp::export]]
double rcpp_compute_ts_refinement_multiclass(IntegerVector y_true, NumericMatrix y_pred,
                                             int num_class, double alpha = 1.0,
                                             bool is_logits = false, int threads = 1) {
  if (num_class <= 0) Rcpp::stop("num_class must be positive.");
  if (y_pred.nrow() != y_true.size() || y_pred.ncol() < num_class) Rcpp::stop("y_pred dimensions must match y_true length and num_class.");
  int n = y_true.size();
  if (n == 0) return 0.0;
  return evofe::compute_ts_refinement_multiclass_impl(
      y_true.begin(), y_pred.begin(), n, num_class, alpha, is_logits, threads);
}

//' Fast C++ Calibrated RMSE
//' @keywords internal
// [[Rcpp::export]]
double rcpp_compute_calibrated_rmse(NumericVector y_true, NumericVector y_pred) {
  if (y_pred.size() != y_true.size()) Rcpp::stop("y_pred and y_true must have the same length.");
  int n = y_true.size();
  if (n <= 1) return 0.0;
  return evofe::compute_calibrated_rmse_impl(y_true.begin(), y_pred.begin(), n);
}

//' Fast C++ Calibrated MAE
//' @keywords internal
// [[Rcpp::export]]
double rcpp_compute_calibrated_mae(NumericVector y_true, NumericVector y_pred) {
  if (y_pred.size() != y_true.size()) Rcpp::stop("y_pred and y_true must have the same length.");
  int n = y_true.size();
  if (n <= 1) return 0.0;
  return evofe::compute_calibrated_mae_impl(y_true.begin(), y_pred.begin(), n);
}

//' Fast C++ Post-Hoc Calibrator Parameter Fitting
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
List rcpp_fit_post_calibrator(SEXP y_true, SEXP y_pred, std::string task, std::string metric,
                              int num_class = 0, double alpha = 1.0, bool is_logits = false,
                              int threads = 1) {
  if (metric == "ts_refinement" && task == "classification") {
    NumericVector yt(y_true);
    NumericVector yp(y_pred);
    int n = yt.size();
    double beta = 1.0;
    if (n > 0 && yp.size() == n) {
      evofe::compute_ts_refinement_binary_impl(yt.begin(), yp.begin(), n, alpha, is_logits, threads, &beta);
    }
    return List::create(_["type"] = "ts_binary", _["beta"] = beta);
  } else if (metric == "ts_refinement" && task == "multiclass") {
    IntegerVector yt(y_true);
    NumericMatrix yp(y_pred);
    int n = yt.size();
    double beta = 1.0;
    if (n > 0 && num_class > 0 && yp.nrow() == n && yp.ncol() >= num_class) {
      evofe::compute_ts_refinement_multiclass_impl(yt.begin(), yp.begin(), n, num_class, alpha, is_logits, threads, &beta);
    }
    return List::create(_["type"] = "ts_multiclass", _["beta"] = beta);
  } else if (metric == "cal_rmse" && task == "regression") {
    NumericVector yt(y_true);
    NumericVector yp(y_pred);
    int n = yt.size();
    double a = 0.0, b = 1.0;
    if (n > 1 && yp.size() == n) {
      evofe::compute_calibrated_rmse_impl(yt.begin(), yp.begin(), n, &a, &b);
    }
    return List::create(_["type"] = "linear", _["a"] = a, _["b"] = b);
  } else if (metric == "cal_mae" && task == "regression") {
    NumericVector yt(y_true);
    NumericVector yp(y_pred);
    int n = yt.size();
    double a = 0.0, b = 1.0;
    if (n > 1 && yp.size() == n) {
      evofe::compute_calibrated_mae_impl(yt.begin(), yp.begin(), n, &a, &b);
    }
    return List::create(_["type"] = "linear", _["a"] = a, _["b"] = b);
  }
  return R_NilValue;
}

