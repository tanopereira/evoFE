#include <Rcpp.h>
#include "metrics_calibration.h"

using namespace Rcpp;

//' Fast C++ Temperature-Scaled Refinement for Binary Classification
//' @keywords internal
// [[Rcpp::export]]
double rcpp_compute_ts_refinement_binary(NumericVector y_true, NumericVector y_pred,
                                         double alpha = 1.0, bool is_logits = false,
                                         int threads = 1) {
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
  int n = y_true.size();
  if (n == 0 || num_class <= 0) return 0.0;
  return evofe::compute_ts_refinement_multiclass_impl(
      y_true.begin(), y_pred.begin(), n, num_class, alpha, is_logits, threads);
}
