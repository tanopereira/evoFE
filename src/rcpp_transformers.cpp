#include <Rcpp.h>
#include "transformers_fused.h"

using namespace Rcpp;

namespace {

inline std::vector<const double*> extract_pointers(List cols, int& n, int& k) {
  k = cols.size();
  if (k == 0) {
    n = 0;
    return {};
  }
  NumericVector col0 = cols[0];
  n = col0.size();
  std::vector<const double*> ptrs(k);
  for (int j = 0; j < k; ++j) {
    NumericVector c_j = cols[j];
    if (c_j.size() != n) {
      stop("All input columns must have identical length.");
    }
    ptrs[j] = c_j.begin();
  }
  return ptrs;
}

} // anonymous namespace

//' Fast C++ Row-wise Minimum
//' @keywords internal
// [[Rcpp::export]]
NumericVector rcpp_fused_row_min(List cols, int threads = 1) {
  int n = 0, k = 0;
  auto ptrs = extract_pointers(cols, n, k);
  if (n == 0 || k == 0) return NumericVector(0);
  NumericVector out(n);
  evofe::fused_row_min_impl(ptrs, k, n, out.begin(), threads);
  return out;
}

//' Fast C++ Row-wise Maximum
//' @keywords internal
// [[Rcpp::export]]
NumericVector rcpp_fused_row_max(List cols, int threads = 1) {
  int n = 0, k = 0;
  auto ptrs = extract_pointers(cols, n, k);
  if (n == 0 || k == 0) return NumericVector(0);
  NumericVector out(n);
  evofe::fused_row_max_impl(ptrs, k, n, out.begin(), threads);
  return out;
}

//' Fast C++ Geometric Mean
//' @keywords internal
// [[Rcpp::export]]
NumericVector rcpp_fused_geometric_mean(List cols, double eps = 1e-6, int threads = 1) {
  int n = 0, k = 0;
  auto ptrs = extract_pointers(cols, n, k);
  if (n == 0 || k == 0) return NumericVector(0);
  NumericVector out(n);
  evofe::fused_geometric_mean_impl(ptrs, k, n, eps, out.begin(), threads);
  return out;
}

//' Fast C++ Harmonic Mean
//' @keywords internal
// [[Rcpp::export]]
NumericVector rcpp_fused_harmonic_mean(List cols, double eps = 1e-6, int threads = 1) {
  int n = 0, k = 0;
  auto ptrs = extract_pointers(cols, n, k);
  if (n == 0 || k == 0) return NumericVector(0);
  NumericVector out(n);
  evofe::fused_harmonic_mean_impl(ptrs, k, n, eps, out.begin(), threads);
  return out;
}

//' Fast C++ Pythagorean Imbalance (AM - HM)
//' @keywords internal
// [[Rcpp::export]]
NumericVector rcpp_fused_pythagorean_imbalance(List cols, double eps = 1e-6, int threads = 1) {
  int n = 0, k = 0;
  auto ptrs = extract_pointers(cols, n, k);
  if (n == 0 || k == 0) return NumericVector(0);
  NumericVector out(n);
  evofe::fused_pythagorean_imbalance_impl(ptrs, k, n, eps, out.begin(), threads);
  return out;
}

//' Fast C++ Relative Rating (Primary vs Mean of Others)
//' @keywords internal
// [[Rcpp::export]]
NumericVector rcpp_fused_relative_rating(List cols, int threads = 1) {
  int n = 0, k = 0;
  auto ptrs = extract_pointers(cols, n, k);
  if (n == 0 || k == 0) return NumericVector(0);
  NumericVector out(n);
  evofe::fused_relative_rating_impl(ptrs, k, n, out.begin(), threads);
  return out;
}
