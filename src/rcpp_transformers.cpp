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

//' Fast C++ Yeo-Johnson Power Transformation Fit
//' @keywords internal
// [[Rcpp::export]]
List rcpp_fit_yeo_johnson(NumericVector x) {
  double lambda = 1.0, mean = 0.0, sd = 1.0;
  evofe::fit_yeo_johnson_impl(x.begin(), x.size(), lambda, mean, sd);
  return List::create(
    Named("lambda") = lambda,
    Named("mean") = mean,
    Named("sd") = sd
  );
}

//' Fast C++ Yeo-Johnson Power Transformation Apply
//' @keywords internal
// [[Rcpp::export]]
NumericVector rcpp_apply_yeo_johnson(NumericVector x, double lambda, double mean, double sd, int threads = 1) {
  int n = x.size();
  if (n == 0) return NumericVector(0);
  NumericVector out(n);
  evofe::apply_yeo_johnson_impl(x.begin(), n, lambda, mean, sd, out.begin(), threads);
  return out;
}

//' Fast C++ Shifted Box-Cox Power Transformation Fit
//' @keywords internal
// [[Rcpp::export]]
List rcpp_fit_box_cox(NumericVector x) {
  double lambda = 1.0, shift = 0.0, mean = 0.0, sd = 1.0;
  evofe::fit_box_cox_impl(x.begin(), x.size(), lambda, shift, mean, sd);
  return List::create(
    Named("lambda") = lambda,
    Named("shift") = shift,
    Named("mean") = mean,
    Named("sd") = sd
  );
}

//' Fast C++ Shifted Box-Cox Power Transformation Apply
//' @keywords internal
// [[Rcpp::export]]
NumericVector rcpp_apply_box_cox(NumericVector x, double lambda, double shift, double mean, double sd, int threads = 1) {
  int n = x.size();
  if (n == 0) return NumericVector(0);
  NumericVector out(n);
  evofe::apply_box_cox_impl(x.begin(), n, lambda, shift, mean, sd, out.begin(), threads);
  return out;
}

//' Fast C++ Single-Pass Numeric Vector Sanitizer (Zero-Copy When Already Clean)
//' @keywords internal
// [[Rcpp::export]]
SEXP rcpp_sanitize_numeric_vec(SEXP x) {
  if (TYPEOF(x) != REALSXP) return x;
  R_xlen_t n = Rf_xlength(x);
  if (n == 0) return x;
  const double MAX_FLOAT = 3.402823e38;
  double* ptr = REAL(x);

  // Fast scan to first non-finite (non-NA) or out-of-range value
  R_xlen_t first_bad = -1;
  for (R_xlen_t i = 0; i < n; ++i) {
    double v = ptr[i];
    if (!R_IsNA(v) && (!std::isfinite(v) || std::abs(v) > MAX_FLOAT)) {
      first_bad = i;
      break;
    }
  }
  if (first_bad < 0) {
    return x; // Already 100% clean: zero allocation, zero copy
  }

  SEXP out = MAYBE_REFERENCED(x) ? PROTECT(Rf_duplicate(x)) : x;
  double* out_ptr = REAL(out);
  for (R_xlen_t i = first_bad; i < n; ++i) {
    double v = out_ptr[i];
    if (!std::isfinite(v) || std::abs(v) > MAX_FLOAT) {
      out_ptr[i] = NA_REAL;
    }
  }
  if (out != x) UNPROTECT(1);
  return out;
}

//' Fast C++ Single-Pass Feature Matrix Converter & Sanitizer
//' @keywords internal
// [[Rcpp::export]]
SEXP rcpp_sanitize_feature_matrix(SEXP x, int threads = 1) {
  if (Rf_isNull(x)) return R_NilValue;
  const double MAX_FLOAT = 3.402823e38;
  int n_threads = (threads > 0) ? threads : 1;
#if defined(_OPENMP)
  n_threads = std::min(n_threads, omp_get_max_threads());
#endif

  // Case 1: Input is already a matrix (double, integer, or logical)
  if (Rf_isMatrix(x)) {
    int type = TYPEOF(x);
    if (type != REALSXP && type != INTSXP && type != LGLSXP) {
      return R_NilValue; // Fallback to R for character/complex matrices
    }
    SEXP dims = Rf_getAttrib(x, R_DimSymbol);
    int n = INTEGER(dims)[0];
    int p = INTEGER(dims)[1];
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(p);

    NumericMatrix out(no_init(n, p));
    double* dst = out.begin();

    if (type == REALSXP) {
      const double* src = REAL(x);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (total >= 16384)
#endif
      for (size_t i = 0; i < total; ++i) {
        double v = src[i];
        dst[i] = (std::isfinite(v) && std::abs(v) <= MAX_FLOAT) ? v : NA_REAL;
      }
    } else {
      const int* src = INTEGER(x);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (total >= 16384)
#endif
      for (size_t i = 0; i < total; ++i) {
        int v = src[i];
        dst[i] = (v == NA_INTEGER) ? NA_REAL : static_cast<double>(v);
      }
    }

    SEXP dn = Rf_getAttrib(x, R_DimNamesSymbol);
    if (!Rf_isNull(dn)) {
      Rf_setAttrib(out, R_DimNamesSymbol, dn);
    }
    Rf_setAttrib(out, Rf_install("sanitized"), Rf_ScalarLogical(1));
    return out;
  }

  // Case 2: Input is a data.frame / data.table (VECSXP)
  if (TYPEOF(x) == VECSXP) {
    int p = Rf_length(x);
    int n = 0;
    if (p > 0) {
      n = Rf_length(VECTOR_ELT(x, 0));
    } else {
      SEXP rn = Rf_getAttrib(x, R_RowNamesSymbol);
      if (!Rf_isNull(rn)) {
        n = (Rf_length(rn) == 2 && TYPEOF(rn) == INTSXP && INTEGER(rn)[0] == NA_INTEGER)
              ? std::abs(INTEGER(rn)[1])
              : Rf_length(rn);
      }
    }

    std::vector<const double*> dbl_ptrs(p, nullptr);
    std::vector<const int*> int_ptrs(p, nullptr);
    for (int j = 0; j < p; ++j) {
      SEXP col = VECTOR_ELT(x, j);
      if (Rf_length(col) != n || Rf_isObject(col) && !Rf_isFactor(col) && !Rf_inherits(col, "POSIXct") && !Rf_inherits(col, "Date")) {
        // Fallback to R data.matrix if column is a custom S3/S4 object other than factor/Date/POSIXct
        return R_NilValue;
      }
      int ct = TYPEOF(col);
      if (ct == REALSXP) {
        dbl_ptrs[j] = REAL(col);
      } else if (ct == INTSXP || ct == LGLSXP) {
        int_ptrs[j] = INTEGER(col);
      } else {
        return R_NilValue; // Character or list column: fallback to R data.matrix
      }
    }

    NumericMatrix out(no_init(n, p));
    double* dst_base = out.begin();
    size_t total = static_cast<size_t>(n) * static_cast<size_t>(p);

#if defined(_OPENMP)
#pragma omp parallel for num_threads(n_threads) schedule(static) if (total >= 16384 && p >= 2)
#endif
    for (int j = 0; j < p; ++j) {
      double* col_dst = dst_base + static_cast<size_t>(j) * static_cast<size_t>(n);
      if (dbl_ptrs[j] != nullptr) {
        const double* src = dbl_ptrs[j];
        for (int i = 0; i < n; ++i) {
          double v = src[i];
          col_dst[i] = (std::isfinite(v) && std::abs(v) <= MAX_FLOAT) ? v : NA_REAL;
        }
      } else {
        const int* src = int_ptrs[j];
        for (int i = 0; i < n; ++i) {
          int v = src[i];
          col_dst[i] = (v == NA_INTEGER) ? NA_REAL : static_cast<double>(v);
        }
      }
    }

    SEXP col_names = Rf_getAttrib(x, R_NamesSymbol);
    if (!Rf_isNull(col_names)) {
      SEXP dn = PROTECT(Rf_allocVector(VECSXP, 2));
      SET_VECTOR_ELT(dn, 0, R_NilValue);
      SET_VECTOR_ELT(dn, 1, col_names);
      Rf_setAttrib(out, R_DimNamesSymbol, dn);
      UNPROTECT(1);
    }
    Rf_setAttrib(out, Rf_install("sanitized"), Rf_ScalarLogical(1));
    return out;
  }

  return R_NilValue;
}

namespace {

inline double compute_pair_abs_cor_dbl(const double* nc, const double* ev, int n, int step) {
  double sum_x = 0.0, sum_y = 0.0;
  int cnt = 0;
  for (int i = 0; i < n; i += step) {
    double x = nc[i];
    double y = ev[i];
    if (std::isfinite(x) && std::isfinite(y)) {
      sum_x += x;
      sum_y += y;
      cnt++;
    }
  }
  if (cnt <= 2) return 0.0;
  double mx = sum_x / cnt;
  double my = sum_y / cnt;
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  for (int i = 0; i < n; i += step) {
    double x = nc[i];
    double y = ev[i];
    if (std::isfinite(x) && std::isfinite(y)) {
      double dx = x - mx;
      double dy = y - my;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }
  }
  if (sxx <= 0.0 || syy <= 0.0) return 0.0;
  return std::abs(sxy) / std::sqrt(sxx * syy);
}

inline double compute_pair_abs_cor_int(const double* nc, const int* ev, int n, int step) {
  double sum_x = 0.0, sum_y = 0.0;
  int cnt = 0;
  for (int i = 0; i < n; i += step) {
    double x = nc[i];
    int yi = ev[i];
    if (std::isfinite(x) && yi != NA_INTEGER) {
      sum_x += x;
      sum_y += static_cast<double>(yi);
      cnt++;
    }
  }
  if (cnt <= 2) return 0.0;
  double mx = sum_x / cnt;
  double my = sum_y / cnt;
  double sxx = 0.0, syy = 0.0, sxy = 0.0;
  for (int i = 0; i < n; i += step) {
    double x = nc[i];
    int yi = ev[i];
    if (std::isfinite(x) && yi != NA_INTEGER) {
      double dx = x - mx;
      double dy = static_cast<double>(yi) - my;
      sxx += dx * dx;
      syy += dy * dy;
      sxy += dx * dy;
    }
  }
  if (sxx <= 0.0 || syy <= 0.0) return 0.0;
  return std::abs(sxy) / std::sqrt(sxx * syy);
}

} // anonymous namespace

//' Fast C++ Gene Constant & Redundancy Validator
//' Returns: 0 = valid, 1 = constant column, 2 = redundant column (cor >= threshold)
//' @keywords internal
// [[Rcpp::export]]
int rcpp_validate_numeric_gene(NumericVector new_col, SEXP train_data, CharacterVector exclude_cols,
                               double cor_threshold = 0.99, int max_cor_rows = 8192) {
  int n = new_col.size();
  if (n <= 1) return 1;
  const double* nc = new_col.begin();

  // 1. Single-pass constant & finite variance check
  double first_val = NA_REAL;
  bool has_first = false;
  bool has_distinct = false;
  int n_finite = 0;

  for (int i = 0; i < n; ++i) {
    double v = nc[i];
    if (std::isfinite(v)) {
      n_finite++;
      if (!has_first) {
        first_val = v;
        has_first = true;
      } else if (!has_distinct && v != first_val) {
        has_distinct = true;
      }
    }
  }

  if (!has_distinct) return 1; // Constant column (0 or 1 unique non-NA value)
  if (n_finite <= 2 || cor_threshold >= 1.0 || TYPEOF(train_data) != VECSXP) return 0;

  // Check if SD > 0 on finite values
  double sum_nc = 0.0;
  for (int i = 0; i < n; ++i) {
    if (std::isfinite(nc[i])) sum_nc += nc[i];
  }
  double mean_nc = sum_nc / n_finite;
  double var_nc = 0.0;
  for (int i = 0; i < n; ++i) {
    if (std::isfinite(nc[i])) {
      double d = nc[i] - mean_nc;
      var_nc += d * d;
    }
  }
  if (var_nc <= 0.0 || !std::isfinite(var_nc)) return 0;

  // 2. Pairwise correlation check against existing numeric columns in train_data
  SEXP col_names = Rf_getAttrib(train_data, R_NamesSymbol);
  int n_cols = Rf_length(train_data);
  int n_excl = exclude_cols.size();
  int step = (max_cor_rows > 0 && n > max_cor_rows) ? (n / max_cor_rows) : 1;

  for (int j = 0; j < n_cols; ++j) {
    if (!Rf_isNull(col_names)) {
      const char* nm = CHAR(STRING_ELT(col_names, j));
      bool excluded = false;
      for (int e = 0; e < n_excl; ++e) {
        if (std::strcmp(nm, CHAR(exclude_cols[e])) == 0) {
          excluded = true;
          break;
        }
      }
      if (excluded) continue;
    }

    SEXP col = VECTOR_ELT(train_data, j);
    if (Rf_length(col) != n || Rf_isFactor(col)) continue;
    int ct = TYPEOF(col);
    if (ct == REALSXP) {
      const double* ev = REAL(col);
      double r = compute_pair_abs_cor_dbl(nc, ev, n, step);
      if (step > 1 && r >= cor_threshold - 0.02) {
        r = compute_pair_abs_cor_dbl(nc, ev, n, 1);
      }
      if (std::isfinite(r) && r >= cor_threshold) return 2;
    } else if (ct == INTSXP) {
      const int* ev = INTEGER(col);
      double r = compute_pair_abs_cor_int(nc, ev, n, step);
      if (step > 1 && r >= cor_threshold - 0.02) {
        r = compute_pair_abs_cor_int(nc, ev, n, 1);
      }
      if (std::isfinite(r) && r >= cor_threshold) return 2;
    }
  }

  return 0;
}

