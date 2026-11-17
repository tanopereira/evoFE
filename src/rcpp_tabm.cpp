#include <Rcpp.h>
#include <RcppEigen.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "rcpp_nn_bridge.h"
#include "tabm_core.h"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using evofe::nn::MetricResult;
using evofe::nn::ThreadGuard;
using evofe::nn::compute_val_metric;
using evofe::nn::copy_eigen_fast;
using evofe::nn::cosine_warmup_schedule;
using evofe::nn::to_eigen_vec;
using evofe::nn::to_rcpp;
using evofe::nn::to_rcpp_vec;
using evofe::tabm::PLEBinMatrix;
using evofe::tabm::PLEFeature;
using evofe::tabm::TabMInferWorkspace;
using evofe::tabm::TabMModel;
using evofe::tabm::TabMWorkspace;

namespace {

List tabm_model_to_list(const TabMModel& m) {
  int n_num = static_cast<int>(m.num_col_indices.size());
  IntegerVector num_cols_1based(no_init(n_num));
  NumericVector num_means_r(no_init(n_num));
  NumericVector num_stds_r(no_init(n_num));
  for (int j = 0; j < n_num; ++j) {
    num_cols_1based[j] = m.num_col_indices[j] + 1;
  }
  if (n_num > 0) {
    std::memcpy(num_means_r.begin(), m.num_means.data(), static_cast<size_t>(n_num) * sizeof(double));
    std::memcpy(num_stds_r.begin(), m.num_stds.data(), static_cast<size_t>(n_num) * sizeof(double));
  }

  int n_ple = static_cast<int>(m.ple_features.size());
  IntegerVector ple_bins(no_init(n_ple));
  IntegerVector ple_offsets(no_init(n_ple));
  size_t total_edges = 0, total_ple_W = 0, total_ple_b = 0;
  for (int j = 0; j < n_ple; ++j) {
    const PLEFeature& pf = m.ple_features[j];
    ple_bins[j] = pf.active ? pf.n_bins : 0;
    ple_offsets[j] = pf.out_col_offset;
    if (pf.active) {
      total_edges += pf.edges.size();
      total_ple_W += static_cast<size_t>(pf.W.val.size());
      total_ple_b += static_cast<size_t>(pf.b.val.size());
    }
  }

  NumericVector ple_edges_packed(no_init(total_edges));
  NumericVector ple_W_packed(no_init(total_ple_W));
  NumericVector ple_b_packed(no_init(total_ple_b));
  size_t off_e = 0, off_w = 0, off_b = 0;
  for (int j = 0; j < n_ple; ++j) {
    const PLEFeature& pf = m.ple_features[j];
    if (!pf.active) continue;
    size_t ne = pf.edges.size();
    Eigen::Map<Eigen::VectorXd>(ple_edges_packed.begin() + off_e, static_cast<Eigen::Index>(ne)) =
        Eigen::Map<const Eigen::VectorXf>(pf.edges.data(), static_cast<Eigen::Index>(ne)).cast<double>();
    off_e += ne;
    size_t nw = static_cast<size_t>(pf.W.val.size());
    Eigen::Map<Eigen::VectorXd>(ple_W_packed.begin() + off_w, static_cast<Eigen::Index>(nw)) =
        Eigen::Map<const Eigen::VectorXf>(pf.W.val.data(), static_cast<Eigen::Index>(nw)).cast<double>();
    off_w += nw;
    size_t nb = static_cast<size_t>(pf.b.val.size());
    Eigen::Map<Eigen::VectorXd>(ple_b_packed.begin() + off_b, static_cast<Eigen::Index>(nb)) =
        Eigen::Map<const Eigen::VectorXf>(pf.b.val.data(), static_cast<Eigen::Index>(nb)).cast<double>();
    off_b += nb;
  }

  int n_cat = static_cast<int>(m.cat_col_indices.size());
  IntegerVector cat_cols_1based(no_init(n_cat));
  IntegerVector cat_cards_r(no_init(n_cat));
  size_t total_cat = 0;
  for (int c = 0; c < n_cat; ++c) {
    cat_cols_1based[c] = m.cat_col_indices[c] + 1;
    cat_cards_r[c] = m.cat_cardinalities[c];
    total_cat += static_cast<size_t>(m.cat_tables[c].val.size());
  }
  NumericVector cat_tables_packed(no_init(total_cat));
  size_t off_c = 0;
  for (int c = 0; c < n_cat; ++c) {
    size_t nc = static_cast<size_t>(m.cat_tables[c].val.size());
    Eigen::Map<Eigen::VectorXd>(cat_tables_packed.begin() + off_c, static_cast<Eigen::Index>(nc)) =
        Eigen::Map<const Eigen::VectorXf>(m.cat_tables[c].val.data(), static_cast<Eigen::Index>(nc)).cast<double>();
    off_c += nc;
  }

  int n_blks = static_cast<int>(m.blocks.size());
  List blocks_W(n_blks);
  List blocks_b(n_blks);
  for (int l = 0; l < n_blks; ++l) {
    blocks_W[l] = to_rcpp(m.blocks[l].W.val);
    blocks_b[l] = to_rcpp(m.blocks[l].b.val);
  }

  return List::create(
    Named("task") = m.task,
    Named("num_emb") = m.num_emb,
    Named("n_features") = m.n_features,
    Named("output_dim") = m.output_dim,
    Named("k_ensemble") = m.k_ensemble,
    Named("d_block") = m.d_block,
    Named("n_blocks") = m.n_blocks,
    Named("d_embedding") = m.d_embedding,
    Named("n_bins") = m.n_bins,
    Named("dropout") = m.dropout,
    Named("y_mean") = m.y_mean,
    Named("y_std") = m.y_std,
    Named("D_num_rep") = m.D_num_rep,
    Named("D_cat_rep") = m.D_cat_rep,
    Named("D_in") = m.D_in,
    Named("has_dummy_in") = m.has_dummy_in,
    Named("num_cols") = num_cols_1based,
    Named("num_means") = num_means_r,
    Named("num_stds") = num_stds_r,
    Named("ple_bins") = ple_bins,
    Named("ple_offsets") = ple_offsets,
    Named("ple_edges_packed") = ple_edges_packed,
    Named("ple_W_packed") = ple_W_packed,
    Named("ple_b_packed") = ple_b_packed,
    Named("cat_cols") = cat_cols_1based,
    Named("cat_cards") = cat_cards_r,
    Named("cat_tables_packed") = cat_tables_packed,
    Named("R_in") = to_rcpp(m.R_in.val),
    Named("blocks_W") = blocks_W,
    Named("blocks_b") = blocks_b,
    Named("W_out") = to_rcpp(m.W_out.val),
    Named("b_out") = to_rcpp(m.b_out.val)
  );
}

TabMModel list_to_tabm_model(const List& lst) {
  TabMModel m;
  m.task = as<std::string>(lst["task"]);
  m.num_emb = as<std::string>(lst["num_emb"]);
  m.n_features = as<int>(lst["n_features"]);
  m.output_dim = as<int>(lst["output_dim"]);
  m.k_ensemble = as<int>(lst["k_ensemble"]);
  m.d_block = as<int>(lst["d_block"]);
  m.n_blocks = as<int>(lst["n_blocks"]);
  m.d_embedding = as<int>(lst["d_embedding"]);
  m.n_bins = as<int>(lst["n_bins"]);
  m.dropout = as<double>(lst["dropout"]);
  m.y_mean = as<double>(lst["y_mean"]);
  m.y_std = as<double>(lst["y_std"]);
  m.D_num_rep = as<int>(lst["D_num_rep"]);
  m.D_cat_rep = as<int>(lst["D_cat_rep"]);
  m.D_in = as<int>(lst["D_in"]);
  m.has_dummy_in = as<bool>(lst["has_dummy_in"]);

  IntegerVector num_cols_r = lst["num_cols"];
  NumericVector num_means_r = lst["num_means"];
  NumericVector num_stds_r = lst["num_stds"];
  int n_num = num_cols_r.size();
  m.num_col_indices.resize(n_num);
  m.num_means.resize(n_num);
  m.num_stds.resize(n_num);
  m.num_inv_stds.resize(n_num);
  if (n_num > 0) {
    std::memcpy(m.num_means.data(), num_means_r.begin(), static_cast<size_t>(n_num) * sizeof(double));
    std::memcpy(m.num_stds.data(), num_stds_r.begin(), static_cast<size_t>(n_num) * sizeof(double));
  }
  for (int j = 0; j < n_num; ++j) {
    m.num_col_indices[j] = num_cols_r[j] - 1;
    double s = m.num_stds[j];
    m.num_inv_stds[j] = (s < 1e-5 || !std::isfinite(s)) ? 1.0 : (1.0 / s);
  }

  IntegerVector ple_bins = lst["ple_bins"];
  IntegerVector ple_offsets = lst["ple_offsets"];
  NumericVector ple_edges_packed = lst["ple_edges_packed"];
  NumericVector ple_W_packed = lst["ple_W_packed"];
  NumericVector ple_b_packed = lst["ple_b_packed"];
  int n_ple = ple_bins.size();
  m.ple_features.resize(n_ple);
  size_t off_e = 0, off_w = 0, off_b = 0;
  for (int j = 0; j < n_ple; ++j) {
    int nb = ple_bins[j];
    if (nb > 0) {
      size_t ne = static_cast<size_t>(nb + 1);
      std::vector<float> fv(ne);
      Eigen::Map<Eigen::VectorXf>(fv.data(), static_cast<Eigen::Index>(ne)) =
          Eigen::Map<const Eigen::VectorXd>(ple_edges_packed.begin() + off_e, static_cast<Eigen::Index>(ne)).cast<float>();
      off_e += ne;

      size_t nw = static_cast<size_t>(nb) * static_cast<size_t>(m.d_embedding);
      Eigen::MatrixXf Wm(nb, m.d_embedding);
      Eigen::Map<Eigen::VectorXf>(Wm.data(), static_cast<Eigen::Index>(nw)) =
          Eigen::Map<const Eigen::VectorXd>(ple_W_packed.begin() + off_w, static_cast<Eigen::Index>(nw)).cast<float>();
      off_w += nw;

      size_t n_b = static_cast<size_t>(m.d_embedding);
      Eigen::MatrixXf bm(m.d_embedding, 1);
      Eigen::Map<Eigen::VectorXf>(bm.data(), static_cast<Eigen::Index>(n_b)) =
          Eigen::Map<const Eigen::VectorXd>(ple_b_packed.begin() + off_b, static_cast<Eigen::Index>(n_b)).cast<float>();
      off_b += n_b;

      m.ple_features[j].init_from_saved(fv, Wm, bm);
      m.ple_features[j].out_col_offset = ple_offsets[j];
    }
  }

  IntegerVector cat_cols_r = lst["cat_cols"];
  IntegerVector cat_cards_r = lst["cat_cards"];
  NumericVector cat_tables_packed = lst["cat_tables_packed"];
  int n_cat = cat_cols_r.size();
  m.cat_col_indices.resize(n_cat);
  m.cat_cardinalities.resize(n_cat);
  m.cat_tables.resize(n_cat);
  size_t off_c = 0;
  for (int c = 0; c < n_cat; ++c) {
    m.cat_col_indices[c] = cat_cols_r[c] - 1;
    int card = cat_cards_r[c];
    m.cat_cardinalities[c] = card;
    int n_rows = card + 1;
    m.cat_tables[c].val.resize(n_rows, m.d_embedding);
    size_t nc = static_cast<size_t>(n_rows) * static_cast<size_t>(m.d_embedding);
    Eigen::Map<Eigen::VectorXf>(m.cat_tables[c].val.data(), static_cast<Eigen::Index>(nc)) =
        Eigen::Map<const Eigen::VectorXd>(cat_tables_packed.begin() + off_c, static_cast<Eigen::Index>(nc)).cast<float>();
    off_c += nc;
  }

  NumericMatrix r_mat = lst["R_in"];
  m.R_in.val = copy_eigen_fast(r_mat);

  List blocks_W = lst["blocks_W"];
  List blocks_b = lst["blocks_b"];
  int n_blks = blocks_W.size();
  m.blocks.resize(n_blks);
  for (int l = 0; l < n_blks; ++l) {
    NumericMatrix Wm = blocks_W[l];
    NumericMatrix bm = blocks_b[l];
    m.blocks[l].W.val = copy_eigen_fast(Wm);
    m.blocks[l].b.val = copy_eigen_fast(bm);
    m.blocks[l].d_in = static_cast<int>(m.blocks[l].W.val.rows());
    m.blocks[l].d_out = static_cast<int>(m.blocks[l].W.val.cols());
  }

  NumericMatrix w_out_r = lst["W_out"];
  m.W_out.val = copy_eigen_fast(w_out_r);
  NumericMatrix b_out_r = lst["b_out"];
  m.b_out.val = copy_eigen_fast(b_out_r);

  return m;
}

void resolve_cat_schema_args(const NumericMatrix& x_train,
                             int D,
                             Nullable<IntegerVector> cat_cols,
                             Nullable<IntegerVector> cat_cardinalities,
                             std::vector<int>& out_cols_0based,
                             std::vector<int>& out_cards) {
  out_cols_0based.clear();
  out_cards.clear();

  if (cat_cols.isNotNull()) {
    IntegerVector cc(cat_cols);
    int n_cat = cc.size();
    out_cols_0based.reserve(n_cat);
    for (int i = 0; i < n_cat; ++i) {
      int idx1 = cc[i];
      if (idx1 == NA_INTEGER || idx1 < 1 || idx1 > D) {
        stop("cat_cols contains out-of-range or NA 1-based column index.");
      }
      out_cols_0based.push_back(idx1 - 1);
    }
    if (cat_cardinalities.isNotNull()) {
      IntegerVector cards(cat_cardinalities);
      if (cards.size() != n_cat) {
        stop("cat_cardinalities length must match cat_cols length.");
      }
      out_cards.reserve(n_cat);
      for (int i = 0; i < n_cat; ++i) {
        int cv = cards[i];
        out_cards.push_back((cv == NA_INTEGER || cv < 1) ? 1 : cv);
      }
    }
    return;
  }

  // Fallback: inspect cat_schema attribute if attached by rcpp_sanitize_feature_matrix
  SEXP cs_attr = x_train.attr("cat_schema");
  if (!Rf_isNull(cs_attr) && TYPEOF(cs_attr) == VECSXP) {
    List cs(cs_attr);
    if (cs.containsElementNamed("cols")) {
      IntegerVector cc = cs["cols"];
      int n_cat = cc.size();
      out_cols_0based.reserve(n_cat);
      for (int i = 0; i < n_cat; ++i) {
        int idx1 = cc[i];
        if (idx1 != NA_INTEGER && idx1 >= 1 && idx1 <= D) {
          out_cols_0based.push_back(idx1 - 1);
        }
      }
      if (cs.containsElementNamed("cardinalities") && static_cast<int>(out_cols_0based.size()) == n_cat) {
        IntegerVector cards = cs["cardinalities"];
        if (cards.size() == n_cat) {
          out_cards.reserve(n_cat);
          for (int i = 0; i < n_cat; ++i) {
            int cv = cards[i];
            out_cards.push_back((cv == NA_INTEGER || cv < 1) ? 1 : cv);
          }
        }
      }
    }
  }
}

} // anonymous namespace

//' Train TabM (TabM-mini BatchEnsemble MLP) Model in C++
//'
//' Native C++17 FP32 BatchEnsemble MLP with Quantile Piecewise Linear Encoding (PLE),
//' lookup categorical embeddings, decoupled AdamW optimization, dropout, and chunked inference.
//'
//' @param x_train Numeric feature matrix (N x D).
//' @param y_train Numeric target vector of length N.
//' @param x_val Optional numeric validation matrix.
//' @param y_val Optional numeric validation target vector.
//' @param task One of `"regression"`, `"classification"`, or `"multiclass"`.
//' @param n_epochs Maximum number of training epochs (default 100).
//' @param batch_size Minibatch size (`<= 0` selects data-adaptive default).
//' @param lr Base learning rate (default 0.002).
//' @param weight_decay Decoupled AdamW weight decay (default 1e-4).
//' @param early_stopping_rounds Patience for early stopping (`0` = disabled, default 16).
//' @param seed Random seed (default 42).
//' @param verbose Verbosity level (`0` = silent).
//' @param num_classes Number of classes for `"multiclass"`.
//' @param threads Number of threads for parallel execution (default 1).
//' @param metric Validation metric name (default `"default"`).
//' @param k_ensemble Number of BatchEnsemble members (default 16).
//' @param d_block Hidden layer width (default 128).
//' @param n_blocks Number of hidden MLP blocks (default 2).
//' @param dropout Dropout probability (default 0.1).
//' @param num_emb Numeric embedding scheme: `"pwl"` (Quantile PLE + Linear) or `"none"`.
//' @param n_bins Target number of quantile bins for `"pwl"` (default 48).
//' @param cat_cols Optional 1-based integer indices of categorical columns in `x_train`.
//' @param cat_cardinalities Optional integer cardinalities aligned with `cat_cols`.
//' @param d_embedding Embedding dimension per numeric/categorical feature (default 8).
//' @param lr_warmup_frac Linear warmup fraction for cosine schedule (default 0.05).
//' @param stop_epoch Optional epoch cap while preserving the full `n_epochs` cosine LR horizon (`0` = disabled).
//' @param compute_importances Logical: whether to compute feature importances via mean occlusion (default TRUE).
//' @return A List with `predictions`, `best_epoch`, `best_score`, `importances`, and `model_state`.
//' @keywords internal
// [[Rcpp::export]]
List rcpp_tabm_train(NumericMatrix x_train,
                     NumericVector y_train,
                     Nullable<NumericMatrix> x_val = R_NilValue,
                     Nullable<NumericVector> y_val = R_NilValue,
                     std::string task = "regression",
                     int n_epochs = 100,
                     int batch_size = -1,
                     double lr = 0.002,
                     double weight_decay = 1e-4,
                     int early_stopping_rounds = 16,
                     int seed = 42,
                     int verbose = 0,
                     int num_classes = 0,
                     int threads = 1,
                     std::string metric = "default",
                     int k_ensemble = 16,
                     int d_block = 128,
                     int n_blocks = 2,
                     double dropout = 0.1,
                     std::string num_emb = "pwl",
                     int n_bins = 48,
                     Nullable<IntegerVector> cat_cols = R_NilValue,
                     Nullable<IntegerVector> cat_cardinalities = R_NilValue,
                     int d_embedding = 8,
                     double lr_warmup_frac = 0.05,
                     int stop_epoch = 0,
                     bool compute_importances = true) {

  int N = x_train.nrow();
  int D = x_train.ncol();
  if (N <= 0 || D <= 0) stop("x_train must have positive dimensions.");
  if (y_train.size() != N) stop("y_train length must match nrow(x_train).");

  bool is_reg = (task == "regression");
  bool is_cls = (task == "classification");
  bool is_multi = (task == "multiclass");
  if (!is_reg && !is_cls && !is_multi) {
    stop("Unsupported task '%s' for TabM.", task);
  }

  int out_dim = 1;
  if (is_multi) {
    if (num_classes <= 1) stop("num_classes must be >= 2 for multiclass task.");
    out_dim = num_classes;
  }

  double base_lr = (lr > 0.0 && std::isfinite(lr)) ? lr : 0.002;
  double wd = (weight_decay > 0.0 && std::isfinite(weight_decay)) ? weight_decay : 0.0;
  int actual_epochs = std::max(0, n_epochs);
  int actual_K = std::max(1, k_ensemble);
  int actual_d_block = std::max(1, d_block);
  int actual_n_blocks = std::max(1, n_blocks);
  int actual_d_emb = std::max(1, d_embedding);
  int actual_n_bins = std::max(1, n_bins);

  // K >= 2: single-threaded BLAS/Eigen so outer OpenMP over ensemble members drives AMX without GCD contention
  ThreadGuard thread_guard(threads, actual_K >= 2, actual_K >= 2);

  // Target preprocessing
  Eigen::MatrixXf Y_tr(N, out_dim);
  double y_mean_val = 0.0;
  double y_std_val = 1.0;

  if (is_reg) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
      if (!std::isfinite(y_train[i])) stop("y_train contains non-finite values.");
      sum += y_train[i];
    }
    y_mean_val = sum / N;
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
      double d = y_train[i] - y_mean_val;
      sum_sq += d * d;
    }
    y_std_val = (N > 1) ? std::sqrt(sum_sq / static_cast<double>(N - 1)) : 1.0;
    if (!std::isfinite(y_std_val) || y_std_val < 1e-8) y_std_val = 1.0;
    for (int i = 0; i < N; ++i) {
      Y_tr(i, 0) = static_cast<float>((y_train[i] - y_mean_val) / y_std_val);
    }
  } else if (is_cls) {
    for (int i = 0; i < N; ++i) {
      if (!std::isfinite(y_train[i])) stop("y_train contains non-finite values.");
      Y_tr(i, 0) = (y_train[i] > 0.0) ? 1.0f : 0.0f;
    }
  } else {
    Y_tr.setZero();
    for (int i = 0; i < N; ++i) {
      if (!std::isfinite(y_train[i])) stop("y_train contains non-finite values.");
      int c = static_cast<int>(std::lround(y_train[i]));
      if (c >= 0 && c < out_dim) {
        Y_tr(i, c) = 1.0f;
      }
    }
  }

  // Schema & categorical resolution
  std::vector<int> cat_cols_0based;
  std::vector<int> cat_cards;
  resolve_cat_schema_args(x_train, D, cat_cols, cat_cardinalities, cat_cols_0based, cat_cards);

  TabMModel model;
  Eigen::MatrixXf X_num_tr;
  Eigen::MatrixXi X_cat_tr;
  model.prepare_schema_and_encode_train(
    x_train.begin(), N, D, cat_cols_0based, cat_cards,
    num_emb, actual_n_bins, actual_d_emb, seed, X_num_tr, X_cat_tr
  );
  model.init_mlp(out_dim, task, actual_K, actual_d_block, actual_n_blocks, dropout, seed);
  model.y_mean = y_mean_val;
  model.y_std = y_std_val;

  // Validation data preparation (extract once outside epoch loop)
  bool has_x_val = x_val.isNotNull();
  bool has_val = (has_x_val && y_val.isNotNull());
  Eigen::MatrixXf X_num_val;
  Eigen::MatrixXi X_cat_val;
  NumericVector y_v_vec;
  const double* y_v_ptr = nullptr;
  int N_val = 0;

  if (has_x_val) {
    NumericMatrix xv(x_val);
    if (xv.ncol() != D) stop("x_val ncol must match x_train ncol.");
    N_val = xv.nrow();
    if (has_val) {
      y_v_vec = NumericVector(y_val);
      if (y_v_vec.size() != N_val) stop("y_val length must match nrow(x_val).");
      y_v_ptr = y_v_vec.begin();
    }
    if (N_val > 0) {
      int n_num = static_cast<int>(model.num_col_indices.size());
      int n_cat = static_cast<int>(model.cat_col_indices.size());
      if (n_num > 0) X_num_val.resize(N_val, n_num);
      if (n_cat > 0) X_cat_val.resize(N_val, n_cat);
      model.extract_raw_chunk(xv.begin(), N_val, 0, N_val, X_num_val, X_cat_val);
    }
  }

  // Precompute static compact PLE bin indices (uint8/uint16) & fractions once across train and validation matrices
  PLEBinMatrix B_tr, B_val;
  Eigen::MatrixXf F_tr, F_val;
  if (model.num_emb == "pwl" && !model.num_col_indices.empty()) {
    model.encode_ple_matrix(X_num_tr, B_tr, F_tr);
    if (has_x_val && N_val > 0) {
      model.encode_ple_matrix(X_num_val, B_val, F_val);
    }
  }

  // K-aware adaptive batch sizing targeting L2-resident K * B workspace rows (~2K-8K)
  int actual_B = batch_size;
  if (actual_B <= 0) {
    if (N < 512) {
      actual_B = (N >= 64) ? std::min(128, std::max(16, N / 4)) : std::max(1, N);
    } else if (N < 4096) {
      actual_B = 64;
    } else if (N < 12288) {
      actual_B = 128;
    } else if (N < 65536) {
      actual_B = 256;
    } else {
      actual_B = 512;
    }
    if (actual_K >= 16 && N >= 512 && N < 65536) {
      actual_B = std::min(actual_B, 256);
    }
  }
  actual_B = std::clamp(actual_B, 1, N);
  int n_batches = (N + actual_B - 1) / actual_B;
  int total_steps = std::max(1, actual_epochs * n_batches);
  int max_epochs = (stop_epoch > 0 && stop_epoch < actual_epochs) ? stop_epoch : actual_epochs;

  int max_omp_threads = 1;
#if defined(_OPENMP)
  max_omp_threads = std::max(1, omp_get_max_threads());
#endif
  int max_bins = model.max_ple_bins();

  TabMWorkspace ws;
  ws.allocate(N, actual_B, actual_K, model.D_in, actual_d_block, actual_n_blocks, out_dim,
              static_cast<int>(model.num_col_indices.size()),
              static_cast<int>(model.cat_col_indices.size()),
              actual_d_emb, model.num_emb, max_bins,
              model.ple_features, model.cat_cardinalities,
              max_omp_threads);

  TabMInferWorkspace val_ws;
  if (has_x_val && N_val > 0) {
    val_ws.allocate(N_val, actual_K, model.D_in, actual_d_block, actual_n_blocks, out_dim,
                    static_cast<int>(model.num_col_indices.size()),
                    static_cast<int>(model.cat_col_indices.size()),
                    max_bins, max_omp_threads);
  }

  std::mt19937 shuffle_rng(static_cast<unsigned int>(seed + 701));
  std::vector<int> perm(N);
  for (int i = 0; i < N; ++i) perm[i] = i;

  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps = 1e-8;

  double best_val_score = std::numeric_limits<double>::quiet_NaN();
  std::string val_metric_nm = "Val metric";
  TabMModel::StateSnapshot best_snapshot;
  bool has_best_snapshot = false;
  int no_improve_epochs = 0;
  bool stopped_early = false;
  int stopped_epoch = max_epochs;
  int best_epoch = max_epochs;
  int log_interval = (verbose >= 3) ? 1 : ((verbose >= 2) ? 8 : 16);

  if (verbose >= 1) {
    std::string stop_info = (max_epochs < actual_epochs) ? (" (stop=" + std::to_string(max_epochs) + ")") : "";
    Rprintf("    [TabM C++ %s] Training: %d rows (%d num, %d cat) | k=%d, d_block=%d, n_blocks=%d, emb=%s, epochs=%d%s, batch=%d\n",
            task.c_str(), N, static_cast<int>(model.num_col_indices.size()),
            static_cast<int>(model.cat_col_indices.size()),
            actual_K, actual_d_block, actual_n_blocks, model.num_emb.c_str(), actual_epochs,
            stop_info.c_str(),
            actual_B);
    R_FlushConsole();
  }

  const PLEBinMatrix* B_tr_ptr = (B_tr.rows() == N) ? &B_tr : nullptr;
  const Eigen::MatrixXf* F_tr_ptr = (F_tr.rows() == N) ? &F_tr : nullptr;
  const PLEBinMatrix* B_val_ptr = (has_x_val && N_val > 0 && B_val.rows() == N_val) ? &B_val : nullptr;
  const Eigen::MatrixXf* F_val_ptr = (has_x_val && N_val > 0 && F_val.rows() == N_val) ? &F_val : nullptr;

  int step = 0;
  for (int epoch = 0; epoch < max_epochs; ++epoch) {
    std::shuffle(perm.begin(), perm.end(), shuffle_rng);
    model.shuffle_epoch_data(X_num_tr, B_tr_ptr, F_tr_ptr, X_cat_tr, Y_tr, perm, ws);
    double epoch_loss_sum = 0.0;

    for (int b = 0; b < n_batches; ++b) {
      step++;
      int start = b * actual_B;
      int cur_B = std::min(actual_B, N - start);
      if (cur_B <= 0) break;

      double t_norm = (static_cast<double>(step) - 0.5) / static_cast<double>(total_steps);
      double cur_lr = base_lr * cosine_warmup_schedule(t_norm, lr_warmup_frac, 0.01);
      epoch_loss_sum += model.train_step_shuffled(
        start, cur_B, step, cur_lr, wd, beta1, beta2, eps, ws
      );
    }

    double avg_loss = epoch_loss_sum / std::max(1, n_batches);
    double val_score = std::numeric_limits<double>::quiet_NaN();
    bool improved = false;

    if (has_val && N_val > 0) {
      model.predict_preprocessed_chunked(X_num_val, X_cat_val, N_val, val_ws, B_val_ptr, F_val_ptr);
      MetricResult m_res = compute_val_metric(val_ws.preds, y_v_ptr, task, metric, out_dim, &val_ws.auc_pairs, threads);
      val_score = m_res.score;
      val_metric_nm = m_res.name;

      if (early_stopping_rounds > 0) {
        bool is_better = false;
        if (!std::isfinite(best_val_score)) {
          is_better = std::isfinite(val_score);
        } else {
          double tol = 1e-7 * std::max(1.0, std::abs(best_val_score));
          is_better = m_res.higher_is_better ? (val_score > best_val_score + tol)
                                             : (val_score < best_val_score - tol);
        }
        if (is_better) {
          best_val_score = val_score;
          best_snapshot.capture_from(model);
          has_best_snapshot = true;
          best_epoch = epoch + 1;
          no_improve_epochs = 0;
          improved = true;
        } else {
          no_improve_epochs++;
        }
      } else {
        best_val_score = val_score;
        best_epoch = epoch + 1;
      }
    }

    int cur_epoch = epoch + 1;
    if (verbose >= 1 && (cur_epoch == 1 || cur_epoch % log_interval == 0 || cur_epoch == max_epochs || (verbose >= 2 && improved))) {
      if (has_val && N_val > 0) {
        Rprintf("    [TabM C++ %s] Epoch %3d/%d | Train loss: %.4f | %s: %.4f (best: %.4f%s)\n",
                task.c_str(), cur_epoch, actual_epochs, avg_loss, val_metric_nm.c_str(),
                val_score, best_val_score, improved ? "*" : "");
      } else {
        Rprintf("    [TabM C++ %s] Epoch %3d/%d | Train loss: %.4f\n",
                task.c_str(), cur_epoch, actual_epochs, avg_loss);
      }
      R_FlushConsole();
    }

    if (early_stopping_rounds > 0 && has_val && N_val > 0 && no_improve_epochs >= early_stopping_rounds) {
      stopped_early = true;
      stopped_epoch = cur_epoch;
      if (verbose >= 1) {
        Rprintf("    [TabM C++ %s] Early stopping at epoch %d (best epoch=%d, best val=%.4f)\n",
                task.c_str(), cur_epoch, best_epoch, best_val_score);
        R_FlushConsole();
      }
      break;
    }

    if (epoch % 8 == 0) {
      R_CheckUserInterrupt();
    }
  }

  if (early_stopping_rounds > 0 && has_val && has_best_snapshot) {
    best_snapshot.restore_to(model);
  }

  Rcpp::RObject val_preds_sexp = R_NilValue;
  if (has_x_val) {
    if (N_val > 0) {
      model.predict_preprocessed_chunked(X_num_val, X_cat_val, N_val, val_ws, B_val_ptr, F_val_ptr);
      val_preds_sexp = to_rcpp(val_ws.preds);
    } else {
      val_preds_sexp = NumericMatrix(0, out_dim);
    }
  }

  // Feature importances via mean occlusion in embedding space mapped to ncol(x_train)
  std::vector<double> importances;
  if (compute_importances) {
    bool use_val_imp = (has_val && N_val > 0 && y_v_ptr != nullptr);
    const Eigen::MatrixXf& X_num_imp_src = use_val_imp ? X_num_val : X_num_tr;
    const Eigen::MatrixXi& X_cat_imp_src = use_val_imp ? X_cat_val : X_cat_tr;
    const double* y_imp_ptr = use_val_imp ? y_v_ptr : y_train.begin();
    int N_avail = use_val_imp ? N_val : N;
    int N_eval = std::min(N_avail, 1000);

    if (N_eval <= 0) {
      importances.assign(D, 1.0 / std::max(1, D));
    } else if (N_eval == N_avail) {
      const PLEBinMatrix* B_imp_ptr = use_val_imp ? B_val_ptr : B_tr_ptr;
      const Eigen::MatrixXf* F_imp_ptr = use_val_imp ? F_val_ptr : F_tr_ptr;
      importances = model.compute_importances(X_num_imp_src, X_cat_imp_src, y_imp_ptr, N_eval, B_imp_ptr, F_imp_ptr);
    } else {
      std::vector<int> imp_idx(N_avail);
      for (int i = 0; i < N_avail; ++i) imp_idx[i] = i;
      std::mt19937 imp_rng(static_cast<unsigned int>(seed + 777));
      for (int i = 0; i < N_eval; ++i) {
        std::uniform_int_distribution<int> dist(i, N_avail - 1);
        std::swap(imp_idx[i], imp_idx[dist(imp_rng)]);
      }
      std::sort(imp_idx.begin(), imp_idx.begin() + N_eval);

      int n_num = static_cast<int>(model.num_col_indices.size());
      int n_cat = static_cast<int>(model.cat_col_indices.size());
      Eigen::MatrixXf Xn_sub;
      Eigen::MatrixXi Xc_sub;
      if (n_num > 0) {
        Xn_sub.resize(N_eval, n_num);
        for (int j = 0; j < n_num; ++j) {
          for (int i = 0; i < N_eval; ++i) Xn_sub(i, j) = X_num_imp_src(imp_idx[i], j);
        }
      }
      if (n_cat > 0) {
        Xc_sub.resize(N_eval, n_cat);
        for (int c = 0; c < n_cat; ++c) {
          for (int i = 0; i < N_eval; ++i) Xc_sub(i, c) = X_cat_imp_src(imp_idx[i], c);
        }
      }
      std::vector<double> y_eval(N_eval);
      for (int i = 0; i < N_eval; ++i) y_eval[i] = y_imp_ptr[imp_idx[i]];
      importances = model.compute_importances(Xn_sub, Xc_sub, y_eval.data(), N_eval);
    }
  }

  return List::create(
    Named("model_state") = tabm_model_to_list(model),
    Named("predictions") = val_preds_sexp,
    Named("val_preds") = val_preds_sexp,
    Named("best_epoch") = best_epoch,
    Named("best_iteration") = best_epoch,
    Named("best_score") = best_val_score,
    Named("best_val_score") = best_val_score,
    Named("importances") = wrap(importances),
    Named("stopped_early") = stopped_early,
    Named("stopped_epoch") = stopped_epoch,
    Named("base_lr") = base_lr
  );
}

//' Predict with TabM Model in C++
//'
//' @param model_state List returned by `rcpp_tabm_train`.
//' @param x_new Numeric matrix of new input features.
//' @param threads Integer: number of threads for parallel prediction (default 0 = unchanged).
//' @return A NumericMatrix of predictions.
//' @keywords internal
// [[Rcpp::export]]
NumericMatrix rcpp_tabm_predict(List model_state, NumericMatrix x_new, int threads = 0) {
  TabMModel model = list_to_tabm_model(model_state);
  const bool single_blas = (model.k_ensemble >= 2);
  ThreadGuard guard(threads, single_blas, single_blas);
  TabMInferWorkspace ws;
  model.predict_raw_chunked(x_new.begin(), x_new.nrow(), x_new.ncol(), ws);
  return to_rcpp(ws.preds);
}
