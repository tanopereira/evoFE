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
using evofe::tabm::PLEFeature;
using evofe::tabm::TabMInferWorkspace;
using evofe::tabm::TabMModel;
using evofe::tabm::TabMWorkspace;

namespace {

List tabm_model_to_list(const TabMModel& m) {
  int n_num = static_cast<int>(m.num_col_indices.size());
  IntegerVector num_cols_1based(n_num);
  NumericVector num_means_r(n_num);
  NumericVector num_stds_r(n_num);
  for (int j = 0; j < n_num; ++j) {
    num_cols_1based[j] = m.num_col_indices[j] + 1;
    num_means_r[j] = m.num_means[j];
    num_stds_r[j] = m.num_stds[j];
  }

  int n_ple = static_cast<int>(m.ple_features.size());
  LogicalVector ple_active(n_ple);
  IntegerVector ple_offsets(n_ple);
  List ple_edges(n_ple);
  List ple_W(n_ple);
  List ple_b(n_ple);
  for (int j = 0; j < n_ple; ++j) {
    const PLEFeature& pf = m.ple_features[j];
    ple_active[j] = pf.active;
    ple_offsets[j] = pf.out_col_offset;
    if (pf.active) {
      NumericVector ev(pf.edges.size());
      for (size_t k = 0; k < pf.edges.size(); ++k) ev[k] = static_cast<double>(pf.edges[k]);
      ple_edges[j] = ev;
      ple_W[j] = to_rcpp(pf.W.val);
      ple_b[j] = to_rcpp(pf.b.val);
    } else {
      ple_edges[j] = NumericVector(0);
      ple_W[j] = NumericMatrix(0, 0);
      ple_b[j] = NumericMatrix(0, 0);
    }
  }

  int n_cat = static_cast<int>(m.cat_col_indices.size());
  IntegerVector cat_cols_1based(n_cat);
  IntegerVector cat_cards_r(n_cat);
  List cat_tables_r(n_cat);
  for (int c = 0; c < n_cat; ++c) {
    cat_cols_1based[c] = m.cat_col_indices[c] + 1;
    cat_cards_r[c] = m.cat_cardinalities[c];
    cat_tables_r[c] = to_rcpp(m.cat_tables[c].val);
  }

  int n_blks = static_cast<int>(m.blocks.size());
  List blocks_W(n_blks);
  List blocks_b(n_blks);
  for (int l = 0; l < n_blks; ++l) {
    blocks_W[l] = to_rcpp(m.blocks[l].W.val);
    blocks_b[l] = to_rcpp(m.blocks[l].b.val);
  }

  int K = static_cast<int>(m.W_out.size());
  List W_out_r(K);
  for (int k = 0; k < K; ++k) {
    W_out_r[k] = to_rcpp(m.W_out[k].val);
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
    Named("ple_active") = ple_active,
    Named("ple_offsets") = ple_offsets,
    Named("ple_edges") = ple_edges,
    Named("ple_W") = ple_W,
    Named("ple_b") = ple_b,
    Named("cat_cols") = cat_cols_1based,
    Named("cat_cards") = cat_cards_r,
    Named("cat_tables") = cat_tables_r,
    Named("R_in") = to_rcpp(m.R_in.val),
    Named("blocks_W") = blocks_W,
    Named("blocks_b") = blocks_b,
    Named("W_out") = W_out_r,
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
  for (int j = 0; j < n_num; ++j) {
    m.num_col_indices[j] = num_cols_r[j] - 1;
    m.num_means[j] = num_means_r[j];
    double s = num_stds_r[j];
    m.num_stds[j] = s;
    m.num_inv_stds[j] = (s < 1e-5 || !std::isfinite(s)) ? 1.0 : (1.0 / s);
  }

  LogicalVector ple_active = lst["ple_active"];
  IntegerVector ple_offsets = lst["ple_offsets"];
  List ple_edges = lst["ple_edges"];
  List ple_W = lst["ple_W"];
  List ple_b = lst["ple_b"];
  int n_ple = ple_active.size();
  m.ple_features.resize(n_ple);
  for (int j = 0; j < n_ple; ++j) {
    if (ple_active[j]) {
      NumericVector ev = ple_edges[j];
      std::vector<float> fv(ev.size());
      for (int k = 0; k < ev.size(); ++k) fv[k] = static_cast<float>(ev[k]);
      NumericMatrix Wm = ple_W[j];
      NumericMatrix bm = ple_b[j];
      m.ple_features[j].init_from_saved(fv, copy_eigen_fast(Wm), copy_eigen_fast(bm));
      m.ple_features[j].out_col_offset = ple_offsets[j];
    }
  }

  IntegerVector cat_cols_r = lst["cat_cols"];
  IntegerVector cat_cards_r = lst["cat_cards"];
  List cat_tables_r = lst["cat_tables"];
  int n_cat = cat_cols_r.size();
  m.cat_col_indices.resize(n_cat);
  m.cat_cardinalities.resize(n_cat);
  m.cat_tables.resize(n_cat);
  for (int c = 0; c < n_cat; ++c) {
    m.cat_col_indices[c] = cat_cols_r[c] - 1;
    m.cat_cardinalities[c] = cat_cards_r[c];
    NumericMatrix tm = cat_tables_r[c];
    m.cat_tables[c].val = copy_eigen_fast(tm);
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

  List W_out_r = lst["W_out"];
  int K = W_out_r.size();
  m.W_out.resize(K);
  for (int k = 0; k < K; ++k) {
    NumericMatrix Wm = W_out_r[k];
    m.W_out[k].val = copy_eigen_fast(Wm);
  }
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
                     double lr_warmup_frac = 0.05) {

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

  ThreadGuard thread_guard(threads);

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

  // Adaptive batch sizing bounded for K * B workspace memory
  int actual_B = 256;
  if (batch_size > 0) {
    actual_B = std::min(batch_size, N);
  } else if (N >= 131072) {
    actual_B = 1024;
  } else if (N >= 16384) {
    actual_B = 512;
  } else if (N >= 1024) {
    actual_B = 256;
  } else if (N > 64) {
    actual_B = std::min(128, std::max(16, N / 4));
  } else {
    actual_B = std::max(1, N);
  }
  actual_B = std::max(1, actual_B);
  int n_batches = (N + actual_B - 1) / actual_B;
  int total_steps = std::max(1, actual_epochs * n_batches);

  TabMWorkspace ws;
  ws.allocate(actual_B, actual_K, model.D_in, actual_d_block, actual_n_blocks, out_dim,
              static_cast<int>(model.num_col_indices.size()),
              static_cast<int>(model.cat_col_indices.size()),
              actual_d_emb, model.ple_features, model.cat_cardinalities);

  TabMInferWorkspace val_ws;
  if (has_x_val && N_val > 0) {
    val_ws.allocate(N_val, actual_K, model.D_in, actual_d_block, actual_n_blocks, out_dim,
                    static_cast<int>(model.num_col_indices.size()),
                    static_cast<int>(model.cat_col_indices.size()));
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
  int stopped_epoch = actual_epochs;
  int best_epoch = actual_epochs;
  int log_interval = (verbose >= 3) ? 1 : ((verbose >= 2) ? 16 : 32);

  if (verbose >= 1) {
    Rprintf("    [TabM C++ %s] Training: %d rows (%d num, %d cat) | k=%d, d_block=%d, n_blocks=%d, emb=%s, epochs=%d, batch=%d\n",
            task.c_str(), N, static_cast<int>(model.num_col_indices.size()),
            static_cast<int>(model.cat_col_indices.size()),
            actual_K, actual_d_block, actual_n_blocks, model.num_emb.c_str(), actual_epochs, actual_B);
  }

  int step = 0;
  for (int epoch = 0; epoch < actual_epochs; ++epoch) {
    std::shuffle(perm.begin(), perm.end(), shuffle_rng);
    double epoch_loss_sum = 0.0;

    for (int b = 0; b < n_batches; ++b) {
      step++;
      int start = b * actual_B;
      int cur_B = std::min(actual_B, N - start);
      if (cur_B <= 0) break;

      double t_norm = static_cast<double>(step) / static_cast<double>(total_steps);
      double cur_lr = base_lr * cosine_warmup_schedule(t_norm, lr_warmup_frac, 0.0);
      epoch_loss_sum += model.train_step(
        X_num_tr, X_cat_tr, Y_tr, perm, start, cur_B,
        step, cur_lr, wd, beta1, beta2, eps, ws
      );
    }

    double avg_loss = epoch_loss_sum / std::max(1, n_batches);
    double val_score = std::numeric_limits<double>::quiet_NaN();
    bool improved = false;

    if (has_val && N_val > 0) {
      model.predict_preprocessed_chunked(X_num_val, X_cat_val, N_val, val_ws);
      MetricResult m_res = compute_val_metric(val_ws.preds, y_v_ptr, task, metric, out_dim, &val_ws.auc_pairs, threads);
      val_score = m_res.score;
      val_metric_nm = m_res.name;

      if (early_stopping_rounds > 0) {
        if (std::isnan(best_val_score)) {
          best_val_score = m_res.higher_is_better ? -std::numeric_limits<double>::infinity()
                                                  : std::numeric_limits<double>::infinity();
        }
        bool is_better = m_res.higher_is_better ? (val_score > best_val_score + 1e-6)
                                                : (val_score < best_val_score - 1e-6);
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
    if (verbose >= 1 && (cur_epoch == 1 || cur_epoch % log_interval == 0 || cur_epoch == actual_epochs || (verbose >= 2 && improved))) {
      if (has_val && N_val > 0) {
        Rprintf("    [TabM C++ %s] Epoch %3d/%d | Train loss: %.4f | %s: %.4f (best: %.4f%s)\n",
                task.c_str(), cur_epoch, actual_epochs, avg_loss, val_metric_nm.c_str(),
                val_score, best_val_score, improved ? "*" : "");
      } else {
        Rprintf("    [TabM C++ %s] Epoch %3d/%d | Train loss: %.4f\n",
                task.c_str(), cur_epoch, actual_epochs, avg_loss);
      }
    }

    if (early_stopping_rounds > 0 && has_val && N_val > 0 && no_improve_epochs >= early_stopping_rounds) {
      stopped_early = true;
      stopped_epoch = cur_epoch;
      if (verbose >= 1) {
        Rprintf("    [TabM C++ %s] Early stopping at epoch %d (best epoch=%d, best val=%.4f)\n",
                task.c_str(), cur_epoch, best_epoch, best_val_score);
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
      model.predict_preprocessed_chunked(X_num_val, X_cat_val, N_val, val_ws);
      val_preds_sexp = to_rcpp(val_ws.preds);
    } else {
      val_preds_sexp = NumericMatrix(0, out_dim);
    }
  }

  // Feature importances via mean occlusion in embedding space mapped to ncol(x_train)
  bool use_val_imp = (has_val && N_val > 0 && y_v_ptr != nullptr);
  const Eigen::MatrixXf& X_num_imp_src = use_val_imp ? X_num_val : X_num_tr;
  const Eigen::MatrixXi& X_cat_imp_src = use_val_imp ? X_cat_val : X_cat_tr;
  const double* y_imp_ptr = use_val_imp ? y_v_ptr : y_train.begin();
  int N_avail = use_val_imp ? N_val : N;
  int N_eval = std::min(N_avail, 1000);

  std::vector<double> importances;
  if (N_eval <= 0) {
    importances.assign(D, 1.0 / std::max(1, D));
  } else if (N_eval == N_avail) {
    importances = model.compute_importances(X_num_imp_src, X_cat_imp_src, y_imp_ptr, N_eval);
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
  ThreadGuard guard(threads);
  TabMModel model = list_to_tabm_model(model_state);
  TabMInferWorkspace ws;
  model.predict_raw_chunked(x_new.begin(), x_new.nrow(), x_new.ncol(), ws);
  return to_rcpp(ws.preds);
}
