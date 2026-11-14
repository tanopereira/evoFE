#ifndef EVOFE_TABM_CORE_H
#define EVOFE_TABM_CORE_H

#include "nn_common.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace evofe {
namespace tabm {

using evofe::nn::AdamParam;
using evofe::nn::FiniteColumnStats;
using evofe::nn::compute_finite_column_stats;
using evofe::nn::cosine_warmup_schedule;
using evofe::nn::gemm_f32_raw;

// Deterministic fast xorshift32 PRNG for thread-safe dropout masks
static inline uint32_t hash_seed32(uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
  uint32_t x = a ^ 0x9e3779b9u;
  x ^= b * 0x85ebca6bu + 0xc2b2ae35u;
  x ^= c * 0xc2b2ae35u + 0x27d4eb2fu;
  x ^= d * 0x165667b1u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return (x == 0u) ? 0x6d2b79f5u : x;
}

static inline uint32_t xorshift32_step(uint32_t& s) noexcept {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

// Single numeric feature Piecewise Linear Encoding + learnable linear projection (d_emb x T_j)
// Stores W as (d_emb x T_j) ColMajor for AdamW state compatibility and maintains transposed
// S_T ((T_j + 1) x d_emb) and W_T (T_j x d_emb) so forward lookup over e is unit-stride in L1 cache.
struct PLEFeature {
  std::vector<float> edges;       // length n_bins + 1, strictly increasing
  std::vector<float> inv_widths;  // length n_bins
  AdamParam W;                    // d_emb x n_bins
  AdamParam b;                    // d_emb x 1
  Eigen::MatrixXf S;              // d_emb x (n_bins + 1), prefix sums: S[:, 0] = b, S[:, r] = b + sum_{0..r-1} W[:, j]
  Eigen::MatrixXf S_T;            // (n_bins + 1) x d_emb, S.transpose() for contiguous unit-stride lookup
  Eigen::MatrixXf W_T;            // n_bins x d_emb, W.val.transpose() for contiguous unit-stride lookup
  int n_bins = 0;
  int d_emb = 0;
  int out_col_offset = -1;        // column offset into E_batch (-1 if constant/inactive)
  bool active = false;

  void rebuild_prefix() {
    if (!active || n_bins <= 0 || d_emb <= 0) return;
    if (S.rows() != d_emb || S.cols() != n_bins + 1) {
      S.resize(d_emb, n_bins + 1);
    }
    S.col(0) = b.val.col(0);
    for (int r = 0; r < n_bins; ++r) {
      S.col(r + 1).noalias() = S.col(r) + W.val.col(r);
    }
    if (S_T.rows() != n_bins + 1 || S_T.cols() != d_emb) {
      S_T.resize(n_bins + 1, d_emb);
    }
    if (W_T.rows() != n_bins || W_T.cols() != d_emb) {
      W_T.resize(n_bins, d_emb);
    }
    S_T.noalias() = S.transpose();
    W_T.noalias() = W.val.transpose();
  }

  void init_from_edges(const std::vector<float>& strict_edges, int d_embedding, std::mt19937& rng) {
    active = false;
    edges.clear();
    inv_widths.clear();
    d_emb = std::max(1, d_embedding);
    if (strict_edges.size() < 2) {
      n_bins = 0;
      return;
    }
    edges = strict_edges;
    n_bins = static_cast<int>(edges.size()) - 1;
    inv_widths.resize(n_bins);
    for (int r = 0; r < n_bins; ++r) {
      float w = edges[r + 1] - edges[r];
      inv_widths[r] = (w > 1e-12f) ? (1.0f / w) : 0.0f;
    }
    W.init(d_emb, n_bins);
    b.init(d_emb, 1);

    float bound = 1.0f / std::sqrt(static_cast<float>(std::max(1, n_bins)));
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (int r = 0; r < n_bins; ++r) {
      for (int e = 0; e < d_emb; ++e) {
        W.val(e, r) = dist(rng);
      }
    }
    b.val.setZero();
    active = true;
    rebuild_prefix();
  }

  void init_from_saved(const std::vector<float>& saved_edges,
                       const Eigen::MatrixXf& saved_W,
                       const Eigen::MatrixXf& saved_b) {
    edges = saved_edges;
    if (edges.size() < 2 || saved_W.cols() != static_cast<int>(edges.size()) - 1) {
      active = false;
      n_bins = 0;
      return;
    }
    n_bins = static_cast<int>(edges.size()) - 1;
    d_emb = static_cast<int>(saved_W.rows());
    inv_widths.resize(n_bins);
    for (int r = 0; r < n_bins; ++r) {
      float w = edges[r + 1] - edges[r];
      inv_widths[r] = (w > 1e-12f) ? (1.0f / w) : 0.0f;
    }
    W.val = saved_W;
    b.val = saved_b;
    active = true;
    rebuild_prefix();
  }

  inline void lookup(float x, int& out_bin, float& out_frac) const noexcept {
    if (x <= edges[0]) {
      out_bin = 0;
      out_frac = 0.0f;
      return;
    }
    if (x >= edges[n_bins]) {
      out_bin = n_bins - 1;
      out_frac = 1.0f;
      return;
    }
    int r = 0;
    if (n_bins > 1) {
      const float* it = std::upper_bound(edges.data() + 1, edges.data() + n_bins + 1, x);
      r = static_cast<int>(it - edges.data()) - 1;
      if (r < 0) r = 0;
      if (r >= n_bins) r = n_bins - 1;
    }
    float f = (x - edges[r]) * inv_widths[r];
    out_bin = r;
    out_frac = std::clamp(f, 0.0f, 1.0f);
  }
};

// Pre-allocated workspace for TabM training batches (zero heap allocations in training loop)
struct TabMWorkspace {
  int max_B = 0;
  int K = 0;
  int max_KB = 0;
  int D_in = 0;
  int d_block = 0;
  int n_blocks = 0;
  int out_dim = 0;
  int n_num = 0;
  int n_cat = 0;
  int d_emb = 0;
  int max_bins = 0;

  // Minibatch gathered inputs (avoids duplicating entire 650K dataset every epoch)
  Eigen::MatrixXf batch_X_num;  // max_B x n_num
  Eigen::MatrixXi batch_X_cat;  // max_B x n_cat
  Eigen::MatrixXf batch_Y;      // max_B x out_dim

  // PLE lookup cache for current batch
  Eigen::MatrixXi bin_idx;      // max_B x n_num
  Eigen::MatrixXf bin_frac;     // max_B x n_num

  // Shared representation before BatchEnsemble expansion
  Eigen::MatrixXf E_batch;      // max_B x D_in
  Eigen::MatrixXf grad_E;       // max_B x D_in

  // MLP forward activations: H[0] is (max_KB x D_in), H[1..n_blocks] are (max_KB x d_block)
  std::vector<Eigen::MatrixXf> H;
  Eigen::MatrixXf Out;          // max_KB x out_dim
  Eigen::MatrixXf P;            // max_KB x out_dim
  Eigen::MatrixXf grad_Out;     // max_KB x out_dim

  // Ping-pong backward deltas, sized max(d_block, D_in) columns
  Eigen::MatrixXf delta_a;
  Eigen::MatrixXf delta_b;

  // Weight gradients
  Eigen::MatrixXf grad_R_in;
  std::vector<Eigen::MatrixXf> grad_W_blocks;
  std::vector<Eigen::MatrixXf> grad_b_blocks;
  std::vector<Eigen::MatrixXf> grad_W_out;
  Eigen::MatrixXf grad_b_out;

  // Per-feature PLE & categorical scratch buffers (parallel safe per feature index)
  std::vector<Eigen::MatrixXf> ple_grad_W;    // d_emb x n_bins (for AdamW update matching PLEFeature::W)
  std::vector<Eigen::MatrixXf> ple_grad_W_T;  // n_bins x d_emb (unit-stride L1-resident accumulators)
  std::vector<Eigen::MatrixXf> ple_grad_b;    // d_emb x 1
  std::vector<Eigen::MatrixXf> ple_diff;      // n_bins x d_emb (suffix-sum scratch per embedding coordinate)
  std::vector<Eigen::MatrixXf> cat_grad_E;

  void allocate(int batch_cap, int k_ens, int d_in_dim, int d_blk, int n_blks, int out_d,
                int n_num_cols, int n_cat_cols, int d_embedding,
                const std::vector<PLEFeature>& ple_feats,
                const std::vector<int>& cat_cards) {
    max_B = std::max(1, batch_cap);
    K = std::max(1, k_ens);
    max_KB = K * max_B;
    D_in = std::max(1, d_in_dim);
    d_block = std::max(1, d_blk);
    n_blocks = std::max(1, n_blks);
    out_dim = std::max(1, out_d);
    n_num = n_num_cols;
    n_cat = n_cat_cols;
    d_emb = std::max(1, d_embedding);

    if (n_num > 0) {
      batch_X_num.resize(max_B, n_num);
      bin_idx.resize(max_B, n_num);
      bin_frac.resize(max_B, n_num);
    }
    if (n_cat > 0) {
      batch_X_cat.resize(max_B, n_cat);
    }
    batch_Y.resize(max_B, out_dim);

    E_batch.resize(max_B, D_in);
    grad_E.resize(max_B, D_in);

    H.resize(n_blocks + 1);
    H[0].resize(max_KB, D_in);
    for (int l = 0; l < n_blocks; ++l) {
      H[l + 1].resize(max_KB, d_block);
    }
    Out.resize(max_KB, out_dim);
    P.resize(max_KB, out_dim);
    grad_Out.resize(max_KB, out_dim);

    int max_delta_cols = std::max(d_block, D_in);
    delta_a.resize(max_KB, max_delta_cols);
    delta_b.resize(max_KB, max_delta_cols);

    grad_R_in.resize(K, D_in);
    grad_W_blocks.resize(n_blocks);
    grad_b_blocks.resize(n_blocks);
    for (int l = 0; l < n_blocks; ++l) {
      int din_l = (l == 0) ? D_in : d_block;
      grad_W_blocks[l].resize(din_l, d_block);
      grad_b_blocks[l].resize(K, d_block);
    }
    grad_W_out.resize(K);
    for (int k = 0; k < K; ++k) {
      grad_W_out[k].resize(d_block, out_dim);
    }
    grad_b_out.resize(K, out_dim);

    ple_grad_W.resize(ple_feats.size());
    ple_grad_W_T.resize(ple_feats.size());
    ple_grad_b.resize(ple_feats.size());
    ple_diff.resize(ple_feats.size());
    for (size_t j = 0; j < ple_feats.size(); ++j) {
      if (ple_feats[j].active) {
        ple_grad_W[j].resize(d_emb, ple_feats[j].n_bins);
        ple_grad_W_T[j].resize(ple_feats[j].n_bins, d_emb);
        ple_grad_b[j].resize(d_emb, 1);
        ple_diff[j].resize(ple_feats[j].n_bins, d_emb);
      }
    }

    cat_grad_E.resize(cat_cards.size());
    for (size_t c = 0; c < cat_cards.size(); ++c) {
      int n_rows = std::max(1, cat_cards[c]) + 1;
      cat_grad_E[c].resize(d_emb, n_rows);
    }
  }
};

// Bounded-memory chunked inference workspace (keeps 650K-row eval/predict under ~80 MB RAM)
struct TabMInferWorkspace {
  static constexpr int DEFAULT_CHUNK_ROWS = 4096;
  int chunk_cap = 0;
  int K = 0;
  int D_in = 0;
  int d_block = 0;
  int n_blocks = 0;
  int out_dim = 0;
  int n_num = 0;
  int n_cat = 0;

  Eigen::MatrixXf E_chunk;
  Eigen::MatrixXf X_num_chunk;
  Eigen::MatrixXi X_cat_chunk;
  Eigen::MatrixXi bin_idx_chunk;
  Eigen::MatrixXf bin_frac_chunk;
  std::vector<Eigen::MatrixXf> H;
  Eigen::MatrixXf Out;
  Eigen::MatrixXd preds;
  std::vector<std::pair<double, int>> auc_pairs;

  void allocate(int n_total_rows, int k_ens, int d_in_dim, int d_blk, int n_blks, int out_d,
                int n_num_cols, int n_cat_cols, int requested_chunk = DEFAULT_CHUNK_ROWS) {
    K = std::max(1, k_ens);
    // Cap KB chunk so even large K values keep workspace bounded (~80 MB)
    int max_rows_by_ram = std::max(256, 65536 / K);
    int cap = std::min(std::max(1, n_total_rows), std::min(requested_chunk, max_rows_by_ram));
    chunk_cap = std::max(1, cap);
    D_in = std::max(1, d_in_dim);
    d_block = std::max(1, d_blk);
    n_blocks = std::max(1, n_blks);
    out_dim = std::max(1, out_d);
    n_num = n_num_cols;
    n_cat = n_cat_cols;

    int max_KB = K * chunk_cap;
    E_chunk.resize(chunk_cap, D_in);
    if (n_num > 0) {
      X_num_chunk.resize(chunk_cap, n_num);
      bin_idx_chunk.resize(chunk_cap, n_num);
      bin_frac_chunk.resize(chunk_cap, n_num);
    }
    if (n_cat > 0) X_cat_chunk.resize(chunk_cap, n_cat);

    H.resize(n_blocks + 1);
    H[0].resize(max_KB, D_in);
    for (int l = 0; l < n_blocks; ++l) {
      H[l + 1].resize(max_KB, d_block);
    }
    Out.resize(max_KB, out_dim);
    preds.resize(n_total_rows, out_dim);
    if (static_cast<int>(auc_pairs.size()) < n_total_rows) {
      auc_pairs.resize(n_total_rows);
    }
  }
};

struct TabMBlock {
  AdamParam W;  // d_in x d_out
  AdamParam b;  // K x d_out
  int d_in = 0;
  int d_out = 0;
};

struct TabMModel {
  std::string task = "regression";
  std::string num_emb = "pwl";  // "pwl" or "none"
  int n_features = 0;
  int output_dim = 1;
  int k_ensemble = 16;
  int d_block = 128;
  int n_blocks = 2;
  int d_embedding = 8;
  int n_bins = 48;
  double dropout = 0.1;
  double y_mean = 0.0;
  double y_std = 1.0;

  // Schema partition (0-based indices into raw ncol(x) matrix)
  std::vector<int> num_col_indices;
  std::vector<double> num_means;
  std::vector<double> num_stds;
  std::vector<double> num_inv_stds;
  std::vector<PLEFeature> ple_features;

  std::vector<int> cat_col_indices;
  std::vector<int> cat_cardinalities;
  std::vector<AdamParam> cat_tables;  // d_embedding x (card + 1)

  int D_num_rep = 0;
  int D_cat_rep = 0;
  int D_in = 1;
  bool has_dummy_in = false;

  // BatchEnsemble parameters
  AdamParam R_in;                   // K x D_in (Rademacher {-1, +1} init)
  std::vector<TabMBlock> blocks;    // n_blocks
  std::vector<AdamParam> W_out;     // K heads, each d_block x output_dim
  AdamParam b_out;                  // K x output_dim

  struct StateSnapshot {
    std::vector<Eigen::MatrixXf> ple_W;
    std::vector<Eigen::MatrixXf> ple_b;
    std::vector<Eigen::MatrixXf> cat_tables;
    Eigen::MatrixXf R_in;
    std::vector<Eigen::MatrixXf> blocks_W;
    std::vector<Eigen::MatrixXf> blocks_b;
    std::vector<Eigen::MatrixXf> W_out;
    Eigen::MatrixXf b_out;

    void capture_from(const TabMModel& m) {
      ple_W.resize(m.ple_features.size());
      ple_b.resize(m.ple_features.size());
      for (size_t j = 0; j < m.ple_features.size(); ++j) {
        if (m.ple_features[j].active) {
          ple_W[j] = m.ple_features[j].W.val;
          ple_b[j] = m.ple_features[j].b.val;
        }
      }
      cat_tables.resize(m.cat_tables.size());
      for (size_t c = 0; c < m.cat_tables.size(); ++c) {
        cat_tables[c] = m.cat_tables[c].val;
      }
      R_in = m.R_in.val;
      blocks_W.resize(m.blocks.size());
      blocks_b.resize(m.blocks.size());
      for (size_t l = 0; l < m.blocks.size(); ++l) {
        blocks_W[l] = m.blocks[l].W.val;
        blocks_b[l] = m.blocks[l].b.val;
      }
      W_out.resize(m.W_out.size());
      for (size_t k = 0; k < m.W_out.size(); ++k) {
        W_out[k] = m.W_out[k].val;
      }
      b_out = m.b_out.val;
    }

    void restore_to(TabMModel& m) const {
      for (size_t j = 0; j < m.ple_features.size() && j < ple_W.size(); ++j) {
        if (m.ple_features[j].active) {
          m.ple_features[j].W.val = ple_W[j];
          m.ple_features[j].b.val = ple_b[j];
          m.ple_features[j].rebuild_prefix();
        }
      }
      for (size_t c = 0; c < m.cat_tables.size() && c < cat_tables.size(); ++c) {
        m.cat_tables[c].val = cat_tables[c];
      }
      m.R_in.val = R_in;
      for (size_t l = 0; l < m.blocks.size() && l < blocks_W.size(); ++l) {
        m.blocks[l].W.val = blocks_W[l];
        m.blocks[l].b.val = blocks_b[l];
      }
      for (size_t k = 0; k < m.W_out.size() && k < W_out.size(); ++k) {
        m.W_out[k].val = W_out[k];
      }
      m.b_out.val = b_out;
    }
  };

  void prepare_schema_and_encode_train(
      const double* x_ptr, int N, int D,
      const std::vector<int>& cat_cols_0based,
      const std::vector<int>& cat_cards_in,
      const std::string& num_emb_req,
      int n_bins_req,
      int d_emb_req,
      int seed,
      Eigen::MatrixXf& X_num_out,
      Eigen::MatrixXi& X_cat_out) {

    n_features = D;
    d_embedding = std::max(1, d_emb_req);
    n_bins = std::max(1, n_bins_req);

    std::string ne = num_emb_req;
    std::transform(ne.begin(), ne.end(), ne.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (ne != "pwl" && ne != "none") {
      throw std::invalid_argument("Unsupported num_emb '" + num_emb_req + "'. Expected 'pwl' or 'none'.");
    }
    num_emb = ne;

    std::vector<unsigned char> is_cat(D, 0);
    cat_col_indices = cat_cols_0based;
    cat_cardinalities = cat_cards_in;
    for (int c_idx : cat_col_indices) {
      if (c_idx < 0 || c_idx >= D) {
        throw std::invalid_argument("cat_cols index out of range.");
      }
      if (is_cat[c_idx]) {
        throw std::invalid_argument("cat_cols contains duplicate column indices.");
      }
      is_cat[c_idx] = 1;
    }

    num_col_indices.clear();
    for (int j = 0; j < D; ++j) {
      if (!is_cat[j]) num_col_indices.push_back(j);
    }

    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());

    // 1. Preprocess numeric columns (finite mean/SD + optional quantile PLE bins)
    num_means.assign(n_num, 0.0);
    num_stds.assign(n_num, 1.0);
    num_inv_stds.assign(n_num, 1.0);
    if (n_num > 0) {
      X_num_out.resize(N, n_num);
    }
    ple_features.clear();
    if (num_emb == "pwl") {
      ple_features.resize(n_num);
    }

    std::mt19937 ple_rng(static_cast<unsigned int>(seed + 13));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2)
#endif
    for (int j = 0; j < n_num; ++j) {
      int orig_col = num_col_indices[j];
      const double* col_ptr = x_ptr + static_cast<size_t>(orig_col) * static_cast<size_t>(N);
      FiniteColumnStats st = compute_finite_column_stats(col_ptr, N);
      num_means[j] = st.mean;
      num_stds[j] = st.sd;
      num_inv_stds[j] = st.inv_std;

      float* dst = X_num_out.col(j).data();
      for (int i = 0; i < N; ++i) {
        double v = col_ptr[i];
        double clean_v = std::isfinite(v) ? v : st.mean;
        double z = (clean_v - st.mean) * st.inv_std;
        dst[i] = static_cast<float>(std::clamp(z, -30.0, 30.0));
      }
    }

    // Build PLE bins deterministically outside parallel RNG mutations
    D_num_rep = 0;
    if (num_emb == "pwl") {
      std::vector<float> fin_sample;
      for (int j = 0; j < n_num; ++j) {
        int orig_col = num_col_indices[j];
        const double* col_ptr = x_ptr + static_cast<size_t>(orig_col) * static_cast<size_t>(N);
        double m = num_means[j];
        double inv_s = num_inv_stds[j];

        fin_sample.clear();
        // Count finite entries first to pick deterministic stride <= 100k samples
        int n_fin = 0;
        for (int i = 0; i < N; ++i) {
          if (std::isfinite(col_ptr[i])) n_fin++;
        }
        int step = (n_fin > 100000) ? (n_fin / 100000) : 1;
        fin_sample.reserve(std::min(n_fin, 100000));
        int seen = 0;
        for (int i = 0; i < N; ++i) {
          double v = col_ptr[i];
          if (std::isfinite(v)) {
            if (step == 1 || (seen % step) == 0) {
              double z = (v - m) * inv_s;
              fin_sample.push_back(static_cast<float>(std::clamp(z, -30.0, 30.0)));
            }
            seen++;
          }
        }
        if (fin_sample.size() < 2) {
          continue;
        }
        std::sort(fin_sample.begin(), fin_sample.end());
        if (fin_sample.back() - fin_sample.front() <= 1e-7f) {
          continue; // Constant column
        }

        int nb = std::max(1, n_bins);
        std::vector<float> edges;
        edges.reserve(nb + 1);
        for (int k = 0; k <= nb; ++k) {
          double q = static_cast<double>(k) / static_cast<double>(nb);
          double pos = q * static_cast<double>(fin_sample.size() - 1);
          size_t idx0 = static_cast<size_t>(std::floor(pos));
          size_t idx1 = std::min(fin_sample.size() - 1, idx0 + 1);
          double frac = pos - static_cast<double>(idx0);
          float e = static_cast<float>((1.0 - frac) * fin_sample[idx0] + frac * fin_sample[idx1]);
          if (edges.empty()) {
            edges.push_back(e);
          } else {
            float tol = 1e-6f * std::max(1.0f, std::abs(edges.back()));
            if (e > edges.back() + tol) {
              edges.push_back(e);
            }
          }
        }
        if (edges.size() == 1 && fin_sample.back() > edges[0] + 1e-6f) {
          edges.push_back(fin_sample.back());
        } else if (edges.size() >= 2 && fin_sample.back() > edges.back()) {
          edges.back() = fin_sample.back();
        }
        if (edges.size() >= 2) {
          ple_features[j].init_from_edges(edges, d_embedding, ple_rng);
          if (ple_features[j].active) {
            ple_features[j].out_col_offset = D_num_rep;
            D_num_rep += d_embedding;
          }
        }
      }
    } else {
      D_num_rep = n_num;
    }

    // 2. Encode categorical columns as 0-based codes with card == missing/OOD bucket
    cat_tables.clear();
    if (n_cat > 0) {
      X_cat_out.resize(N, n_cat);
      if (static_cast<int>(cat_cardinalities.size()) != n_cat) {
        cat_cardinalities.assign(n_cat, 1);
        for (int c = 0; c < n_cat; ++c) {
          const double* col_ptr = x_ptr + static_cast<size_t>(cat_col_indices[c]) * static_cast<size_t>(N);
          int max_code = 1;
          for (int i = 0; i < N; ++i) {
            double v = col_ptr[i];
            if (std::isfinite(v)) {
              int iv = static_cast<int>(std::lround(v));
              if (iv > max_code && iv <= 1000000) max_code = iv;
            }
          }
          cat_cardinalities[c] = max_code;
        }
      } else {
        for (int c = 0; c < n_cat; ++c) {
          cat_cardinalities[c] = std::max(1, cat_cardinalities[c]);
        }
      }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_cat >= 2)
#endif
      for (int c = 0; c < n_cat; ++c) {
        int card = cat_cardinalities[c];
        const double* col_ptr = x_ptr + static_cast<size_t>(cat_col_indices[c]) * static_cast<size_t>(N);
        int* dst = X_cat_out.col(c).data();
        for (int i = 0; i < N; ++i) {
          double v = col_ptr[i];
          int code = card; // missing / OOD
          if (std::isfinite(v)) {
            long iv = std::lround(v);
            if (iv >= 1 && iv <= card) {
              code = static_cast<int>(iv) - 1;
            }
          }
          dst[i] = code;
        }
      }

      cat_tables.resize(n_cat);
      std::mt19937 cat_rng(static_cast<unsigned int>(seed + 29));
      float bound = 1.0f / std::sqrt(static_cast<float>(d_embedding));
      std::uniform_real_distribution<float> dist(-bound, bound);
      for (int c = 0; c < n_cat; ++c) {
        int n_rows = cat_cardinalities[c] + 1;
        cat_tables[c].init(d_embedding, n_rows);
        for (int r = 0; r < n_rows; ++r) {
          for (int e = 0; e < d_embedding; ++e) {
            cat_tables[c].val(e, r) = dist(cat_rng);
          }
        }
      }
    }
    D_cat_rep = n_cat * d_embedding;

    has_dummy_in = (D_num_rep + D_cat_rep == 0);
    D_in = has_dummy_in ? 1 : (D_num_rep + D_cat_rep);
  }

  // Bin a contiguous row slice [row_offset .. row_offset + cur_B) from standardized X_num
  // into B_out (rows >= cur_B, cols == n_num) and F_out (rows >= cur_B, cols == n_num).
  void encode_ple_chunk(const float* x_num_ptr, int x_num_lda, int row_offset, int cur_B,
                        Eigen::MatrixXi& B_out, Eigen::MatrixXf& F_out) const {
    if (num_emb != "pwl" || cur_B <= 0) return;
    int n_num = static_cast<int>(num_col_indices.size());
    if (n_num <= 0) return;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B >= 32)
#endif
    for (int j = 0; j < n_num; ++j) {
      const PLEFeature& pf = ple_features[j];
      if (!pf.active) continue;
      const float* x_col = x_num_ptr + static_cast<size_t>(j) * static_cast<size_t>(x_num_lda) + row_offset;
      int* b_col = B_out.col(j).data();
      float* f_col = F_out.col(j).data();
      for (int i = 0; i < cur_B; ++i) {
        pf.lookup(x_col[i], b_col[i], f_col[i]);
      }
    }
  }

  // Precompute static PLE (bin_idx, bin_frac) once across an entire matrix outside epoch loops.
  void encode_ple_matrix(const Eigen::MatrixXf& X_num,
                         Eigen::MatrixXi& B_out,
                         Eigen::MatrixXf& F_out) const {
    int N = static_cast<int>(X_num.rows());
    int n_num = static_cast<int>(num_col_indices.size());
    if (num_emb != "pwl" || n_num <= 0 || N <= 0) {
      B_out.resize(0, 0);
      F_out.resize(0, 0);
      return;
    }
    B_out.resize(N, n_num);
    F_out.resize(N, n_num);
    encode_ple_chunk(X_num.data(), static_cast<int>(X_num.outerStride()), 0, N, B_out, F_out);
  }

  void init_mlp(int out_dim_req, const std::string& task_req, int k_ens, int d_blk, int n_blks, double drop_rate, int seed) {
    task = task_req;
    output_dim = std::max(1, out_dim_req);
    k_ensemble = std::max(1, k_ens);
    d_block = std::max(1, d_blk);
    n_blocks = std::max(1, n_blks);
    dropout = std::clamp(drop_rate, 0.0, 0.99);

    std::mt19937 rng(static_cast<unsigned int>(seed + 101));

    // Multiplicative BatchEnsemble input adapter R_in (K x D_in) initialized Rademacher {-1, +1}
    R_in.init(k_ensemble, D_in);
    std::uniform_int_distribution<int> rad_dist(0, 1);
    for (int d = 0; d < D_in; ++d) {
      R_in.val(0, d) = 1.0f;
      for (int k = 1; k < k_ensemble; ++k) {
        R_in.val(k, d) = rad_dist(rng) ? 1.0f : -1.0f;
      }
    }

    blocks.resize(n_blocks);
    for (int l = 0; l < n_blocks; ++l) {
      int din_l = (l == 0) ? D_in : d_block;
      blocks[l].d_in = din_l;
      blocks[l].d_out = d_block;
      blocks[l].W.init(din_l, d_block);
      blocks[l].b.init(k_ensemble, d_block);
      float std_he = std::sqrt(2.0f / static_cast<float>(std::max(1, din_l)));
      std::normal_distribution<float> norm_dist(0.0f, std_he);
      for (int c = 0; c < d_block; ++c) {
        for (int r = 0; r < din_l; ++r) {
          blocks[l].W.val(r, c) = norm_dist(rng);
        }
      }
      blocks[l].b.val.setZero();
    }

    W_out.resize(k_ensemble);
    b_out.init(k_ensemble, output_dim);
    float xavier_bound = std::sqrt(6.0f / static_cast<float>(d_block + output_dim));
    std::uniform_real_distribution<float> out_dist(-xavier_bound, xavier_bound);
    for (int k = 0; k < k_ensemble; ++k) {
      W_out[k].init(d_block, output_dim);
      for (int c = 0; c < output_dim; ++c) {
        for (int r = 0; r < d_block; ++r) {
          W_out[k].val(r, c) = out_dist(rng);
        }
      }
    }
    b_out.val.setZero();
  }

  // Extract a contiguous slice [start_row .. start_row + cur_B) from raw ColMajor double matrix
  // into standardized FP32 X_num_buf and int32 X_cat_buf without duplicating the full matrix.
  void extract_raw_chunk(const double* x_raw_ptr, int N_total, int start_row, int cur_B,
                         Eigen::MatrixXf& X_num_buf, Eigen::MatrixXi& X_cat_buf) const {
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B >= 64)
#endif
    for (int j = 0; j < n_num; ++j) {
      const double* col_ptr = x_raw_ptr + static_cast<size_t>(num_col_indices[j]) * static_cast<size_t>(N_total) + start_row;
      double m = num_means[j];
      double inv_s = num_inv_stds[j];
      float* dst = X_num_buf.col(j).data();
      for (int i = 0; i < cur_B; ++i) {
        double v = col_ptr[i];
        double clean_v = std::isfinite(v) ? v : m;
        double z = (clean_v - m) * inv_s;
        dst[i] = static_cast<float>(std::clamp(z, -30.0, 30.0));
      }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_cat >= 2 && cur_B >= 64)
#endif
    for (int c = 0; c < n_cat; ++c) {
      int card = cat_cardinalities[c];
      const double* col_ptr = x_raw_ptr + static_cast<size_t>(cat_col_indices[c]) * static_cast<size_t>(N_total) + start_row;
      int* dst = X_cat_buf.col(c).data();
      for (int i = 0; i < cur_B; ++i) {
        double v = col_ptr[i];
        int code = card;
        if (std::isfinite(v)) {
          long iv = std::lround(v);
          if (iv >= 1 && iv <= card) {
            code = static_cast<int>(iv) - 1;
          }
        }
        dst[i] = code;
      }
    }
  }

  // Evaluate feature embeddings using optional precomputed PLE (bin_idx, bin_frac) and contiguous outer-e loops.
  // Set x_num_ptr == nullptr when num_emb != "pwl" and caller already wrote raw features into E_out[0..cur_B, 0..n_num).
  void forward_embeddings_prebinned(const float* x_num_ptr, int x_num_lda,
                                    const int* b_ptr, int b_lda,
                                    const float* f_ptr, int f_lda,
                                    const int* x_cat_ptr, int x_cat_lda,
                                    int row_offset, int cur_B,
                                    Eigen::MatrixXf& E_out) const {
    if (has_dummy_in) {
      E_out.block(0, 0, cur_B, 1).setZero();
      return;
    }

    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    int e_stride = static_cast<int>(E_out.outerStride());
    float* e_base = E_out.data();

    if (num_emb == "pwl" && n_num > 0 && b_ptr != nullptr && f_ptr != nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B >= 64)
#endif
      for (int j = 0; j < n_num; ++j) {
        const PLEFeature& pf = ple_features[j];
        if (!pf.active) continue;
        const int* b_col = b_ptr + static_cast<size_t>(j) * static_cast<size_t>(b_lda) + row_offset;
        const float* f_col = f_ptr + static_cast<size_t>(j) * static_cast<size_t>(f_lda) + row_offset;
        int d_e = pf.d_emb;
        int col_off = pf.out_col_offset;

        for (int e = 0; e < d_e; ++e) {
          const float* s_col = pf.S_T.col(e).data();
          const float* w_col = pf.W_T.col(e).data();
          float* e_col = e_base + static_cast<size_t>(col_off + e) * static_cast<size_t>(e_stride);
          for (int i = 0; i < cur_B; ++i) {
            const int b_star = b_col[i];
            e_col[i] = s_col[b_star] + f_col[i] * w_col[b_star];
          }
        }
      }
    } else if (n_num > 0 && x_num_ptr != nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B >= 64)
#endif
      for (int j = 0; j < n_num; ++j) {
        std::memcpy(e_base + static_cast<size_t>(j) * static_cast<size_t>(e_stride),
                    x_num_ptr + static_cast<size_t>(j) * static_cast<size_t>(x_num_lda) + row_offset,
                    static_cast<size_t>(cur_B) * sizeof(float));
      }
    }

    if (n_cat > 0 && x_cat_ptr != nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_cat >= 2 && cur_B >= 64)
#endif
      for (int c = 0; c < n_cat; ++c) {
        const Eigen::MatrixXf& tbl = cat_tables[c].val;
        const float* t_data = tbl.data();
        int t_stride = static_cast<int>(tbl.outerStride());
        int card = cat_cardinalities[c];
        const int* code_col = x_cat_ptr + static_cast<size_t>(c) * static_cast<size_t>(x_cat_lda) + row_offset;
        int col_off = D_num_rep + c * d_embedding;
        for (int i = 0; i < cur_B; ++i) {
          int code = code_col[i];
          if (code < 0 || code > card) code = card;
          const float* emb = t_data + code * t_stride;
          for (int e = 0; e < d_embedding; ++e) {
            e_base[i + static_cast<size_t>(col_off + e) * static_cast<size_t>(e_stride)] = emb[e];
          }
        }
      }
    }
  }

  // Forward feature embedding from dense batch X_num / X_cat (bins X_num on demand if pwl)
  void forward_embeddings(const Eigen::MatrixXf& X_num, const Eigen::MatrixXi& X_cat, int cur_B,
                          Eigen::MatrixXf& E_out,
                          Eigen::MatrixXi* bin_idx_cache = nullptr,
                          Eigen::MatrixXf* bin_frac_cache = nullptr) const {
    if (has_dummy_in) {
      E_out.block(0, 0, cur_B, 1).setZero();
      return;
    }
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    Eigen::MatrixXi tmp_b;
    Eigen::MatrixXf tmp_f;
    Eigen::MatrixXi* b_buf = bin_idx_cache;
    Eigen::MatrixXf* f_buf = bin_frac_cache;
    if (num_emb == "pwl" && n_num > 0) {
      if (b_buf == nullptr || f_buf == nullptr) {
        tmp_b.resize(cur_B, n_num);
        tmp_f.resize(cur_B, n_num);
        b_buf = &tmp_b;
        f_buf = &tmp_f;
      }
      encode_ple_chunk(X_num.data(), static_cast<int>(X_num.outerStride()), 0, cur_B, *b_buf, *f_buf);
    }
    forward_embeddings_prebinned(
      (n_num > 0) ? X_num.data() : nullptr, (n_num > 0) ? static_cast<int>(X_num.outerStride()) : 0,
      (b_buf != nullptr && n_num > 0) ? b_buf->data() : nullptr, (b_buf != nullptr && n_num > 0) ? static_cast<int>(b_buf->outerStride()) : 0,
      (f_buf != nullptr && n_num > 0) ? f_buf->data() : nullptr, (f_buf != nullptr && n_num > 0) ? static_cast<int>(f_buf->outerStride()) : 0,
      (n_cat > 0) ? X_cat.data() : nullptr, (n_cat > 0) ? static_cast<int>(X_cat.outerStride()) : 0,
      0, cur_B, E_out
    );
  }

  // Forward MLP from shared E (cur_B x D_in) -> Out ((K * cur_B) x out_dim)
  // H_bufs must have size n_blocks + 1 with matching outerStrides.
  void forward_mlp_from_emb(const Eigen::MatrixXf& E_in, int cur_B, bool is_train, int step_id,
                            std::vector<Eigen::MatrixXf>& H_bufs, Eigen::MatrixXf& Out_buf) const {
    int K = k_ensemble;
    int cur_KB = K * cur_B;

    // 1. Expand E (cur_B x D_in) * R_in (K x D_in) -> H[0] (cur_KB x D_in)
    Eigen::MatrixXf& H0 = H_bufs[0];
    int e_stride = static_cast<int>(E_in.outerStride());
    int h0_stride = static_cast<int>(H0.outerStride());
    const float* e_data = E_in.data();
    float* h0_data = H0.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (D_in >= 4 && cur_KB >= 512)
#endif
    for (int d = 0; d < D_in; ++d) {
      const float* e_col = e_data + d * e_stride;
      float* h_col = h0_data + d * h0_stride;
      for (int k = 0; k < K; ++k) {
        float r_kd = R_in.val(k, d);
        float* dst = h_col + k * cur_B;
        for (int i = 0; i < cur_B; ++i) {
          dst[i] = e_col[i] * r_kd;
        }
      }
    }

    // 2. Shared MLP blocks with per-member biases + ReLU + optional branchless 4-lane Dropout
    bool do_drop = (is_train && dropout > 0.0);
    float keep_prob = static_cast<float>(1.0 - dropout);
    float inv_keep = do_drop ? (1.0f / keep_prob) : 1.0f;
    uint32_t drop_thresh8 = static_cast<uint32_t>(std::clamp(1.0 - dropout, 0.0, 1.0) * 256.0);

    for (int l = 0; l < n_blocks; ++l) {
      const TabMBlock& blk = blocks[l];
      const Eigen::MatrixXf& H_prev = H_bufs[l];
      Eigen::MatrixXf& H_cur = H_bufs[l + 1];

      gemm_f32_raw(false, false, cur_KB, blk.d_out, blk.d_in,
                   H_prev.data(), static_cast<int>(H_prev.outerStride()),
                   blk.W.val.data(), static_cast<int>(blk.W.val.outerStride()),
                   H_cur.data(), static_cast<int>(H_cur.outerStride()),
                   1.0f, 0.0f);

      int h_stride = static_cast<int>(H_cur.outerStride());
      float* h_data = H_cur.data();
      int b_stride = static_cast<int>(blk.b.val.outerStride());
      const float* b_data = blk.b.val.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (blk.d_out >= 4 && cur_KB >= 512)
#endif
      for (int c = 0; c < blk.d_out; ++c) {
        float* h_col = h_data + c * h_stride;
        const float* b_col = b_data + c * b_stride;
        if (do_drop) {
          uint32_t rng_st = hash_seed32(static_cast<uint32_t>(step_id),
                                        static_cast<uint32_t>(l + 1),
                                        static_cast<uint32_t>(c + 1),
                                        0x9e3779b9u);
          if (rng_st == 0u) rng_st = 0x9e3779b9u;
          for (int k = 0; k < K; ++k) {
            float bk = b_col[k];
            float* dst = h_col + k * cur_B;
            int i = 0;
            for (; i + 3 < cur_B; i += 4) {
              uint32_t r = xorshift32_step(rng_st);
              uint32_t u0 = r & 0xFFu;
              uint32_t u1 = (r >> 8) & 0xFFu;
              uint32_t u2 = (r >> 16) & 0xFFu;
              uint32_t u3 = r >> 24;
              float z0 = dst[i]     + bk;
              float z1 = dst[i + 1] + bk;
              float z2 = dst[i + 2] + bk;
              float z3 = dst[i + 3] + bk;
              float v0 = (z0 > 0.0f) ? z0 : 0.0f;
              float v1 = (z1 > 0.0f) ? z1 : 0.0f;
              float v2 = (z2 > 0.0f) ? z2 : 0.0f;
              float v3 = (z3 > 0.0f) ? z3 : 0.0f;
              dst[i]     = (u0 < drop_thresh8) ? (v0 * inv_keep) : 0.0f;
              dst[i + 1] = (u1 < drop_thresh8) ? (v1 * inv_keep) : 0.0f;
              dst[i + 2] = (u2 < drop_thresh8) ? (v2 * inv_keep) : 0.0f;
              dst[i + 3] = (u3 < drop_thresh8) ? (v3 * inv_keep) : 0.0f;
            }
            for (; i < cur_B; ++i) {
              uint32_t u = xorshift32_step(rng_st) >> 24;
              float z = dst[i] + bk;
              float v = (z > 0.0f) ? z : 0.0f;
              dst[i] = (u < drop_thresh8) ? (v * inv_keep) : 0.0f;
            }
          }
        } else {
          for (int k = 0; k < K; ++k) {
            float bk = b_col[k];
            float* dst = h_col + k * cur_B;
            for (int i = 0; i < cur_B; ++i) {
              float v = dst[i] + bk;
              dst[i] = (v > 0.0f) ? v : 0.0f;
            }
          }
        }
      }
    }

    // 3. Per-member output heads
    const Eigen::MatrixXf& H_last = H_bufs[n_blocks];
    int hlast_stride = static_cast<int>(H_last.outerStride());
    int out_stride = static_cast<int>(Out_buf.outerStride());
    const float* hlast_data = H_last.data();
    float* out_data = Out_buf.data();

    if (output_dim == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (K >= 2 && cur_KB >= 1024)
#endif
      for (int k = 0; k < K; ++k) {
        float* dst = out_data + k * cur_B;
        float bk = b_out.val(k, 0);
        for (int i = 0; i < cur_B; ++i) {
          dst[i] = bk;
        }
        const float* w_k = W_out[k].val.data();
        const float* hk_base = hlast_data + k * cur_B;
        for (int d = 0; d < d_block; ++d) {
          float wd = w_k[d];
          const float* h_col = hk_base + static_cast<size_t>(d) * static_cast<size_t>(hlast_stride);
          for (int i = 0; i < cur_B; ++i) {
            dst[i] += h_col[i] * wd;
          }
        }
      }
    } else {
      for (int k = 0; k < K; ++k) {
        gemm_f32_raw(false, false, cur_B, output_dim, d_block,
                     hlast_data + k * cur_B, hlast_stride,
                     W_out[k].val.data(), static_cast<int>(W_out[k].val.outerStride()),
                     out_data + k * cur_B, out_stride,
                     1.0f, 0.0f);
        for (int c = 0; c < output_dim; ++c) {
          float bk = b_out.val(k, c);
          float* dst = out_data + c * out_stride + k * cur_B;
          for (int i = 0; i < cur_B; ++i) {
            dst[i] += bk;
          }
        }
      }
    }
  }

  // Average ensemble predictions for slice cur_B from Out_buf (cur_KB x out_dim) into preds_dst[start_row..start_row+cur_B)
  void ensemble_reduce_chunk(const Eigen::MatrixXf& Out_buf, int cur_B, int start_row,
                             Eigen::MatrixXd& preds_dst) const {
    int K = k_ensemble;
    double inv_K = 1.0 / static_cast<double>(K);
    int out_stride = static_cast<int>(Out_buf.outerStride());
    const float* out_data = Out_buf.data();

    if (task == "regression") {
      double ys = (std::isfinite(y_std) && y_std > 1e-8) ? y_std : 1.0;
      double ym = std::isfinite(y_mean) ? y_mean : 0.0;
      double* dst = preds_dst.col(0).data() + start_row;
      std::fill(dst, dst + cur_B, 0.0);
      for (int k = 0; k < K; ++k) {
        const float* src = out_data + k * cur_B;
        for (int i = 0; i < cur_B; ++i) {
          float z = std::clamp(src[i], -50.0f, 50.0f);
          dst[i] += static_cast<double>(z);
        }
      }
      double scale = inv_K * ys;
      for (int i = 0; i < cur_B; ++i) {
        dst[i] = dst[i] * scale + ym;
      }
    } else if (task == "classification") {
      double* dst = preds_dst.col(0).data() + start_row;
      std::fill(dst, dst + cur_B, 0.0);
      for (int k = 0; k < K; ++k) {
        const float* src = out_data + k * cur_B;
        for (int i = 0; i < cur_B; ++i) {
          float z = std::clamp(src[i], -30.0f, 30.0f);
          dst[i] += static_cast<double>(1.0f / (1.0f + evofe::nn::fast_expf(-z)));
        }
      }
      for (int i = 0; i < cur_B; ++i) {
        dst[i] *= inv_K;
      }
    } else {
      double* p_data = preds_dst.data();
      int p_stride = static_cast<int>(preds_dst.outerStride());
#if defined(_OPENMP)
#pragma omp parallel if (cur_B * K * output_dim >= 2048)
#endif
      {
        float stack_exps[64];
        std::vector<float> heap_exps;
        float* exps = (output_dim <= 64) ? stack_exps : (heap_exps.resize(output_dim), heap_exps.data());
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < cur_B; ++i) {
          for (int c = 0; c < output_dim; ++c) {
            p_data[start_row + i + static_cast<size_t>(c) * static_cast<size_t>(p_stride)] = 0.0;
          }
          for (int k = 0; k < K; ++k) {
            int r = k * cur_B + i;
            float max_v = out_data[r];
            for (int c = 1; c < output_dim; ++c) {
              float v = out_data[r + static_cast<size_t>(c) * static_cast<size_t>(out_stride)];
              if (v > max_v) max_v = v;
            }
            float sum_e = 0.0f;
            for (int c = 0; c < output_dim; ++c) {
              float ep = evofe::nn::fast_expf(out_data[r + static_cast<size_t>(c) * static_cast<size_t>(out_stride)] - max_v);
              exps[c] = ep;
              sum_e += ep;
            }
            if (sum_e > 0.0f) {
              double inv_sum = inv_K / static_cast<double>(sum_e);
              for (int c = 0; c < output_dim; ++c) {
                p_data[start_row + i + static_cast<size_t>(c) * static_cast<size_t>(p_stride)] += static_cast<double>(exps[c]) * inv_sum;
              }
            } else {
              double unif = inv_K / static_cast<double>(output_dim);
              for (int c = 0; c < output_dim; ++c) {
                p_data[start_row + i + static_cast<size_t>(c) * static_cast<size_t>(p_stride)] += unif;
              }
            }
          }
        }
      }
    }
  }

  // Evaluate raw column-major double matrix using chunked workspace (constant RAM on 650K rows)
  void predict_raw_chunked(const double* x_raw_ptr, int N_total, int n_cols_in, TabMInferWorkspace& ws) const {
    if (n_cols_in != n_features) {
      throw std::invalid_argument("Input matrix column count does not match TabM model n_features.");
    }
    if (N_total <= 0) {
      ws.preds.resize(0, output_dim);
      return;
    }
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    if (ws.chunk_cap <= 0 || ws.K != k_ensemble || ws.D_in != D_in ||
        ws.d_block != d_block || ws.n_blocks != n_blocks || ws.out_dim != output_dim ||
        ws.preds.rows() != N_total || ws.preds.cols() != output_dim) {
      ws.allocate(N_total, k_ensemble, D_in, d_block, n_blocks, output_dim, n_num, n_cat);
    }

    for (int start = 0; start < N_total; start += ws.chunk_cap) {
      int cur_B = std::min(ws.chunk_cap, N_total - start);
      extract_raw_chunk(x_raw_ptr, N_total, start, cur_B, ws.X_num_chunk, ws.X_cat_chunk);
      if (num_emb == "pwl" && n_num > 0) {
        encode_ple_chunk(ws.X_num_chunk.data(), static_cast<int>(ws.X_num_chunk.outerStride()),
                         0, cur_B, ws.bin_idx_chunk, ws.bin_frac_chunk);
      }
      forward_embeddings_prebinned(
        n_num > 0 ? ws.X_num_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.X_num_chunk.outerStride()) : 0,
        n_num > 0 ? ws.bin_idx_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.bin_idx_chunk.outerStride()) : 0,
        n_num > 0 ? ws.bin_frac_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.bin_frac_chunk.outerStride()) : 0,
        n_cat > 0 ? ws.X_cat_chunk.data() : nullptr, n_cat > 0 ? static_cast<int>(ws.X_cat_chunk.outerStride()) : 0,
        0, cur_B, ws.E_chunk
      );
      forward_mlp_from_emb(ws.E_chunk, cur_B, false, 0, ws.H, ws.Out);
      ensemble_reduce_chunk(ws.Out, cur_B, start, ws.preds);
    }
  }

  // Evaluate pre-standardized X_num / X_cat (and optional precomputed PLE bins) zero-copy across chunks
  void predict_preprocessed_chunked(const Eigen::MatrixXf& X_num, const Eigen::MatrixXi& X_cat,
                                    int N_total, TabMInferWorkspace& ws,
                                    const Eigen::MatrixXi* B_pre = nullptr,
                                    const Eigen::MatrixXf* F_pre = nullptr) const {
    if (N_total <= 0) {
      ws.preds.resize(0, output_dim);
      return;
    }
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    if (ws.chunk_cap <= 0 || ws.K != k_ensemble || ws.D_in != D_in ||
        ws.d_block != d_block || ws.n_blocks != n_blocks || ws.out_dim != output_dim ||
        ws.preds.rows() != N_total || ws.preds.cols() != output_dim) {
      ws.allocate(N_total, k_ensemble, D_in, d_block, n_blocks, output_dim, n_num, n_cat);
    }

    const bool has_pre = (num_emb == "pwl" && B_pre != nullptr && F_pre != nullptr &&
                          B_pre->rows() == N_total && F_pre->rows() == N_total);
    const int x_num_lda = (n_num > 0) ? static_cast<int>(X_num.outerStride()) : 0;
    const float* x_num_ptr = (n_num > 0) ? X_num.data() : nullptr;
    const int x_cat_lda = (n_cat > 0) ? static_cast<int>(X_cat.outerStride()) : 0;
    const int* x_cat_ptr = (n_cat > 0) ? X_cat.data() : nullptr;

    for (int start = 0; start < N_total; start += ws.chunk_cap) {
      int cur_B = std::min(ws.chunk_cap, N_total - start);
      if (has_pre) {
        forward_embeddings_prebinned(
          x_num_ptr, x_num_lda,
          n_num > 0 ? B_pre->data() : nullptr, n_num > 0 ? static_cast<int>(B_pre->outerStride()) : 0,
          n_num > 0 ? F_pre->data() : nullptr, n_num > 0 ? static_cast<int>(F_pre->outerStride()) : 0,
          x_cat_ptr, x_cat_lda,
          start, cur_B, ws.E_chunk
        );
      } else {
        if (num_emb == "pwl" && n_num > 0) {
          encode_ple_chunk(x_num_ptr, x_num_lda, start, cur_B, ws.bin_idx_chunk, ws.bin_frac_chunk);
        }
        forward_embeddings_prebinned(
          x_num_ptr, x_num_lda,
          n_num > 0 ? ws.bin_idx_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.bin_idx_chunk.outerStride()) : 0,
          n_num > 0 ? ws.bin_frac_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.bin_frac_chunk.outerStride()) : 0,
          x_cat_ptr, x_cat_lda,
          start, cur_B, ws.E_chunk
        );
      }
      forward_mlp_from_emb(ws.E_chunk, cur_B, false, 0, ws.H, ws.Out);
      ensemble_reduce_chunk(ws.Out, cur_B, start, ws.preds);
    }
  }

  // Single minibatch training step: gather perm[start..start+cur_B), forward, loss, backward, AdamW update.
  // Accepts optional precomputed training PLE bin matrices B_tr, F_tr to avoid repeated binary search per epoch.
  double train_step(const Eigen::MatrixXf& X_num_tr,
                    const Eigen::MatrixXi& X_cat_tr,
                    const Eigen::MatrixXf& Y_tr,
                    const std::vector<int>& perm,
                    int start,
                    int cur_B,
                    int step_id,
                    double cur_lr,
                    double weight_decay,
                    double beta1,
                    double beta2,
                    double eps,
                    TabMWorkspace& ws,
                    const Eigen::MatrixXi* B_tr = nullptr,
                    const Eigen::MatrixXf* F_tr = nullptr) {
    int K = k_ensemble;
    int cur_KB = K * cur_B;
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    const int* p_idx = perm.data() + start;
    const bool has_pre_pwl = (num_emb == "pwl" && B_tr != nullptr && F_tr != nullptr &&
                              B_tr->rows() == X_num_tr.rows() && F_tr->rows() == X_num_tr.rows());

    // 1. Gather only current minibatch cur_B rows directly into prebinned or embedding buffers
    if (n_num > 0) {
      if (has_pre_pwl) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B >= 64)
#endif
        for (int j = 0; j < n_num; ++j) {
          if (!ple_features[j].active) continue;
          const int* b_src = B_tr->col(j).data();
          const float* f_src = F_tr->col(j).data();
          int* b_dst = ws.bin_idx.col(j).data();
          float* f_dst = ws.bin_frac.col(j).data();
          for (int i = 0; i < cur_B; ++i) {
            int row = p_idx[i];
            b_dst[i] = b_src[row];
            f_dst[i] = f_src[row];
          }
        }
      } else if (num_emb == "pwl") {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B >= 64)
#endif
        for (int j = 0; j < n_num; ++j) {
          if (!ple_features[j].active) continue;
          const float* src = X_num_tr.col(j).data();
          float* dst = ws.batch_X_num.col(j).data();
          for (int i = 0; i < cur_B; ++i) {
            dst[i] = src[p_idx[i]];
          }
        }
        encode_ple_chunk(ws.batch_X_num.data(), static_cast<int>(ws.batch_X_num.outerStride()),
                         0, cur_B, ws.bin_idx, ws.bin_frac);
      } else {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B >= 64)
#endif
        for (int j = 0; j < n_num; ++j) {
          const float* src = X_num_tr.col(j).data();
          float* dst = ws.E_batch.col(j).data();
          for (int i = 0; i < cur_B; ++i) {
            dst[i] = src[p_idx[i]];
          }
        }
      }
    }

    if (n_cat > 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_cat >= 2 && cur_B >= 64)
#endif
      for (int c = 0; c < n_cat; ++c) {
        const int* src = X_cat_tr.col(c).data();
        int* dst = ws.batch_X_cat.col(c).data();
        for (int i = 0; i < cur_B; ++i) {
          dst[i] = src[p_idx[i]];
        }
      }
    }

    for (int c = 0; c < output_dim; ++c) {
      const float* src = Y_tr.col(c).data();
      float* dst = ws.batch_Y.col(c).data();
      for (int i = 0; i < cur_B; ++i) {
        dst[i] = src[p_idx[i]];
      }
    }

    // 2. Forward embeddings (skip raw numeric copy when num_emb != "pwl" since gathered straight into E_batch) + MLP
    forward_embeddings_prebinned(
      nullptr, 0,
      (num_emb == "pwl" && n_num > 0) ? ws.bin_idx.data() : nullptr,
      (num_emb == "pwl" && n_num > 0) ? static_cast<int>(ws.bin_idx.outerStride()) : 0,
      (num_emb == "pwl" && n_num > 0) ? ws.bin_frac.data() : nullptr,
      (num_emb == "pwl" && n_num > 0) ? static_cast<int>(ws.bin_frac.outerStride()) : 0,
      n_cat > 0 ? ws.batch_X_cat.data() : nullptr, n_cat > 0 ? static_cast<int>(ws.batch_X_cat.outerStride()) : 0,
      0, cur_B, ws.E_batch
    );
    forward_mlp_from_emb(ws.E_batch, cur_B, true, step_id, ws.H, ws.Out);

    // 3. Per-member TabM training loss, output gradient & in-loop gradient norm tracking
    double batch_loss = 0.0;
    double inv_KB = 1.0 / static_cast<double>(cur_KB);
    float inv_KB_f = static_cast<float>(inv_KB);
    int out_stride = static_cast<int>(ws.Out.outerStride());
    int g_stride = static_cast<int>(ws.grad_Out.outerStride());
    const float* out_ptr = ws.Out.data();
    float* g_ptr = ws.grad_Out.data();
    std::vector<float> max_g_k(K, 0.0f);

    if (task == "regression") {
      double loss_sum = 0.0;
      const float* y_ptr = ws.batch_Y.data();
      float two_inv_KB = 2.0f * inv_KB_f;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+:loss_sum) if (K >= 2 && cur_KB >= 2048)
#endif
      for (int k = 0; k < K; ++k) {
        int r0 = k * cur_B;
        double k_loss = 0.0;
        float k_max_g = 0.0f;
        for (int i = 0; i < cur_B; ++i) {
          float diff = out_ptr[r0 + i] - y_ptr[i];
          k_loss += static_cast<double>(diff) * static_cast<double>(diff);
          float gv = two_inv_KB * diff;
          g_ptr[r0 + i] = gv;
          float ag = std::abs(gv);
          if (ag > k_max_g) k_max_g = ag;
        }
        loss_sum += k_loss;
        max_g_k[k] = k_max_g;
      }
      batch_loss = loss_sum * inv_KB;
    } else if (task == "classification") {
      double loss_sum = 0.0;
      const float* y_ptr = ws.batch_Y.data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+:loss_sum) if (K >= 2 && cur_KB >= 2048)
#endif
      for (int k = 0; k < K; ++k) {
        int r0 = k * cur_B;
        double k_loss = 0.0;
        float k_max_g = 0.0f;
        for (int i = 0; i < cur_B; ++i) {
          float z = std::clamp(out_ptr[r0 + i], -30.0f, 30.0f);
          float p = 1.0f / (1.0f + evofe::nn::fast_expf(-z));
          float y = y_ptr[i];
          float gv = inv_KB_f * (p - y);
          g_ptr[r0 + i] = gv;
          float ag = std::abs(gv);
          if (ag > k_max_g) k_max_g = ag;
          float az = (z >= 0.0f) ? z : -z;
          float pos_z = (z > 0.0f) ? z : 0.0f;
          k_loss += static_cast<double>(pos_z - y * z + std::log1pf(evofe::nn::fast_expf(-az)));
        }
        loss_sum += k_loss;
        max_g_k[k] = k_max_g;
      }
      batch_loss = loss_sum * inv_KB;
    } else {
      double loss_sum = 0.0;
      int p_stride = static_cast<int>(ws.P.outerStride());
      float* p_ptr = ws.P.data();
      int y_stride = static_cast<int>(ws.batch_Y.outerStride());
      const float* y_ptr = ws.batch_Y.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(+:loss_sum) if (K >= 2 && cur_KB * output_dim >= 2048)
#endif
      for (int k = 0; k < K; ++k) {
        int r0 = k * cur_B;
        double k_loss = 0.0;
        float k_max_g = 0.0f;
        for (int i = 0; i < cur_B; ++i) {
          int r = r0 + i;
          float max_v = out_ptr[r];
          for (int c = 1; c < output_dim; ++c) {
            float v = out_ptr[r + static_cast<size_t>(c) * static_cast<size_t>(out_stride)];
            if (v > max_v) max_v = v;
          }
          float sum_e = 0.0f;
          for (int c = 0; c < output_dim; ++c) {
            float ep = evofe::nn::fast_expf(out_ptr[r + static_cast<size_t>(c) * static_cast<size_t>(out_stride)] - max_v);
            p_ptr[r + static_cast<size_t>(c) * static_cast<size_t>(p_stride)] = ep;
            sum_e += ep;
          }
          float inv_sum = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
          float unif = 1.0f / static_cast<float>(output_dim);
          for (int c = 0; c < output_dim; ++c) {
            float pr = (sum_e > 0.0f) ? (p_ptr[r + static_cast<size_t>(c) * static_cast<size_t>(p_stride)] * inv_sum) : unif;
            float yc = y_ptr[i + static_cast<size_t>(c) * static_cast<size_t>(y_stride)];
            if (yc > 0.0f) {
              k_loss -= static_cast<double>(yc) * std::log(std::max(1e-15, static_cast<double>(pr)));
            }
            float gv = inv_KB_f * (pr - yc);
            g_ptr[r + static_cast<size_t>(c) * static_cast<size_t>(g_stride)] = gv;
            float ag = std::abs(gv);
            if (ag > k_max_g) k_max_g = ag;
          }
        }
        loss_sum += k_loss;
        max_g_k[k] = k_max_g;
      }
      batch_loss = loss_sum * inv_KB;
    }

    float max_g = 0.0f;
    for (int k = 0; k < K; ++k) {
      if (max_g_k[k] > max_g) max_g = max_g_k[k];
    }
    if (std::isfinite(max_g) && max_g > 10.0f) {
      ws.grad_Out.topRows(cur_KB) *= (10.0f / max_g);
    }

    // 4. Backward through K output heads
    Eigen::MatrixXf* delta_cur = &ws.delta_a;
    Eigen::MatrixXf* delta_next = &ws.delta_b;

    const Eigen::MatrixXf& H_last = ws.H[n_blocks];
    int hlast_stride = static_cast<int>(H_last.outerStride());
    int dcur_stride = static_cast<int>(delta_cur->outerStride());
    const float* hlast_data = H_last.data();
    float* dcur_data = delta_cur->data();

    if (output_dim == 1) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (K >= 2 && cur_KB >= 1024)
#endif
      for (int k = 0; k < K; ++k) {
        const float* gk = g_ptr + k * cur_B;
        float bsum = 0.0f;
        for (int i = 0; i < cur_B; ++i) {
          bsum += gk[i];
        }
        ws.grad_b_out(k, 0) = bsum;

        const float* hk_base = hlast_data + k * cur_B;
        float* dk_base = dcur_data + k * cur_B;
        const float* w_k = W_out[k].val.data();
        float* gw_k = ws.grad_W_out[k].data();
        for (int d = 0; d < d_block; ++d) {
          float wd = w_k[d];
          const float* h_col = hk_base + static_cast<size_t>(d) * static_cast<size_t>(hlast_stride);
          float* d_col = dk_base + static_cast<size_t>(d) * static_cast<size_t>(dcur_stride);
          float gsum = 0.0f;
          for (int i = 0; i < cur_B; ++i) {
            float gi = gk[i];
            gsum += h_col[i] * gi;
            d_col[i] = gi * wd;
          }
          gw_k[d] = gsum;
        }
      }
    } else {
      for (int k = 0; k < K; ++k) {
        gemm_f32_raw(true, false, d_block, output_dim, cur_B,
                     hlast_data + k * cur_B, hlast_stride,
                     g_ptr + k * cur_B, g_stride,
                     ws.grad_W_out[k].data(), static_cast<int>(ws.grad_W_out[k].outerStride()),
                     1.0f, 0.0f);
        for (int c = 0; c < output_dim; ++c) {
          const float* gc = g_ptr + c * g_stride + k * cur_B;
          float sum = 0.0f;
          for (int i = 0; i < cur_B; ++i) sum += gc[i];
          ws.grad_b_out(k, c) = sum;
        }
        gemm_f32_raw(false, true, cur_B, d_block, output_dim,
                     g_ptr + k * cur_B, g_stride,
                     W_out[k].val.data(), static_cast<int>(W_out[k].val.outerStride()),
                     dcur_data + k * cur_B, dcur_stride,
                     1.0f, 0.0f);
      }
    }

    // 5. Backward through shared MLP blocks (branchless mask reconstruction from post-act H[l+1] > 0)
    float inv_keep = (dropout > 0.0) ? static_cast<float>(1.0 / (1.0 - dropout)) : 1.0f;

    for (int l = n_blocks - 1; l >= 0; --l) {
      const TabMBlock& blk = blocks[l];
      const Eigen::MatrixXf& H_out_l = ws.H[l + 1];
      const Eigen::MatrixXf& H_in_l = ws.H[l];
      int hout_stride = static_cast<int>(H_out_l.outerStride());
      const float* hout_data = H_out_l.data();
      dcur_stride = static_cast<int>(delta_cur->outerStride());
      dcur_data = delta_cur->data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (blk.d_out >= 4 && cur_KB >= 512)
#endif
      for (int c = 0; c < blk.d_out; ++c) {
        const float* h_col = hout_data + c * hout_stride;
        float* d_col = dcur_data + c * dcur_stride;
        for (int k = 0; k < K; ++k) {
          float bsum = 0.0f;
          int off = k * cur_B;
          const float* hk = h_col + off;
          float* dk = d_col + off;
          for (int i = 0; i < cur_B; ++i) {
            float keep = (hk[i] > 0.0f) ? inv_keep : 0.0f;
            float d = dk[i] * keep;
            dk[i] = d;
            bsum += d;
          }
          ws.grad_b_blocks[l](k, c) = bsum;
        }
      }

      gemm_f32_raw(true, false, blk.d_in, blk.d_out, cur_KB,
                   H_in_l.data(), static_cast<int>(H_in_l.outerStride()),
                   dcur_data, dcur_stride,
                   ws.grad_W_blocks[l].data(), static_cast<int>(ws.grad_W_blocks[l].outerStride()),
                   1.0f, 0.0f);

      gemm_f32_raw(false, true, cur_KB, blk.d_in, blk.d_out,
                   dcur_data, dcur_stride,
                   blk.W.val.data(), static_cast<int>(blk.W.val.outerStride()),
                   delta_next->data(), static_cast<int>(delta_next->outerStride()),
                   1.0f, 0.0f);

      std::swap(delta_cur, delta_next);
    }

    // Now delta_cur holds delta_H0 (cur_KB x D_in)
    // 6. Backward through BatchEnsemble R_in (K x D_in) -> grad_R_in & grad_E
    const float* dH0_data = delta_cur->data();
    int dH0_stride = static_cast<int>(delta_cur->outerStride());
    const float* e_data = ws.E_batch.data();
    int e_stride = static_cast<int>(ws.E_batch.outerStride());
    float* ge_data = ws.grad_E.data();
    int ge_stride = static_cast<int>(ws.grad_E.outerStride());

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (D_in >= 4 && cur_KB >= 512)
#endif
    for (int d = 0; d < D_in; ++d) {
      const float* dh_col = dH0_data + d * dH0_stride;
      const float* e_col = e_data + d * e_stride;
      float* ge_col = ge_data + d * ge_stride;
      for (int i = 0; i < cur_B; ++i) {
        ge_col[i] = 0.0f;
      }
      for (int k = 0; k < K; ++k) {
        float r_kd = R_in.val(k, d);
        const float* dh_k = dh_col + k * cur_B;
        float sum_r = 0.0f;
        for (int i = 0; i < cur_B; ++i) {
          float dh = dh_k[i];
          sum_r += dh * e_col[i];
          ge_col[i] += dh * r_kd;
        }
        ws.grad_R_in(k, d) = sum_r;
      }
    }

    // 7. AdamW updates (decoupled weight_decay on weights/embeddings; 0.0 on biases and R_in)
    double b1_corr = 1.0 - std::pow(beta1, step_id);
    double b2_corr = 1.0 - std::pow(beta2, step_id);

    R_in.update(ws.grad_R_in, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, 0.0);
    for (int l = 0; l < n_blocks; ++l) {
      blocks[l].W.update(ws.grad_W_blocks[l], cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay);
      blocks[l].b.update_serial(ws.grad_b_blocks[l], cur_lr, beta1, beta2, eps, b1_corr, b2_corr, 0.0);
    }
    for (int k = 0; k < K; ++k) {
      W_out[k].update_serial(ws.grad_W_out[k], cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay);
    }
    b_out.update_serial(ws.grad_b_out, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, 0.0);

    // 8. Backward & AdamW for PLE features (unit-stride outer-e suffix-sum gradient) + Categorical tables
    if (!has_dummy_in) {
      if (num_emb == "pwl" && n_num > 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2)
#endif
        for (int j = 0; j < n_num; ++j) {
          PLEFeature& pf = ple_features[j];
          if (!pf.active) continue;
          Eigen::MatrixXf& gW = ws.ple_grad_W[j];
          Eigen::MatrixXf& gW_T = ws.ple_grad_W_T[j];
          Eigen::MatrixXf& gb = ws.ple_grad_b[j];
          Eigen::MatrixXf& diff = ws.ple_diff[j];
          gW_T.setZero();
          diff.setZero();

          const int* b_col = ws.bin_idx.col(j).data();
          const float* f_col = ws.bin_frac.col(j).data();
          int d_e = pf.d_emb;
          int col_off = pf.out_col_offset;
          float* gW_T_data = gW_T.data();
          int gWT_stride = static_cast<int>(gW_T.outerStride());
          float* diff_data = diff.data();
          int diff_stride = static_cast<int>(diff.outerStride());

          for (int e = 0; e < d_e; ++e) {
            const float* ge_col = ge_data + static_cast<size_t>(col_off + e) * static_cast<size_t>(ge_stride);
            float* gw_col = gW_T_data + static_cast<size_t>(e) * static_cast<size_t>(gWT_stride);
            float* df_col = diff_data + static_cast<size_t>(e) * static_cast<size_t>(diff_stride);
            for (int i = 0; i < cur_B; ++i) {
              int b_star = b_col[i];
              float g = ge_col[i];
              gw_col[b_star] += f_col[i] * g;
              df_col[b_star] += g;
            }
            float acc = 0.0f;
            for (int r = pf.n_bins - 1; r >= 1; --r) {
              acc += df_col[r];
              gw_col[r - 1] += acc;
            }
            gb(e, 0) = acc + df_col[0];
          }
          gW.noalias() = gW_T.transpose();
          pf.W.update_serial(gW, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay);
          pf.b.update_serial(gb, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, 0.0);
          pf.rebuild_prefix();
        }
      }

      if (n_cat > 0) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_cat >= 2)
#endif
        for (int c = 0; c < n_cat; ++c) {
          Eigen::MatrixXf& gE = ws.cat_grad_E[c];
          gE.setZero();
          const int* code_col = ws.batch_X_cat.col(c).data();
          int gE_stride = static_cast<int>(gE.outerStride());
          float* gE_data = gE.data();
          int col_off = D_num_rep + c * d_embedding;
          for (int i = 0; i < cur_B; ++i) {
            int code = code_col[i];
            float* dst = gE_data + code * gE_stride;
            for (int e = 0; e < d_embedding; ++e) {
              dst[e] += ge_data[i + (col_off + e) * ge_stride];
            }
          }
          cat_tables[c].update_serial(gE, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay);
        }
      }
    }

    return batch_loss;
  }

  // Occlusion importance in shared representation space E mapped back to original D columns
  std::vector<double> compute_importances(const Eigen::MatrixXf& X_num_src,
                                          const Eigen::MatrixXi& X_cat_src,
                                          const double* y_eval,
                                          int N_eval) const {
    int D = n_features;
    std::vector<double> imp(D, 1.0 / std::max(1, D));
    if (D <= 0 || N_eval <= 0 || y_eval == nullptr || has_dummy_in) {
      return imp;
    }

    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());

    TabMInferWorkspace ws;
    ws.allocate(N_eval, k_ensemble, D_in, d_block, n_blocks, output_dim, n_num, n_cat, N_eval);

    // Base embeddings for all N_eval rows (capped <= 1000 so single chunk!)
    Eigen::MatrixXf E_base(N_eval, D_in);
    const float* xn_ptr = (n_num > 0) ? X_num_src.data() : nullptr;
    const int xn_lda = (n_num > 0) ? static_cast<int>(X_num_src.outerStride()) : 0;
    const int* xc_ptr = (n_cat > 0) ? X_cat_src.data() : nullptr;
    const int xc_lda = (n_cat > 0) ? static_cast<int>(X_cat_src.outerStride()) : 0;
    if (num_emb == "pwl" && n_num > 0) {
      encode_ple_chunk(xn_ptr, xn_lda, 0, N_eval, ws.bin_idx_chunk, ws.bin_frac_chunk);
    }
    forward_embeddings_prebinned(
      xn_ptr, xn_lda,
      n_num > 0 ? ws.bin_idx_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.bin_idx_chunk.outerStride()) : 0,
      n_num > 0 ? ws.bin_frac_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.bin_frac_chunk.outerStride()) : 0,
      xc_ptr, xc_lda, 0, N_eval, E_base
    );

    Eigen::MatrixXf E_occ = E_base;
    Eigen::MatrixXd base_preds(N_eval, output_dim);
    forward_mlp_from_emb(E_base, N_eval, false, 0, ws.H, ws.Out);
    ensemble_reduce_chunk(ws.Out, N_eval, 0, base_preds);

    auto eval_loss = [&](const Eigen::MatrixXd& preds) -> double {
      if (task == "regression") {
        double ss = 0.0;
        for (int i = 0; i < N_eval; ++i) {
          double d = preds(i, 0) - y_eval[i];
          ss += d * d;
        }
        return ss / N_eval;
      } else if (task == "classification") {
        double ll = 0.0;
        for (int i = 0; i < N_eval; ++i) {
          double p = std::clamp(preds(i, 0), 1e-15, 1.0 - 1e-15);
          double y = (y_eval[i] > 0.0) ? 1.0 : 0.0;
          ll -= (y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        }
        return ll / N_eval;
      } else {
        double ll = 0.0;
        for (int i = 0; i < N_eval; ++i) {
          int c = static_cast<int>(y_eval[i]);
          double p = 1e-15;
          if (c >= 0 && c < output_dim) {
            p = std::clamp(preds(i, c), 1e-15, 1.0);
          }
          ll -= std::log(p);
        }
        return ll / N_eval;
      }
    };

    double base_loss = eval_loss(base_preds);

    auto score_slice = [&](int col_start, int width) -> double {
      if (width <= 0 || col_start < 0 || col_start + width > D_in) return 0.0;
      Eigen::RowVectorXf mean_slice = E_base.block(0, col_start, N_eval, width).colwise().mean();
      E_occ.block(0, col_start, N_eval, width).rowwise() = mean_slice;
      forward_mlp_from_emb(E_occ, N_eval, false, 0, ws.H, ws.Out);
      ensemble_reduce_chunk(ws.Out, N_eval, 0, ws.preds);
      double occ_loss = eval_loss(ws.preds);
      double pred_shift = (ws.preds.topRows(N_eval) - base_preds).cwiseAbs().mean();
      E_occ.block(0, col_start, N_eval, width) = E_base.block(0, col_start, N_eval, width);
      return std::max(0.0, occ_loss - base_loss) + 0.1 * pred_shift;
    };

    std::fill(imp.begin(), imp.end(), 0.0);
    for (int j = 0; j < n_num; ++j) {
      int orig_col = num_col_indices[j];
      if (num_emb == "pwl") {
        if (ple_features[j].active) {
          imp[orig_col] = score_slice(ple_features[j].out_col_offset, d_embedding);
        }
      } else {
        imp[orig_col] = score_slice(j, 1);
      }
    }
    for (int c = 0; c < n_cat; ++c) {
      int orig_col = cat_col_indices[c];
      imp[orig_col] = score_slice(D_num_rep + c * d_embedding, d_embedding);
    }

    double sum_imp = 0.0;
    for (double v : imp) sum_imp += v;
    if (sum_imp > 1e-12 && std::isfinite(sum_imp)) {
      for (int d = 0; d < D; ++d) imp[d] /= sum_imp;
    } else {
      std::fill(imp.begin(), imp.end(), 1.0 / D);
    }
    return imp;
  }
};

} // namespace tabm
} // namespace evofe

#endif // EVOFE_TABM_CORE_H
