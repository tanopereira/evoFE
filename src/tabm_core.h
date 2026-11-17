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
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace evofe {
namespace tabm {

using evofe::nn::AdamParam;
using evofe::nn::FiniteColumnStats;
using evofe::nn::compute_finite_column_stats;
using evofe::nn::cosine_warmup_schedule;
using evofe::nn::gemm_f32_raw;

constexpr double MAX_FLOAT32_VAL = 3.402823e38;

// RAII guard forcing single-threaded BLAS/Eigen inside TabM so outer OpenMP over k_ensemble
// drives Apple AMX / multi-core execution without Accelerate GCD or OpenBLAS thread contention.
struct TabMBlasSingleThreadGuard {
  int old_eigen = -1;
  int old_openblas = -1;
  int old_accelerate = -1;
  bool active = false;

  explicit TabMBlasSingleThreadGuard(bool enable = true) noexcept {
    if (!enable) return;
    active = true;
    old_eigen = Eigen::nbThreads();
    if (old_eigen != 1) Eigen::setNbThreads(1);
#if !defined(_WIN32)
    typedef int (*ob_get_fn)();
    typedef void (*ob_set_fn)(int);
    ob_get_fn g_ob = reinterpret_cast<ob_get_fn>(dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
    ob_set_fn s_ob = reinterpret_cast<ob_set_fn>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    if (g_ob && s_ob) {
      old_openblas = g_ob();
      if (old_openblas != 1) s_ob(1);
    }
#endif
#if defined(__APPLE__)
    typedef int (*acc_get_fn)();
    typedef int (*acc_set_fn)(int);
    acc_get_fn g_acc = reinterpret_cast<acc_get_fn>(dlsym(RTLD_DEFAULT, "BLASGetThreading"));
    acc_set_fn s_acc = reinterpret_cast<acc_set_fn>(dlsym(RTLD_DEFAULT, "BLASSetThreading"));
    if (g_acc && s_acc) {
      old_accelerate = g_acc();
      if (old_accelerate != 1) s_acc(1);  // BLAS_THREADING_SINGLE_THREADED = 1
    }
#endif
  }

  ~TabMBlasSingleThreadGuard() {
    if (!active) return;
    if (old_eigen > 0 && old_eigen != 1) {
      Eigen::setNbThreads(old_eigen);
    }
#if !defined(_WIN32)
    if (old_openblas > 0 && old_openblas != 1) {
      typedef void (*ob_set_fn)(int);
      ob_set_fn s_ob = reinterpret_cast<ob_set_fn>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
      if (s_ob) s_ob(old_openblas);
    }
#endif
#if defined(__APPLE__)
    if (old_accelerate >= 0 && old_accelerate != 1) {
      typedef int (*acc_set_fn)(int);
      acc_set_fn s_acc = reinterpret_cast<acc_set_fn>(dlsym(RTLD_DEFAULT, "BLASSetThreading"));
      if (s_acc) s_acc(old_accelerate);
    }
#endif
  }
};

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

// Vectorized ReLU + bias + optional deterministic dropout for a single ensemble member k
inline void apply_relu_dropout_fwd_k(float* __restrict__ h_data, int h_lda,
                                     const float* __restrict__ b_col,
                                     int cur_B, int d_out,
                                     bool do_drop, float inv_keep, uint32_t drop_thresh8,
                                     int step_id, int layer_id, int k_id) noexcept {
  if (!do_drop) {
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float32x4_t vzero = vdupq_n_f32(0.0f);
    for (int c = 0; c < d_out; ++c) {
      float bk = b_col[c];
      const float32x4_t vbk = vdupq_n_f32(bk);
      float* dst = h_data + static_cast<size_t>(c) * static_cast<size_t>(h_lda);
      int i = 0;
      for (; i + 15 < cur_B; i += 16) {
        float32x4_t v0 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i), vbk), vzero);
        float32x4_t v1 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i + 4), vbk), vzero);
        float32x4_t v2 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i + 8), vbk), vzero);
        float32x4_t v3 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i + 12), vbk), vzero);
        vst1q_f32(dst + i, v0);
        vst1q_f32(dst + i + 4, v1);
        vst1q_f32(dst + i + 8, v2);
        vst1q_f32(dst + i + 12, v3);
      }
      for (; i + 3 < cur_B; i += 4) {
        float32x4_t v0 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i), vbk), vzero);
        vst1q_f32(dst + i, v0);
      }
      for (; i < cur_B; ++i) {
        float v = dst[i] + bk;
        dst[i] = (v > 0.0f) ? v : 0.0f;
      }
    }
#else
    for (int c = 0; c < d_out; ++c) {
      float bk = b_col[c];
      float* dst = h_data + static_cast<size_t>(c) * static_cast<size_t>(h_lda);
      for (int i = 0; i < cur_B; ++i) {
        float v = dst[i] + bk;
        dst[i] = (v > 0.0f) ? v : 0.0f;
      }
    }
#endif
    return;
  }

  uint32_t sd[4] = {
    hash_seed32(static_cast<uint32_t>(step_id), static_cast<uint32_t>(layer_id), static_cast<uint32_t>(k_id), 0x9e3779b9u),
    hash_seed32(static_cast<uint32_t>(step_id), static_cast<uint32_t>(layer_id), static_cast<uint32_t>(k_id), 0x85ebca6bu),
    hash_seed32(static_cast<uint32_t>(step_id), static_cast<uint32_t>(layer_id), static_cast<uint32_t>(k_id), 0xc2b2ae35u),
    hash_seed32(static_cast<uint32_t>(step_id), static_cast<uint32_t>(layer_id), static_cast<uint32_t>(k_id), 0x27d4eb2fu)
  };
#if defined(__ARM_NEON) || defined(__aarch64__)
  const float32x4_t vzero = vdupq_n_f32(0.0f);
  const float32x4_t vinv = vdupq_n_f32(inv_keep);
  const uint32x4_t vth8 = vdupq_n_u32(drop_thresh8);
  const uint32x4_t vff = vdupq_n_u32(0xFFu);
  uint32x4_t s0 = vld1q_u32(sd);

  for (int c = 0; c < d_out; ++c) {
    float bk = b_col[c];
    const float32x4_t vbk = vdupq_n_f32(bk);
    float* dst = h_data + static_cast<size_t>(c) * static_cast<size_t>(h_lda);
    int i = 0;
    for (; i + 15 < cur_B; i += 16) {
      s0 = veorq_u32(s0, vshlq_n_u32(s0, 13));
      s0 = veorq_u32(s0, vshrq_n_u32(s0, 17));
      s0 = veorq_u32(s0, vshlq_n_u32(s0, 5));
      uint32x4_t m0 = vcltq_u32(vandq_u32(s0, vff), vth8);
      uint32x4_t m1 = vcltq_u32(vandq_u32(vshrq_n_u32(s0, 8), vff), vth8);
      uint32x4_t m2 = vcltq_u32(vandq_u32(vshrq_n_u32(s0, 16), vff), vth8);
      uint32x4_t m3 = vcltq_u32(vshrq_n_u32(s0, 24), vth8);
      float32x4_t v0 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i), vbk), vzero);
      float32x4_t v1 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i + 4), vbk), vzero);
      float32x4_t v2 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i + 8), vbk), vzero);
      float32x4_t v3 = vmaxq_f32(vaddq_f32(vld1q_f32(dst + i + 12), vbk), vzero);
      vst1q_f32(dst + i,      vreinterpretq_f32_u32(vandq_u32(m0, vreinterpretq_u32_f32(vmulq_f32(v0, vinv)))));
      vst1q_f32(dst + i + 4,  vreinterpretq_f32_u32(vandq_u32(m1, vreinterpretq_u32_f32(vmulq_f32(v1, vinv)))));
      vst1q_f32(dst + i + 8,  vreinterpretq_f32_u32(vandq_u32(m2, vreinterpretq_u32_f32(vmulq_f32(v2, vinv)))));
      vst1q_f32(dst + i + 12, vreinterpretq_f32_u32(vandq_u32(m3, vreinterpretq_u32_f32(vmulq_f32(v3, vinv)))));
    }
    if (i < cur_B) {
      uint32_t tail_rng = vgetq_lane_u32(s0, 0);
      for (; i < cur_B; ++i) {
        uint32_t u = xorshift32_step(tail_rng) >> 24;
        float z = dst[i] + bk;
        float v = (z > 0.0f) ? z : 0.0f;
        dst[i] = (u < drop_thresh8) ? (v * inv_keep) : 0.0f;
      }
      s0 = vsetq_lane_u32(tail_rng, s0, 0);
    }
  }
#else
  uint32_t s0 = sd[0], s1 = sd[1], s2 = sd[2], s3 = sd[3];
  for (int c = 0; c < d_out; ++c) {
    float bk = b_col[c];
    float* dst = h_data + static_cast<size_t>(c) * static_cast<size_t>(h_lda);
    int i = 0;
    for (; i + 15 < cur_B; i += 16) {
      uint32_t r0 = xorshift32_step(s0);
      uint32_t r1 = xorshift32_step(s1);
      uint32_t r2 = xorshift32_step(s2);
      uint32_t r3 = xorshift32_step(s3);
      uint32_t rs[4] = { r0, r1, r2, r3 };
      for (int lane = 0; lane < 4; ++lane) {
        uint32_t r = rs[lane];
        int base_i = i + lane * 4;
        float z0 = dst[base_i]     + bk;
        float z1 = dst[base_i + 1] + bk;
        float z2 = dst[base_i + 2] + bk;
        float z3 = dst[base_i + 3] + bk;
        float v0 = (z0 > 0.0f) ? (z0 * inv_keep) : 0.0f;
        float v1 = (z1 > 0.0f) ? (z1 * inv_keep) : 0.0f;
        float v2 = (z2 > 0.0f) ? (z2 * inv_keep) : 0.0f;
        float v3 = (z3 > 0.0f) ? (z3 * inv_keep) : 0.0f;
        dst[base_i]     = ((r & 0xFFu) < drop_thresh8) ? v0 : 0.0f;
        dst[base_i + 1] = (((r >> 8) & 0xFFu) < drop_thresh8) ? v1 : 0.0f;
        dst[base_i + 2] = (((r >> 16) & 0xFFu) < drop_thresh8) ? v2 : 0.0f;
        dst[base_i + 3] = ((r >> 24) < drop_thresh8) ? v3 : 0.0f;
      }
    }
    for (; i < cur_B; ++i) {
      uint32_t u = xorshift32_step(s0) >> 24;
      float z = dst[i] + bk;
      float v = (z > 0.0f) ? z : 0.0f;
      dst[i] = (u < drop_thresh8) ? (v * inv_keep) : 0.0f;
    }
  }
#endif
}

// Vectorized ReLU+dropout mask backward + bias gradient accumulation for member k
inline void apply_relu_dropout_bwd_k(const float* __restrict__ h_data, int h_lda,
                                     float* __restrict__ d_data, int d_lda,
                                     float* __restrict__ gb_col,
                                     int cur_B, int d_out, float inv_keep) noexcept {
#if defined(__ARM_NEON) || defined(__aarch64__)
  const float32x4_t vzero = vdupq_n_f32(0.0f);
  const float32x4_t vinv = vdupq_n_f32(inv_keep);
  const bool no_scale = (inv_keep == 1.0f);
  for (int c = 0; c < d_out; ++c) {
    const float* hk = h_data + static_cast<size_t>(c) * static_cast<size_t>(h_lda);
    float* dk = d_data + static_cast<size_t>(c) * static_cast<size_t>(d_lda);
    float32x4_t s0 = vzero, s1 = vzero, s2 = vzero, s3 = vzero;
    int i = 0;
    if (no_scale) {
      for (; i + 15 < cur_B; i += 16) {
        uint32x4_t m0 = vcgtq_f32(vld1q_f32(hk + i), vzero);
        uint32x4_t m1 = vcgtq_f32(vld1q_f32(hk + i + 4), vzero);
        uint32x4_t m2 = vcgtq_f32(vld1q_f32(hk + i + 8), vzero);
        uint32x4_t m3 = vcgtq_f32(vld1q_f32(hk + i + 12), vzero);
        float32x4_t d0 = vreinterpretq_f32_u32(vandq_u32(m0, vreinterpretq_u32_f32(vld1q_f32(dk + i))));
        float32x4_t d1 = vreinterpretq_f32_u32(vandq_u32(m1, vreinterpretq_u32_f32(vld1q_f32(dk + i + 4))));
        float32x4_t d2 = vreinterpretq_f32_u32(vandq_u32(m2, vreinterpretq_u32_f32(vld1q_f32(dk + i + 8))));
        float32x4_t d3 = vreinterpretq_f32_u32(vandq_u32(m3, vreinterpretq_u32_f32(vld1q_f32(dk + i + 12))));
        vst1q_f32(dk + i, d0);
        vst1q_f32(dk + i + 4, d1);
        vst1q_f32(dk + i + 8, d2);
        vst1q_f32(dk + i + 12, d3);
        s0 = vaddq_f32(s0, d0);
        s1 = vaddq_f32(s1, d1);
        s2 = vaddq_f32(s2, d2);
        s3 = vaddq_f32(s3, d3);
      }
    } else {
      for (; i + 15 < cur_B; i += 16) {
        uint32x4_t m0 = vcgtq_f32(vld1q_f32(hk + i), vzero);
        uint32x4_t m1 = vcgtq_f32(vld1q_f32(hk + i + 4), vzero);
        uint32x4_t m2 = vcgtq_f32(vld1q_f32(hk + i + 8), vzero);
        uint32x4_t m3 = vcgtq_f32(vld1q_f32(hk + i + 12), vzero);
        float32x4_t d0 = vreinterpretq_f32_u32(vandq_u32(m0, vreinterpretq_u32_f32(vmulq_f32(vld1q_f32(dk + i), vinv))));
        float32x4_t d1 = vreinterpretq_f32_u32(vandq_u32(m1, vreinterpretq_u32_f32(vmulq_f32(vld1q_f32(dk + i + 4), vinv))));
        float32x4_t d2 = vreinterpretq_f32_u32(vandq_u32(m2, vreinterpretq_u32_f32(vmulq_f32(vld1q_f32(dk + i + 8), vinv))));
        float32x4_t d3 = vreinterpretq_f32_u32(vandq_u32(m3, vreinterpretq_u32_f32(vmulq_f32(vld1q_f32(dk + i + 12), vinv))));
        vst1q_f32(dk + i, d0);
        vst1q_f32(dk + i + 4, d1);
        vst1q_f32(dk + i + 8, d2);
        vst1q_f32(dk + i + 12, d3);
        s0 = vaddq_f32(s0, d0);
        s1 = vaddq_f32(s1, d1);
        s2 = vaddq_f32(s2, d2);
        s3 = vaddq_f32(s3, d3);
      }
    }
    float bsum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < cur_B; ++i) {
      float dv = (hk[i] > 0.0f) ? (dk[i] * inv_keep) : 0.0f;
      dk[i] = dv;
      bsum += dv;
    }
    gb_col[c] = bsum;
  }
#else
  for (int c = 0; c < d_out; ++c) {
    const float* hk = h_data + static_cast<size_t>(c) * static_cast<size_t>(h_lda);
    float* dk = d_data + static_cast<size_t>(c) * static_cast<size_t>(d_lda);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    int i = 0;
    for (; i + 7 < cur_B; i += 8) {
      float d0 = (hk[i]     > 0.0f) ? (dk[i]     * inv_keep) : 0.0f;
      float d1 = (hk[i + 1] > 0.0f) ? (dk[i + 1] * inv_keep) : 0.0f;
      float d2 = (hk[i + 2] > 0.0f) ? (dk[i + 2] * inv_keep) : 0.0f;
      float d3 = (hk[i + 3] > 0.0f) ? (dk[i + 3] * inv_keep) : 0.0f;
      float d4 = (hk[i + 4] > 0.0f) ? (dk[i + 4] * inv_keep) : 0.0f;
      float d5 = (hk[i + 5] > 0.0f) ? (dk[i + 5] * inv_keep) : 0.0f;
      float d6 = (hk[i + 6] > 0.0f) ? (dk[i + 6] * inv_keep) : 0.0f;
      float d7 = (hk[i + 7] > 0.0f) ? (dk[i + 7] * inv_keep) : 0.0f;
      dk[i] = d0; dk[i + 1] = d1; dk[i + 2] = d2; dk[i + 3] = d3;
      dk[i + 4] = d4; dk[i + 5] = d5; dk[i + 6] = d6; dk[i + 7] = d7;
      a0 += d0; a1 += d1; a2 += d2; a3 += d3;
      a4 += d4; a5 += d5; a6 += d6; a7 += d7;
    }
    float bsum = ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7));
    for (; i < cur_B; ++i) {
      float dv = (hk[i] > 0.0f) ? (dk[i] * inv_keep) : 0.0f;
      dk[i] = dv;
      bsum += dv;
    }
    gb_col[c] = bsum;
  }
#endif
}

// Single numeric feature Piecewise Linear Encoding + learnable linear projection.
// Native ColMajor layout: W is (n_bins x d_emb), S is ((n_bins + 1) x d_emb) so
// lookup and gradient accumulation over bins are unit-stride without any per-batch transpose.
struct PLEFeature {
  std::vector<float> edges;       // length n_bins + 1, strictly increasing
  std::vector<float> inv_widths;  // length n_bins
  AdamParam W;                    // n_bins x d_emb
  AdamParam b;                    // d_emb x 1
  Eigen::MatrixXf S;              // (n_bins + 1) x d_emb, S(0, e) = b(e), S(r, e) = b(e) + sum_{0..r-1} W(j, e)
  int n_bins = 0;
  int d_emb = 0;
  int out_col_offset = -1;        // column offset into E_batch (-1 if constant/inactive)
  bool active = false;

  void rebuild_prefix() {
    if (!active || n_bins <= 0 || d_emb <= 0) return;
    if (S.rows() != n_bins + 1 || S.cols() != d_emb) {
      S.resize(n_bins + 1, d_emb);
    }
    for (int e = 0; e < d_emb; ++e) {
      float* s_col = S.col(e).data();
      const float* w_col = W.val.col(e).data();
      float acc = b.val(e, 0);
      s_col[0] = acc;
      for (int r = 0; r < n_bins; ++r) {
        acc += w_col[r];
        s_col[r + 1] = acc;
      }
    }
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
    W.init(n_bins, d_emb);
    b.init(d_emb, 1);

    float bound = 1.0f / std::sqrt(static_cast<float>(std::max(1, n_bins)));
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (int r = 0; r < n_bins; ++r) {
      for (int e = 0; e < d_emb; ++e) {
        W.val(r, e) = dist(rng);
      }
    }
    b.val.setZero();

    // Channel 0 preserves standardized input x cleanly over [edges[0], edges[n_bins]] at init;
    // remaining random PWL channels are zero-centered across uniform training quantiles.
    int e_center_start = 0;
    if (d_emb >= 2) {
      for (int r = 0; r < n_bins; ++r) {
        W.val(r, 0) = edges[r + 1] - edges[r];
      }
      b.val(0, 0) = edges[0];
      e_center_start = 1;
    }
    for (int e = e_center_start; e < d_emb; ++e) {
      double mean_s = 0.0;
      double acc = 0.0;
      for (int r = 0; r < n_bins; ++r) {
        double w = static_cast<double>(W.val(r, e));
        mean_s += acc + 0.5 * w;
        acc += w;
      }
      b.val(e, 0) = static_cast<float>(-mean_s / static_cast<double>(n_bins));
    }

    active = true;
    rebuild_prefix();
  }

  void init_from_saved(const std::vector<float>& saved_edges,
                       const Eigen::MatrixXf& saved_W,
                       const Eigen::MatrixXf& saved_b) {
    edges = saved_edges;
    if (edges.size() < 2 || saved_W.rows() != static_cast<int>(edges.size()) - 1) {
      active = false;
      n_bins = 0;
      return;
    }
    n_bins = static_cast<int>(edges.size()) - 1;
    d_emb = static_cast<int>(saved_W.cols());
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
      const float* base = edges.data() + 1;
      int len = n_bins;
      while (len > 1) {
        int half = len >> 1;
        base += (base[half - 1] <= x) ? half : 0;
        len -= half;
      }
      r = static_cast<int>(base - edges.data()) - 1 + ((*base <= x) ? 1 : 0);
      if (r < 0) r = 0;
      if (r >= n_bins) r = n_bins - 1;
    }
    float f = (x - edges[r]) * inv_widths[r];
    out_bin = r;
    out_frac = std::clamp(f, 0.0f, 1.0f);
  }
};

// Compact column-major PLE bin index storage (uint8_t <= 255 bins, uint16_t <= 65535, int32_t fallback)
struct PLEBinMatrix {
  int rows_ = 0;
  int cols_ = 0;
  int elem_bytes = 1;  // 1, 2, or 4
  std::vector<uint8_t> u8;
  std::vector<uint16_t> u16;
  std::vector<int32_t> i32;

  void resize(int r, int c, int max_bins) {
    rows_ = std::max(0, r);
    cols_ = std::max(0, c);
    size_t tot = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
    if (max_bins <= 255) {
      elem_bytes = 1;
      u8.resize(tot);
      u16.clear();
      i32.clear();
    } else if (max_bins <= 65535) {
      elem_bytes = 2;
      u16.resize(tot);
      u8.clear();
      i32.clear();
    } else {
      elem_bytes = 4;
      i32.resize(tot);
      u8.clear();
      u16.clear();
    }
  }

  int rows() const noexcept { return rows_; }
  int cols() const noexcept { return cols_; }
  int lda() const noexcept { return rows_; }
  bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

  uint8_t* col_u8(int j) noexcept { return u8.data() + static_cast<size_t>(j) * static_cast<size_t>(rows_); }
  const uint8_t* col_u8(int j) const noexcept { return u8.data() + static_cast<size_t>(j) * static_cast<size_t>(rows_); }
  uint16_t* col_u16(int j) noexcept { return u16.data() + static_cast<size_t>(j) * static_cast<size_t>(rows_); }
  const uint16_t* col_u16(int j) const noexcept { return u16.data() + static_cast<size_t>(j) * static_cast<size_t>(rows_); }
  int32_t* col_i32(int j) noexcept { return i32.data() + static_cast<size_t>(j) * static_cast<size_t>(rows_); }
  const int32_t* col_i32(int j) const noexcept { return i32.data() + static_cast<size_t>(j) * static_cast<size_t>(rows_); }
};

// Pre-allocated workspace for TabM training batches (zero heap allocations in training loop)
struct TabMWorkspace {
  struct alignas(128) ThreadScratch {
    std::vector<Eigen::MatrixXf> grad_W_blocks;  // per block l: din_l x d_block
    Eigen::MatrixXf grad_E;                      // max_B x D_in
    Eigen::MatrixXf delta_a;                     // max_B x max(d_block, D_in)
    Eigen::MatrixXf delta_b;                     // max_B x max(d_block, D_in)
    std::vector<float> col_ge;                   // max_B single-col grad_E reduction buffer (L1 resident)
    double loss_sum = 0.0;
    float max_g = 0.0f;
    bool has_work = false;
  };

  int max_B = 0;
  int N_train = 0;
  int K = 0;
  int D_in = 0;
  int d_block = 0;
  int n_blocks = 0;
  int out_dim = 0;
  int n_num = 0;
  int n_cat = 0;
  int d_emb = 0;
  int max_threads = 1;
  int b_out_pad_stride = 32;
  int r_in_pad_stride = 32;
  int b_blk_pad_stride = 32;

  // Epoch-shuffled contiguous training buffers (eliminates per-batch random row gather)
  Eigen::MatrixXf X_num_shuf;   // N_train x n_num (when num_emb != "pwl")
  PLEBinMatrix B_shuf;          // N_train x n_num compact uint8/uint16 (when num_emb == "pwl")
  Eigen::MatrixXf F_shuf;       // N_train x n_num (when num_emb == "pwl")
  Eigen::MatrixXi Cat_shuf;     // N_train x n_cat
  Eigen::MatrixXf Y_shuf;       // N_train x out_dim

  // Shared representation before BatchEnsemble expansion
  Eigen::MatrixXf E_batch;      // max_B x D_in

  // Per-member MLP forward activations & outputs:
  // H_k[k][0] is (max_B x D_in), H_k[k][1..n_blocks] are (max_B x d_block)
  std::vector<std::vector<Eigen::MatrixXf>> H_k;
  std::vector<Eigen::MatrixXf> Out_k;       // K x (max_B x out_dim)
  std::vector<Eigen::MatrixXf> grad_Out_k;  // K x (max_B x out_dim)
  std::vector<Eigen::MatrixXf> P_k;         // K x (max_B x out_dim) for multiclass

  // Thread-local backward scratch (cache-line aligned, eliminates 128B false sharing)
  std::vector<ThreadScratch> tls;

  // Weight gradients (contiguous, matching TabMModel parameter shapes)
  Eigen::MatrixXf grad_R_in;               // D_in x K
  std::vector<float> grad_R_in_pad;        // K * r_in_pad_stride
  std::vector<Eigen::MatrixXf> grad_b_blocks;  // n_blocks x (d_block x K)
  std::vector<std::vector<float>> grad_b_blocks_pad;
  Eigen::MatrixXf grad_W_out;              // d_block x (K * out_dim)
  Eigen::MatrixXf grad_b_out;              // out_dim x K
  std::vector<float> grad_b_out_pad;       // K * b_out_pad_stride

  // Per-feature PLE & categorical scratch buffers (parallel safe per feature index)
  std::vector<Eigen::MatrixXf> ple_grad_W;  // n_bins x d_emb
  std::vector<Eigen::MatrixXf> ple_grad_b;  // d_emb x 1
  std::vector<Eigen::MatrixXf> ple_diff;    // n_bins x d_emb
  std::vector<Eigen::MatrixXf> cat_grad_E;  // (card + 1) x d_emb

  void ensure_tls(int n_threads) {
    if (n_threads <= static_cast<int>(tls.size())) return;
    int old_sz = static_cast<int>(tls.size());
    tls.resize(n_threads);
    int max_delta_cols = std::max(d_block, D_in);
    for (int t = old_sz; t < n_threads; ++t) {
      tls[t].grad_W_blocks.resize(n_blocks);
      for (int l = 0; l < n_blocks; ++l) {
        int din_l = (l == 0) ? D_in : d_block;
        tls[t].grad_W_blocks[l].resize(din_l, d_block);
      }
      tls[t].grad_E.resize(max_B, D_in);
      tls[t].delta_a.resize(max_B, max_delta_cols);
      tls[t].delta_b.resize(max_B, max_delta_cols);
      tls[t].col_ge.resize(max_B);
    }
    max_threads = n_threads;
  }

  void allocate(int n_train_rows, int batch_cap, int k_ens, int d_in_dim, int d_blk, int n_blks, int out_d,
                int n_num_cols, int n_cat_cols, int d_embedding,
                const std::string& num_emb_mode, int max_ple_bins,
                const std::vector<PLEFeature>& ple_feats,
                const std::vector<int>& cat_cards,
                int num_threads = 1) {
    N_train = std::max(0, n_train_rows);
    max_B = std::max(1, batch_cap);
    K = std::max(1, k_ens);
    D_in = std::max(1, d_in_dim);
    d_block = std::max(1, d_blk);
    n_blocks = std::max(1, n_blks);
    out_dim = std::max(1, out_d);
    n_num = n_num_cols;
    n_cat = n_cat_cols;
    d_emb = std::max(1, d_embedding);
    max_threads = std::max(1, num_threads);

    if (N_train > 0) {
      if (n_num > 0) {
        if (num_emb_mode == "pwl") {
          B_shuf.resize(N_train, n_num, max_ple_bins);
          F_shuf.resize(N_train, n_num);
          X_num_shuf.resize(0, 0);
        } else {
          X_num_shuf.resize(N_train, n_num);
          B_shuf.resize(0, 0, 1);
          F_shuf.resize(0, 0);
        }
      }
      if (n_cat > 0) {
        Cat_shuf.resize(N_train, n_cat);
      }
      Y_shuf.resize(N_train, out_dim);
    }

    E_batch.resize(max_B, D_in);

    H_k.resize(K);
    Out_k.resize(K);
    grad_Out_k.resize(K);
    P_k.resize(K);
    for (int k = 0; k < K; ++k) {
      H_k[k].resize(n_blocks + 1);
      H_k[k][0].resize(max_B, D_in);
      for (int l = 0; l < n_blocks; ++l) {
        H_k[k][l + 1].resize(max_B, d_block);
      }
      Out_k[k].resize(max_B, out_dim);
      grad_Out_k[k].resize(max_B, out_dim);
      if (out_dim > 1) {
        P_k[k].resize(max_B, out_dim);
      }
    }

    tls.clear();
    ensure_tls(max_threads);

    grad_R_in.resize(D_in, K);
    r_in_pad_stride = ((D_in + 31) / 32) * 32;
    grad_R_in_pad.assign(static_cast<size_t>(K) * static_cast<size_t>(r_in_pad_stride), 0.0f);

    grad_b_blocks.resize(n_blocks);
    b_blk_pad_stride = ((d_block + 31) / 32) * 32;
    grad_b_blocks_pad.resize(n_blocks);
    for (int l = 0; l < n_blocks; ++l) {
      grad_b_blocks[l].resize(d_block, K);
      if (d_block < 32 || (d_block & 31) != 0) {
        grad_b_blocks_pad[l].assign(static_cast<size_t>(K) * static_cast<size_t>(b_blk_pad_stride), 0.0f);
      }
    }

    grad_W_out.resize(d_block, K * out_dim);
    grad_b_out.resize(out_dim, K);
    b_out_pad_stride = ((out_dim + 31) / 32) * 32;
    grad_b_out_pad.assign(static_cast<size_t>(K) * static_cast<size_t>(b_out_pad_stride), 0.0f);

    ple_grad_W.resize(ple_feats.size());
    ple_grad_b.resize(ple_feats.size());
    ple_diff.resize(ple_feats.size());
    for (size_t j = 0; j < ple_feats.size(); ++j) {
      if (ple_feats[j].active) {
        ple_grad_W[j].resize(ple_feats[j].n_bins, d_emb);
        ple_grad_b[j].resize(d_emb, 1);
        ple_diff[j].resize(ple_feats[j].n_bins, d_emb);
      }
    }

    cat_grad_E.resize(cat_cards.size());
    for (size_t c = 0; c < cat_cards.size(); ++c) {
      int n_rows = std::max(1, cat_cards[c]) + 1;
      cat_grad_E[c].resize(n_rows, d_emb);
    }
  }
};

// Bounded-memory chunked inference workspace (drops H[0] via W_eff, uses per-thread ping-pong H_a/H_b)
struct TabMInferWorkspace {
  static constexpr int DEFAULT_CHUNK_ROWS = 1024;

  struct alignas(128) InferThreadScratch {
    Eigen::MatrixXf H_a;           // chunk_cap x d_block
    Eigen::MatrixXf H_b;           // chunk_cap x d_block
    Eigen::MatrixXf Out_k;         // chunk_cap x out_dim
    std::vector<double> preds_acc; // chunk_cap * out_dim
    std::vector<float> soft_exps;  // out_dim (for multiclass)
  };

  int chunk_cap = 0;
  int K = 0;
  int D_in = 0;
  int d_block = 0;
  int n_blocks = 0;
  int out_dim = 0;
  int n_num = 0;
  int n_cat = 0;
  int max_threads = 1;

  Eigen::MatrixXf E_chunk;
  Eigen::MatrixXf X_num_chunk;
  Eigen::MatrixXi X_cat_chunk;
  PLEBinMatrix bin_idx_chunk;
  Eigen::MatrixXf bin_frac_chunk;
  std::vector<Eigen::MatrixXf> W_eff;  // K x (D_in x d_block): diag(R_in[:, k]) * W_0
  std::vector<InferThreadScratch> tls;
  Eigen::MatrixXd preds;
  std::vector<std::pair<double, int>> auc_pairs;  // Lazily allocated only if metric requires AUC/PR-AUC

  void ensure_tls(int n_threads) {
    if (n_threads <= static_cast<int>(tls.size())) return;
    int old_sz = static_cast<int>(tls.size());
    tls.resize(n_threads);
    for (int t = old_sz; t < n_threads; ++t) {
      tls[t].H_a.resize(chunk_cap, d_block);
      tls[t].H_b.resize(chunk_cap, d_block);
      tls[t].Out_k.resize(chunk_cap, out_dim);
      tls[t].preds_acc.resize(static_cast<size_t>(chunk_cap) * static_cast<size_t>(out_dim));
      if (out_dim > 1) tls[t].soft_exps.resize(out_dim);
    }
    max_threads = n_threads;
  }

  void allocate(int n_total_rows, int k_ens, int d_in_dim, int d_blk, int n_blks, int out_d,
                int n_num_cols, int n_cat_cols, int max_ple_bins = 48,
                int num_threads = 1, int requested_chunk = DEFAULT_CHUNK_ROWS) {
    K = std::max(1, k_ens);
    int max_rows_by_ram = std::max(256, 32768 / K);
    int cap = std::min(std::max(1, n_total_rows), std::min(requested_chunk, max_rows_by_ram));
    chunk_cap = std::max(1, cap);
    D_in = std::max(1, d_in_dim);
    d_block = std::max(1, d_blk);
    n_blocks = std::max(1, n_blks);
    out_dim = std::max(1, out_d);
    n_num = n_num_cols;
    n_cat = n_cat_cols;
    max_threads = std::max(1, num_threads);

    E_chunk.resize(chunk_cap, D_in);
    if (n_num > 0) {
      X_num_chunk.resize(chunk_cap, n_num);
      bin_idx_chunk.resize(chunk_cap, n_num, max_ple_bins);
      bin_frac_chunk.resize(chunk_cap, n_num);
    }
    if (n_cat > 0) X_cat_chunk.resize(chunk_cap, n_cat);

    W_eff.resize(K);
    for (int k = 0; k < K; ++k) {
      W_eff[k].resize(D_in, d_block);
    }
    tls.clear();
    ensure_tls(max_threads);
    preds.resize(n_total_rows, out_dim);
  }
};

struct TabMBlock {
  AdamParam W;  // d_in x d_out
  AdamParam b;  // d_out x K (column k is contiguous bias for member k)
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
  std::vector<AdamParam> cat_tables;  // (card + 1) x d_embedding (row card == missing/OOD)

  int D_num_rep = 0;
  int D_cat_rep = 0;
  int D_in = 1;
  bool has_dummy_in = false;

  // BatchEnsemble parameters (column-contiguous per member k)
  AdamParam R_in;                   // D_in x K (Rademacher {-1, +1} init)
  std::vector<TabMBlock> blocks;    // n_blocks
  AdamParam W_out;                  // d_block x (K * output_dim) contiguous packed heads
  AdamParam b_out;                  // output_dim x K

  struct StateSnapshot {
    std::vector<Eigen::MatrixXf> ple_W;
    std::vector<Eigen::MatrixXf> ple_b;
    std::vector<Eigen::MatrixXf> cat_tables;
    Eigen::MatrixXf R_in;
    std::vector<Eigen::MatrixXf> blocks_W;
    std::vector<Eigen::MatrixXf> blocks_b;
    Eigen::MatrixXf W_out;
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
      W_out = m.W_out.val;
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
      m.W_out.val = W_out;
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
    std::vector<int> n_finite_counts(n_num, 0);
    if (n_num > 0) {
      X_num_out.resize(N, n_num);
    }
    ple_features.clear();
    if (num_emb == "pwl") {
      ple_features.resize(n_num);
    }

    std::mt19937 ple_rng(static_cast<unsigned int>(seed + 13));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && N >= 256)
#endif
    for (int j = 0; j < n_num; ++j) {
      int orig_col = num_col_indices[j];
      const double* col_ptr = x_ptr + static_cast<size_t>(orig_col) * static_cast<size_t>(N);
      FiniteColumnStats st = compute_finite_column_stats(col_ptr, N);
      num_means[j] = st.mean;
      num_stds[j] = st.sd;
      num_inv_stds[j] = st.inv_std;
      n_finite_counts[j] = st.n_finite;

      float* dst = X_num_out.col(j).data();
      for (int i = 0; i < N; ++i) {
        double v = col_ptr[i];
        double clean_v = (std::isfinite(v) && std::abs(v) <= MAX_FLOAT32_VAL) ? v : st.mean;
        double z = (clean_v - st.mean) * st.inv_std;
        dst[i] = static_cast<float>(std::clamp(z, -30.0, 30.0));
      }
    }

    // Build PLE edges in parallel across n_num, then init weights deterministically with ple_rng
    D_num_rep = 0;
    if (num_emb == "pwl" && n_num > 0) {
      std::vector<std::vector<float>> all_edges(n_num);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && N >= 256)
#endif
      for (int j = 0; j < n_num; ++j) {
        int n_fin = n_finite_counts[j];
        if (n_fin < 2) continue;

        int orig_col = num_col_indices[j];
        const double* col_ptr = x_ptr + static_cast<size_t>(orig_col) * static_cast<size_t>(N);
        const float* x_std_col = X_num_out.col(j).data();
        int step = (n_fin > 100000) ? ((n_fin + 99999) / 100000) : 1;

        std::vector<float> fin_sample;
        fin_sample.reserve(std::min(n_fin, 100000));
        if (n_fin == N) {
          for (int i = 0; i < N; i += step) {
            fin_sample.push_back(x_std_col[i]);
          }
        } else {
          int seen = 0;
          for (int i = 0; i < N; ++i) {
            double v = col_ptr[i];
            if (std::isfinite(v) && std::abs(v) <= MAX_FLOAT32_VAL) {
              if (step == 1 || (seen % step) == 0) {
                fin_sample.push_back(x_std_col[i]);
              }
              seen++;
            }
          }
        }
        if (fin_sample.size() < 2) continue;
        std::sort(fin_sample.begin(), fin_sample.end());
        if (fin_sample.back() - fin_sample.front() <= 1e-7f) continue;

        int nb = std::max(1, n_bins);
        std::vector<float>& edges = all_edges[j];
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
      }

      for (int j = 0; j < n_num; ++j) {
        if (all_edges[j].size() >= 2) {
          ple_features[j].init_from_edges(all_edges[j], d_embedding, ple_rng);
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
#pragma omp parallel for schedule(static) if (n_cat >= 2 && N >= 256)
#endif
      for (int c = 0; c < n_cat; ++c) {
        int card = cat_cardinalities[c];
        const double* col_ptr = x_ptr + static_cast<size_t>(cat_col_indices[c]) * static_cast<size_t>(N);
        int* dst = X_cat_out.col(c).data();
        for (int i = 0; i < N; ++i) {
          double v = col_ptr[i];
          int code = card;  // missing / OOD
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
        int card = cat_cardinalities[c];
        int n_rows = card + 1;
        cat_tables[c].init(n_rows, d_embedding);
        for (int r = 0; r < card; ++r) {
          for (int e = 0; e < d_embedding; ++e) {
            cat_tables[c].val(r, e) = dist(cat_rng);
          }
        }
        // Zero-initialize unseen/OOD row `card` (learns only if NAs occur in train)
        for (int e = 0; e < d_embedding; ++e) {
          cat_tables[c].val(card, e) = 0.0f;
        }
      }
    }
    D_cat_rep = n_cat * d_embedding;

    has_dummy_in = (D_num_rep + D_cat_rep == 0);
    D_in = has_dummy_in ? 1 : (D_num_rep + D_cat_rep);
  }

  int max_ple_bins() const noexcept {
    int mb = 1;
    for (const auto& pf : ple_features) {
      if (pf.active && pf.n_bins > mb) mb = pf.n_bins;
    }
    return mb;
  }

  template <typename BinT>
  void encode_ple_chunk_t(const float* x_num_ptr, int x_num_lda, int row_offset, int cur_B,
                          BinT* b_base, int b_lda, float* f_base, int f_lda) const {
    int n_num = static_cast<int>(num_col_indices.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B * n_num >= 16384)
#endif
    for (int j = 0; j < n_num; ++j) {
      const PLEFeature& pf = ple_features[j];
      if (!pf.active) continue;
      const float* x_col = x_num_ptr + static_cast<size_t>(j) * static_cast<size_t>(x_num_lda) + row_offset;
      BinT* b_col = b_base + static_cast<size_t>(j) * static_cast<size_t>(b_lda);
      float* f_col = f_base + static_cast<size_t>(j) * static_cast<size_t>(f_lda);
      for (int i = 0; i < cur_B; ++i) {
        int b_val = 0;
        pf.lookup(x_col[i], b_val, f_col[i]);
        b_col[i] = static_cast<BinT>(b_val);
      }
    }
  }

  // Bin a contiguous row slice [row_offset .. row_offset + cur_B) from standardized X_num
  void encode_ple_chunk(const float* x_num_ptr, int x_num_lda, int row_offset, int cur_B,
                        PLEBinMatrix& B_out, Eigen::MatrixXf& F_out) const {
    if (num_emb != "pwl" || cur_B <= 0) return;
    int n_num = static_cast<int>(num_col_indices.size());
    if (n_num <= 0) return;
    int b_lda = B_out.lda();
    int f_lda = static_cast<int>(F_out.outerStride());
    if (B_out.elem_bytes == 1) {
      encode_ple_chunk_t<uint8_t>(x_num_ptr, x_num_lda, row_offset, cur_B, B_out.u8.data(), b_lda, F_out.data(), f_lda);
    } else if (B_out.elem_bytes == 2) {
      encode_ple_chunk_t<uint16_t>(x_num_ptr, x_num_lda, row_offset, cur_B, B_out.u16.data(), b_lda, F_out.data(), f_lda);
    } else {
      encode_ple_chunk_t<int32_t>(x_num_ptr, x_num_lda, row_offset, cur_B, B_out.i32.data(), b_lda, F_out.data(), f_lda);
    }
  }

  void encode_ple_matrix(const Eigen::MatrixXf& X_num,
                         PLEBinMatrix& B_out,
                         Eigen::MatrixXf& F_out) const {
    int N = static_cast<int>(X_num.rows());
    int n_num = static_cast<int>(num_col_indices.size());
    if (num_emb != "pwl" || n_num <= 0 || N <= 0) {
      B_out.resize(0, 0, 1);
      F_out.resize(0, 0);
      return;
    }
    B_out.resize(N, n_num, max_ple_bins());
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

    // Multiplicative BatchEnsemble input adapter R_in (D_in x K) initialized Rademacher {-1, +1}
    R_in.init(D_in, k_ensemble);
    std::uniform_int_distribution<int> rad_dist(0, 1);
    for (int d = 0; d < D_in; ++d) {
      R_in.val(d, 0) = 1.0f;
      for (int k = 1; k < k_ensemble; ++k) {
        R_in.val(d, k) = rad_dist(rng) ? 1.0f : -1.0f;
      }
    }

    blocks.resize(n_blocks);
    for (int l = 0; l < n_blocks; ++l) {
      int din_l = (l == 0) ? D_in : d_block;
      blocks[l].d_in = din_l;
      blocks[l].d_out = d_block;
      blocks[l].W.init(din_l, d_block);
      blocks[l].b.init(d_block, k_ensemble);
      float std_he = std::sqrt(2.0f / static_cast<float>(std::max(1, din_l)));
      std::normal_distribution<float> norm_dist(0.0f, std_he);
      for (int c = 0; c < d_block; ++c) {
        for (int r = 0; r < din_l; ++r) {
          blocks[l].W.val(r, c) = norm_dist(rng);
        }
      }
      blocks[l].b.val.setZero();
    }

    W_out.init(d_block, k_ensemble * output_dim);
    b_out.init(output_dim, k_ensemble);
    float xavier_bound = std::sqrt(6.0f / static_cast<float>(d_block + output_dim));
    std::uniform_real_distribution<float> out_dist(-xavier_bound, xavier_bound);
    for (int k = 0; k < k_ensemble; ++k) {
      for (int c = 0; c < output_dim; ++c) {
        int col = k * output_dim + c;
        for (int r = 0; r < d_block; ++r) {
          W_out.val(r, col) = out_dist(rng);
        }
      }
    }
    b_out.val.setZero();
  }

  void extract_raw_chunk(const double* x_raw_ptr, int N_total, int start_row, int cur_B,
                         Eigen::MatrixXf& X_num_buf, Eigen::MatrixXi& X_cat_buf) const {
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B * n_num >= 16384)
#endif
    for (int j = 0; j < n_num; ++j) {
      const double* col_ptr = x_raw_ptr + static_cast<size_t>(num_col_indices[j]) * static_cast<size_t>(N_total) + start_row;
      double m = num_means[j];
      double inv_s = num_inv_stds[j];
      float* dst = X_num_buf.col(j).data();
      for (int i = 0; i < cur_B; ++i) {
        double v = col_ptr[i];
        double clean_v = (std::isfinite(v) && std::abs(v) <= MAX_FLOAT32_VAL) ? v : m;
        double z = (clean_v - m) * inv_s;
        dst[i] = static_cast<float>(std::clamp(z, -30.0, 30.0));
      }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_cat >= 2 && cur_B * n_cat >= 16384)
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

  template <typename BinT>
  inline void forward_ple_single_col_t(int j,
                                       const BinT* b_base, int b_lda,
                                       const float* f_base, int f_lda,
                                       int row_offset, int cur_B,
                                       float* e_base, int e_stride) const noexcept {
    const PLEFeature& pf = ple_features[j];
    if (!pf.active) return;
    const BinT* b_col = b_base + static_cast<size_t>(j) * static_cast<size_t>(b_lda) + row_offset;
    const float* f_col = f_base + static_cast<size_t>(j) * static_cast<size_t>(f_lda) + row_offset;
    int d_e = pf.d_emb;
    int col_off = pf.out_col_offset;
    for (int e = 0; e < d_e; ++e) {
      const float* s_col = pf.S.col(e).data();
      const float* w_col = pf.W.val.col(e).data();
      float* e_col = e_base + static_cast<size_t>(col_off + e) * static_cast<size_t>(e_stride);
      for (int i = 0; i < cur_B; ++i) {
        const int b_star = static_cast<int>(b_col[i]);
        e_col[i] = s_col[b_star] + f_col[i] * w_col[b_star];
      }
    }
  }

  inline void forward_cat_single_col(int c,
                                     const int* x_cat_ptr, int x_cat_lda,
                                     int row_offset, int cur_B,
                                     float* e_base, int e_stride) const noexcept {
    const Eigen::MatrixXf& tbl = cat_tables[c].val;
    int card = cat_cardinalities[c];
    const int* code_col = x_cat_ptr + static_cast<size_t>(c) * static_cast<size_t>(x_cat_lda) + row_offset;
    int col_off = D_num_rep + c * d_embedding;
    for (int e = 0; e < d_embedding; ++e) {
      const float* t_col = tbl.col(e).data();
      float* e_col = e_base + static_cast<size_t>(col_off + e) * static_cast<size_t>(e_stride);
      for (int i = 0; i < cur_B; ++i) {
        int code = code_col[i];
        if (static_cast<unsigned int>(code) > static_cast<unsigned int>(card)) code = card;
        e_col[i] = t_col[code];
      }
    }
  }

  template <typename BinT>
  void forward_ple_cols_t(const BinT* b_base, int b_lda,
                          const float* f_base, int f_lda,
                          int row_offset, int cur_B,
                          float* e_base, int e_stride) const {
    int n_num = static_cast<int>(num_col_indices.size());
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B * n_num * d_embedding >= 65536)
#endif
    for (int j = 0; j < n_num; ++j) {
      forward_ple_single_col_t<BinT>(j, b_base, b_lda, f_base, f_lda, row_offset, cur_B, e_base, e_stride);
    }
  }

  void forward_embeddings_prebinned(const float* x_num_ptr, int x_num_lda,
                                    const PLEBinMatrix* B_ptr,
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

    if (num_emb == "pwl" && n_num > 0 && B_ptr != nullptr && !B_ptr->empty() && f_ptr != nullptr) {
      if (B_ptr->elem_bytes == 1) {
        forward_ple_cols_t<uint8_t>(B_ptr->u8.data(), B_ptr->lda(), f_ptr, f_lda, row_offset, cur_B, e_base, e_stride);
      } else if (B_ptr->elem_bytes == 2) {
        forward_ple_cols_t<uint16_t>(B_ptr->u16.data(), B_ptr->lda(), f_ptr, f_lda, row_offset, cur_B, e_base, e_stride);
      } else {
        forward_ple_cols_t<int32_t>(B_ptr->i32.data(), B_ptr->lda(), f_ptr, f_lda, row_offset, cur_B, e_base, e_stride);
      }
    } else if (n_num > 0 && x_num_ptr != nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_num >= 2 && cur_B * n_num >= 65536)
#endif
      for (int j = 0; j < n_num; ++j) {
        std::memcpy(e_base + static_cast<size_t>(j) * static_cast<size_t>(e_stride),
                    x_num_ptr + static_cast<size_t>(j) * static_cast<size_t>(x_num_lda) + row_offset,
                    static_cast<size_t>(cur_B) * sizeof(float));
      }
    }

    if (n_cat > 0 && x_cat_ptr != nullptr) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (n_cat >= 2 && cur_B * n_cat * d_embedding >= 65536)
#endif
      for (int c = 0; c < n_cat; ++c) {
        forward_cat_single_col(c, x_cat_ptr, x_cat_lda, row_offset, cur_B, e_base, e_stride);
      }
    }
  }

  void forward_embeddings(const Eigen::MatrixXf& X_num, const Eigen::MatrixXi& X_cat, int cur_B,
                          Eigen::MatrixXf& E_out,
                          PLEBinMatrix* bin_idx_cache = nullptr,
                          Eigen::MatrixXf* bin_frac_cache = nullptr) const {
    if (has_dummy_in) {
      E_out.block(0, 0, cur_B, 1).setZero();
      return;
    }
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    PLEBinMatrix tmp_b;
    Eigen::MatrixXf tmp_f;
    PLEBinMatrix* b_buf = bin_idx_cache;
    Eigen::MatrixXf* f_buf = bin_frac_cache;
    if (num_emb == "pwl" && n_num > 0) {
      if (b_buf == nullptr || f_buf == nullptr) {
        tmp_b.resize(cur_B, n_num, max_ple_bins());
        tmp_f.resize(cur_B, n_num);
        b_buf = &tmp_b;
        f_buf = &tmp_f;
      }
      encode_ple_chunk(X_num.data(), static_cast<int>(X_num.outerStride()), 0, cur_B, *b_buf, *f_buf);
    }
    forward_embeddings_prebinned(
      (n_num > 0) ? X_num.data() : nullptr, (n_num > 0) ? static_cast<int>(X_num.outerStride()) : 0,
      (b_buf != nullptr && n_num > 0) ? b_buf : nullptr,
      (f_buf != nullptr && n_num > 0) ? f_buf->data() : nullptr, (f_buf != nullptr && n_num > 0) ? static_cast<int>(f_buf->outerStride()) : 0,
      (n_cat > 0) ? X_cat.data() : nullptr, (n_cat > 0) ? static_cast<int>(X_cat.outerStride()) : 0,
      0, cur_B, E_out
    );
  }

  // Output head forward for a single member k: H_last (cur_B x d_block) -> Out_k (cur_B x output_dim)
  inline void forward_out_member(const Eigen::MatrixXf& H_last, int cur_B, int k,
                                 Eigen::MatrixXf& Out_k_buf) const noexcept {
    int hlast_stride = static_cast<int>(H_last.outerStride());
    int out_stride = static_cast<int>(Out_k_buf.outerStride());
    const float* hlast_data = H_last.data();
    float* out_data = Out_k_buf.data();
    const float* w_k = W_out.val.data() + static_cast<size_t>(k) * static_cast<size_t>(output_dim) * static_cast<size_t>(d_block);
    const float* b_k = b_out.val.col(k).data();

    if (output_dim == 1) {
      float bk = b_k[0];
      for (int i = 0; i < cur_B; ++i) out_data[i] = bk;
      int d = 0;
      for (; d + 3 < d_block; d += 4) {
        float w0 = w_k[d], w1 = w_k[d + 1], w2 = w_k[d + 2], w3 = w_k[d + 3];
        const float* h0 = hlast_data + static_cast<size_t>(d) * static_cast<size_t>(hlast_stride);
        const float* h1 = h0 + hlast_stride;
        const float* h2 = h1 + hlast_stride;
        const float* h3 = h2 + hlast_stride;
        for (int i = 0; i < cur_B; ++i) {
          out_data[i] += h0[i] * w0 + h1[i] * w1 + h2[i] * w2 + h3[i] * w3;
        }
      }
      for (; d < d_block; ++d) {
        float wd = w_k[d];
        const float* h_col = hlast_data + static_cast<size_t>(d) * static_cast<size_t>(hlast_stride);
        for (int i = 0; i < cur_B; ++i) {
          out_data[i] += h_col[i] * wd;
        }
      }
    } else {
      gemm_f32_raw(false, false, cur_B, output_dim, d_block,
                   hlast_data, hlast_stride,
                   w_k, static_cast<int>(W_out.val.outerStride()),
                   out_data, out_stride,
                   1.0f, 0.0f);
      for (int c = 0; c < output_dim; ++c) {
        float bk = b_k[c];
        float* dst = out_data + static_cast<size_t>(c) * static_cast<size_t>(out_stride);
        for (int i = 0; i < cur_B; ++i) {
          dst[i] += bk;
        }
      }
    }
  }

  // Forward MLP for a single ensemble member k during training (caches H_bufs[0..n_blocks])
  inline void forward_mlp_member(const Eigen::MatrixXf& E_in, int cur_B, int k,
                                 bool do_drop, float inv_keep, uint32_t drop_thresh8, int step_id,
                                 std::vector<Eigen::MatrixXf>& H_bufs,
                                 Eigen::MatrixXf& Out_k_buf) const {
    Eigen::MatrixXf& H0 = H_bufs[0];
    int e_stride = static_cast<int>(E_in.outerStride());
    int h0_stride = static_cast<int>(H0.outerStride());
    const float* e_data = E_in.data();
    float* h0_data = H0.data();
    const float* r_col = R_in.val.col(k).data();
    for (int d = 0; d < D_in; ++d) {
      float r_kd = r_col[d];
      const float* e_col = e_data + static_cast<size_t>(d) * static_cast<size_t>(e_stride);
      float* dst = h0_data + static_cast<size_t>(d) * static_cast<size_t>(h0_stride);
      for (int i = 0; i < cur_B; ++i) {
        dst[i] = e_col[i] * r_kd;
      }
    }

    for (int l = 0; l < n_blocks; ++l) {
      const TabMBlock& blk = blocks[l];
      const Eigen::MatrixXf& H_prev = H_bufs[l];
      Eigen::MatrixXf& H_cur = H_bufs[l + 1];
      int h_stride = static_cast<int>(H_cur.outerStride());

      gemm_f32_raw(false, false, cur_B, blk.d_out, blk.d_in,
                   H_prev.data(), static_cast<int>(H_prev.outerStride()),
                   blk.W.val.data(), static_cast<int>(blk.W.val.outerStride()),
                   H_cur.data(), h_stride,
                   1.0f, 0.0f);

      apply_relu_dropout_fwd_k(H_cur.data(), h_stride, blk.b.val.col(k).data(),
                               cur_B, blk.d_out, do_drop, inv_keep, drop_thresh8,
                               step_id, l + 1, k + 1);
    }

    forward_out_member(H_bufs[n_blocks], cur_B, k, Out_k_buf);
  }

  // Fold R_in[:, k] into effective layer-0 weight matrix W_eff[k] = diag(R_in[:, k]) * W_0
  void build_w_eff(std::vector<Eigen::MatrixXf>& W_eff) const {
    W_eff.resize(k_ensemble);
    if (n_blocks <= 0) return;
    const Eigen::MatrixXf& W0 = blocks[0].W.val;
    int w0_stride = static_cast<int>(W0.outerStride());
    const float* w0_data = W0.data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (k_ensemble >= 2 && k_ensemble * D_in * d_block >= 32768)
#endif
    for (int k = 0; k < k_ensemble; ++k) {
      if (W_eff[k].rows() != D_in || W_eff[k].cols() != d_block) {
        W_eff[k].resize(D_in, d_block);
      }
      const float* r_col = R_in.val.col(k).data();
      float* we_data = W_eff[k].data();
      int we_stride = static_cast<int>(W_eff[k].outerStride());
      for (int c = 0; c < d_block; ++c) {
        const float* src = w0_data + static_cast<size_t>(c) * static_cast<size_t>(w0_stride);
        float* dst = we_data + static_cast<size_t>(c) * static_cast<size_t>(we_stride);
        for (int d = 0; d < D_in; ++d) {
          dst[d] = src[d] * r_col[d];
        }
      }
    }
  }

  // Accumulate single-member logits/probs into per-thread FP64 accumulator (cur_B x output_dim ColMajor)
  inline void accumulate_member_preds(int cur_B, const Eigen::MatrixXf& Out_k_buf,
                                      double* preds_acc, float* soft_exps) const noexcept {
    const float* out_data = Out_k_buf.data();
    int out_stride = static_cast<int>(Out_k_buf.outerStride());
    if (task == "regression") {
      for (int i = 0; i < cur_B; ++i) {
        float z = std::clamp(out_data[i], -50.0f, 50.0f);
        preds_acc[i] += static_cast<double>(z);
      }
    } else if (task == "classification") {
      for (int i = 0; i < cur_B; ++i) {
        float z = std::clamp(out_data[i], -30.0f, 30.0f);
        preds_acc[i] += static_cast<double>(1.0f / (1.0f + evofe::nn::fast_expf(-z)));
      }
    } else {
      for (int i = 0; i < cur_B; ++i) {
        float max_v = out_data[i];
        for (int c = 1; c < output_dim; ++c) {
          float v = out_data[i + static_cast<size_t>(c) * static_cast<size_t>(out_stride)];
          if (v > max_v) max_v = v;
        }
        float sum_e = 0.0f;
        for (int c = 0; c < output_dim; ++c) {
          float ep = evofe::nn::fast_expf(out_data[i + static_cast<size_t>(c) * static_cast<size_t>(out_stride)] - max_v);
          soft_exps[c] = ep;
          sum_e += ep;
        }
        if (sum_e > 0.0f) {
          double inv_sum = 1.0 / static_cast<double>(sum_e);
          for (int c = 0; c < output_dim; ++c) {
            preds_acc[i + static_cast<size_t>(c) * static_cast<size_t>(cur_B)] += static_cast<double>(soft_exps[c]) * inv_sum;
          }
        } else {
          double unif = 1.0 / static_cast<double>(output_dim);
          for (int c = 0; c < output_dim; ++c) {
            preds_acc[i + static_cast<size_t>(c) * static_cast<size_t>(cur_B)] += unif;
          }
        }
      }
    }
  }

  // Forward member k starting from pre-activation A1 in *H_cur, using ping-pong *H_cur <-> *H_next
  inline void forward_from_a1_member(int k, int cur_B,
                                     Eigen::MatrixXf* H_cur,
                                     Eigen::MatrixXf* H_next,
                                     Eigen::MatrixXf& Out_k_buf,
                                     double* preds_acc,
                                     float* soft_exps) const noexcept {
    apply_relu_dropout_fwd_k(H_cur->data(), static_cast<int>(H_cur->outerStride()),
                             blocks[0].b.val.col(k).data(), cur_B, blocks[0].d_out,
                             false, 1.0f, 256u, 0, 1, k + 1);
    for (int l = 1; l < n_blocks; ++l) {
      const TabMBlock& blk = blocks[l];
      int h_stride = static_cast<int>(H_next->outerStride());
      gemm_f32_raw(false, false, cur_B, blk.d_out, blk.d_in,
                   H_cur->data(), static_cast<int>(H_cur->outerStride()),
                   blk.W.val.data(), static_cast<int>(blk.W.val.outerStride()),
                   H_next->data(), h_stride, 1.0f, 0.0f);
      apply_relu_dropout_fwd_k(H_next->data(), h_stride, blk.b.val.col(k).data(),
                               cur_B, blk.d_out, false, 1.0f, 256u, 0, l + 1, k + 1);
      std::swap(H_cur, H_next);
    }
    forward_out_member(*H_cur, cur_B, k, Out_k_buf);
    accumulate_member_preds(cur_B, Out_k_buf, preds_acc, soft_exps);
  }

  // Finalize per-thread accumulated member predictions into preds_dst[start_row .. start_row + cur_B)
  inline void finalize_preds_rows(int cur_B, int start_row,
                                  const std::vector<const double*>& src_ptrs,
                                  int n_srcs,
                                  Eigen::MatrixXd& preds_dst) const noexcept {
    double inv_K = 1.0 / static_cast<double>(k_ensemble);
    if (task == "regression") {
      double ys = (std::isfinite(y_std) && y_std > 1e-8) ? y_std : 1.0;
      double ym = std::isfinite(y_mean) ? y_mean : 0.0;
      double scale = inv_K * ys;
      double* dst = preds_dst.col(0).data() + start_row;
      for (int i = 0; i < cur_B; ++i) {
        double s = 0.0;
        for (int t = 0; t < n_srcs; ++t) s += src_ptrs[t][i];
        dst[i] = s * scale + ym;
      }
    } else if (task == "classification") {
      double* dst = preds_dst.col(0).data() + start_row;
      for (int i = 0; i < cur_B; ++i) {
        double s = 0.0;
        for (int t = 0; t < n_srcs; ++t) s += src_ptrs[t][i];
        dst[i] = s * inv_K;
      }
    } else {
      double* p_data = preds_dst.data();
      int p_stride = static_cast<int>(preds_dst.outerStride());
      for (int c = 0; c < output_dim; ++c) {
        double* dst = p_data + start_row + static_cast<size_t>(c) * static_cast<size_t>(p_stride);
        size_t off = static_cast<size_t>(c) * static_cast<size_t>(cur_B);
        for (int i = 0; i < cur_B; ++i) {
          double s = 0.0;
          for (int t = 0; t < n_srcs; ++t) s += src_ptrs[t][off + i];
          dst[i] = s * inv_K;
        }
      }
    }
  }

  void eval_chunk_w_eff(const Eigen::MatrixXf& E_chunk, int cur_B, int start_row,
                        TabMInferWorkspace& ws) const {
    int K = k_ensemble;
    int omp_nt = 1;
#if defined(_OPENMP)
    omp_nt = std::max(1, omp_get_max_threads());
#endif
    bool use_omp = (omp_nt > 1 && K >= 2);
    int n_threads = use_omp ? std::min(omp_nt, K) : 1;
    ws.ensure_tls(n_threads);

#if defined(_OPENMP)
#pragma omp parallel num_threads(n_threads) if (use_omp)
#endif
    {
#if defined(_OPENMP)
      int tid = use_omp ? omp_get_thread_num() : 0;
#else
      int tid = 0;
#endif
      TabMInferWorkspace::InferThreadScratch& ts = ws.tls[tid];
      std::fill(ts.preds_acc.begin(),
                ts.preds_acc.begin() + static_cast<size_t>(cur_B) * static_cast<size_t>(output_dim),
                0.0);
      float* exps_p = (output_dim > 1) ? ts.soft_exps.data() : nullptr;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
      for (int k = 0; k < K; ++k) {
        gemm_f32_raw(false, false, cur_B, d_block, D_in,
                     E_chunk.data(), static_cast<int>(E_chunk.outerStride()),
                     ws.W_eff[k].data(), static_cast<int>(ws.W_eff[k].outerStride()),
                     ts.H_a.data(), static_cast<int>(ts.H_a.outerStride()),
                     1.0f, 0.0f);
        forward_from_a1_member(k, cur_B, &ts.H_a, &ts.H_b, ts.Out_k, ts.preds_acc.data(), exps_p);
      }
    }

    std::vector<const double*> src_ptrs(n_threads);
    for (int t = 0; t < n_threads; ++t) src_ptrs[t] = ws.tls[t].preds_acc.data();
    finalize_preds_rows(cur_B, start_row, src_ptrs, n_threads, ws.preds);
  }

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
    int omp_nt = 1;
#if defined(_OPENMP)
    omp_nt = std::max(1, omp_get_max_threads());
#endif
    if (ws.chunk_cap <= 0 || ws.K != k_ensemble || ws.D_in != D_in ||
        ws.d_block != d_block || ws.n_blocks != n_blocks || ws.out_dim != output_dim ||
        ws.preds.rows() != N_total || ws.preds.cols() != output_dim) {
      ws.allocate(N_total, k_ensemble, D_in, d_block, n_blocks, output_dim, n_num, n_cat, max_ple_bins(), omp_nt);
    }

    build_w_eff(ws.W_eff);

    for (int start = 0; start < N_total; start += ws.chunk_cap) {
      int cur_B = std::min(ws.chunk_cap, N_total - start);
      extract_raw_chunk(x_raw_ptr, N_total, start, cur_B, ws.X_num_chunk, ws.X_cat_chunk);
      if (num_emb == "pwl" && n_num > 0) {
        encode_ple_chunk(ws.X_num_chunk.data(), static_cast<int>(ws.X_num_chunk.outerStride()),
                         0, cur_B, ws.bin_idx_chunk, ws.bin_frac_chunk);
      }
      forward_embeddings_prebinned(
        n_num > 0 ? ws.X_num_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.X_num_chunk.outerStride()) : 0,
        n_num > 0 ? &ws.bin_idx_chunk : nullptr,
        n_num > 0 ? ws.bin_frac_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.bin_frac_chunk.outerStride()) : 0,
        n_cat > 0 ? ws.X_cat_chunk.data() : nullptr, n_cat > 0 ? static_cast<int>(ws.X_cat_chunk.outerStride()) : 0,
        0, cur_B, ws.E_chunk
      );
      eval_chunk_w_eff(ws.E_chunk, cur_B, start, ws);
    }
  }

  void predict_preprocessed_chunked(const Eigen::MatrixXf& X_num, const Eigen::MatrixXi& X_cat,
                                    int N_total, TabMInferWorkspace& ws,
                                    const PLEBinMatrix* B_pre = nullptr,
                                    const Eigen::MatrixXf* F_pre = nullptr) const {
    if (N_total <= 0) {
      ws.preds.resize(0, output_dim);
      return;
    }
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    int omp_nt = 1;
#if defined(_OPENMP)
    omp_nt = std::max(1, omp_get_max_threads());
#endif
    if (ws.chunk_cap <= 0 || ws.K != k_ensemble || ws.D_in != D_in ||
        ws.d_block != d_block || ws.n_blocks != n_blocks || ws.out_dim != output_dim ||
        ws.preds.rows() != N_total || ws.preds.cols() != output_dim) {
      ws.allocate(N_total, k_ensemble, D_in, d_block, n_blocks, output_dim, n_num, n_cat, max_ple_bins(), omp_nt);
    }

    build_w_eff(ws.W_eff);

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
          n_num > 0 ? B_pre : nullptr,
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
          n_num > 0 ? &ws.bin_idx_chunk : nullptr,
          n_num > 0 ? ws.bin_frac_chunk.data() : nullptr, n_num > 0 ? static_cast<int>(ws.bin_frac_chunk.outerStride()) : 0,
          x_cat_ptr, x_cat_lda,
          start, cur_B, ws.E_chunk
        );
      }
      eval_chunk_w_eff(ws.E_chunk, cur_B, start, ws);
    }
  }

  // Parallel column-wise epoch shuffle into contiguous ws.*_shuf buffers
  void shuffle_epoch_data(const Eigen::MatrixXf& X_num_tr,
                          const PLEBinMatrix* B_tr,
                          const Eigen::MatrixXf* F_tr,
                          const Eigen::MatrixXi& X_cat_tr,
                          const Eigen::MatrixXf& Y_tr,
                          const std::vector<int>& perm,
                          TabMWorkspace& ws) const {
    int N = static_cast<int>(perm.size());
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    const int* p_idx = perm.data();
    const bool use_pwl = (num_emb == "pwl" && n_num > 0 && B_tr != nullptr && F_tr != nullptr && B_tr->rows() == N);

    int total_cols = n_num + n_cat + output_dim;
    size_t total_elems = static_cast<size_t>(N) * static_cast<size_t>(total_cols);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (total_cols >= 2 && total_elems >= 32768)
#endif
    for (int task_col = 0; task_col < total_cols; ++task_col) {
      if (task_col < n_num) {
        int j = task_col;
        if (use_pwl) {
          if (!ple_features[j].active) continue;
          const float* f_src = F_tr->col(j).data();
          float* f_dst = ws.F_shuf.col(j).data();
          if (ws.B_shuf.elem_bytes == 1) {
            const uint8_t* b_src = B_tr->col_u8(j);
            uint8_t* b_dst = ws.B_shuf.col_u8(j);
            for (int i = 0; i < N; ++i) {
              int r = p_idx[i];
              b_dst[i] = b_src[r];
              f_dst[i] = f_src[r];
            }
          } else if (ws.B_shuf.elem_bytes == 2) {
            const uint16_t* b_src = B_tr->col_u16(j);
            uint16_t* b_dst = ws.B_shuf.col_u16(j);
            for (int i = 0; i < N; ++i) {
              int r = p_idx[i];
              b_dst[i] = b_src[r];
              f_dst[i] = f_src[r];
            }
          } else {
            const int32_t* b_src = B_tr->col_i32(j);
            int32_t* b_dst = ws.B_shuf.col_i32(j);
            for (int i = 0; i < N; ++i) {
              int r = p_idx[i];
              b_dst[i] = b_src[r];
              f_dst[i] = f_src[r];
            }
          }
        } else {
          const float* src = X_num_tr.col(j).data();
          float* dst = ws.X_num_shuf.col(j).data();
          for (int i = 0; i < N; ++i) {
            dst[i] = src[p_idx[i]];
          }
        }
      } else if (task_col < n_num + n_cat) {
        int c = task_col - n_num;
        const int* src = X_cat_tr.col(c).data();
        int* dst = ws.Cat_shuf.col(c).data();
        for (int i = 0; i < N; ++i) {
          dst[i] = src[p_idx[i]];
        }
      } else {
        int c = task_col - n_num - n_cat;
        const float* src = Y_tr.col(c).data();
        float* dst = ws.Y_shuf.col(c).data();
        for (int i = 0; i < N; ++i) {
          dst[i] = src[p_idx[i]];
        }
      }
    }
  }

  static inline bool reduce_col_grad_e(const TabMWorkspace& ws,
                                       int n_threads,
                                       int col_idx,
                                       int cur_B,
                                       float* __restrict__ ge_buf) noexcept {
    bool first_t = true;
    for (int t = 0; t < n_threads; ++t) {
      if (!ws.tls[t].has_work) continue;
      const float* __restrict__ src = ws.tls[t].grad_E.col(col_idx).data();
      if (first_t) {
        std::memcpy(ge_buf, src, static_cast<size_t>(cur_B) * sizeof(float));
        first_t = false;
      } else {
        for (int i = 0; i < cur_B; ++i) {
          ge_buf[i] += src[i];
        }
      }
    }
    return !first_t;
  }

  template <typename BinT>
  inline void backward_ple_single_col_t(int j,
                                        const BinT* b_base, int b_lda,
                                        const float* f_base, int f_lda,
                                        int row_offset, int cur_B,
                                        int n_threads, float* ge_buf,
                                        double cur_lr, double beta1, double beta2, double eps,
                                        double b1_corr, double b2_corr, double weight_decay,
                                        TabMWorkspace& ws) noexcept {
    PLEFeature& pf = ple_features[j];
    if (!pf.active) return;
    Eigen::MatrixXf& gW = ws.ple_grad_W[j];
    Eigen::MatrixXf& gb = ws.ple_grad_b[j];
    Eigen::MatrixXf& diff = ws.ple_diff[j];
    gW.setZero();
    diff.setZero();

    const BinT* b_col = b_base + static_cast<size_t>(j) * static_cast<size_t>(b_lda) + row_offset;
    const float* f_col = f_base + static_cast<size_t>(j) * static_cast<size_t>(f_lda) + row_offset;
    int d_e = pf.d_emb;
    int col_off = pf.out_col_offset;

    for (int e = 0; e < d_e; ++e) {
      if (!reduce_col_grad_e(ws, n_threads, col_off + e, cur_B, ge_buf)) {
        gb(e, 0) = 0.0f;
        continue;
      }
      float* gw_col = gW.col(e).data();
      float* df_col = diff.col(e).data();
      for (int i = 0; i < cur_B; ++i) {
        int b_star = static_cast<int>(b_col[i]);
        float g = ge_buf[i];
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
    pf.W.update_serial(gW, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay);
    pf.b.update_serial(gb, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, 0.0);
    pf.rebuild_prefix();
  }

  inline void backward_cat_single_col(int c,
                                      const int* code_col,
                                      int cur_B,
                                      int n_threads, float* ge_buf,
                                      double cur_lr, double beta1, double beta2, double eps,
                                      double b1_corr, double b2_corr, double weight_decay,
                                      TabMWorkspace& ws) noexcept {
    Eigen::MatrixXf& gE = ws.cat_grad_E[c];
    gE.setZero();
    int col_off = D_num_rep + c * d_embedding;
    for (int e = 0; e < d_embedding; ++e) {
      if (!reduce_col_grad_e(ws, n_threads, col_off + e, cur_B, ge_buf)) continue;
      float* gE_col = gE.col(e).data();
      for (int i = 0; i < cur_B; ++i) {
        gE_col[code_col[i]] += ge_buf[i];
      }
    }
    cat_tables[c].update_serial(gE, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay);
  }

  template <bool UsePwl, typename BinT>
  double train_step_shuffled_impl(int start,
                                  int cur_B,
                                  int step_id,
                                  double cur_lr,
                                  double weight_decay,
                                  double beta1,
                                  double beta2,
                                  double eps,
                                  TabMWorkspace& ws) {
    int K = k_ensemble;
    int cur_KB = K * cur_B;
    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    int total_emb_cols = n_num + n_cat;

    bool do_drop = (dropout > 0.0);
    float keep_prob = static_cast<float>(1.0 - dropout);
    float inv_keep = do_drop ? (1.0f / keep_prob) : 1.0f;
    uint32_t drop_thresh8 = static_cast<uint32_t>(std::clamp(1.0 - dropout, 0.0, 1.0) * 256.0);

    double inv_KB = 1.0 / static_cast<double>(cur_KB);
    float inv_KB_f = static_cast<float>(inv_KB);
    float two_inv_KB = 2.0f * inv_KB_f;

    int omp_nt = 1;
#if defined(_OPENMP)
    omp_nt = std::max(1, omp_get_max_threads());
#endif
    bool use_omp = (omp_nt > 1 && (K >= 2 || total_emb_cols >= 2));
    int n_threads = use_omp ? omp_nt : 1;
    ws.ensure_tls(n_threads);

    double total_loss_sum = 0.0;

    const double b1_corr = 1.0 - std::pow(beta1, step_id);
    const double b2_corr = 1.0 - std::pow(beta2, step_id);
    const float alpha = static_cast<float>(cur_lr * std::sqrt(b2_corr) / b1_corr);
    const float eps_scaled = static_cast<float>(eps * std::sqrt(b2_corr));
    const float b1_f = static_cast<float>(beta1);
    const float one_m_b1 = static_cast<float>(1.0 - beta1);
    const float b2_f = static_cast<float>(beta2);
    const float one_m_b2 = static_cast<float>(1.0 - beta2);
    const float wd_step = static_cast<float>(cur_lr * weight_decay);

    float* e_data = ws.E_batch.data();
    const int e_stride = static_cast<int>(ws.E_batch.outerStride());
    const float* y_data = ws.Y_shuf.data() + start;
    const int y_stride = static_cast<int>(ws.Y_shuf.outerStride());
    const bool use_b_blk_pad = (d_block < 32 || (d_block & 31) != 0);

    const BinT* b_base = nullptr;
    int b_lda = 0;
    const float* f_base = nullptr;
    int f_lda = 0;
    if constexpr (UsePwl) {
      if constexpr (std::is_same_v<BinT, uint8_t>) {
        b_base = ws.B_shuf.u8.data();
      } else if constexpr (std::is_same_v<BinT, uint16_t>) {
        b_base = ws.B_shuf.u16.data();
      } else {
        b_base = ws.B_shuf.i32.data();
      }
      b_lda = ws.B_shuf.lda();
      f_base = ws.F_shuf.data();
      f_lda = static_cast<int>(ws.F_shuf.outerStride());
    }
    const float* xnum_base = (!UsePwl && n_num > 0) ? ws.X_num_shuf.data() : nullptr;
    const int xnum_lda = (!UsePwl && n_num > 0) ? static_cast<int>(ws.X_num_shuf.outerStride()) : 0;
    const int* xcat_base = (n_cat > 0) ? ws.Cat_shuf.data() : nullptr;
    const int xcat_lda = (n_cat > 0) ? static_cast<int>(ws.Cat_shuf.outerStride()) : 0;
    const bool w_out_big = (static_cast<int>(W_out.val.size()) >= 32768);
    const bool parallel_emb_fwd = use_omp && !has_dummy_in && (total_emb_cols >= 2) &&
        (UsePwl ? (static_cast<size_t>(cur_B) * static_cast<size_t>(D_in) >= 16384)
                : (static_cast<size_t>(cur_B) * static_cast<size_t>(n_cat * d_embedding + n_num) >= 32768));

    auto run_emb_fwd_col = [&](int task_col) noexcept {
      if (task_col < n_num) {
        if constexpr (UsePwl) {
          forward_ple_single_col_t<BinT>(task_col, b_base, b_lda, f_base, f_lda, start, cur_B, e_data, e_stride);
        } else {
          std::memcpy(e_data + static_cast<size_t>(task_col) * static_cast<size_t>(e_stride),
                      xnum_base + static_cast<size_t>(task_col) * static_cast<size_t>(xnum_lda) + start,
                      static_cast<size_t>(cur_B) * sizeof(float));
        }
      } else {
        forward_cat_single_col(task_col - n_num, xcat_base, xcat_lda, start, cur_B, e_data, e_stride);
      }
    };

    if (has_dummy_in) {
      std::memset(e_data, 0, static_cast<size_t>(cur_B) * sizeof(float));
    } else if (!parallel_emb_fwd) {
      for (int task_col = 0; task_col < total_emb_cols; ++task_col) {
        run_emb_fwd_col(task_col);
      }
    }

    // Single OpenMP parallel region covering: [optional parallel Emb Fwd] -> MLP Fwd -> Loss ->
    // MLP Bwd -> R_in Bwd -> W_blocks Reduction+AdamW & Fused grad_E/Embeddings Bwd & Bias/R_in Updates
#if defined(_OPENMP)
#pragma omp parallel num_threads(n_threads) if (use_omp)
#endif
    {
#if defined(_OPENMP)
      int tid = use_omp ? omp_get_thread_num() : 0;
#else
      int tid = 0;
#endif
      // 1. Parallel forward embeddings only when work is large enough to amortize OMP barrier
      if (parallel_emb_fwd) {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int task_col = 0; task_col < total_emb_cols; ++task_col) {
          run_emb_fwd_col(task_col);
        }
      }

      // 2-3. Forward MLP & Loss per member k
      TabMWorkspace::ThreadScratch& ts = ws.tls[tid];
      double k_loss_acc = 0.0;
      float k_max_g_acc = 0.0f;
      ts.has_work = false;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
      for (int k = 0; k < K; ++k) {
        forward_mlp_member(ws.E_batch, cur_B, k, do_drop, inv_keep, drop_thresh8, step_id, ws.H_k[k], ws.Out_k[k]);

        const float* out_ptr = ws.Out_k[k].data();
        float* g_ptr = ws.grad_Out_k[k].data();
        int out_stride = static_cast<int>(ws.Out_k[k].outerStride());
        int g_stride = static_cast<int>(ws.grad_Out_k[k].outerStride());

        if (task == "regression") {
          double k_loss = 0.0;
          for (int i = 0; i < cur_B; ++i) {
            float diff = out_ptr[i] - y_data[i];
            k_loss += static_cast<double>(diff) * static_cast<double>(diff);
            float gv = two_inv_KB * diff;
            g_ptr[i] = gv;
            float ag = std::abs(gv);
            if (ag > k_max_g_acc) k_max_g_acc = ag;
          }
          k_loss_acc += k_loss;
        } else if (task == "classification") {
          double k_loss = 0.0;
          for (int i = 0; i < cur_B; ++i) {
            float z = std::clamp(out_ptr[i], -30.0f, 30.0f);
            float az = (z >= 0.0f) ? z : -z;
            float ez = evofe::nn::fast_expf(-az);
            float p = (z >= 0.0f) ? (1.0f / (1.0f + ez)) : (ez / (1.0f + ez));
            float y = y_data[i];
            float gv = inv_KB_f * (p - y);
            g_ptr[i] = gv;
            float ag = std::abs(gv);
            if (ag > k_max_g_acc) k_max_g_acc = ag;
            float pos_z = (z > 0.0f) ? z : 0.0f;
            k_loss += static_cast<double>(pos_z - y * z + std::log1pf(ez));
          }
          k_loss_acc += k_loss;
        } else {
          double k_loss = 0.0;
          float* p_ptr = ws.P_k[k].data();
          int p_stride = static_cast<int>(ws.P_k[k].outerStride());
          float unif = 1.0f / static_cast<float>(output_dim);
          for (int i = 0; i < cur_B; ++i) {
            float max_v = out_ptr[i];
            for (int c = 1; c < output_dim; ++c) {
              float v = out_ptr[i + static_cast<size_t>(c) * static_cast<size_t>(out_stride)];
              if (v > max_v) max_v = v;
            }
            float sum_e = 0.0f;
            for (int c = 0; c < output_dim; ++c) {
              float ep = evofe::nn::fast_expf(out_ptr[i + static_cast<size_t>(c) * static_cast<size_t>(out_stride)] - max_v);
              p_ptr[i + static_cast<size_t>(c) * static_cast<size_t>(p_stride)] = ep;
              sum_e += ep;
            }
            float inv_sum = (sum_e > 0.0f) ? (1.0f / sum_e) : 0.0f;
            for (int c = 0; c < output_dim; ++c) {
              float pr = (sum_e > 0.0f) ? (p_ptr[i + static_cast<size_t>(c) * static_cast<size_t>(p_stride)] * inv_sum) : unif;
              float yc = y_data[i + static_cast<size_t>(c) * static_cast<size_t>(y_stride)];
              if (yc > 0.0f) {
                k_loss -= static_cast<double>(yc) * std::log(std::max(1e-15, static_cast<double>(pr)));
              }
              float gv = inv_KB_f * (pr - yc);
              g_ptr[i + static_cast<size_t>(c) * static_cast<size_t>(g_stride)] = gv;
              float ag = std::abs(gv);
              if (ag > k_max_g_acc) k_max_g_acc = ag;
            }
          }
          k_loss_acc += k_loss;
        }
      }

      ts.loss_sum = k_loss_acc;
      ts.max_g = k_max_g_acc;

#if defined(_OPENMP)
#pragma omp barrier
#endif

      float local_global_max = 0.0f;
      for (int t = 0; t < n_threads; ++t) {
        if (ws.tls[t].max_g > local_global_max) local_global_max = ws.tls[t].max_g;
      }
      float clip_scale = (std::isfinite(local_global_max) && local_global_max > 10.0f) ? (10.0f / local_global_max) : 1.0f;
#if defined(_OPENMP)
#pragma omp single nowait
#endif
      {
        double lsum = 0.0;
        for (int t = 0; t < n_threads; ++t) lsum += ws.tls[t].loss_sum;
        total_loss_sum = lsum;
      }

      // 4-6. Backward pass over the SAME k assignment
      bool first_k = true;
      Eigen::MatrixXf* delta_cur = &ts.delta_a;
      Eigen::MatrixXf* delta_next = &ts.delta_b;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
      for (int k = 0; k < K; ++k) {
        float* g_ptr = ws.grad_Out_k[k].data();
        int g_stride = static_cast<int>(ws.grad_Out_k[k].outerStride());
        if (clip_scale != 1.0f) {
          for (int c = 0; c < output_dim; ++c) {
            float* gc = g_ptr + static_cast<size_t>(c) * static_cast<size_t>(g_stride);
            for (int i = 0; i < cur_B; ++i) gc[i] *= clip_scale;
          }
        }

        const Eigen::MatrixXf& H_last = ws.H_k[k][n_blocks];
        int hlast_stride = static_cast<int>(H_last.outerStride());
        int dcur_stride = static_cast<int>(delta_cur->outerStride());
        const float* hlast_data = H_last.data();
        float* dcur_data = delta_cur->data();

        const float* w_k = W_out.val.data() + static_cast<size_t>(k) * static_cast<size_t>(output_dim) * static_cast<size_t>(d_block);
        float* gw_k = ws.grad_W_out.data() + static_cast<size_t>(k) * static_cast<size_t>(output_dim) * static_cast<size_t>(d_block);
        float* gb_k_pad = ws.grad_b_out_pad.data() + static_cast<size_t>(k) * static_cast<size_t>(ws.b_out_pad_stride);

        if (output_dim == 1) {
          float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
          int i = 0;
          for (; i + 3 < cur_B; i += 4) {
            b0 += g_ptr[i]; b1 += g_ptr[i + 1]; b2 += g_ptr[i + 2]; b3 += g_ptr[i + 3];
          }
          float bsum = (b0 + b1) + (b2 + b3);
          for (; i < cur_B; ++i) bsum += g_ptr[i];
          gb_k_pad[0] = bsum;

          for (int d = 0; d < d_block; ++d) {
            float wd = w_k[d];
            const float* h_col = hlast_data + static_cast<size_t>(d) * static_cast<size_t>(hlast_stride);
            float* d_col = dcur_data + static_cast<size_t>(d) * static_cast<size_t>(dcur_stride);
#if defined(__ARM_NEON) || defined(__aarch64__)
            float32x4_t gs0 = vdupq_n_f32(0.0f), gs1 = vdupq_n_f32(0.0f);
            float32x4_t vwd = vdupq_n_f32(wd);
            int r = 0;
            for (; r + 7 < cur_B; r += 8) {
              float32x4_t g0 = vld1q_f32(g_ptr + r);
              float32x4_t g1 = vld1q_f32(g_ptr + r + 4);
              gs0 = vfmaq_f32(gs0, vld1q_f32(h_col + r), g0);
              gs1 = vfmaq_f32(gs1, vld1q_f32(h_col + r + 4), g1);
              vst1q_f32(d_col + r, vmulq_f32(g0, vwd));
              vst1q_f32(d_col + r + 4, vmulq_f32(g1, vwd));
            }
            float gsum = vaddvq_f32(vaddq_f32(gs0, gs1));
            for (; r < cur_B; ++r) {
              float gi = g_ptr[r];
              gsum += h_col[r] * gi;
              d_col[r] = gi * wd;
            }
            gw_k[d] = gsum;
#else
            float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
            int r = 0;
            for (; r + 3 < cur_B; r += 4) {
              float gi0 = g_ptr[r], gi1 = g_ptr[r + 1], gi2 = g_ptr[r + 2], gi3 = g_ptr[r + 3];
              g0 += h_col[r] * gi0; g1 += h_col[r + 1] * gi1; g2 += h_col[r + 2] * gi2; g3 += h_col[r + 3] * gi3;
              d_col[r] = gi0 * wd; d_col[r + 1] = gi1 * wd; d_col[r + 2] = gi2 * wd; d_col[r + 3] = gi3 * wd;
            }
            float gsum = (g0 + g1) + (g2 + g3);
            for (; r < cur_B; ++r) {
              float gi = g_ptr[r];
              gsum += h_col[r] * gi;
              d_col[r] = gi * wd;
            }
            gw_k[d] = gsum;
#endif
          }
        } else {
          gemm_f32_raw(true, false, d_block, output_dim, cur_B,
                       hlast_data, hlast_stride,
                       g_ptr, g_stride,
                       gw_k, static_cast<int>(ws.grad_W_out.outerStride()),
                       1.0f, 0.0f);
          for (int c = 0; c < output_dim; ++c) {
            const float* gc = g_ptr + static_cast<size_t>(c) * static_cast<size_t>(g_stride);
            float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            int i = 0;
            for (; i + 3 < cur_B; i += 4) {
              b0 += gc[i]; b1 += gc[i + 1]; b2 += gc[i + 2]; b3 += gc[i + 3];
            }
            float bsum = (b0 + b1) + (b2 + b3);
            for (; i < cur_B; ++i) bsum += gc[i];
            gb_k_pad[c] = bsum;
          }
          gemm_f32_raw(false, true, cur_B, d_block, output_dim,
                       g_ptr, g_stride,
                       w_k, static_cast<int>(W_out.val.outerStride()),
                       dcur_data, dcur_stride,
                       1.0f, 0.0f);
        }

        float beta_gw = first_k ? 0.0f : 1.0f;
        for (int l = n_blocks - 1; l >= 0; --l) {
          const TabMBlock& blk = blocks[l];
          const Eigen::MatrixXf& H_out_l = ws.H_k[k][l + 1];
          const Eigen::MatrixXf& H_in_l = ws.H_k[k][l];
          dcur_stride = static_cast<int>(delta_cur->outerStride());
          dcur_data = delta_cur->data();

          float* gb_dst = use_b_blk_pad
              ? (ws.grad_b_blocks_pad[l].data() + static_cast<size_t>(k) * static_cast<size_t>(ws.b_blk_pad_stride))
              : ws.grad_b_blocks[l].col(k).data();
          apply_relu_dropout_bwd_k(H_out_l.data(), static_cast<int>(H_out_l.outerStride()),
                                   dcur_data, dcur_stride, gb_dst,
                                   cur_B, blk.d_out, inv_keep);

          gemm_f32_raw(true, false, blk.d_in, blk.d_out, cur_B,
                       H_in_l.data(), static_cast<int>(H_in_l.outerStride()),
                       dcur_data, dcur_stride,
                       ts.grad_W_blocks[l].data(), static_cast<int>(ts.grad_W_blocks[l].outerStride()),
                       1.0f, beta_gw);

          gemm_f32_raw(false, true, cur_B, blk.d_in, blk.d_out,
                       dcur_data, dcur_stride,
                       blk.W.val.data(), static_cast<int>(blk.W.val.outerStride()),
                       delta_next->data(), static_cast<int>(delta_next->outerStride()),
                       1.0f, 0.0f);

          std::swap(delta_cur, delta_next);
        }

        const float* dH0_data = delta_cur->data();
        int dH0_stride = static_cast<int>(delta_cur->outerStride());
        float* ge_tls_data = ts.grad_E.data();
        int ge_tls_stride = static_cast<int>(ts.grad_E.outerStride());
        const float* r_col = R_in.val.col(k).data();
        float* gr_dst = ws.grad_R_in_pad.data() + static_cast<size_t>(k) * static_cast<size_t>(ws.r_in_pad_stride);

        const int ge_start = (UsePwl || has_dummy_in) ? 0 : (n_cat > 0 ? D_num_rep : D_in);
        for (int d = 0; d < ge_start; ++d) {
          const float* dh_col = dH0_data + static_cast<size_t>(d) * static_cast<size_t>(dH0_stride);
          const float* e_col = e_data + static_cast<size_t>(d) * static_cast<size_t>(e_stride);
#if defined(__ARM_NEON) || defined(__aarch64__)
          float32x4_t sr0 = vdupq_n_f32(0.0f), sr1 = vdupq_n_f32(0.0f);
          int i = 0;
          for (; i + 7 < cur_B; i += 8) {
            sr0 = vfmaq_f32(sr0, vld1q_f32(dh_col + i), vld1q_f32(e_col + i));
            sr1 = vfmaq_f32(sr1, vld1q_f32(dh_col + i + 4), vld1q_f32(e_col + i + 4));
          }
          float sum_r = vaddvq_f32(vaddq_f32(sr0, sr1));
          for (; i < cur_B; ++i) sum_r += dh_col[i] * e_col[i];
          gr_dst[d] = sum_r;
#else
          float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
          int i = 0;
          for (; i + 3 < cur_B; i += 4) {
            s0 += dh_col[i] * e_col[i]; s1 += dh_col[i + 1] * e_col[i + 1];
            s2 += dh_col[i + 2] * e_col[i + 2]; s3 += dh_col[i + 3] * e_col[i + 3];
          }
          float sum_r = (s0 + s1) + (s2 + s3);
          for (; i < cur_B; ++i) sum_r += dh_col[i] * e_col[i];
          gr_dst[d] = sum_r;
#endif
        }

        for (int d = ge_start; d < D_in; ++d) {
          float r_kd = r_col[d];
          const float* dh_col = dH0_data + static_cast<size_t>(d) * static_cast<size_t>(dH0_stride);
          const float* e_col = e_data + static_cast<size_t>(d) * static_cast<size_t>(e_stride);
          float* ge_col = ge_tls_data + static_cast<size_t>(d) * static_cast<size_t>(ge_tls_stride);
#if defined(__ARM_NEON) || defined(__aarch64__)
          float32x4_t sr0 = vdupq_n_f32(0.0f), sr1 = vdupq_n_f32(0.0f);
          float32x4_t vr = vdupq_n_f32(r_kd);
          int i = 0;
          if (first_k) {
            for (; i + 7 < cur_B; i += 8) {
              float32x4_t dh0 = vld1q_f32(dh_col + i);
              float32x4_t dh1 = vld1q_f32(dh_col + i + 4);
              sr0 = vfmaq_f32(sr0, dh0, vld1q_f32(e_col + i));
              sr1 = vfmaq_f32(sr1, dh1, vld1q_f32(e_col + i + 4));
              vst1q_f32(ge_col + i, vmulq_f32(dh0, vr));
              vst1q_f32(ge_col + i + 4, vmulq_f32(dh1, vr));
            }
            float sum_r = vaddvq_f32(vaddq_f32(sr0, sr1));
            for (; i < cur_B; ++i) {
              float dh = dh_col[i];
              sum_r += dh * e_col[i];
              ge_col[i] = dh * r_kd;
            }
            gr_dst[d] = sum_r;
          } else {
            for (; i + 7 < cur_B; i += 8) {
              float32x4_t dh0 = vld1q_f32(dh_col + i);
              float32x4_t dh1 = vld1q_f32(dh_col + i + 4);
              sr0 = vfmaq_f32(sr0, dh0, vld1q_f32(e_col + i));
              sr1 = vfmaq_f32(sr1, dh1, vld1q_f32(e_col + i + 4));
              vst1q_f32(ge_col + i, vfmaq_f32(vld1q_f32(ge_col + i), dh0, vr));
              vst1q_f32(ge_col + i + 4, vfmaq_f32(vld1q_f32(ge_col + i + 4), dh1, vr));
            }
            float sum_r = vaddvq_f32(vaddq_f32(sr0, sr1));
            for (; i < cur_B; ++i) {
              float dh = dh_col[i];
              sum_r += dh * e_col[i];
              ge_col[i] += dh * r_kd;
            }
            gr_dst[d] = sum_r;
          }
#else
          float sum_r = 0.0f;
          if (first_k) {
            for (int i = 0; i < cur_B; ++i) {
              float dh = dh_col[i];
              sum_r += dh * e_col[i];
              ge_col[i] = dh * r_kd;
            }
          } else {
            for (int i = 0; i < cur_B; ++i) {
              float dh = dh_col[i];
              sum_r += dh * e_col[i];
              ge_col[i] += dh * r_kd;
            }
          }
          gr_dst[d] = sum_r;
#endif
        }
        first_k = false;
        ts.has_work = true;
      }

      int active_t[64];
      int n_act = 0;
      if (n_threads <= 64) {
        for (int t = 0; t < n_threads; ++t) {
          if (ws.tls[t].has_work) active_t[n_act++] = t;
        }
      }

      // 7a. Fused reduction of thread-local grad_W_blocks[l] + AdamW update (nowait)
      for (int l = 0; l < n_blocks; ++l) {
        int sz = static_cast<int>(blocks[l].W.val.size());
        float* __restrict__ val_p = blocks[l].W.val.data();
        float* __restrict__ m_p = blocks[l].W.m.data();
        float* __restrict__ v_p = blocks[l].W.v.data();
        const float* gw_ptrs[64];
        if (n_threads <= 64) {
          for (int a = 0; a < n_act; ++a) {
            gw_ptrs[a] = ws.tls[active_t[a]].grad_W_blocks[l].data();
          }
        }

#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
        for (int idx = 0; idx < sz; ++idx) {
          float g = 0.0f;
          if (n_threads <= 64) {
            for (int a = 0; a < n_act; ++a) g += gw_ptrs[a][idx];
          } else {
            for (int t = 0; t < n_threads; ++t) {
              if (ws.tls[t].has_work) g += ws.tls[t].grad_W_blocks[l].data()[idx];
            }
          }
          float m_val = b1_f * m_p[idx] + one_m_b1 * g;
          float v_val = b2_f * v_p[idx] + one_m_b2 * g * g;
          m_p[idx] = m_val;
          v_p[idx] = v_val;
          float u = alpha * m_val / (std::sqrt(std::max(0.0f, v_val)) + eps_scaled);
          if (wd_step > 0.0f) {
            val_p[idx] = val_p[idx] * (1.0f - wd_step) - u;
          } else {
            val_p[idx] -= u;
          }
        }
      }

      // 8. Fused col grad_E reduction + Backward & AdamW for PLE + Cat tables (no intermediate barrier)
      const int bwd_emb_cols = has_dummy_in ? 0 : (UsePwl ? (n_num + n_cat) : n_cat);
      if (bwd_emb_cols > 0) {
        float* col_ge_buf = ts.col_ge.data();
#if defined(_OPENMP)
#pragma omp for schedule(static) nowait
#endif
        for (int task_col = 0; task_col < bwd_emb_cols; ++task_col) {
          if constexpr (UsePwl) {
            if (task_col < n_num) {
              backward_ple_single_col_t<BinT>(task_col, b_base, b_lda, f_base, f_lda,
                                              start, cur_B, n_threads, col_ge_buf,
                                              cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay, ws);
            } else {
              int c = task_col - n_num;
              backward_cat_single_col(c, xcat_base + static_cast<size_t>(c) * static_cast<size_t>(xcat_lda) + start,
                                      cur_B, n_threads, col_ge_buf,
                                      cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay, ws);
            }
          } else {
            int c = task_col;
            backward_cat_single_col(c, xcat_base + static_cast<size_t>(c) * static_cast<size_t>(xcat_lda) + start,
                                    cur_B, n_threads, col_ge_buf,
                                    cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay, ws);
          }
        }
      }

#if defined(_OPENMP)
#pragma omp single nowait
#endif
      {
        for (int k = 0; k < K; ++k) {
          std::memcpy(ws.grad_R_in.col(k).data(),
                      ws.grad_R_in_pad.data() + static_cast<size_t>(k) * static_cast<size_t>(ws.r_in_pad_stride),
                      static_cast<size_t>(D_in) * sizeof(float));
          std::memcpy(ws.grad_b_out.col(k).data(),
                      ws.grad_b_out_pad.data() + static_cast<size_t>(k) * static_cast<size_t>(ws.b_out_pad_stride),
                      static_cast<size_t>(output_dim) * sizeof(float));
          if (use_b_blk_pad) {
            for (int l = 0; l < n_blocks; ++l) {
              std::memcpy(ws.grad_b_blocks[l].col(k).data(),
                          ws.grad_b_blocks_pad[l].data() + static_cast<size_t>(k) * static_cast<size_t>(ws.b_blk_pad_stride),
                          static_cast<size_t>(d_block) * sizeof(float));
            }
          }
        }
        R_in.update_serial(ws.grad_R_in, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, 0.0);
        for (int l = 0; l < n_blocks; ++l) {
          blocks[l].b.update_serial(ws.grad_b_blocks[l], cur_lr, beta1, beta2, eps, b1_corr, b2_corr, 0.0);
        }
        if (!w_out_big) {
          W_out.update_serial(ws.grad_W_out, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay);
        }
        b_out.update_serial(ws.grad_b_out, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, 0.0);
      }
    } // end single omp parallel region

    if (w_out_big) {
      W_out.update(ws.grad_W_out, cur_lr, beta1, beta2, eps, b1_corr, b2_corr, weight_decay);
    }

    return total_loss_sum * inv_KB;
  }

  // Single minibatch training step on contiguous epoch-shuffled batch [start..start+cur_B)
  double train_step_shuffled(int start,
                             int cur_B,
                             int step_id,
                             double cur_lr,
                             double weight_decay,
                             double beta1,
                             double beta2,
                             double eps,
                             TabMWorkspace& ws) {
    int n_num = static_cast<int>(num_col_indices.size());
    const bool use_pwl = (num_emb == "pwl" && n_num > 0 && !ws.B_shuf.empty());
    if (use_pwl) {
      if (ws.B_shuf.elem_bytes == 1) {
        return train_step_shuffled_impl<true, uint8_t>(start, cur_B, step_id, cur_lr, weight_decay, beta1, beta2, eps, ws);
      } else if (ws.B_shuf.elem_bytes == 2) {
        return train_step_shuffled_impl<true, uint16_t>(start, cur_B, step_id, cur_lr, weight_decay, beta1, beta2, eps, ws);
      } else {
        return train_step_shuffled_impl<true, int32_t>(start, cur_B, step_id, cur_lr, weight_decay, beta1, beta2, eps, ws);
      }
    }
    return train_step_shuffled_impl<false, uint8_t>(start, cur_B, step_id, cur_lr, weight_decay, beta1, beta2, eps, ws);
  }

  // Incremental delta-A1 occlusion importance in shared representation space E mapped back to original D columns
  std::vector<double> compute_importances(const Eigen::MatrixXf& X_num_src,
                                          const Eigen::MatrixXi& X_cat_src,
                                          const double* y_eval,
                                          int N_eval,
                                          const PLEBinMatrix* B_pre = nullptr,
                                          const Eigen::MatrixXf* F_pre = nullptr) const {
    int D = n_features;
    std::vector<double> imp(D, 1.0 / std::max(1, D));
    if (D <= 0 || N_eval <= 0 || y_eval == nullptr || has_dummy_in) {
      return imp;
    }

    int n_num = static_cast<int>(num_col_indices.size());
    int n_cat = static_cast<int>(cat_col_indices.size());
    int K = k_ensemble;

    Eigen::MatrixXf E_base(N_eval, D_in);
    const float* xn_ptr = (n_num > 0) ? X_num_src.data() : nullptr;
    const int xn_lda = (n_num > 0) ? static_cast<int>(X_num_src.outerStride()) : 0;
    const int* xc_ptr = (n_cat > 0) ? X_cat_src.data() : nullptr;
    const int xc_lda = (n_cat > 0) ? static_cast<int>(X_cat_src.outerStride()) : 0;
    const bool has_pre = (num_emb == "pwl" && B_pre != nullptr && F_pre != nullptr &&
                          B_pre->rows() == N_eval && F_pre->rows() == N_eval);

    if (has_pre) {
      forward_embeddings_prebinned(
        xn_ptr, xn_lda,
        n_num > 0 ? B_pre : nullptr,
        n_num > 0 ? F_pre->data() : nullptr, n_num > 0 ? static_cast<int>(F_pre->outerStride()) : 0,
        xc_ptr, xc_lda, 0, N_eval, E_base
      );
    } else {
      PLEBinMatrix tmp_b;
      Eigen::MatrixXf tmp_f;
      if (num_emb == "pwl" && n_num > 0) {
        tmp_b.resize(N_eval, n_num, max_ple_bins());
        tmp_f.resize(N_eval, n_num);
        encode_ple_chunk(xn_ptr, xn_lda, 0, N_eval, tmp_b, tmp_f);
      }
      forward_embeddings_prebinned(
        xn_ptr, xn_lda,
        (num_emb == "pwl" && n_num > 0) ? &tmp_b : nullptr,
        (num_emb == "pwl" && n_num > 0) ? tmp_f.data() : nullptr,
        (num_emb == "pwl" && n_num > 0) ? static_cast<int>(tmp_f.outerStride()) : 0,
        xc_ptr, xc_lda, 0, N_eval, E_base
      );
    }

    std::vector<Eigen::MatrixXf> W_eff;
    build_w_eff(W_eff);

    std::vector<Eigen::MatrixXf> A1_base(K);
    for (int k = 0; k < K; ++k) {
      A1_base[k].resize(N_eval, d_block);
    }

    int omp_nt = 1;
#if defined(_OPENMP)
    omp_nt = std::max(1, omp_get_max_threads());
#endif
    int n_t_base = (omp_nt > 1 && K >= 2) ? std::min(omp_nt, K) : 1;
    std::vector<std::vector<double>> base_accs(n_t_base, std::vector<double>(static_cast<size_t>(N_eval) * static_cast<size_t>(output_dim), 0.0));

#if defined(_OPENMP)
#pragma omp parallel num_threads(n_t_base) if (n_t_base > 1)
#endif
    {
#if defined(_OPENMP)
      int tid = (n_t_base > 1) ? omp_get_thread_num() : 0;
#else
      int tid = 0;
#endif
      Eigen::MatrixXf H_a(N_eval, d_block);
      Eigen::MatrixXf H_b(N_eval, d_block);
      Eigen::MatrixXf Out_buf(N_eval, output_dim);
      std::vector<float> exps(output_dim > 1 ? output_dim : 0);
      double* p_acc = base_accs[tid].data();
      float* exps_p = (output_dim > 1) ? exps.data() : nullptr;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
      for (int k = 0; k < K; ++k) {
        gemm_f32_raw(false, false, N_eval, d_block, D_in,
                     E_base.data(), static_cast<int>(E_base.outerStride()),
                     W_eff[k].data(), static_cast<int>(W_eff[k].outerStride()),
                     A1_base[k].data(), static_cast<int>(A1_base[k].outerStride()),
                     1.0f, 0.0f);
        H_a = A1_base[k];
        forward_from_a1_member(k, N_eval, &H_a, &H_b, Out_buf, p_acc, exps_p);
      }
    }

    Eigen::MatrixXd base_preds(N_eval, output_dim);
    std::vector<const double*> base_ptrs(n_t_base);
    for (int t = 0; t < n_t_base; ++t) base_ptrs[t] = base_accs[t].data();
    finalize_preds_rows(N_eval, 0, base_ptrs, n_t_base, base_preds);

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

    struct SliceTask {
      int orig_col;
      int col_start;
      int width;
    };
    std::vector<SliceTask> slices;
    slices.reserve(n_num + n_cat);
    int max_width = 1;
    for (int j = 0; j < n_num; ++j) {
      int orig_col = num_col_indices[j];
      if (num_emb == "pwl") {
        if (ple_features[j].active) {
          slices.push_back({ orig_col, ple_features[j].out_col_offset, d_embedding });
          if (d_embedding > max_width) max_width = d_embedding;
        }
      } else {
        slices.push_back({ orig_col, j, 1 });
      }
    }
    for (int c = 0; c < n_cat; ++c) {
      int orig_col = cat_col_indices[c];
      slices.push_back({ orig_col, D_num_rep + c * d_embedding, d_embedding });
      if (d_embedding > max_width) max_width = d_embedding;
    }

    std::fill(imp.begin(), imp.end(), 0.0);
    int n_slices = static_cast<int>(slices.size());
    int n_t_imp = (omp_nt > 1 && n_slices >= 2) ? std::min(omp_nt, n_slices) : 1;

#if defined(_OPENMP)
#pragma omp parallel num_threads(n_t_imp) if (n_t_imp > 1)
#endif
    {
      Eigen::MatrixXf Delta_E(N_eval, max_width);
      Eigen::MatrixXf H_a(N_eval, d_block);
      Eigen::MatrixXf H_b(N_eval, d_block);
      Eigen::MatrixXf Out_buf(N_eval, output_dim);
      std::vector<float> exps(output_dim > 1 ? output_dim : 0);
      std::vector<double> acc(static_cast<size_t>(N_eval) * static_cast<size_t>(output_dim), 0.0);
      Eigen::MatrixXd occ_preds(N_eval, output_dim);
      const double* acc_ptr = acc.data();
      std::vector<const double*> one_ptr(1, acc_ptr);
      float* exps_p = (output_dim > 1) ? exps.data() : nullptr;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
      for (int s = 0; s < n_slices; ++s) {
        int col_start = slices[s].col_start;
        int width = slices[s].width;
        if (width <= 0 || col_start < 0 || col_start + width > D_in) continue;

        for (int u = 0; u < width; ++u) {
          const float* src = E_base.col(col_start + u).data();
          float sum = 0.0f;
          for (int i = 0; i < N_eval; ++i) sum += src[i];
          float mean_u = sum / static_cast<float>(N_eval);
          float* dst = Delta_E.col(u).data();
          for (int i = 0; i < N_eval; ++i) dst[i] = mean_u - src[i];
        }

        std::fill(acc.begin(), acc.end(), 0.0);
        for (int k = 0; k < K; ++k) {
          std::memcpy(H_a.data(), A1_base[k].data(), static_cast<size_t>(A1_base[k].size()) * sizeof(float));
          gemm_f32_raw(false, false, N_eval, d_block, width,
                       Delta_E.data(), static_cast<int>(Delta_E.outerStride()),
                       W_eff[k].data() + col_start, static_cast<int>(W_eff[k].outerStride()),
                       H_a.data(), static_cast<int>(H_a.outerStride()),
                       1.0f, 1.0f);
          forward_from_a1_member(k, N_eval, &H_a, &H_b, Out_buf, acc.data(), exps_p);
        }

        finalize_preds_rows(N_eval, 0, one_ptr, 1, occ_preds);
        double occ_loss = eval_loss(occ_preds);
        double pred_shift = (occ_preds - base_preds).cwiseAbs().mean();
        imp[slices[s].orig_col] = std::max(0.0, occ_loss - base_loss) + 0.1 * pred_shift;
      }
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
