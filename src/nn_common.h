#ifndef EVOFE_NN_COMMON_H
#define EVOFE_NN_COMMON_H

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

// M_PI is POSIX, not C++17 standard — provide fallback for strict compilers
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if defined(__APPLE__)
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
// Include cblas_new.h and vForce.h directly instead of <Accelerate/Accelerate.h>
// because Accelerate.h pulls in Fortran blas_new.h which conflicts with RcppEigen's blas.h.
#include <vecLib/cblas_new.h>
#include <vecLib/vForce.h>
#endif

namespace evofe {
namespace nn {

// Fast SIMD-vectorizable IEEE-754 + 5th-degree minimax polynomial expf (relative error < 2e-7, NaN-safe)
static inline float fast_expf(float x) noexcept {
  if (!(x >= -87.3f)) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  float z = x * 1.4426950408889634f;
  float k = std::nearbyintf(z);
  // Two-step Cody-Waite range reduction for ln(2)
  float f = (x - k * 0.693145751953125f) - k * 1.428606765330187e-6f;
  float p = 1.0f + f * (1.0f + f * (0.5000009f + f * (0.1666680f + f * (0.0416520f + f * 0.0083298f))));
  int32_t ki = static_cast<int32_t>(k) + 127;
  uint32_t bits = static_cast<uint32_t>(ki) << 23;
  float scale;
  std::memcpy(&scale, &bits, sizeof(float));
  return p * scale;
}

// Raw-pointer ColMajor FP32 GEMM wrapper: dispatches to Apple Accelerate cblas_sgemm$NEWLAPACK
// on macOS (bypassing Fortran sgemm_ shadowing by R's libopenblas), with Eigen fallback elsewhere.
inline void gemm_f32_raw(bool transA, bool transB, int M, int N, int K,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float* C, int ldc,
                         float alpha = 1.0f, float beta = 0.0f) {
  if (M <= 0 || N <= 0 || K <= 0) return;
#if defined(__APPLE__)
  cblas_sgemm(CblasColMajor,
              transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans,
              M, N, K, alpha,
              A, lda,
              B, ldb,
              beta,
              C, ldc);
#else
  using ConstMatMap = Eigen::Map<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>;
  using MatMap = Eigen::Map<Eigen::MatrixXf, 0, Eigen::OuterStride<>>;
  ConstMatMap A_map(A, transA ? K : M, transA ? M : K, Eigen::OuterStride<>(lda));
  ConstMatMap B_map(B, transB ? N : K, transB ? K : N, Eigen::OuterStride<>(ldb));
  MatMap C_map(C, M, N, Eigen::OuterStride<>(ldc));
  if (beta == 0.0f) {
    if (!transA && !transB) {
      C_map.noalias() = (A_map * B_map) * alpha;
    } else if (!transA && transB) {
      C_map.noalias() = (A_map * B_map.transpose()) * alpha;
    } else if (transA && !transB) {
      C_map.noalias() = (A_map.transpose() * B_map) * alpha;
    } else {
      C_map.noalias() = (A_map.transpose() * B_map.transpose()) * alpha;
    }
  } else {
    if (beta != 1.0f) {
      C_map *= beta;
    }
    if (!transA && !transB) {
      C_map.noalias() += (A_map * B_map) * alpha;
    } else if (!transA && transB) {
      C_map.noalias() += (A_map * B_map.transpose()) * alpha;
    } else if (transA && !transB) {
      C_map.noalias() += (A_map.transpose() * B_map) * alpha;
    } else {
      C_map.noalias() += (A_map.transpose() * B_map.transpose()) * alpha;
    }
  }
#endif
}

// Eigen::MatrixXf convenience wrapper preserving realmlp call sites (beta defaults to 0.0f)
inline void gemm_f32(bool transA, bool transB, int M, int N, int K,
                     const Eigen::MatrixXf& A, const Eigen::MatrixXf& B,
                     Eigen::MatrixXf& C, float alpha = 1.0f, float beta = 0.0f) {
  gemm_f32_raw(transA, transB, M, N, K,
               A.data(), static_cast<int>(A.outerStride()),
               B.data(), static_cast<int>(B.outerStride()),
               C.data(), static_cast<int>(C.outerStride()),
               alpha, beta);
}

// Symmetric rank-k update into lower triangle of ColMajor N x N matrix C:
// C := alpha * op(A) * op(A)^T + beta * C, where op(A) is N x K.
inline void syrk_lower_f32(bool trans, int N, int K,
                           const float* A, int lda,
                           float* C, int ldc,
                           float alpha = 1.0f, float beta = 0.0f) {
  if (N <= 0 || K <= 0) return;
#if defined(__APPLE__)
  cblas_ssyrk(CblasColMajor, CblasLower,
              trans ? CblasTrans : CblasNoTrans,
              N, K, alpha, A, lda, beta, C, ldc);
#else
  using ConstMatMap = Eigen::Map<const Eigen::MatrixXf, 0, Eigen::OuterStride<>>;
  using MatMap = Eigen::Map<Eigen::MatrixXf, 0, Eigen::OuterStride<>>;
  ConstMatMap A_map(A, trans ? K : N, trans ? N : K, Eigen::OuterStride<>(lda));
  MatMap C_map(C, N, N, Eigen::OuterStride<>(ldc));
  if (beta == 0.0f) {
    C_map.triangularView<Eigen::Lower>().setZero();
  } else if (beta != 1.0f) {
    C_map.triangularView<Eigen::Lower>() *= beta;
  }
  if (trans) {
    C_map.selfadjointView<Eigen::Lower>().rankUpdate(A_map.transpose(), alpha);
  } else {
    C_map.selfadjointView<Eigen::Lower>().rankUpdate(A_map, alpha);
  }
#endif
}

// Vectorized FP32 math wrappers (Apple Accelerate vForce on macOS, scalar fallback elsewhere)
inline void vcos_f32(int n, const float* in, float* out) noexcept {
  if (n <= 0) return;
#if defined(__APPLE__)
  vvcosf(out, in, &n);
#else
  for (int i = 0; i < n; ++i) {
    out[i] = std::cos(in[i]);
  }
#endif
}

inline void vsin_f32(int n, const float* in, float* out) noexcept {
  if (n <= 0) return;
#if defined(__APPLE__)
  vvsinf(out, in, &n);
#else
  for (int i = 0; i < n; ++i) {
    out[i] = std::sin(in[i]);
  }
#endif
}

inline void vexp_f32(int n, const float* in, float* out) noexcept {
  if (n <= 0) return;
#if defined(__APPLE__)
  vvexpf(out, in, &n);
#else
  for (int i = 0; i < n; ++i) {
    out[i] = std::exp(in[i]);
  }
#endif
}

// Supported FP32 neural network activations
enum class ActType : int {
  SELU = 0,
  Mish = 1,
  ReLU = 2,
  SiLU = 3
};

namespace detail {

constexpr float SELU_LAMBDA = 1.0507009873554804934193349852946f;
constexpr float SELU_ALPHA  = 1.6732632423543772848170429916717f;
constexpr float SELU_LAMBDA_ALPHA = SELU_LAMBDA * SELU_ALPHA;

template <ActType Act>
static inline float eval_act_scalar(float z) noexcept {
  if constexpr (Act == ActType::SELU) {
    return (z > 0.0f) ? (SELU_LAMBDA * z) : (SELU_LAMBDA_ALPHA * (fast_expf(z) - 1.0f));
  } else if constexpr (Act == ActType::Mish) {
    if (z > 20.0f) return z;
    if (z < -20.0f) return z * fast_expf(z);
    float ez = fast_expf(z);
    float w = 1.0f + ez;
    float tsp = 1.0f - 2.0f / (w * w + 1.0f);
    return z * tsp;
  } else if constexpr (Act == ActType::ReLU) {
    return (z > 0.0f) ? z : 0.0f;
  } else {
    if (z > 20.0f) return z;
    if (z < -20.0f) return z * fast_expf(z);
    return z / (1.0f + fast_expf(-z));
  }
}

template <ActType Act>
static inline float eval_act_grad_scalar(float a) noexcept {
  if constexpr (Act == ActType::SELU) {
    return (a > 0.0f) ? SELU_LAMBDA : (SELU_LAMBDA_ALPHA * fast_expf(a));
  } else if constexpr (Act == ActType::Mish) {
    if (a > 20.0f) return 1.0f;
    if (a < -20.0f) return fast_expf(a) * (1.0f + a);
    float ea = fast_expf(a);
    float w = 1.0f + ea;
    float tsp = 1.0f - 2.0f / (w * w + 1.0f);
    float sig = ea / w;
    return tsp + a * sig * (1.0f - tsp * tsp);
  } else if constexpr (Act == ActType::ReLU) {
    return (a > 0.0f) ? 1.0f : 0.0f;
  } else {
    if (a > 20.0f) return 1.0f;
    if (a < -20.0f) return fast_expf(a) * (1.0f + a);
    float sig = 1.0f / (1.0f + fast_expf(-a));
    return sig * (1.0f + a * (1.0f - sig));
  }
}

template <ActType Act, bool HasBias>
inline void apply_activation_impl(const Eigen::Ref<const Eigen::MatrixXf>& in,
                                  const float* b_data,
                                  Eigen::Ref<Eigen::MatrixXf> out) {
  int R = static_cast<int>(in.rows());
  int C = static_cast<int>(in.cols());
  int in_stride = static_cast<int>(in.outerStride());
  int out_stride = static_cast<int>(out.outerStride());
  const float* in_data = in.data();
  float* out_data = out.data();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (C >= (Act == ActType::SELU ? 4 : 2))
#endif
  for (int c = 0; c < C; ++c) {
    float b = HasBias ? b_data[c] : 0.0f;
    const float* in_col = in_data + c * in_stride;
    float* out_col = out_data + c * out_stride;
    for (int r = 0; r < R; ++r) {
      float z = HasBias ? (in_col[r] + b) : in_col[r];
      out_col[r] = eval_act_scalar<Act>(z);
    }
  }
}

template <ActType Act, bool HasBias, bool AccumBias>
inline void apply_activation_grad_impl(const Eigen::Ref<const Eigen::MatrixXf>& act_in,
                                       const float* b_data,
                                       Eigen::Ref<Eigen::MatrixXf> delta,
                                       float* gb_data,
                                       const Eigen::Ref<const Eigen::MatrixXf>* act_out) {
  int R = static_cast<int>(delta.rows());
  int C = static_cast<int>(delta.cols());
  int d_stride = static_cast<int>(delta.outerStride());
  float* d_data = delta.data();

  // SELU & ReLU derivatives depend only on forward output H, avoiding expf or pre-activation reads.
  if constexpr (Act == ActType::SELU || Act == ActType::ReLU) {
    if (act_out != nullptr) {
      int h_stride = static_cast<int>(act_out->outerStride());
      const float* h_data = act_out->data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (C >= 2)
#endif
      for (int c = 0; c < C; ++c) {
        const float* h_col = h_data + c * h_stride;
        float* d_col = d_data + c * d_stride;
        float sum = 0.0f;
        for (int r = 0; r < R; ++r) {
          float h = h_col[r];
          float factor;
          if constexpr (Act == ActType::SELU) {
            factor = (h > 0.0f) ? SELU_LAMBDA : (h + SELU_LAMBDA_ALPHA);
          } else {
            factor = (h > 0.0f) ? 1.0f : 0.0f;
          }
          float d = d_col[r] * factor;
          d_col[r] = d;
          if constexpr (AccumBias) {
            sum += d;
          }
        }
        if constexpr (AccumBias) {
          gb_data[c] = sum;
        }
      }
      return;
    }
  }

  int a_stride = static_cast<int>(act_in.outerStride());
  const float* a_data = act_in.data();
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (C >= 2)
#endif
  for (int c = 0; c < C; ++c) {
    float b = HasBias ? b_data[c] : 0.0f;
    const float* a_col = a_data + c * a_stride;
    float* d_col = d_data + c * d_stride;
    float sum = 0.0f;
    for (int r = 0; r < R; ++r) {
      float a = HasBias ? (a_col[r] + b) : a_col[r];
      float d = d_col[r] * eval_act_grad_scalar<Act>(a);
      d_col[r] = d;
      if constexpr (AccumBias) {
        sum += d;
      }
    }
    if constexpr (AccumBias) {
      gb_data[c] = sum;
    }
  }
}

template <bool Parallel, bool WithWD>
inline void adamw_step_impl(float* __restrict__ val_ptr,
                            float* __restrict__ m_ptr,
                            float* __restrict__ v_ptr,
                            const float* __restrict__ g_ptr,
                            int sz,
                            float alpha, float eps_scaled,
                            float b1_f, float one_m_b1,
                            float b2_f, float one_m_b2,
                            float wd_step) noexcept {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (Parallel && sz >= 1024)
#endif
  for (int i = 0; i < sz; ++i) {
    float g = g_ptr[i];
    float m_val = b1_f * m_ptr[i] + one_m_b1 * g;
    float v_val = b2_f * v_ptr[i] + one_m_b2 * g * g;
    m_ptr[i] = m_val;
    v_ptr[i] = v_val;
    float step = alpha * m_val / (std::sqrt(std::max(0.0f, v_val)) + eps_scaled);
    if constexpr (WithWD) {
      val_ptr[i] = val_ptr[i] * (1.0f - wd_step) - step;
    } else {
      val_ptr[i] -= step;
    }
  }
}

} // namespace detail

inline void apply_activation(const Eigen::Ref<const Eigen::MatrixXf>& in,
                             Eigen::Ref<Eigen::MatrixXf> out,
                             ActType act) {
  switch (act) {
    case ActType::SELU: detail::apply_activation_impl<ActType::SELU, false>(in, nullptr, out); break;
    case ActType::Mish: detail::apply_activation_impl<ActType::Mish, false>(in, nullptr, out); break;
    case ActType::ReLU: detail::apply_activation_impl<ActType::ReLU, false>(in, nullptr, out); break;
    case ActType::SiLU: detail::apply_activation_impl<ActType::SiLU, false>(in, nullptr, out); break;
  }
}

inline void apply_activation_grad_inplace(const Eigen::Ref<const Eigen::MatrixXf>& act_in,
                                          Eigen::Ref<Eigen::MatrixXf> delta,
                                          ActType act,
                                          const Eigen::Ref<const Eigen::MatrixXf>* act_out = nullptr) {
  switch (act) {
    case ActType::SELU: detail::apply_activation_grad_impl<ActType::SELU, false, false>(act_in, nullptr, delta, nullptr, act_out); break;
    case ActType::Mish: detail::apply_activation_grad_impl<ActType::Mish, false, false>(act_in, nullptr, delta, nullptr, act_out); break;
    case ActType::ReLU: detail::apply_activation_grad_impl<ActType::ReLU, false, false>(act_in, nullptr, delta, nullptr, act_out); break;
    case ActType::SiLU: detail::apply_activation_grad_impl<ActType::SiLU, false, false>(act_in, nullptr, delta, nullptr, act_out); break;
  }
}

inline void apply_activation_with_bias(const Eigen::Ref<const Eigen::MatrixXf>& in,
                                       const Eigen::Ref<const Eigen::MatrixXf>& bias,
                                       Eigen::Ref<Eigen::MatrixXf> out,
                                       ActType act) {
  const float* b_data = bias.data();
  switch (act) {
    case ActType::SELU: detail::apply_activation_impl<ActType::SELU, true>(in, b_data, out); break;
    case ActType::Mish: detail::apply_activation_impl<ActType::Mish, true>(in, b_data, out); break;
    case ActType::ReLU: detail::apply_activation_impl<ActType::ReLU, true>(in, b_data, out); break;
    case ActType::SiLU: detail::apply_activation_impl<ActType::SiLU, true>(in, b_data, out); break;
  }
}

inline void apply_activation_grad_and_accum_bias(const Eigen::Ref<const Eigen::MatrixXf>& act_in,
                                                 const Eigen::Ref<const Eigen::MatrixXf>& bias,
                                                 Eigen::Ref<Eigen::MatrixXf> delta,
                                                 Eigen::MatrixXf& grad_b,
                                                 ActType act,
                                                 const Eigen::Ref<const Eigen::MatrixXf>* act_out = nullptr) {
  const float* b_data = bias.data();
  float* gb_data = grad_b.data();
  switch (act) {
    case ActType::SELU: detail::apply_activation_grad_impl<ActType::SELU, true, true>(act_in, b_data, delta, gb_data, act_out); break;
    case ActType::Mish: detail::apply_activation_grad_impl<ActType::Mish, true, true>(act_in, b_data, delta, gb_data, act_out); break;
    case ActType::ReLU: detail::apply_activation_grad_impl<ActType::ReLU, true, true>(act_in, b_data, delta, gb_data, act_out); break;
    case ActType::SiLU: detail::apply_activation_grad_impl<ActType::SiLU, true, true>(act_in, b_data, delta, gb_data, act_out); break;
  }
}

// Multi-cycle Coslog schedule as defined in PyTabKit (David Holzmüller et al.):
// Combines cosine annealing with logarithmic cycle layout across n_cycles.
inline double coslog_schedule(double t, int n_cycles = 4) {
  t = std::clamp(t, 0.0, 1.0);
  double scale = std::pow(2.0, n_cycles) - 1.0;
  double u = std::log2(1.0 + scale * t);
  return 0.5 * (1.0 - std::cos(2.0 * M_PI * u));
}

inline double coslog4_schedule(double t) {
  return coslog_schedule(t, 4);
}

// Linear warmup + cosine decay schedule (for AdamW training in TabM)
inline double cosine_warmup_schedule(double t, double warmup_frac = 0.05, double min_lr_ratio = 0.0) noexcept {
  t = std::clamp(t, 0.0, 1.0);
  double wf = std::clamp(warmup_frac, 0.0, 0.5);
  if (wf > 0.0 && t < wf) {
    return t / wf;
  }
  double denom = std::max(1e-12, 1.0 - wf);
  double u = (t - wf) / denom;
  double cos_decay = 0.5 * (1.0 + std::cos(M_PI * u));
  return min_lr_ratio + (1.0 - min_lr_ratio) * cos_decay;
}

// Single-element Adam step helper (used in elementwise fused embedding updates)
static inline void adam_scalar_step(float& val, float& m, float& v, float g,
                                    float alpha, float eps_scaled,
                                    float b1_f, float one_m_b1,
                                    float b2_f, float one_m_b2) noexcept {
  float m_val = b1_f * m + one_m_b1 * g;
  float v_val = b2_f * v + one_m_b2 * g * g;
  m = m_val;
  v = v_val;
  val -= alpha * m_val / (std::sqrt(std::max(0.0f, v_val)) + eps_scaled);
}

// Zero-allocation, vectorized in-place Adam / AdamW optimizer state (FP32)
struct AdamParam {
  Eigen::MatrixXf val;
  Eigen::MatrixXf m;
  Eigen::MatrixXf v;

  void init(int rows, int cols) {
    val = Eigen::MatrixXf::Zero(rows, cols);
    m = Eigen::MatrixXf::Zero(rows, cols);
    v = Eigen::MatrixXf::Zero(rows, cols);
  }

  void update(const Eigen::MatrixXf& grad, double lr, double beta1, double beta2, double eps,
              double b1_corr, double b2_corr, double weight_decay = 0.0) {
    float alpha = static_cast<float>(lr * std::sqrt(b2_corr) / b1_corr);
    float eps_scaled = static_cast<float>(eps * std::sqrt(b2_corr));
    float b1_f = static_cast<float>(beta1);
    float one_m_b1 = static_cast<float>(1.0 - beta1);
    float b2_f = static_cast<float>(beta2);
    float one_m_b2 = static_cast<float>(1.0 - beta2);
    int sz = static_cast<int>(val.size());

    if (weight_decay > 0.0) {
      float wd_step = static_cast<float>(lr * weight_decay);
      detail::adamw_step_impl<true, true>(val.data(), m.data(), v.data(), grad.data(),
                                          sz, alpha, eps_scaled, b1_f, one_m_b1, b2_f, one_m_b2, wd_step);
    } else {
      detail::adamw_step_impl<true, false>(val.data(), m.data(), v.data(), grad.data(),
                                           sz, alpha, eps_scaled, b1_f, one_m_b1, b2_f, one_m_b2, 0.0f);
    }
  }

  void update_serial(const Eigen::MatrixXf& grad, double lr, double beta1, double beta2, double eps,
                     double b1_corr, double b2_corr, double weight_decay = 0.0) {
    float alpha = static_cast<float>(lr * std::sqrt(b2_corr) / b1_corr);
    float eps_scaled = static_cast<float>(eps * std::sqrt(b2_corr));
    float b1_f = static_cast<float>(beta1);
    float one_m_b1 = static_cast<float>(1.0 - beta1);
    float b2_f = static_cast<float>(beta2);
    float one_m_b2 = static_cast<float>(1.0 - beta2);
    int sz = static_cast<int>(val.size());

    if (weight_decay > 0.0) {
      float wd_step = static_cast<float>(lr * weight_decay);
      detail::adamw_step_impl<false, true>(val.data(), m.data(), v.data(), grad.data(),
                                           sz, alpha, eps_scaled, b1_f, one_m_b1, b2_f, one_m_b2, wd_step);
    } else {
      detail::adamw_step_impl<false, false>(val.data(), m.data(), v.data(), grad.data(),
                                            sz, alpha, eps_scaled, b1_f, one_m_b1, b2_f, one_m_b2, 0.0f);
    }
  }
};

// Column statistics container for standardized scaling (non-finite mapped to 0 before mean)
struct ColumnStats {
  double mean;
  double inv_std;
  double std_val;
};

// 2-Pass SIMD Reduction Kernel: Computes column mean and sample standard deviation
// Pass 1: Vectorized sum with fused NaN/Inf replacement (0.0)
// Pass 2: Vectorized sum of squared differences
// Variance floor: If s < 1e-5 or !isfinite(s), s is floored to 1.0 and inv_s = 1.0
// Strictly matching the behavior of rcpp_realmlp.cpp:353
inline ColumnStats compute_column_stats(const double* __restrict__ col, int N) {
  if (N <= 0) return { 0.0, 1.0, 1.0 };

  double sum = 0.0;
  for (int i = 0; i < N; ++i) {
    double v = col[i];
    sum += std::isfinite(v) ? v : 0.0;
  }
  double mean = sum / N;

  double sum_sq = 0.0;
  for (int i = 0; i < N; ++i) {
    double v = col[i];
    double clean_v = std::isfinite(v) ? v : 0.0;
    double diff = clean_v - mean;
    sum_sq += diff * diff;
  }
  double var = sum_sq / std::max(1, N - 1);
  double s = std::sqrt(var);

  if (s < 1e-5 || !std::isfinite(s)) {
    s = 1.0;
  }
  double inv_s = 1.0 / s;
  return { mean, inv_s, s };
}

// Column statistics for NA-mean-imputation: ignores non-finite values in mean,
// then computes exact post-imputation sample SD (zero squared diff on imputed entries).
struct FiniteColumnStats {
  double mean;
  double sd;
  double inv_std;
  int n_finite;
};

inline FiniteColumnStats compute_finite_column_stats(const double* __restrict__ col, int N) {
  if (N <= 0) return { 0.0, 1.0, 1.0, 0 };

  double sum = 0.0;
  int n_fin = 0;
  for (int i = 0; i < N; ++i) {
    double v = col[i];
    if (std::isfinite(v)) {
      sum += v;
      n_fin++;
    }
  }
  double mean = (n_fin > 0) ? (sum / n_fin) : 0.0;
  if (!std::isfinite(mean)) mean = 0.0;

  double sum_sq = 0.0;
  for (int i = 0; i < N; ++i) {
    double v = col[i];
    if (std::isfinite(v)) {
      double diff = v - mean;
      sum_sq += diff * diff;
    }
  }
  double sd = (N > 1 && n_fin > 0) ? std::sqrt(sum_sq / static_cast<double>(N - 1)) : 1.0;
  if (!std::isfinite(sd) || sd == 0.0) {
    sd = 1.0;
  }
  double inv_s = (sd < 1e-5) ? 1.0 : (1.0 / sd);
  return { mean, sd, inv_s, n_fin };
}

// Single-Pass Fused Column Standardization:
// Centers, multiplies by precomputed inv_std, and clamps in-place into output buffer
inline void fuse_standardize_column(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int N,
    double mean,
    double inv_std,
    double clamp_min = -30.0,
    double clamp_max = 30.0) {
  for (int i = 0; i < N; ++i) {
    double v = src[i];
    double clean_v = std::isfinite(v) ? v : 0.0;
    double z = (clean_v - mean) * inv_std;
    dst[i] = std::clamp(z, clamp_min, clamp_max);
  }
}

// Single-Pass Fused Matrix Standardization into FP32 (float):
// Computes standardization in FP64 for numerical accuracy against near-zero variance columns,
// clamps to [-30, 30], and stores directly as FP32 for AMX/SIMD acceleration.
inline void fuse_standardize_matrix_f32(
    const double* __restrict__ src,
    float* __restrict__ dst,
    int N, int D,
    const double* __restrict__ x_mean,
    const double* __restrict__ x_inv_std) {
  if (N <= 0 || D <= 0) return;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (D >= 2)
#endif
  for (int j = 0; j < D; ++j) {
    size_t offset = static_cast<size_t>(j) * N;
    const double* src_col = src + offset;
    float* dst_col = dst + offset;
    double m = x_mean[j];
    double inv_s = x_inv_std[j];

    for (int i = 0; i < N; ++i) {
      double v = src_col[i];
      double clean_v = std::isfinite(v) ? v : 0.0;
      double z = (clean_v - m) * inv_s;
      dst_col[i] = static_cast<float>(std::clamp(z, -30.0, 30.0));
    }
  }
}

// Fused standardize matrix helper into FP32 accepting Eigen::Ref with zero dynamic reallocations
inline Eigen::MatrixXf fuse_standardize_matrix_f32(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::VectorXd& x_mean,
    const Eigen::VectorXd& x_std) {
  int B = static_cast<int>(X.rows());
  int D = static_cast<int>(X.cols());
  Eigen::MatrixXf X_norm(B, D);
  Eigen::VectorXd x_inv_std(D);
  for (int j = 0; j < D; ++j) {
    double s = x_std(j);
    if (s < 1e-5 || !std::isfinite(s)) {
      x_inv_std(j) = 1.0;
    } else {
      x_inv_std(j) = 1.0 / s;
    }
  }
  if (X.innerStride() == 1 && X.outerStride() == B) {
    fuse_standardize_matrix_f32(X.data(), X_norm.data(), B, D, x_mean.data(), x_inv_std.data());
  } else {
    for (int j = 0; j < D; ++j) {
      double m = x_mean(j);
      double inv_s = x_inv_std(j);
      for (int i = 0; i < B; ++i) {
        double v = X(i, j);
        double clean_v = std::isfinite(v) ? v : 0.0;
        double z = (clean_v - m) * inv_s;
        X_norm(i, j) = static_cast<float>(std::clamp(z, -30.0, 30.0));
      }
    }
  }
  return X_norm;
}

} // namespace nn
} // namespace evofe

#endif // EVOFE_NN_COMMON_H
