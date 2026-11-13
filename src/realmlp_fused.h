#ifndef EVOFE_REALMLP_FUSED_H
#define EVOFE_REALMLP_FUSED_H

#include "nn_common.h"

namespace realmlp {

using evofe::nn::ColumnStats;
using evofe::nn::compute_column_stats;
using evofe::nn::fuse_standardize_column;
using evofe::nn::fuse_standardize_matrix_f32;

} // namespace realmlp

#endif // EVOFE_REALMLP_FUSED_H
