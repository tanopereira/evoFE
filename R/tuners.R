# Bayesian Optimization Hyperparameter Tuners for Built-In Evaluators
#
# Evaluators that tune hyperparameters using Bayesian Optimization
# via 'mlr3mbo', 'paradox', and 'bbotk' delegating to make_tunable().

make_tunable(
  base_model_name = "lightgbm",
  param_ranges = list(
    learning_rate = list(type = "numeric", lower = 0.01, upper = 0.3, log = TRUE),
    num_leaves = list(type = "integer", lower = 7, upper = 63),
    max_depth = list(type = "integer", lower = 3, upper = 10),
    feature_fraction = list(type = "numeric", lower = 0.5, upper = 1.0)
  ),
  tuner_name = "lightgbm_mbo"
)

make_tunable(
  base_model_name = "xgboost",
  param_ranges = list(
    eta = list(type = "numeric", lower = 0.01, upper = 0.3, log = TRUE),
    max_depth = list(type = "integer", lower = 3, upper = 10),
    subsample = list(type = "numeric", lower = 0.5, upper = 1.0),
    colsample_bytree = list(type = "numeric", lower = 0.5, upper = 1.0)
  ),
  tuner_name = "xgboost_mbo"
)

make_tunable(
  base_model_name = "catboost",
  param_ranges = list(
    learning_rate = list(type = "numeric", lower = 0.01, upper = 0.3, log = TRUE),
    depth = list(type = "integer", lower = 4, upper = 10),
    l2_leaf_reg = list(type = "numeric", lower = 1.0, upper = 10.0)
  ),
  tuner_name = "catboost_mbo"
)

make_tunable(
  base_model_name = "realmlp",
  param_ranges = list(
    realmlp_lr = list(type = "numeric", lower = 0.01, upper = 0.35, log = TRUE),
    hidden_dim = list(type = "integer", lower = 64, upper = 512),
    realmlp_n_cycles = list(type = "integer", lower = 1L, upper = 4L)
  ),
  tuner_name = "realmlp_mbo"
)

make_tunable(
  base_model_name = "lm",
  param_ranges = list(
    alpha = list(type = "numeric", lower = 0.0, upper = 1.0),
    lambda = list(type = "numeric", lower = 1e-6, upper = 1.0, log = TRUE)
  ),
  tuner_name = "lm_mbo"
)

# ─────────────────────────────────────────────────────────────
# TabArena Hyperparameter Spaces (ML_STRATEGY.md Section 10)
# ─────────────────────────────────────────────────────────────

make_tunable(
  base_model_name = "lightgbm",
  param_ranges = list(
    num_leaves       = list(type = "integer", lower = 15, upper = 255),
    max_depth        = list(type = "integer", lower = 3, upper = 12),
    min_data_in_leaf = list(type = "integer", lower = 10, upper = 100),
    feature_fraction = list(type = "numeric", lower = 0.40, upper = 0.95),
    bagging_fraction = list(type = "numeric", lower = 0.60, upper = 1.00),
    lambda_l1        = list(type = "numeric", lower = 0.0, upper = 10.0),
    lambda_l2        = list(type = "numeric", lower = 0.0, upper = 10.0)
  ),
  tuner_name = "lightgbm_tabarena"
)

make_tunable(
  base_model_name = "xgboost",
  param_ranges = list(
    max_depth        = list(type = "integer", lower = 3, upper = 10),
    min_child_weight = list(type = "numeric", lower = 0.5, upper = 20.0),
    colsample_bytree = list(type = "numeric", lower = 0.40, upper = 0.95),
    subsample        = list(type = "numeric", lower = 0.60, upper = 1.00),
    gamma            = list(type = "numeric", lower = 0.0, upper = 5.0),
    alpha            = list(type = "numeric", lower = 0.0, upper = 10.0),
    lambda           = list(type = "numeric", lower = 0.0, upper = 10.0)
  ),
  tuner_name = "xgboost_tabarena"
)

make_tunable(
  base_model_name = "catboost",
  param_ranges = list(
    depth               = list(type = "integer", lower = 4, upper = 10),
    l2_leaf_reg         = list(type = "numeric", lower = 1.0, upper = 10.0),
    random_strength     = list(type = "numeric", lower = 1e-9, upper = 10.0, log = TRUE),
    bagging_temperature = list(type = "numeric", lower = 0.0, upper = 1.0)
  ),
  tuner_name = "catboost_tabarena"
)

make_tunable(
  base_model_name = "realmlp",
  param_ranges = list(
    hidden_dim       = list(type = "integer", lower = 128L, upper = 512L),
    realmlp_lr       = list(type = "numeric", lower = 0.01, upper = 0.35, log = TRUE),
    realmlp_n_cycles = list(type = "integer", lower = 1L, upper = 4L),
    batch_size       = list(type = "discrete", values = c(256L, 512L, 1024L, 2048L))
  ),
  tuner_name = "realmlp_tabarena"
)

make_tunable(
  base_model_name = "lm",
  param_ranges = list(
    alpha = list(type = "numeric", lower = 0.0, upper = 1.0),
    s = list(type = "discrete", values = c("lambda.min", "lambda.1se"))
  ),
  tuner_name = "lm_tabarena"
)

# 6. TabM Tuners (MBO & TabArena)
make_tunable(
  base_model_name = "tabm",
  param_ranges = list(
    lr = list(type = "numeric", lower = 1e-4, upper = 1e-2, log = TRUE),
    d_block = list(type = "integer", lower = 64L, upper = 384L),
    n_blocks = list(type = "integer", lower = 1L, upper = 4L),
    dropout = list(type = "numeric", lower = 0.0, upper = 0.4),
    k_ensemble = list(type = "integer", lower = 8L, upper = 32L)
  ),
  tuner_name = "tabm_mbo"
)

make_tunable(
  base_model_name = "tabm",
  param_ranges = list(
    lr = list(type = "numeric", lower = 1e-4, upper = 1e-2, log = TRUE),
    d_block = list(type = "discrete", values = c(64L, 128L, 256L, 384L)),
    n_blocks = list(type = "integer", lower = 1L, upper = 4L),
    dropout = list(type = "numeric", lower = 0.0, upper = 0.45),
    weight_decay = list(type = "numeric", lower = 1e-6, upper = 1e-2, log = TRUE),
    k_ensemble = list(type = "discrete", values = c(8L, 16L, 32L)),
    batch_size = list(type = "discrete", values = c(256L, 512L, 1024L, 2048L)),
    n_bins = list(type = "discrete", values = c(16L, 32L, 48L, 64L)),
    d_embedding = list(type = "discrete", values = c(4L, 8L, 16L, 32L))
  ),
  tuner_name = "tabm_tabarena"
)

# 7. RFF / Nystrom Tuners (MBO & TabArena)
make_tunable(
  base_model_name = "rff",
  param_ranges = list(
    sigma = list(type = "numeric", lower = 0.05, upper = 20.0, log = TRUE),
    lambda = list(type = "numeric", lower = 1e-5, upper = 100.0, log = TRUE),
    n_features = list(type = "integer", lower = 128L, upper = 1024L)
  ),
  tuner_name = "rff_mbo"
)

make_tunable(
  base_model_name = "rff",
  param_ranges = list(
    sigma = list(type = "numeric", lower = 0.05, upper = 25.0, log = TRUE),
    lambda = list(type = "numeric", lower = 1e-5, upper = 100.0, log = TRUE),
    n_features = list(type = "discrete", values = c(256L, 512L, 1024L, 2048L)),
    kernel = list(type = "discrete", values = c("rbf", "laplacian")),
    approx_method = list(type = "discrete", values = c("rff", "nystrom"))
  ),
  tuner_name = "rff_tabarena"
)

# 8. Spline-LM Tuners (MBO & TabArena)
make_tunable(
  base_model_name = "spline_lm",
  param_ranges = list(
    alpha = list(type = "numeric", lower = 0.0, upper = 1.0),
    lambda = list(type = "numeric", lower = 1e-6, upper = 10.0, log = TRUE),
    n_knots = list(type = "integer", lower = 3L, upper = 12L),
    degree = list(type = "integer", lower = 1L, upper = 3L)
  ),
  tuner_name = "spline_lm_mbo"
)

make_tunable(
  base_model_name = "spline_lm",
  param_ranges = list(
    alpha = list(type = "numeric", lower = 0.0, upper = 1.0),
    lambda = list(type = "numeric", lower = 1e-6, upper = 10.0, log = TRUE),
    n_knots = list(type = "integer", lower = 3L, upper = 15L),
    degree = list(type = "discrete", values = c(1L, 2L, 3L)),
    include_linear = list(type = "discrete", values = c(TRUE, FALSE))
  ),
  tuner_name = "spline_lm_tabarena"
)

# 9. LightGBM Extra-Trees Random Forest Tuners (MBO & TabArena)
make_tunable(
  base_model_name = "lightgbm_rf",
  param_ranges = list(
    num_leaves = list(type = "integer", lower = 15L, upper = 255L),
    max_depth = list(type = "integer", lower = 4L, upper = 16L),
    feature_fraction = list(type = "numeric", lower = 0.3, upper = 0.9),
    bagging_fraction = list(type = "numeric", lower = 0.5, upper = 0.95),
    min_data_in_leaf = list(type = "integer", lower = 5L, upper = 80L)
  ),
  tuner_name = "lightgbm_rf_mbo"
)

make_tunable(
  base_model_name = "lightgbm_rf",
  param_ranges = list(
    num_leaves = list(type = "integer", lower = 31L, upper = 511L),
    max_depth = list(type = "integer", lower = 5L, upper = 20L),
    feature_fraction = list(type = "numeric", lower = 0.25, upper = 0.9),
    bagging_fraction = list(type = "numeric", lower = 0.5, upper = 0.95),
    min_data_in_leaf = list(type = "integer", lower = 2L, upper = 100L),
    lambda_l2 = list(type = "numeric", lower = 0.0, upper = 10.0)
  ),
  tuner_name = "lightgbm_rf_tabarena"
)

# 10. Ranger Extra-Trees Random Forest Tuners (MBO & TabArena)
make_tunable(
  base_model_name = "ranger",
  param_ranges = list(
    mtry_ratio = list(type = "numeric", lower = 0.15, upper = 0.9),
    min.node.size = list(type = "integer", lower = 1L, upper = 30L),
    sample.fraction = list(type = "numeric", lower = 0.5, upper = 1.0)
  ),
  tuner_name = "ranger_mbo"
)

make_tunable(
  base_model_name = "ranger",
  param_ranges = list(
    mtry_ratio = list(type = "numeric", lower = 0.1, upper = 0.95),
    min.node.size = list(type = "integer", lower = 1L, upper = 50L),
    sample.fraction = list(type = "numeric", lower = 0.5, upper = 1.0),
    max.depth = list(type = "integer", lower = 0L, upper = 25L),
    num.random.splits = list(type = "integer", lower = 1L, upper = 10L),
    replace = list(type = "discrete", values = c(TRUE, FALSE))
  ),
  tuner_name = "ranger_tabarena"
)

