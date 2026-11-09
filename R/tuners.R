# Bayesian Optimization Hyperparameter Tuners for Built-In Evaluators
#
# Evaluators that tune hyperparameters using Bayesian Optimization
# via 'mlr3mbo', 'paradox', and 'bbotk' delegating to make_tunable().

make_tunable(
  base_model_name = "lightgbm",
  param_ranges = list(
    learning_rate = list(type = "numeric", lower = 0.01, upper = 0.3),
    num_leaves = list(type = "integer", lower = 7, upper = 63),
    max_depth = list(type = "integer", lower = 3, upper = 10),
    feature_fraction = list(type = "numeric", lower = 0.5, upper = 1.0)
  ),
  tuner_name = "lightgbm_mbo"
)

make_tunable(
  base_model_name = "xgboost",
  param_ranges = list(
    eta = list(type = "numeric", lower = 0.01, upper = 0.3),
    max_depth = list(type = "integer", lower = 3, upper = 10),
    subsample = list(type = "numeric", lower = 0.5, upper = 1.0),
    colsample_bytree = list(type = "numeric", lower = 0.5, upper = 1.0)
  ),
  tuner_name = "xgboost_mbo"
)

make_tunable(
  base_model_name = "catboost",
  param_ranges = list(
    learning_rate = list(type = "numeric", lower = 0.01, upper = 0.3),
    depth = list(type = "integer", lower = 4, upper = 10),
    l2_leaf_reg = list(type = "numeric", lower = 1.0, upper = 10.0)
  ),
  tuner_name = "catboost_mbo"
)

make_tunable(
  base_model_name = "realmlp",
  param_ranges = list(
    realmlp_lr = list(type = "numeric", lower = 0.01, upper = 0.20),
    hidden_dim = list(type = "integer", lower = 64, upper = 512),
    realmlp_n_cycles = list(type = "integer", lower = 1, upper = 6)
  ),
  tuner_name = "realmlp_mbo"
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
    random_strength     = list(type = "numeric", lower = 1e-9, upper = 10.0),
    bagging_temperature = list(type = "numeric", lower = 0.0, upper = 1.0)
  ),
  tuner_name = "catboost_tabarena"
)

make_tunable(
  base_model_name = "realmlp",
  param_ranges = list(
    hidden_dim = list(type = "integer", lower = 128, upper = 512),
    realmlp_lr = list(type = "numeric", lower = 0.001, upper = 0.03),
    batch_size = list(type = "discrete", values = c(1024, 2048, 4096))
  ),
  tuner_name = "realmlp_tabarena"
)

