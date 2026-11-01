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
