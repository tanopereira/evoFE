# Bayesian Optimization Hyperparameter Tuner for LightGBM
#
# Evaluator that tunes LightGBM hyperparameters using Bayesian Optimization
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
