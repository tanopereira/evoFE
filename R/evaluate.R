# R/evaluate.R
# Decomposed into modular components in Phase 3:
# - R/pipeline_apply.R (apply_gene, apply_individual)
# - R/metrics_classification.R (compute_exp_neg_logloss, compute_exp_neg_multiclass_logloss, compute_auc, compute_multiclass_auc, compute_f1)
# - R/metrics_regression.R (compute_mae, compute_rmse)
# - R/metrics_calibration.R (compute_ts_refinement, compute_calibrated_rmse, compute_calibrated_mae)
# - R/metrics_dispatcher.R (compute_metric)
# - R/evaluation_complexity.R (compute_complexity_penalty)
# - R/evaluation_fitness.R (evaluate_fitness)
# - R/evaluation_holdout.R (evaluate_holdout_fitness)
