# Suppress R CMD check NOTEs for data.table non-standard evaluation symbols
utils::globalVariables(c(
  ".",           # data.table's .() alias for list()
  ".N",          # data.table row count symbol
  ".SD",         # data.table Subset of Data symbol
  "b_idx",       # numeric_target_encode quantile bin index
  "c",           # data.table component index
  "cat",         # data.table category symbol
  "cat_q",       # groupby_quantile transformer
  "dist",        # mst_score transformer
  "dist_events", # woe_encode transformer
  "dist_nonevents", # woe_encode transformer
  "grp",         # groupby aggregations
  "mean_val",    # cat_zscore transformer
  "mean_y",      # target_encode transformer
  "N",           # freq_encode transformer
  "n",           # target_encode n column
  "n_events",    # woe_encode transformer
  "n_nonevents", # woe_encode transformer
  "node",        # mst_score transformer
  "score",       # mst_score transformer
  "sd_val",      # cat_zscore transformer
  "smoothed",    # target_encode fit/apply
  "sum_y",       # smoothed target_encode helper
  "val",         # cat_mean/cat_sd/cat_max/cat_min transformers
  "var",         # pooled_target_encode transformer
  "woe",         # woe_encode transformer
  "x",           # target_encode by column
  "x_key",       # numeric_target_encode exact key column
  "x_joint",     # cat_interaction_target_encode
  "y"            # target_encode target column
))
