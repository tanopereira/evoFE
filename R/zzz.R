# Suppress R CMD check NOTEs for data.table non-standard evaluation symbols
utils::globalVariables(c(
  ".",           # data.table's .() alias for list()
  ".N",          # data.table row count symbol
  ".SD",         # data.table Subset of Data symbol
  "smoothed",    # target_encode fit/apply
  "val",         # cat_mean/cat_sd/cat_max/cat_min transformers
  "mean_val",    # cat_zscore transformer
  "sd_val",      # cat_zscore transformer
  "node",        # mst_score transformer
  "dist",        # mst_score transformer
  "score",       # mst_score transformer
  "N",           # freq_encode transformer
  "n",           # target_encode n column
  "x",           # target_encode by column
  "y"            # target_encode target column
))
