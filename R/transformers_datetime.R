# Datetime transformers: component extraction, cyclic encoding, date differences.
# Split out of transformers.R; loaded after it (alphabetical file order).

evo_transformers$datetime_extract <- create_transformer(
  name = "datetime_extract",
  type = "unary",
  input_type = "datetime",
  output_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt_parsed <- tryCatch({
      if (inherits(x, c("POSIXct", "POSIXlt", "Date"))) {
        as.POSIXct(x)
      } else {
        as.POSIXct(as.character(x), tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"))
      }
    }, error = function(e) {
      as.POSIXct(rep(NA, length(x)))
    })
    comp <- if (!is.null(gene$params$component)) gene$params$component else "month"
    res <- switch(comp,
      year = as.integer(format(dt_parsed, "%Y")),
      month = as.integer(format(dt_parsed, "%m")),
      day = as.integer(format(dt_parsed, "%d")),
      hour = as.integer(format(dt_parsed, "%H")),
      day_of_week = as.integer(format(dt_parsed, "%u")),
      weekend = as.integer(format(dt_parsed, "%u") %in% c("6", "7")),
      rep(0L, length(x))
    )
    res[is.na(res)] <- 0L
    as.numeric(res)
  },
  name_generator = function(gene) .gene_col_name(gene, "dt")
)

# Date Difference (binary datetime -> numeric, stateless)
# Days elapsed between two datetime columns; sign encodes order.
evo_transformers$date_diff <- create_transformer(
  name = "date_diff",
  type = "binary",
  input_type = "datetime",
  output_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    .to_posix <- function(x) {
      tryCatch({
        if (inherits(x, c("POSIXct", "POSIXlt", "Date"))) {
          as.POSIXct(x)
        } else {
          as.POSIXct(as.character(x), tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"))
        }
      }, error = function(e) as.POSIXct(rep(NA, length(x))))
    }
    a <- .to_posix(data[[input_cols[1]]])
    b <- .to_posix(data[[input_cols[2]]])
    res <- as.numeric(difftime(a, b, units = "days"))
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "ddiff")
)

# Multiclass Target Encoding
evo_transformers$datetime_cyclic <- create_transformer(
  name = "datetime_cyclic",
  type = "unary",
  input_type = "datetime",
  output_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt_parsed <- tryCatch({
      if (inherits(x, c("POSIXct", "POSIXlt", "Date"))) {
        as.POSIXct(x)
      } else {
        as.POSIXct(as.character(x), tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"))
      }
    }, error = function(e) {
      as.POSIXct(rep(NA, length(x)))
    })
    comp <- if (!is.null(gene$params$component)) gene$params$component else "hour_sin"

    res <- switch(comp,
      hour_sin = sin(2 * pi * as.numeric(format(dt_parsed, "%H")) / 24),
      hour_cos = cos(2 * pi * as.numeric(format(dt_parsed, "%H")) / 24),
      wday_sin = sin(2 * pi * (as.numeric(format(dt_parsed, "%u")) - 1) / 7),
      wday_cos = cos(2 * pi * (as.numeric(format(dt_parsed, "%u")) - 1) / 7),
      month_sin = sin(2 * pi * (as.numeric(format(dt_parsed, "%m")) - 1) / 12),
      month_cos = cos(2 * pi * (as.numeric(format(dt_parsed, "%m")) - 1) / 12),
      yday_sin = sin(2 * pi * (as.numeric(format(dt_parsed, "%j")) - 1) / 365.25),
      yday_cos = cos(2 * pi * (as.numeric(format(dt_parsed, "%j")) - 1) / 365.25),
      rep(0, length(x))
    )
    res[is.na(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "dt_cyc")
)

# Target Quantile Encoding
