# Datetime transformers: component extraction, cyclic encoding, date differences.
# Split out of transformers.R; loaded after it (alphabetical file order).

.to_posixlt <- function(x) {
  tryCatch({
    if (inherits(x, "POSIXlt")) {
      x
    } else if (inherits(x, c("POSIXct", "Date"))) {
      as.POSIXlt(x)
    } else {
      as.POSIXlt(as.character(x), tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"))
    }
  }, error = function(e) {
    as.POSIXlt(rep(NA_character_, length(x)))
  })
}

evo_transformers$datetime_extract <- create_transformer(
  name = "datetime_extract",
  type = "unary",
  input_type = "datetime",
  output_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    lt <- .to_posixlt(x)
    comp <- if (!is.null(gene$params$component)) gene$params$component else "month"
    res <- switch(comp,
      year = lt$year + 1900L,
      month = lt$mon + 1L,
      day = lt$mday,
      hour = lt$hour,
      day_of_week = lt$wday + 1L,
      weekend = as.integer(lt$wday %in% c(0L, 6L)),
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
    a <- as.POSIXct(.to_posixlt(data[[input_cols[1]]]))
    b <- as.POSIXct(.to_posixlt(data[[input_cols[2]]]))
    res <- as.numeric(difftime(a, b, units = "days"))
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "ddiff")
)

# Datetime Cyclic Encoding
evo_transformers$datetime_cyclic <- create_transformer(
  name = "datetime_cyclic",
  type = "unary",
  input_type = "datetime",
  output_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    lt <- .to_posixlt(x)
    comp <- if (!is.null(gene$params$component)) gene$params$component else "hour_sin"

    res <- switch(comp,
      hour_sin = sin(2 * pi * lt$hour / 24),
      hour_cos = cos(2 * pi * lt$hour / 24),
      wday_sin = sin(2 * pi * lt$wday / 7),
      wday_cos = cos(2 * pi * lt$wday / 7),
      month_sin = sin(2 * pi * lt$mon / 12),
      month_cos = cos(2 * pi * lt$mon / 12),
      yday_sin = sin(2 * pi * lt$yday / 365.25),
      yday_cos = cos(2 * pi * lt$yday / 365.25),
      rep(0, length(x))
    )
    res[is.na(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "dt_cyc")
)
