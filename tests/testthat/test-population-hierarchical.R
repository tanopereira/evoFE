test_that("initialize_population produces no hierarchical genes in Generation 1", {
  library(data.table)

  set.seed(42)
  numeric_cols <- c("x1", "x2", "x3")
  categorical_cols <- c("cat1", "cat2")
  datetime_cols <- character(0)

  pop <- initialize_population(
    pop_size = 10,
    numeric_cols = numeric_cols,
    categorical_cols = categorical_cols,
    datetime_cols = datetime_cols,
    initial_genes = 4,
    task = "regression"
  )

  raw_cols <- c(numeric_cols, categorical_cols, datetime_cols)

  for (ind in pop) {
    if (length(ind$genes) > 0) {
      for (gene in ind$genes) {
        # Check that every input_col of initial genes is a raw base column
        for (col in gene$input_cols) {
          expect_true(col %in% raw_cols,
                      info = sprintf("Gene '%s' (%s) used non-raw column '%s' in Gen 1",
                                     gene$output_col, gene_to_formula(gene), col))
        }
      }
    }
  }
})
