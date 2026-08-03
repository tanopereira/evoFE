# Evolutionary Feature Engineering on Employee Salaries Dataset (OpenML 42125)
#
# Demonstrates 10-island Ring topology migration with full-individual recipe payload
# for regression on annual salary prediction using LightGBM.

if (!requireNamespace("farff", quietly = TRUE)) {
  install.packages("farff")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table")
}

# Download & load OpenML dataset 42125 (employee_salaries)
url <- "https://openml.org/data/v1/download/21718841/employee_salaries.arff"
tmp <- tempfile(fileext = ".arff")
download.file(url, tmp, headers = c("User-Agent" = "Mozilla/5.0"), quiet = TRUE)

df <- farff::readARFF(tmp)
data.table::setDT(df)

# Check columns
head(df[, .(employee_position_title, department_name, date_first_hired, current_annual_salary)])
df$full_name <- NULL

library(evoFE)

# Configure 10-island Ring migration topology with full-individual payload
migration <- migration_config(
  topology = topology_ring(islands = 10),
  payload  = "full_individual"
)

set.seed(42)

# Evolve features on employee salaries dataset
recipe <- evolve_features(
  data = df,
  target_col = "current_annual_salary",
  task = "regression",
  evaluator = "lightgbm",
  evaluation_strategy = "split",
  generations = 30,
  pop_size = 5,
  threads = 8,
  cv_folds = 3,
  complexity_penalty = 0,
  early_stopping_generations = 5,
  verbose = TRUE,
  migration_interval = 3,
  record = TRUE,
  row_split_islands = FALSE,
  per_island_validation = FALSE,
  migration = migration
)

# View top evolved features
print(summary(recipe))
