# End-to-End Evolutionary Feature Engineering Workflow with evoFE
# 
# This demo showcases the primary user-facing features of the evoFE package:
# 1. Feature evolution on a binary classification task
# 2. Inspecting the winning recipe using print and summary S3 methods
# 3. Visualizing the fitness curve and feature importances
# 4. Extracting the engineered features and generating model predictions
# 5. Defining and registering a custom feature transformer
#

library(evoFE)

# =====================================================================
# 1. Prepare Data
# =====================================================================
message(">>> Loading and preparing mtcars dataset...")
data(mtcars)
df <- mtcars
# Use 'am' (transmission: 0 = automatic, 1 = manual) as our target column
df$am <- as.integer(df$am)

# =====================================================================
# 2. Evolve Features
# =====================================================================
message("\n>>> Running Evolutionary Feature Engineering (evoFE)...")
# We use small populations/generations for demonstration speed
set.seed(42)
recipe <- evolve_features(
  data = df,
  target_col = "am",
  task = "classification",
  evaluator = "xgboost",
  generations = 3,
  pop_size = 6,
  cv_folds = 3,
  verbose = TRUE
)

# =====================================================================
# 3. Inspect the Recipe
# =====================================================================
message("\n>>> Inspecting the winning recipe object...")

# Printing the recipe overview
message("\n[Print Recipe Overview]")
print(recipe)

# Detailed recipe summary
message("\n[Recipe Summary Details]")
recipe_summary <- summary(recipe)
print(recipe_summary)

# =====================================================================
# 4. Visualizations
# =====================================================================
message("\n>>> Plotting evolution results...")
# Note: In an interactive session, this will open a plotting device.
if (interactive()) {
  # Plot the fitness curve across generations
  plot(recipe, type = "fitness")
  
  # Wait briefly, then plot feature importances of the best individual
  Sys.sleep(2)
  plot(recipe, type = "importance")
} else {
  message("  (Graphics plotting skipped because the session is non-interactive)")
}

# =====================================================================
# 5. Feature Extraction and Prediction
# =====================================================================
message("\n>>> Engineering features and predicting on new data...")

# Select a subset of test rows
test_df <- df[1:5, ]

# Apply the evolved feature recipe to transform newdata
engineered_data <- predict(recipe, newdata = test_df)
message("\n[Engineered Features (Top Columns)]")
print(head(engineered_data))

# Generate predictions directly using the trained model
preds <- predict_model(recipe, newdata = test_df)
message("\n[Model Predictions (Probabilities of manual transmission)]")
print(preds)

# =====================================================================
# 6. Custom Transformer Registration
# =====================================================================
message("\n>>> Registering and verifying a custom transformer...")

# Create a custom scaling transformer
times_hundred_trans <- create_transformer(
  name = "times_hundred",
  type = "unary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    data[[gene$input_cols[1]]] * 100
  },
  name_generator = function(gene) paste0("x100_", gene$input_cols[1])
)

# Register the transformer
register_transformer("times_hundred", times_hundred_trans)

# Verify registration
if (exists("times_hundred", envir = evo_transformers)) {
  message("  Success! 'times_hundred' transformer successfully registered.")
} else {
  message("  Error: 'times_hundred' transformer registration failed.")
}

message("\n>>> Demo completed successfully!")
