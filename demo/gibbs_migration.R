# Dual-Gibbs Pull and Gibbs-Type Island Model Migration Demo with evoFE
# 
# This demo showcases advanced island model migration topologies:
# 1. Dual-Gibbs Pull Demand-Driven Migration ("dual_gibbs_pull")
# 2. Gibbs Stagnation Push Migration ("gibbs_stagnation")
# 3. Gibbs Fitness Push Migration ("gibbs_fitness")
#

library(evoFE)

# =====================================================================
# 1. Prepare Data
# =====================================================================
message(">>> Preparing mtcars classification dataset...")
data(mtcars)
df <- mtcars
df$am <- as.integer(df$am)

# =====================================================================
# 2. Evolve Features using Dual-Gibbs Pull Migration Topology
# =====================================================================
message("\n>>> Running Evolutionary Feature Engineering with Dual-Gibbs Pull Migration...")
message("    Stagnated islands will dynamically pull elites from high-fitness donors.")

set.seed(42)
recipe_pull <- evolve_features(
  data = df,
  target_col = "am",
  task = "classification",
  evaluator = "lightgbm",
  generations = 5,
  pop_size = 6,
  islands = 4,
  migration_interval = 2,
  migration_topology = "dual_gibbs_pull",
  migration_temperature = 0.5,
  pull_stagnation_threshold = 2,
  verbose = TRUE
)

message("\n[Dual-Gibbs Pull Recipe Summary]")
print(recipe_pull)

# =====================================================================
# 3. Evolve Features using Gibbs Stagnation Push Migration Topology
# =====================================================================
message("\n>>> Running Evolutionary Feature Engineering with Gibbs Stagnation Migration...")
message("    Source islands will probabilistically target stagnated islands for rescue.")

set.seed(42)
recipe_stagnation <- evolve_features(
  data = df,
  target_col = "am",
  task = "classification",
  evaluator = "lightgbm",
  generations = 5,
  pop_size = 6,
  islands = 4,
  migration_interval = 2,
  migration_topology = "gibbs_stagnation",
  migration_temperature = 0.5,
  verbose = TRUE
)

message("\n[Gibbs Stagnation Recipe Summary]")
print(recipe_stagnation)

message("\n>>> Gibbs Migration Demo completed successfully!")
