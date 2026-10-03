# R/individual.R
# Decomposed into modular components in Phase 3:
# - R/population_graph.R (topological_sort_genes)
# - R/operators_mutation.R (mutate)
# - R/operators_crossover.R (crossover, union_crossover)
# - R/operators_mask.R (sample_gene_inputs, recalculate_mask, toggle_raw_feature)
# - R/population_init.R (create_gene, create_individual, strip_individual_state)
# - R/pipeline_serialization.R (gene_to_formula, gene_to_state_formula, individual_to_recipe_string)
