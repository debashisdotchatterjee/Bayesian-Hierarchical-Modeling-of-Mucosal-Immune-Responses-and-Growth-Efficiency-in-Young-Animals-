# ============================ #
# Append Metadata to CSV File #
# ============================ #

# Define metadata
metadata <- list(
  "Description" = "Synthetic dataset simulating growth efficiency in young animals, generated using a Bayesian hierarchical model framework.",
  "Seed" = 123,
  "Sample size (n_animals)" = n_animals,
  "Number of groups" = n_groups,
  "True coefficients (mu_beta)" = paste(mu_beta, collapse = ", "),
  "Animal-level random effect SD (sigma_u)" = sigma_u,
  "Group-level random effect SD (sigma_v)" = sigma_v,
  "Residual error SD (sigma_eps)" = sigma_eps,
  "Generated using" = "R 4.3+ with MASS, brms, ggplot2 packages"
)

# Create metadata text for header
metadata_text <- paste0("# ", names(metadata), ": ", unlist(metadata), collapse = "\n")

# Write metadata to a file
metadata_file <- "synthetic_data_output/synthetic_animal_data_with_metadata.csv"
writeLines(metadata_text, con = metadata_file)

# Append the actual data (below the metadata)
suppressWarnings(
  write.table(animal_data, file = metadata_file, append = TRUE, sep = ",",
              row.names = FALSE, col.names = TRUE)
)

# Confirm saved
message("Synthetic dataset with metadata successfully saved at: ", metadata_file)
