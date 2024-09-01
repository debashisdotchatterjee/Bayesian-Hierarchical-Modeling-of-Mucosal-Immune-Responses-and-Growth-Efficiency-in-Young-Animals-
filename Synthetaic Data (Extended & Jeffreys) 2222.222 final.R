# Load necessary libraries
library(MASS)
library(ggplot2)
library(brms)

# Set seed for reproducibility
set.seed(123)

# Parameters for synthetic data generation
n_animals <- 100
n_groups <- 5
mu_beta <- c(0.5, 0.3, -0.2, 0.4, 0.6, 0.3, -0.5)  # True coefficients

# Random effects and standard deviations
sigma_u <- 0.5                # Animal-level random effect
sigma_v <- 0.3                # Group-level random effect
sigma_eps <- 1.0              # Residual error

# Generate random group-level effects
group_effects <- rnorm(n_groups, mean = 0, sd = sigma_v)

# Generate synthetic dataset
animal_data <- data.frame(
  animal_id = 1:n_animals,
  group_id = sample(1:n_groups, n_animals, replace = TRUE),
  diet = rnorm(n_animals, mean = 0, sd = 1),
  stress = rnorm(n_animals, mean = 0, sd = 1),
  genetic = rnorm(n_animals, mean = 0, sd = 1),
  microbiota_diversity = rnorm(n_animals, mean = 0, sd = 1),
  cytokine_levels = rnorm(n_animals, mean = 0, sd = 1),
  villus_height = rnorm(n_animals, mean = 0, sd = 1),
  crypt_depth = rnorm(n_animals, mean = 0, sd = 1)
)

# Simulate the outcome variable (growth efficiency)
animal_data$growth_efficiency <- with(animal_data, 
                                      mu_beta[1] * diet + mu_beta[2] * stress + mu_beta[3] * genetic +
                                        mu_beta[4] * microbiota_diversity + mu_beta[5] * cytokine_levels +
                                        mu_beta[6] * villus_height + mu_beta[7] * crypt_depth +
                                        group_effects[group_id] + rnorm(n_animals, mean = 0, sd = sigma_eps)
)

# Display the first few rows of the dataset
print(head(animal_data))

# Save the synthetic dataset
dir.create("synthetic_data_output", showWarnings = FALSE)
write.csv(animal_data, "synthetic_data_output/synthetic_animal_data.csv", row.names = FALSE)

# Define the model using brms with empirical priors
priors <- c(
  prior(horseshoe(df = 1), class = "b"),
  prior(student_t(3, 0, 2.5), class = "Intercept"),
  prior(student_t(3, 0, 2.5), class = "sigma"),
  prior(normal(0, 1), class = "sd", group = "animal_id"),
  prior(normal(0, 1), class = "sd", group = "group_id")
)

# Fit the model
brms_model <- brm(
  formula = growth_efficiency ~ diet + stress + genetic + microbiota_diversity + cytokine_levels + villus_height + crypt_depth + (1 | group_id) + (1 | animal_id),
  data = animal_data,
  family = gaussian(),
  prior = priors,
  iter = 10000,
  warmup = 5000,
  chains = 4,
  seed = 123
)

# Save the model summary
brms_summary <- summary(brms_model)
capture.output(brms_summary, file = "synthetic_data_output/brms_model_summary.txt")

# Posterior distributions plot (each plot with its corresponding truth line)
posterior_density_plots <- list()
param_names <- c("b_diet", "b_stress", "b_genetic", "b_microbiota_diversity", "b_cytokine_levels", "b_villus_height", "b_crypt_depth")

for (i in 1:length(mu_beta)) {
  plot <- mcmc_dens_overlay(as.array(brms_model), pars = param_names[i]) +
    geom_vline(xintercept = mu_beta[i], color = "red", linetype = "dashed", size = 0.5) +
    labs(title = paste("Posterior Density for", param_names[i]))
  posterior_density_plots[[i]] <- plot
  ggsave(paste0("synthetic_data_output/posterior_density_", param_names[i], ".png"), plot)
  print(plot)
}

# Predicted vs Observed plot with R-squared
predicted <- apply(fitted(brms_model), 1, mean)  # Taking the mean of the posterior predictions
observed <- animal_data$growth_efficiency
r_squared <- round(cor(observed, predicted)^2, 3)

pred_vs_obs_plot <- ggplot(data = NULL, aes(x = observed, y = predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red') +
  labs(title = paste("Predicted vs. Observed Growth Efficiency\nR-squared =", r_squared),
       x = "Observed",
       y = "Predicted")

ggsave("synthetic_data_output/pred_vs_obs.png", pred_vs_obs_plot)
print(pred_vs_obs_plot)

# Posterior predictive check
ppc_plot <- pp_check(brms_model)
ggsave("synthetic_data_output/ppc_plot.png", ppc_plot)
print(ppc_plot)

# Traceplot for MCMC convergence diagnostics
traceplot <- mcmc_trace(as.array(brms_model), pars = param_names)
ggsave("synthetic_data_output/traceplot.png", traceplot)
print(traceplot)

# Save posterior samples
posterior_samples <- as.data.frame(brms_model)
write.csv(posterior_samples, "synthetic_data_output/posterior_samples.csv")

# Sensitivity Analysis: Slightly alter initial values or prior and rerun the sampler
samples_sensitivity <- update(brms_model, prior = priors, seed = 123)

# Compare the posterior means from the original and sensitivity analysis
sensitivity_means <- apply(as.matrix(samples_sensitivity), 2, mean)
comparison <- data.frame(Original = apply(as.matrix(brms_model), 2, mean), Sensitivity = sensitivity_means)
print(comparison)

# Save the sensitivity analysis comparison table
write.csv(comparison, "synthetic_data_output/sensitivity_analysis_comparison.csv")

# Additional Model Assessment: Calculate WAIC and LOO
waic_result <- waic(brms_model)
loo_result <- loo(brms_model)

# Save and print WAIC and LOO results
capture.output(waic_result, file = "synthetic_data_output/waic_result.txt")
capture.output(loo_result, file = "synthetic_data_output/loo_result.txt")

print(waic_result)
print(loo_result)

######################

# Additional Model Assessment: Calculate WAIC and LOO
waic_result <- waic(brms_model)
print(waic_result)

# LOO with moment matching
loo_result <- loo(brms_model, moment_match = TRUE)
print(loo_result)

# Save and print WAIC and LOO results
capture.output(waic_result, file = "synthetic_data_output/waic_result.txt")
capture.output(loo_result, file = "synthetic_data_output/loo_result.txt")

#######################

# Refit the model with save_pars to enable moment matching
brms_model <- brm(
  formula = growth_efficiency ~ diet + stress + genetic + microbiota_diversity + cytokine_levels + villus_height + crypt_depth + (1 | group_id) + (1 | animal_id),
  data = animal_data,
  family = gaussian(),
  prior = priors,
  iter = 10000,
  warmup = 5000,
  chains = 4,
  seed = 123,
  save_pars = save_pars(all = TRUE)
)

# Additional Model Assessment: Calculate WAIC and LOO with moment matching
waic_result <- waic(brms_model)
print(waic_result)


#######################
# Fit the model with increased adapt_delta
brms_model <- brm(
  formula = growth_efficiency ~ diet + stress + genetic + microbiota_diversity + cytokine_levels + villus_height + crypt_depth + (1 | group_id) + (1 | animal_id),
  data = animal_data,
  family = gaussian(),
  prior = priors,
  iter = 10000,
  warmup = 5000,
  chains = 4,
  seed = 123,
  control = list(adapt_delta = 0.95)  # Increase adapt_delta
)

# Calculate the summary statistics
summary_stats <- summary(brms_model)

# Extract and print the Effective Sample Size (ESS) for all parameters
ess_values <- summary_stats$fixed[,"Bulk_ESS"]
print(ess_values)

# Extract the posterior samples for calculating MCSE manually
posterior_samples <- as_draws_df(brms_model)

# Calculate MCSE (Monte Carlo Standard Error)
mcse_values <- apply(posterior_samples, 2, function(x) sd(x) / sqrt(length(x)))
print(mcse_values)

# Save ESS and MCSE values to a CSV file
ess_mcse_df <- data.frame(Parameter = names(ess_values), 
                          ESS = ess_values, 
                          MCSE = mcse_values)
write.csv(ess_mcse_df, "synthetic_data_output/ess_mcse_values.csv")

# Additional diagnostics or plots can follow as needed

kfold_result <- kfold(brms_model, K = 10)
print(kfold_result)

##################


# Assuming 'brms_model' is your fitted model object
posterior_samples <- as_draws_df(brms_model)

# Extract the summary of the model
summary_stats <- summary(brms_model)

# Extract Effective Sample Size (ESS) for all fixed effects parameters
ess_values <- summary_stats$fixed[,"Bulk_ESS"]
print(ess_values)

# Calculate MCSE (Monte Carlo Standard Error)
mcse_values <- apply(posterior_samples, 2, function(x) sd(x) / sqrt(length(x)))
print(mcse_values)
# Create a dataframe with ESS and MCSE values
ess_mcse_df <- data.frame(
  Parameter = names(ess_values),
  ESS = ess_values,
  MCSE = mcse_values[names(ess_values)]
)

# Print the combined ESS and MCSE table
print(ess_mcse_df)

# Save the ESS and MCSE values to a CSV file
write.csv(ess_mcse_df, "synthetic_data_output/ess_mcse_values.csv", row.names = FALSE)

