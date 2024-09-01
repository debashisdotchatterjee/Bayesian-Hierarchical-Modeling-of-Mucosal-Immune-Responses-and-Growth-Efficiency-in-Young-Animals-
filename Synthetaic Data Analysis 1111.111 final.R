# Load and install necessary libraries if required
if (!requireNamespace("rstan", quietly = TRUE)) {
  install.packages("rstan")
}
if (!requireNamespace("brms", quietly = TRUE)) {
  install.packages("brms")
}
if (!requireNamespace("bayesplot", quietly = TRUE)) {
  install.packages("bayesplot")
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

library(brms)
library(bayesplot)
library(ggplot2)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Parameters for synthetic data
n_animals <- 100
n_groups <- 5
mu_beta <- c(0.5, 0.3, -0.2)  # True coefficients for diet, stress, genetic
sigma_u <- 0.5                # Animal-level random effect
sigma_v <- 0.3                # Group-level random effect
sigma_eps <- 1.0              # Residual error

# Generate random group-level effects
group_effects <- rnorm(n_groups, mean = 0, sd = sigma_v)

# Generate random animal-level effects and covariates
animal_data <- data.frame(
  animal_id = 1:n_animals,
  group_id = sample(1:n_groups, n_animals, replace = TRUE),
  diet = rnorm(n_animals, mean = 0, sd = 1),
  stress = rnorm(n_animals, mean = 0, sd = 1),
  genetic = rnorm(n_animals, mean = 0, sd = 1)
)

# Create interaction terms
animal_data$interaction <- with(animal_data, diet * stress)

# Simulate the outcome variable (growth efficiency)
animal_data$growth_efficiency <- with(animal_data, 
                                      mu_beta[1] * diet + mu_beta[2] * stress + mu_beta[3] * genetic + 
                                        group_effects[group_id] + rnorm(n_animals, mean = 0, sd = sigma_eps)
)

# Save the synthetic dataset
dir.create("synthetic_data_output", showWarnings = FALSE)
write.csv(animal_data, "synthetic_data_output/synthetic_animal_data.csv", row.names = FALSE)

# Print the first few rows of the dataset
print(head(animal_data))

# Fit the Bayesian hierarchical model using brms with non-centered parameterization
brms_model <- brm(
  bf(growth_efficiency ~ diet + stress + genetic + (1 | group_id) + (1 | animal_id)),
  data = animal_data,
  family = gaussian(),
  prior = c(set_prior("normal(0, 1)", class = "b"),
            set_prior("normal(0, 5)", class = "Intercept"),
            set_prior("cauchy(0, 2.5)", class = "sigma")),
  chains = 4, iter = 10000, warmup = 5000, seed = 123,
  control = list(adapt_delta = 0.9999, max_treedepth = 15, stepsize = 0.01)
)

# Summary of the model
brms_summary <- summary(brms_model)
print(brms_summary)

# Save the model summary
capture.output(brms_summary, file = "synthetic_data_output/brms_model_summary.txt")

# Posterior distributions plot (focus on key parameters only) with truth lines
posterior_density_combined <- mcmc_dens_overlay(
  as.array(brms_model),
  pars = c("b_diet", "b_stress", "b_genetic")
) +
  geom_vline(xintercept = mu_beta[1], color = "red", linetype = "dashed", size = 0.5) +
  geom_vline(xintercept = mu_beta[2], color = "green", linetype = "dashed", size = 0.5) +
  geom_vline(xintercept = mu_beta[3], color = "blue", linetype = "dashed", size = 0.5) +
  labs(title = "Posterior Densities with Truth Lines for Each Parameter")

ggsave("synthetic_data_output/posterior_density_combined.png", posterior_density_combined)
print(posterior_density_combined)

# Predicted vs Observed plot with R²
predicted <- apply(fitted(brms_model), 1, mean)  # Taking the mean of the posterior predictions
observed <- animal_data$growth_efficiency

# Calculate R²
r_squared <- cor(predicted, observed)^2

pred_vs_obs_plot <- ggplot(data = NULL, aes(x = observed, y = predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red') +
  labs(title = paste("Predicted vs. Observed Growth Efficiency\nR² =", round(r_squared, 3)),
       x = "Observed",
       y = "Predicted")

ggsave("synthetic_data_output/pred_vs_obs.png", pred_vs_obs_plot)
print(pred_vs_obs_plot)

# Posterior predictive check
ppc_plot <- pp_check(brms_model)
ggsave("synthetic_data_output/ppc_plot.png", ppc_plot)
print(ppc_plot)

# Traceplot for MCMC convergence diagnostics
traceplot <- mcmc_trace(as.array(brms_model), pars = c("b_diet", "b_stress", "b_genetic"))
ggsave("synthetic_data_output/traceplot.png", traceplot)
print(traceplot)

# Save posterior samples
posterior_samples <- as.data.frame(brms_model)
write.csv(posterior_samples, "synthetic_data_output/posterior_samples.csv")

# Summary of fixed effects
fixed_effects_summary <- as.data.frame(brms_summary$fixed)
write.csv(fixed_effects_summary, "synthetic_data_output/fixed_effects_summary.csv")
print(fixed_effects_summary)
