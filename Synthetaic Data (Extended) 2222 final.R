# Install and load necessary packages
if (!requireNamespace("brms", quietly = TRUE)) {
  install.packages("brms")
}
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

# Specify Jeffrey's prior for microbiota_diversity, cytokine_levels, villus_height, and crypt_depth
priors <- c(
  prior(normal(0, 1), class = "b"),
  prior(cauchy(0, 2.5), class = "sd"),
  prior(cauchy(0, 2.5), class = "sigma"),
  prior(constant(1), class = "b", coef = "microbiota_diversity"),
  prior(constant(1), class = "b", coef = "cytokine_levels"),
  prior(constant(1), class = "b", coef = "villus_height"),
  prior(constant(1), class = "b", coef = "crypt_depth")
)

# Fit the Bayesian hierarchical model using brms
brms_model <- brm(
  growth_efficiency ~ diet + stress + genetic + microbiota_diversity +
    cytokine_levels + villus_height + crypt_depth +
    (1 | group_id) + (1 | animal_id),
  data = animal_data,
  family = gaussian(),
  prior = priors,
  chains = 1, iter = 10000, warmup = 5000, seed = 123
)

# Save the model summary
brms_summary <- summary(brms_model)
capture.output(brms_summary, file = "synthetic_data_output/brms_model_summary.txt")

# Create and save plots
posterior_density_combined <- mcmc_dens_overlay(
  as.array(brms_model),
  pars = c("b_diet", "b_stress", "b_genetic", "b_microbiota_diversity",
           "b_cytokine_levels", "b_villus_height", "b_crypt_depth")
) +
  geom_vline(xintercept = mu_beta, color = c("red", "green", "blue", "purple", "orange", "brown", "black"), linetype = "dashed", size = 0.5) +
  labs(title = "Posterior Densities with Truth Lines for Each Parameter")
ggsave("synthetic_data_output/posterior_density_combined.png", posterior_density_combined)
print(posterior_density_combined)

# Predicted vs Observed plot
predicted <- apply(fitted(brms_model), 1, mean)
observed <- animal_data$growth_efficiency
pred_vs_obs_plot <- ggplot(data = NULL, aes(x = observed, y = predicted)) +
  geom_point(color = 'blue') +
  geom_abline(slope = 1, intercept = 0, color = 'red') +
  labs(title = "Predicted vs. Observed Growth Efficiency",
       x = "Observed",
       y = "Predicted")
ggsave("synthetic_data_output/pred_vs_obs.png", pred_vs_obs_plot)
print(pred_vs_obs_plot)

# Posterior predictive check
ppc_plot <- pp_check(brms_model)
ggsave("synthetic_data_output/ppc_plot.png", ppc_plot)
print(ppc_plot)

# Traceplot for MCMC convergence diagnostics
traceplot <- mcmc_trace(as.array(brms_model), pars = c("b_diet", "b_stress", "b_genetic", "b_microbiota_diversity",
                                                       "b_cytokine_levels", "b_villus_height", "b_crypt_depth"))
ggsave("synthetic_data_output/traceplot.png", traceplot)
print(traceplot)

# Save posterior samples
posterior_samples <- as.data.frame(brms_model)
write.csv(posterior_samples, "synthetic_data_output/posterior_samples.csv")

# Summary of fixed effects
fixed_effects_summary <- as.data.frame(brms_summary$fixed)
write.csv(fixed_effects_summary, "synthetic_data_output/fixed_effects_summary.csv")
print(fixed_effects_summary)
