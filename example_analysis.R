source("ss_ate_surrogates.R")
data <- read.csv("simulated_analysis_dataset.csv")

library(foreach)
library(progressr)
library(doFuture)

#### Example 1: Using GLMs for treatment and labeling propensity scores
treat_spec <- list(
  method = "glm",
  formula = tx ~ sex_cd + race_cd_updated + age_tx_months + mean_crp + cci_score + num_encounters_q + gcortico + csdmard + other,
  family = binomial()
)

label_spec <- list(
  method = "glm",
  formula = r ~ sex_cd + race_cd_updated + age_tx_months + mean_crp + cci_score + num_encounters_q + gcortico + csdmard + other + tx,
  family = binomial()
)

covariates <- c("sex_cd", "race_cd_updated", "age_tx_months", "mean_crp", "cci_score", "num_encounters_q", "gcortico", "csdmard", "other")

plan(multisession, gc = TRUE, workers = 8)
results_glm <- ss_ate_surrogates(data,
                           kfolds = 5,
                           outcome = "das28_crp3_binary",
                           label = "r",
                           treatment = "tx",
                           naive_imp = "xihat",
                           covariates = covariates,
                           return_analytic_var = TRUE,
                           missing = c("mean_crp"),
                           treat_spec = treat_spec,
                           label_spec = label_spec,
                           prob_bound = 1e-6,
                           return_bootstrap_var = TRUE,
                           nboot = 1000,
                           bootstrap_mode = c("foreach"),
                           bootstrap_seed = 999,
                           mainfit_seed = 999,
                           show_progress = TRUE,
                           diag = FALSE)
plan("sequential")
results_glm

#### Example 2: Using Lasso for treatment and labeling propensity scores
library(glmnet)
treat_spec <- list(
  method = "glmnet",
  xvars = c("sex_cd", "race_cd_updated", "age_tx_months", "mean_crp", "cci_score", "num_encounters_q", "gcortico", "csdmard", "other"),
  alpha = 1,
  nfolds = 5,
  lambda_choice = "lambda.min"
)

label_spec <- list(
  method = "glmnet",
  xvars = c("sex_cd", "race_cd_updated", "age_tx_months", "mean_crp", "cci_score", "num_encounters_q", "gcortico", "csdmard", "other", "tx"),
  alpha = 1,
  nfolds = 5,
  lambda_choice = "lambda.min"
)

covariates <- c("sex_cd", "race_cd_updated", "age_tx_months", "mean_crp", "cci_score", "num_encounters_q", "gcortico", "csdmard", "other")

plan(multisession, gc = TRUE, workers = 8)
results_lasso <- ss_ate_surrogates(data, 
                                   kfolds = 5, 
                                   outcome = "das28_crp3_binary", 
                                   label = "r", 
                                   treatment = "tx", 
                                   naive_imp = "xihat", 
                                   covariates = covariates,
                                   return_analytic_var = TRUE, 
                                   missing = c("mean_crp"), 
                                   treat_spec = treat_spec,
                                   label_spec = label_spec, 
                                   prob_bound = 1e-6,
                                   return_bootstrap_var = TRUE,
                                   nboot = 1000,
                                   bootstrap_mode = c("foreach"),
                                   bootstrap_seed = 999,
                                   mainfit_seed = 999,
                                   show_progress = TRUE,
                                   diag = FALSE)
plan("sequential")
results_lasso


#### Example 3: Using SuperLearner for treatment and labeling propensity scores
# Note this may take a few minutes, hence adjusted nboot = 100 for illustrative purposes
library(SuperLearner)
library(gam)
library(glmnet)
treat_spec <- list(
  method = "superlearner",
  xvars = c("sex_cd", "race_cd_updated", "age_tx_months", "mean_crp", "cci_score", "num_encounters_q", "gcortico", "csdmard", "other"),
  SL.library = c("SL.glm", "SL.glmnet", "SL.gam")
)

label_spec <- list(
  method = "superlearner",
  xvars = c("sex_cd", "race_cd_updated", "age_tx_months", "mean_crp", "cci_score", "num_encounters_q", "gcortico", "csdmard", "other", "tx"),
  SL.library = c("SL.glm", "SL.glmnet", "SL.gam")
)

covariates <- c("sex_cd", "race_cd_updated", "age_tx_months", "mean_crp", "cci_score", "num_encounters_q", "gcortico", "csdmard", "other")

plan(multisession, gc = TRUE, workers = 8)
results_superlearner <- ss_ate_surrogates(data, 
                                           kfolds = 5, 
                                           outcome = "das28_crp3_binary", 
                                           label = "r", 
                                           treatment = "tx", 
                                           naive_imp = "xihat", 
                                           covariates = covariates,
                                           return_analytic_var = TRUE, 
                                           missing = c("mean_crp"), 
                                           treat_spec = treat_spec,
                                           label_spec = label_spec, 
                                           prob_bound = 1e-6,
                                           return_bootstrap_var = TRUE,
                                           nboot = 100,
                                           bootstrap_mode = c("foreach"),
                                           bootstrap_seed = 999,
                                           mainfit_seed = 999,
                                           show_progress = TRUE,
                                           diag = FALSE)
plan("sequential")
results_superlearner