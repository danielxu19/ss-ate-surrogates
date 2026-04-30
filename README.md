# Semi-supervised ATE estimation with surrogates

This repository contains R code accompanying "Semi-Supervised Calibration of Inferred Outcomes for Estimating Average Treatment Effects with Validated Outcomes" for estimating treatment-specific means and the average treatment effect (ATE) in a semi-supervised setting where the primary outcome is only observed for a labeled subset of units.

## Files

- `ss_ate_surrogates.R`: main estimation function `ss_ate_surrogates()`
- `simulated_analysis_dataset.csv`: simulated dataset based on the real data application in the main paper
- `example_analysis.R`: example applying the method to `simulated_analysis_dataset.csv`

## Method summary

The function `ss_ate_surrogates()` combines:
- cross-fitted treatment propensity score estimation
- cross-fitted labeling propensity score estimation
- calibration of a user-supplied naive imputation model
- arm-specific outcome regression

It returns estimates of:
- `mu1 = E[Y(1)]`
- `mu0 = E[Y(0)]`
- `ATE = mu1 - mu0`

Optional analytic and bootstrap variance estimates are also available.

## Required inputs

The input data frame must contain:
- outcome variable
- labeling indicator
- treatment indicator
- naive imputation variable
- covariates used in nuisance models

Assumptions for the current implementation:
- treatment is binary and coded 0/1
- labeling indicator is binary and coded 0/1
- the naive imputation variable is on the probability scale for binary outcomes

## Supported nuisance models

For treatment and labeling propensity scores, the function supports:
- `glm`
- `glmnet`
- `superlearner`

Each nuisance model is specified using a named list such as:

```r
list(method = "glm", formula = ..., family = ...)
list(method = "glmnet", formula = ... or xvars = ..., alpha = 1, nfolds = 5, family = "binomial", lambda_choice = "lambda.min")
list(method = "superlearner", formula = ... or xvars = ..., family = binomial(), SL.library = c(...))
```
