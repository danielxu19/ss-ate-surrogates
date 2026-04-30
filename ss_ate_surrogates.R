#' Estimate ATE using surrogates in a semi-supervised setting
#'
#' @description
#' `ss_ate_surrogates()` estimates the treatment-specific means
#' \eqn{\mu_1 = E[Y(1)]}, \eqn{\mu_0 = E[Y(0)]}, and the average treatment effect
#' \eqn{ATE = \mu_1 - \mu_0} in a semi-supervised setting where the outcome is
#' only observed among labeled units. The function combines:
#' \itemize{
#'   \item cross-fitted treatment propensity score estimation,
#'   \item cross-fitted labeling propensity score estimation,
#'   \item calibration of user-supplied naive imputations for the outcome,
#'   \item outcome regression using user-supplied naive imputations.
#' }
#'
#' Treatment and labeling propensity scores can be estimated using logistic
#' regression (`glm`), lasso logistic regression (`glmnet`), or Super Learner
#' (`superlearner`).
#'
#' @param data A data frame containing all variables needed for estimation.
#'
#' @param kfolds Integer number of folds used for cross-fitting.
#'
#' @param outcome Character string giving the name of the outcome variable.
#' Outcome may be missing for unlabeled observations.
#'
#' @param label Character string giving the name of the labeling indicator.
#' This should be binary, with 1 indicating the outcome is observed.
#'
#' @param treatment Character string giving the name of the binary treatment variable.
#'
#' @param naive_imp Character string giving the name of the naive imputed outcome
#' prediction. This should be a variable already present in `data`, typically a
#' predicted probability or fitted outcome value from an external imputation model.
#'
#' @param covariates Character vector giving the covariate names to use as the
#' main-effect predictors in the outcome calibration and calibrated outcome
#' regression models.
#'
#' @param return_analytic_var Logical. If `TRUE`, analytic variance estimates are
#' returned in addition to point estimates.
#'
#' @param missing Optional character vector of covariate names for which missing
#' values should be mean-imputed before fitting nuisance models.
#'
#' @param treat_spec A named list specifying the treatment propensity model.
#' Supported values:
#' \itemize{
#'   \item `list(method = "glm", formula = ..., family = ...)`
#'   \item `list(method = "glmnet", formula = ... or xvars = ..., alpha = 1,
#'          nfolds = 5, family = "binomial", lambda_choice = "lambda.min")`
#'   \item `list(method = "superlearner", formula = ... or xvars = ...,
#'          family = binomial(), SL.library = c(...))`
#' }
#' If no formula, rhs, or xvars are supplied, the default uses all columns in
#' `data` except the response and excluded variables.
#'
#' @param label_spec A named list specifying the labeling propensity model, with
#' the same structure as `treat_spec`.
#'
#' @param prob_bound Numeric scalar used to truncate estimated propensity scores
#' away from 0 and 1. Default is `1e-6`.
#'
#' @param return_bootstrap_var Logical. If `TRUE`, bootstrap variance estimates
#' are returned.
#'
#' @param nboot Integer number of bootstrap resamples used when
#' `return_bootstrap_var = TRUE`.
#'
#' @param bootstrap_mode Character string, either `"for"` or `"foreach"`.
#' `"for"` uses a standard loop; `"foreach"` uses `%dofuture%` for parallel
#' bootstrap computation.
#'
#' @param bootstrap_seed Integer seed used before bootstrap resampling.
#'
#' @param mainfit_seed Integer seed used before the main sample split / cross-fit.
#'
#' @param show_progress Logical. If `TRUE` and package `progressr` is available,
#' progress bars are displayed during bootstrap.
#'
#' @param diag Logical. If `TRUE`, additionally returns diagnostics from the
#' main estimation run, including nuisance predictions, fold assignments,
#' per-observation estimator contributions, model fits / coefficients, and
#' warnings produced while fitting nuisance models. Bootstrap runs never return
#' diagnostics.
#'
#' @details
#' The function first performs cross-fitting to estimate the treatment propensity
#' score and the labeling propensity score. These nuisance functions are then used
#' to:
#' \enumerate{
#'   \item calibrate the naive imputation model separately for treatment arms 1 and 0,
#'   \item calibrated outcome regression,
#'   \item compute estimators of \eqn{\mu_1}, \eqn{\mu_0}, and the ATE.
#' }
#'
#' The returned point estimators are:
#' \itemize{
#'   \item `ipw.cc`: complete-case inverse probability weighted estimator
#'   \item `aipw.ee`: augmented IPW estimator using naive imputation in the estimating equation
#'   \item `ipw.naive`: IPW estimator based on the naive imputation model
#'   \item `ipw`: IPW estimator based on the calibrated imputation model
#'   \item `aipw`: augmented IPW estimator based on the calibrated imputation model
#' }
#'
#' The analytic variance estimates returned are:
#' \itemize{
#'   \item `var.full`: full influence-function-based variance estimate
#'   \item `var.simplified`: simplified variance component based only on the
#'   residual correction term
#' }
#'
#' @return
#' If `return_bootstrap_var = FALSE`:
#' \itemize{
#'   \item if `return_analytic_var = FALSE`, returns a data frame of point estimates.
#'   \item if `return_analytic_var = TRUE`, returns a named list with:
#'     \itemize{
#'       \item `"Estimates"`: data frame of point estimates
#'       \item `"Analytic Variance"`: data frame of analytic variance estimates
#'     }
#' }
#'
#' If `return_bootstrap_var = TRUE`, returns a named list with:
#' \itemize{
#'   \item `"Estimates"`: data frame of point estimates
#'   \item `"Analytic Variance"`: analytic variance data frame, or `NULL` if
#'   `return_analytic_var = FALSE`
#'   \item `"Bootstrap Variance"`: data frame of bootstrap variances for each
#'   estimator for `mu1`, `mu0`, and `ate`
#' }
#'
#' If `diag = TRUE`, the return value additionally includes `"Diagnostics"`,
#' containing observation-level diagnostics, nuisance model fits / coefficients,
#' and nuisance-model fitting warnings from the main estimation run only.
#'
#' @examples
#' # Default logistic regression nuisance models
#' fit <- ss_ate_surrogates(
#'   data = dat,
#'   kfolds = 5,
#'   outcome = "Y",
#'   label = "R",
#'   treatment = "A",
#'   naive_imp = "mhat",
#'   return_analytic_var = TRUE
#' )
#'
#' # Lasso treatment model and GLM labeling model
#' fit <- ss_ate_surrogates(
#'   data = dat,
#'   kfolds = 5,
#'   outcome = "Y",
#'   label = "R",
#'   treatment = "A",
#'   naive_imp = "mhat",
#'   return_analytic_var = TRUE,
#'   treat_spec = list(
#'     method = "glmnet",
#'     xvars = c("X1", "X2", "X3"),
#'     alpha = 1,
#'     nfolds = 5,
#'     lambda_choice = "lambda.min"
#'   ),
#'   label_spec = list(
#'     method = "glm",
#'     formula = R ~ X1 + X2 + X3
#'   )
#' )
#'
#' # SuperLearner propensity models with bootstrap variance
#' fit <- ss_ate_surrogates(
#'   data = dat,
#'   kfolds = 5,
#'   outcome = "Y",
#'   label = "R",
#'   treatment = "A",
#'   naive_imp = "mhat",
#'   return_analytic_var = TRUE,
#'   return_bootstrap_var = TRUE,
#'   nboot = 100,
#'   bootstrap_mode = "for",
#'   treat_spec = list(
#'     method = "superlearner",
#'     xvars = c("X1", "X2", "X3"),
#'     SL.library = c("SL.glm", "SL.glmnet", "SL.mean")
#'   ),
#'   label_spec = list(
#'     method = "superlearner",
#'     xvars = c("X1", "X2", "X3"),
#'     SL.library = c("SL.glm", "SL.mean")
#'   )
#' )

ss_ate_surrogates_null_coalesce <- function(x, y) if (is.null(x)) y else x

ss_ate_surrogates_expit <- function(x) 1 / (1 + exp(-x))

ss_ate_surrogates_logit <- function(p) log(p / (1 - p))

ss_ate_surrogates_bound_prob <- function(p, prob_bound) {
  pmin(pmax(p, prob_bound), 1 - prob_bound)
}

ss_ate_surrogates_impute_mean <- function(x) {
  x[is.na(x)] <- mean(x, na.rm = TRUE)
  x
}

ss_ate_surrogates_compute_arm_stats <- function(A, pA, R, pl, Y_orig,
                                                Y_imp_naive, Y_imp_cal, Y_out, N) {
  contrib_ipw_cc <- (R * A * Y_orig / (pA * pl))
  contrib_ipw_naive <- (A * Y_imp_naive / pA)
  contrib_ipw <- (A * Y_imp_cal / pA)
  contrib_aipw_ee <- (
    (A * Y_imp_naive / pA) +
      ((1 - A / pA) * Y_out) +
      (A * R * (Y_orig - Y_imp_naive) / (pA * pl))
  )
  contrib_aipw <- (
    (A * Y_imp_cal / pA) +
      (1 - A / pA) * Y_out
  )
  
  ipw_cc <- sum(contrib_ipw_cc) / N
  ipw_naive <- sum(contrib_ipw_naive) / N
  ipw <- sum(contrib_ipw) / N
  aipw_ee <- sum(contrib_aipw_ee) / N
  aipw <- sum(contrib_aipw) / N
  
  var_full_if <- contrib_aipw_ee - aipw
  var_simplified_if <- (R * A * (Y_orig - Y_imp_naive) / (pA * pl))
  
  vars <- c(
    var.full = (1 / N^2) * sum(var_full_if^2),
    var.simplified = (1 / N^2) * sum(var_simplified_if^2)
  )
  
  list(
    mu = c(
      ipw.cc = ipw_cc,
      aipw.ee = aipw_ee,
      ipw.naive = ipw_naive,
      ipw = ipw,
      aipw = aipw
    ),
    vars = vars,
    ifs = list(
      full = var_full_if,
      simplified = var_simplified_if
    ),
    contribs = list(
      ipw_cc = contrib_ipw_cc,
      ipw_naive = contrib_ipw_naive,
      ipw = contrib_ipw,
      aipw_ee = contrib_aipw_ee,
      aipw = contrib_aipw
    )
  )
}

ss_ate_surrogates_make_default_formula <- function(response, data_names, exclude) {
  rhs_vars <- setdiff(data_names, c(response, exclude))
  reformulate(rhs_vars, response = response)
}

ss_ate_surrogates_resolve_formula <- function(spec, response, data, exclude = character()) {
  if (!is.null(spec$formula)) {
    fm <- spec$formula
    if (is.character(fm)) fm <- as.formula(fm)
    return(fm)
  }
  
  if (!is.null(spec$rhs))
    return(as.formula(paste(response, "~", spec$rhs)))
  
  if (!is.null(spec$xvars))
    return(reformulate(spec$xvars, response = response))
  
  ss_ate_surrogates_make_default_formula(response, names(data), exclude = exclude)
}

ss_ate_surrogates_fit_binary_model <- function(train_data, response, spec,
                                               exclude = character()) {
  method <- tolower(ss_ate_surrogates_null_coalesce(spec$method, "glm"))
  
  if (method == "glm") {
    fm <- ss_ate_surrogates_resolve_formula(spec, response, train_data, exclude = exclude)
    fam <- ss_ate_surrogates_null_coalesce(spec$family, binomial())
    fit <- glm(fm, family = fam, data = train_data)
    return(list(method = method, fit = fit))
  }
  
  if (method == "glmnet") {
    if (!requireNamespace("glmnet", quietly = TRUE))
      stop("Package 'glmnet' required.")
    
    fm <- ss_ate_surrogates_resolve_formula(spec, response, train_data, exclude = exclude)
    tt <- terms(fm, data = train_data)
    X <- model.matrix(tt, data = train_data)
    if ("(Intercept)" %in% colnames(X))
      X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
    
    y <- train_data[[response]]
    
    fit <- glmnet::cv.glmnet(
      x = X,
      y = y,
      family = ss_ate_surrogates_null_coalesce(spec$family, "binomial"),
      alpha = ss_ate_surrogates_null_coalesce(spec$alpha, 1),
      nfolds = ss_ate_surrogates_null_coalesce(spec$nfolds, 5)
    )
    
    return(list(
      method = method,
      fit = fit,
      terms = tt,
      lambda_choice = ss_ate_surrogates_null_coalesce(spec$lambda_choice, "lambda.min")
    ))
  }
  
  if (method == "superlearner") {
    if (!requireNamespace("SuperLearner", quietly = TRUE))
      stop("Package 'SuperLearner' required.")
    
    fm <- ss_ate_surrogates_resolve_formula(spec, response, train_data, exclude = exclude)
    tt <- delete.response(terms(fm, data = train_data))
    X <- model.frame(tt, data = train_data, na.action = na.pass)
    y <- train_data[[response]]
    
    fit <- SuperLearner::SuperLearner(
      Y = y,
      X = X,
      family = ss_ate_surrogates_null_coalesce(spec$family, binomial()),
      SL.library = ss_ate_surrogates_null_coalesce(spec$SL.library, c("SL.glm", "SL.mean"))
    )
    
    return(list(method = method, fit = fit, terms = tt))
  }
  
  stop("Unsupported learner")
}

ss_ate_surrogates_predict_binary_model <- function(obj, new_data, prob_bound) {
  method <- obj$method
  
  if (method == "glm") {
    return(ss_ate_surrogates_bound_prob(
      as.numeric(predict(obj$fit, newdata = new_data, type = "response")),
      prob_bound
    ))
  }
  
  if (method == "glmnet") {
    Xnew <- model.matrix(obj$terms, data = new_data)
    if ("(Intercept)" %in% colnames(Xnew))
      Xnew <- Xnew[, colnames(Xnew) != "(Intercept)", drop = FALSE]
    
    p <- predict(obj$fit, newx = Xnew, s = obj$lambda_choice, type = "response")
    return(ss_ate_surrogates_bound_prob(as.numeric(p), prob_bound))
  }
  
  if (method == "superlearner") {
    Xnew <- model.frame(obj$terms, data = new_data, na.action = na.pass)
    p <- predict(obj$fit, newdata = Xnew)$pred
    return(ss_ate_surrogates_bound_prob(as.numeric(p), prob_bound))
  }
  
  stop("Unsupported prediction method")
}

ss_ate_surrogates_capture_warning_value <- function(w) {
  if (!is.null(conditionMessage(w))) {
    return(conditionMessage(w))
  }
  as.character(w)
}

ss_ate_surrogates_fit_with_warnings <- function(expr, model_name, fold = NA_integer_) {
  warnings <- character()
  fit <- withCallingHandlers(
    expr,
    warning = function(w) {
      warnings <<- c(warnings, ss_ate_surrogates_capture_warning_value(w))
      invokeRestart("muffleWarning")
    }
  )
  
  list(
    fit = fit,
    warnings = if (length(warnings) == 0) {
      NULL
    } else {
      data.frame(
        model = rep(model_name, length(warnings)),
        fold = rep(fold, length(warnings)),
        warning = warnings,
        stringsAsFactors = FALSE
      )
    }
  )
}

ss_ate_surrogates_extract_model_coefficients <- function(model_obj) {
  if (is.null(model_obj))
    return(NULL)
  
  method <- ss_ate_surrogates_null_coalesce(model_obj$method, class(model_obj$fit)[1])
  
  if (!is.null(model_obj$method) && model_obj$method == "glm")
    return(stats::coef(model_obj$fit))
  
  if (!is.null(model_obj$method) && model_obj$method == "glmnet")
    return(as.matrix(stats::coef(model_obj$fit, s = model_obj$lambda_choice)))
  
  if (!is.null(model_obj$method) && model_obj$method == "superlearner") {
    coefs <- tryCatch(stats::coef(model_obj$fit), error = function(e) NULL)
    if (is.null(coefs) && !is.null(model_obj$fit$coef))
      coefs <- model_obj$fit$coef
    if (is.null(coefs) && !is.null(model_obj$fit$library.predict))
      coefs <- list(meta_weights = model_obj$fit$coef)
    return(coefs)
  }
  
  if (inherits(model_obj$fit, "glm"))
    return(stats::coef(model_obj$fit))
  
  tryCatch(stats::coef(model_obj$fit), error = function(e) {
    list(unavailable = sprintf("No coefficient extractor implemented for class '%s'", method))
  })
}

ss_ate_surrogates_run_once <- function(data_in, kfolds, outcome, label, treatment,
                                       naive_imp, covariates, missing, treat_spec, label_spec,
                                       prob_bound, diag = FALSE) {
  n_unlabeled <- sum(is.na(data_in[, outcome]))
  n_labeled <- nrow(data_in) - n_unlabeled
  
  data_in$k <- NA_integer_
  
  k_labeled <- sample(
    c(rep(1:(kfolds - 1), each = floor(n_labeled / kfolds)),
      rep(kfolds, times = n_labeled - (kfolds - 1) * floor(n_labeled / kfolds))),
    size = n_labeled, replace = FALSE
  )
  
  k_unlabeled <- sample(
    c(rep(1:(kfolds - 1), each = floor(n_unlabeled / kfolds)),
      rep(kfolds, times = n_unlabeled - (kfolds - 1) * floor(n_unlabeled / kfolds))),
    size = n_unlabeled, replace = FALSE
  )
  
  data_in$k[is.na(data_in[, outcome])] <- k_unlabeled
  data_in$k[!is.na(data_in[, outcome])] <- k_labeled
  
  if (!is.null(missing))
    data_in[, missing] <- lapply(data_in[, missing, drop = FALSE], ss_ate_surrogates_impute_mean)
  
  data_noY <- data_in[, -which(colnames(data_in) == outcome), drop = FALSE]
  
  N <- nrow(data_in)
  K <- data_in$k
  T <- data_in[, treatment]
  R <- data_in[, label]
  Y_obs <- data_in[, outcome]
  Y_imp_naive <- data_in[, naive_imp]
  
  pd <- pl <- rep(NA_real_, N)
  Y_calibrated_T1 <- Y_calibrated_T0 <- rep(NA_real_, N)
  mhat_T1 <- mhat_T0 <- rep(NA_real_, N)
  
  diag_fits <- NULL
  diag_coefficients <- NULL
  diag_warnings <- NULL
  
  if (diag) {
    diag_fits <- list(
      pd = vector("list", kfolds),
      pl = vector("list", kfolds),
      Y_calibrated_T1 = vector("list", kfolds),
      Y_calibrated_T0 = vector("list", kfolds),
      mhat_T1 = vector("list", kfolds),
      mhat_T0 = vector("list", kfolds),
      Y_pred_imp_T1 = NULL,
      Y_pred_imp_T0 = NULL
    )
    diag_coefficients <- list(
      pd = vector("list", kfolds),
      pl = vector("list", kfolds),
      Y_calibrated_T1 = vector("list", kfolds),
      Y_calibrated_T0 = vector("list", kfolds),
      mhat_T1 = vector("list", kfolds),
      mhat_T0 = vector("list", kfolds),
      Y_pred_imp_T1 = NULL,
      Y_pred_imp_T0 = NULL
    )
  }
  
  for (j in seq_len(kfolds)) {
    train_idx <- K != j
    test_idx <- K == j
    
    train_data_noY <- data_noY[train_idx, , drop = FALSE]
    test_data_noY  <- data_noY[test_idx, , drop = FALSE]
    
    treat_fit_info <- ss_ate_surrogates_fit_with_warnings(
      ss_ate_surrogates_fit_binary_model(
        train_data_noY, treatment, treat_spec,
        exclude = c(label, "k", naive_imp)
      ),
      model_name = "pd",
      fold = j
    )
    treat_fit <- treat_fit_info$fit
    pd[test_idx] <- ss_ate_surrogates_predict_binary_model(treat_fit, test_data_noY, prob_bound)
    if (!is.null(treat_fit_info$warnings))
      diag_warnings <- rbind(diag_warnings, treat_fit_info$warnings)
    if (diag) {
      diag_fits$pd[[j]] <- treat_fit
      diag_coefficients$pd[[j]] <- ss_ate_surrogates_extract_model_coefficients(treat_fit)
    }
    
    label_fit_info <- ss_ate_surrogates_fit_with_warnings(
      ss_ate_surrogates_fit_binary_model(
        train_data_noY, label, label_spec,
        exclude = c("k", naive_imp)
      ),
      model_name = "pl",
      fold = j
    )
    label_fit <- label_fit_info$fit
    pl[test_idx] <- ss_ate_surrogates_predict_binary_model(label_fit, test_data_noY, prob_bound)
    if (!is.null(label_fit_info$warnings))
      diag_warnings <- rbind(diag_warnings, label_fit_info$warnings)
    if (diag) {
      diag_fits$pl[[j]] <- label_fit
      diag_coefficients$pl[[j]] <- ss_ate_surrogates_extract_model_coefficients(label_fit)
    }
  }
  
  for (j in seq_len(kfolds)) {
    train_idx <- K != j
    test_idx <- K == j
    
    weights_oc_T1 <- T * R * train_idx / pl
    weights_oc_T0 <- (1 - T) * R * train_idx / pl
    
    outcome_calibration_fm <- reformulate(
      termlabels = covariates,
      response = outcome
    )
    outcome_calibration_fm[[3]] <- call(
      "+",
      outcome_calibration_fm[[3]],
      call("offset", call("ss_ate_surrogates_logit", as.name(naive_imp)))
    )
    
    outcome_calibration_T1_info <- ss_ate_surrogates_fit_with_warnings(
      glm(outcome_calibration_fm, data = data_in,
          weights = weights_oc_T1, family = quasibinomial()),
      model_name = "Y_calibrated_T1",
      fold = j
    )
    outcome_calibration_T1 <- outcome_calibration_T1_info$fit
    
    Y_calibrated_T1[train_idx] <- predict(outcome_calibration_T1,
                                          newdata = data_in[train_idx, ],
                                          type = "response")
    if (!is.null(outcome_calibration_T1_info$warnings))
      diag_warnings <- rbind(diag_warnings, outcome_calibration_T1_info$warnings)
    if (diag) {
      diag_fits$Y_calibrated_T1[[j]] <- outcome_calibration_T1
      diag_coefficients$Y_calibrated_T1[[j]] <- stats::coef(outcome_calibration_T1)
    }
    
    outcome_calibration_T0_info <- ss_ate_surrogates_fit_with_warnings(
      glm(outcome_calibration_fm, data = data_in,
          weights = weights_oc_T0, family = quasibinomial()),
      model_name = "Y_calibrated_T0",
      fold = j
    )
    outcome_calibration_T0 <- outcome_calibration_T0_info$fit
    
    Y_calibrated_T0[train_idx] <- predict(outcome_calibration_T0,
                                          newdata = data_in[train_idx, ],
                                          type = "response")
    if (!is.null(outcome_calibration_T0_info$warnings))
      diag_warnings <- rbind(diag_warnings, outcome_calibration_T0_info$warnings)
    if (diag) {
      diag_fits$Y_calibrated_T0[[j]] <- outcome_calibration_T0
      diag_coefficients$Y_calibrated_T0[[j]] <- stats::coef(outcome_calibration_T0)
    }
    
    outcome_model_T1_info <- ss_ate_surrogates_fit_with_warnings(
      glm(
        reformulate(covariates, response = "Y_calibrated_T1"),
        data = data_noY, weights = T, family = quasibinomial(),
        subset = train_idx
      ),
      model_name = "mhat_T1",
      fold = j
    )
    outcome_model_T1 <- outcome_model_T1_info$fit
    
    mhat_T1[test_idx] <- predict(outcome_model_T1,
                                 newdata = data_in[test_idx, ],
                                 type = "response")
    if (!is.null(outcome_model_T1_info$warnings))
      diag_warnings <- rbind(diag_warnings, outcome_model_T1_info$warnings)
    if (diag) {
      diag_fits$mhat_T1[[j]] <- outcome_model_T1
      diag_coefficients$mhat_T1[[j]] <- stats::coef(outcome_model_T1)
    }
    
    outcome_model_T0_info <- ss_ate_surrogates_fit_with_warnings(
      glm(
        reformulate(covariates, response = "Y_calibrated_T0"),
        data = data_noY, weights = 1 - T, family = quasibinomial(),
        subset = train_idx
      ),
      model_name = "mhat_T0",
      fold = j
    )
    outcome_model_T0 <- outcome_model_T0_info$fit
    
    mhat_T0[test_idx] <- predict(outcome_model_T0,
                                 newdata = data_in[test_idx, ],
                                 type = "response")
    if (!is.null(outcome_model_T0_info$warnings))
      diag_warnings <- rbind(diag_warnings, outcome_model_T0_info$warnings)
    if (diag) {
      diag_fits$mhat_T0[[j]] <- outcome_model_T0
      diag_coefficients$mhat_T0[[j]] <- stats::coef(outcome_model_T0)
    }
  }
  
  U_T1 <- T / pd
  imp1_info <- ss_ate_surrogates_fit_with_warnings(
    glm(Y_obs ~ -1 + U_T1 + offset(ss_ate_surrogates_logit(Y_imp_naive)),
        weights = R / pl, family = quasibinomial()),
    model_name = "Y_pred_imp_T1",
    fold = NA_integer_
  )
  imp1 <- imp1_info$fit
  
  Y_pred_imp_T1 <- ss_ate_surrogates_expit(
    ss_ate_surrogates_logit(Y_imp_naive) + coef(imp1) / pd
  )
  if (!is.null(imp1_info$warnings))
    diag_warnings <- rbind(diag_warnings, imp1_info$warnings)
  if (diag) {
    diag_fits$Y_pred_imp_T1 <- imp1
    diag_coefficients$Y_pred_imp_T1 <- stats::coef(imp1)
  }
  
  U_T0 <- (1 - T) / (1 - pd)
  imp0_info <- ss_ate_surrogates_fit_with_warnings(
    glm(Y_obs ~ -1 + U_T0 + offset(ss_ate_surrogates_logit(Y_imp_naive)),
        weights = R / pl, family = quasibinomial()),
    model_name = "Y_pred_imp_T0",
    fold = NA_integer_
  )
  imp0 <- imp0_info$fit
  
  Y_pred_imp_T0 <- ss_ate_surrogates_expit(
    ss_ate_surrogates_logit(Y_imp_naive) + coef(imp0) / (1 - pd)
  )
  if (!is.null(imp0_info$warnings))
    diag_warnings <- rbind(diag_warnings, imp0_info$warnings)
  if (diag) {
    diag_fits$Y_pred_imp_T0 <- imp0
    diag_coefficients$Y_pred_imp_T0 <- stats::coef(imp0)
  }
  
  Y_original <- ifelse(is.na(Y_obs), 0, Y_obs)
  
  arm1 <- ss_ate_surrogates_compute_arm_stats(T, pd, R, pl, Y_original,
                                              Y_imp_naive, Y_pred_imp_T1, mhat_T1, N)
  
  arm0 <- ss_ate_surrogates_compute_arm_stats(1 - T, 1 - pd, R, pl, Y_original,
                                              Y_imp_naive, Y_pred_imp_T0, mhat_T0, N)
  
  results.mu <- data.frame(mu1 = arm1$mu,
                           mu0 = arm0$mu,
                           ate = arm1$mu - arm0$mu)
  
  results.var <- data.frame(
    mu1 = arm1$vars,
    mu0 = arm0$vars,
    ate = c(
      var.full = (1 / N^2) * sum((arm1$ifs$full - arm0$ifs$full)^2),
      var.simplified = (1 / N^2) * sum((arm1$ifs$simplified - arm0$ifs$simplified)^2)
    )
  )
  
  out <- list(results.mu = results.mu, results.var = results.var)
  
  if (diag) {
    diag_original_data <- data_in[, unique(c(covariates, label, outcome, treatment, naive_imp)), drop = FALSE]
    
    diagnostics_df <- cbind(
      diag_original_data,
      data.frame(
        pd = pd,
        pl = pl,
        Y_calibrated_T1 = Y_calibrated_T1,
        Y_calibrated_T0 = Y_calibrated_T0,
        mhat_T1 = mhat_T1,
        mhat_T0 = mhat_T0,
        Y_pred_imp_T1 = Y_pred_imp_T1,
        Y_pred_imp_T0 = Y_pred_imp_T0,
        k = K,
        mu1_ipw_cc_contrib = arm1$contribs$ipw_cc,
        mu0_ipw_cc_contrib = arm0$contribs$ipw_cc,
        ate_ipw_cc_contrib = arm1$contribs$ipw_cc - arm0$contribs$ipw_cc,
        mu1_ipw_naive_contrib = arm1$contribs$ipw_naive,
        mu0_ipw_naive_contrib = arm0$contribs$ipw_naive,
        ate_ipw_naive_contrib = arm1$contribs$ipw_naive - arm0$contribs$ipw_naive,
        mu1_ipw_contrib = arm1$contribs$ipw,
        mu0_ipw_contrib = arm0$contribs$ipw,
        ate_ipw_contrib = arm1$contribs$ipw - arm0$contribs$ipw,
        mu1_aipw_ee_contrib = arm1$contribs$aipw_ee,
        mu0_aipw_ee_contrib = arm0$contribs$aipw_ee,
        ate_aipw_ee_contrib = arm1$contribs$aipw_ee - arm0$contribs$aipw_ee,
        mu1_aipw_contrib = arm1$contribs$aipw,
        mu0_aipw_contrib = arm0$contribs$aipw,
        ate_aipw_contrib = arm1$contribs$aipw - arm0$contribs$aipw,
        mu1_var_full_if = arm1$ifs$full,
        mu0_var_full_if = arm0$ifs$full,
        ate_var_full_if = arm1$ifs$full - arm0$ifs$full,
        mu1_var_simplified_if = arm1$ifs$simplified,
        mu0_var_simplified_if = arm0$ifs$simplified,
        ate_var_simplified_if = arm1$ifs$simplified - arm0$ifs$simplified
      )
    )
    
    out$diagnostics <- list(
      observation_level = diagnostics_df,
      model_fits = diag_fits,
      model_coefficients = diag_coefficients,
      warnings = if (is.null(diag_warnings)) {
        data.frame(
          model = character(),
          fold = integer(),
          warning = character(),
          stringsAsFactors = FALSE
        )
      } else {
        rownames(diag_warnings) <- NULL
        diag_warnings
      }
    )
  }
  
  out
}


ss_ate_surrogates <- function(data, kfolds, outcome, label, treatment, naive_imp,
                              covariates,
                              return_analytic_var,
                              missing = NULL,
                              treat_spec = list(method = "glm"),
                              label_spec = list(method = "glm"),
                              prob_bound = 1e-6,
                              return_bootstrap_var = FALSE,
                              nboot = 200,
                              bootstrap_mode = c("for", "foreach"),
                              bootstrap_seed = 999,
                              mainfit_seed = 999,
                              show_progress = TRUE,
                              diag = FALSE) {
  
  bootstrap_mode <- match.arg(bootstrap_mode)
  set.seed(mainfit_seed)
  main_fit <- ss_ate_surrogates_run_once(
    data_in = data,
    kfolds = kfolds,
    outcome = outcome,
    label = label,
    treatment = treatment,
    naive_imp = naive_imp,
    covariates = covariates,
    missing = missing,
    treat_spec = treat_spec,
    label_spec = label_spec,
    prob_bound = prob_bound,
    diag = diag
  )
  results.mu <- main_fit$results.mu
  results.var <- main_fit$results.var
  
  if (!return_bootstrap_var) {
    if (!diag) {
      if (return_analytic_var)
        return(list("Estimates" = results.mu, "Analytic Variance" = results.var))
      else
        return(results.mu)
    }
    
    return(list(
      "Estimates" = results.mu,
      "Analytic Variance" = if (return_analytic_var) results.var else NULL,
      "Diagnostics" = main_fit$diagnostics
    ))
  }
  
  boot_names <- as.vector(t(outer(rownames(results.mu),
                                  colnames(results.mu), paste, sep = "::")))
  
  boot_mat <- matrix(NA_real_, nrow = nboot, ncol = length(boot_names))
  colnames(boot_mat) <- boot_names
  
  set.seed(bootstrap_seed)
  
  run_bootstrap <- function(p = NULL) {
    if (bootstrap_mode == "for") {
      for (b in seq_len(nboot)) {
        idx <- sample.int(nrow(data), nrow(data), replace = TRUE)
        boot_fit <- ss_ate_surrogates_run_once(
          data_in = data[idx, , drop = FALSE],
          kfolds = kfolds,
          outcome = outcome,
          label = label,
          treatment = treatment,
          naive_imp = naive_imp,
          covariates = covariates,
          missing = missing,
          treat_spec = treat_spec,
          label_spec = label_spec,
          prob_bound = prob_bound,
          diag = FALSE
        )
        boot_mat[b, ] <<- as.numeric(as.matrix(boot_fit$results.mu))
        if (!is.null(p)) p(sprintf("bootstrap %d/%d", b, nboot))
      }
    }
    
    if (bootstrap_mode == "foreach") {
      if (!requireNamespace("foreach", quietly = TRUE))
        stop("Package 'foreach' required")
      if (!requireNamespace("doFuture", quietly = TRUE))
        stop("Package 'doFuture' required")
      
      doFuture::registerDoFuture()
      
      boot_res <- foreach::foreach(
        b = seq_len(nboot),
        .combine = rbind,
        .options.future = list(
          seed = TRUE,
          packages = c("SuperLearner", "glmnet", "gam")
        )
      ) %dofuture% {
        idx <- sample.int(nrow(data), nrow(data), replace = TRUE)
        boot_fit <- ss_ate_surrogates_run_once(
          data_in = data[idx, , drop = FALSE],
          kfolds = kfolds,
          outcome = outcome,
          label = label,
          treatment = treatment,
          naive_imp = naive_imp,
          covariates = covariates,
          missing = missing,
          treat_spec = treat_spec,
          label_spec = label_spec,
          prob_bound = prob_bound,
          diag = FALSE
        )
        out <- as.numeric(as.matrix(boot_fit$results.mu))
        if (!is.null(p)) p(sprintf("bootstrap %d/%d", b, nboot))
        out
      }
      
      boot_mat[,] <<- boot_res
    }
  }
  
  if (show_progress && requireNamespace("progressr", quietly = TRUE)) {
    progressr::with_progress({
      p <- progressr::progressor(steps = nboot)
      run_bootstrap(p)
    })
  } else {
    run_bootstrap(NULL)
  }
  
  boot_var_vec <- apply(boot_mat, 2, var, na.rm = TRUE)
  
  bootstrap_var_df <- matrix(
    boot_var_vec,
    nrow = nrow(results.mu),
    ncol = ncol(results.mu),
    byrow = FALSE,
    dimnames = list(rownames(results.mu), colnames(results.mu))
  )
  
  bootstrap_var_df <- as.data.frame(bootstrap_var_df)
  
  out <- list(
    "Estimates" = results.mu,
    "Analytic Variance" = if (return_analytic_var) results.var else NULL,
    "Bootstrap Variance" = bootstrap_var_df
  )
  
  if (diag)
    out$Diagnostics <- main_fit$diagnostics
  
  out
}
