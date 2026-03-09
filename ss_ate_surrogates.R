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


ss_ate_surrogates <- function(data, kfolds, outcome, label, treatment, naive_imp,
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
                              show_progress = TRUE) {
  
  bootstrap_mode <- match.arg(bootstrap_mode)
  
  expit <- function(x) 1 / (1 + exp(-x))
  logit <- function(p) log(p / (1 - p))
  bound_prob <- function(p, eps = prob_bound) pmin(pmax(p, eps), 1 - eps)
  `%||%` <- function(x, y) if (is.null(x)) y else x
  
  impute_mean <- function(x) {
    x[is.na(x)] <- mean(x, na.rm = TRUE)
    x
  }
  
  compute_arm_stats <- function(A, pA, R, pl, Y_orig, Y_imp_naive, Y_imp_cal, Y_out, N) {
    ipw_cc <- (1 / N) * sum(R * A * Y_orig / (pA * pl))
    ipw_naive <- (1 / N) * sum(A * Y_imp_naive / pA)
    ipw <- (1 / N) * sum(A * Y_imp_cal / pA)
    aipw_ee <- (1 / N) * sum(
      (A * Y_imp_naive / pA) +
        ((1 - A / pA) * Y_out) +
        (A * R * (Y_orig - Y_imp_naive) / (pA * pl))
    )
    aipw <- (1 / N) * sum(
      (A * Y_imp_cal / pA) +
        (1 - A / pA) * Y_out
    )
    
    var_full_if <- (A * Y_imp_naive / pA) +
      ((1 - A / pA) * Y_out) +
      (A * R * (Y_orig - Y_imp_naive) / (pA * pl)) - aipw
    
    var_simplified_if <- (R * A * (Y_orig - Y_imp_naive)) / (pA * pl)
    
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
      )
    )
  }
  
  make_default_formula <- function(response, data_names, exclude) {
    rhs_vars <- setdiff(data_names, c(response, exclude))
    reformulate(rhs_vars, response = response)
  }
  
  resolve_formula <- function(spec, response, data, exclude = character()) {
    if (!is.null(spec$formula)) {
      fm <- spec$formula
      if (is.character(fm)) fm <- as.formula(fm)
      return(fm)
    }
    
    if (!is.null(spec$rhs))
      return(as.formula(paste(response, "~", spec$rhs)))
    
    if (!is.null(spec$xvars))
      return(reformulate(spec$xvars, response = response))
    
    make_default_formula(response, names(data), exclude = exclude)
  }
  
  fit_binary_model <- function(train_data, response, spec, exclude = character()) {
    method <- tolower(spec$method %||% "glm")
    
    if (method == "glm") {
      fm <- resolve_formula(spec, response, train_data, exclude = exclude)
      fam <- spec$family %||% binomial()
      fit <- glm(fm, family = fam, data = train_data)
      return(list(method = method, fit = fit))
    }
    
    if (method == "glmnet") {
      if (!requireNamespace("glmnet", quietly = TRUE))
        stop("Package 'glmnet' required.")
      
      fm <- resolve_formula(spec, response, train_data, exclude = exclude)
      tt <- terms(fm, data = train_data)
      X <- model.matrix(tt, data = train_data)
      if ("(Intercept)" %in% colnames(X))
        X <- X[, colnames(X) != "(Intercept)", drop = FALSE]
      
      y <- train_data[[response]]
      
      fit <- glmnet::cv.glmnet(
        x = X,
        y = y,
        family = spec$family %||% "binomial",
        alpha = spec$alpha %||% 1,
        nfolds = spec$nfolds %||% 5
      )
      
      return(list(
        method = method,
        fit = fit,
        terms = tt,
        lambda_choice = spec$lambda_choice %||% "lambda.min"
      ))
    }
    
    if (method == "superlearner") {
      if (!requireNamespace("SuperLearner", quietly = TRUE))
        stop("Package 'SuperLearner' required.")
      
      fm <- resolve_formula(spec, response, train_data, exclude = exclude)
      tt <- delete.response(terms(fm, data = train_data))
      X <- model.frame(tt, data = train_data, na.action = na.pass)
      y <- train_data[[response]]
      
      fit <- SuperLearner::SuperLearner(
        Y = y,
        X = X,
        family = spec$family %||% binomial(),
        SL.library = spec$SL.library %||% c("SL.glm", "SL.mean")
      )
      
      return(list(method = method, fit = fit, terms = tt))
    }
    
    stop("Unsupported learner")
  }
  
  predict_binary_model <- function(obj, new_data) {
    method <- obj$method
    
    if (method == "glm")
      return(bound_prob(as.numeric(predict(obj$fit, newdata = new_data, type = "response"))))
    
    if (method == "glmnet") {
      Xnew <- model.matrix(obj$terms, data = new_data)
      if ("(Intercept)" %in% colnames(Xnew))
        Xnew <- Xnew[, colnames(Xnew) != "(Intercept)", drop = FALSE]
      
      p <- predict(obj$fit, newx = Xnew, s = obj$lambda_choice, type = "response")
      return(bound_prob(as.numeric(p)))
    }
    
    if (method == "superlearner") {
      Xnew <- model.frame(obj$terms, data = new_data, na.action = na.pass)
      p <- predict(obj$fit, newdata = Xnew)$pred
      return(bound_prob(as.numeric(p)))
    }
    
    stop("Unsupported prediction method")
  }
  
  run_once <- function(data_in) {
    
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
      data_in[, missing] <- lapply(data_in[, missing, drop = FALSE], impute_mean)
    
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
    
    for (j in seq_len(kfolds)) {
      train_idx <- K != j
      test_idx <- K == j
      
      train_data_noY <- data_noY[train_idx, , drop = FALSE]
      test_data_noY  <- data_noY[test_idx, , drop = FALSE]
      
      treat_fit <- fit_binary_model(train_data_noY, treatment, treat_spec,
                                    exclude = c(label, "k", naive_imp))
      pd[test_idx] <- predict_binary_model(treat_fit, test_data_noY)
      
      label_fit <- fit_binary_model(train_data_noY, label, label_spec,
                                    exclude = c("k", naive_imp))
      pl[test_idx] <- predict_binary_model(label_fit, test_data_noY)
    }
    
    for (j in seq_len(kfolds)) {
      train_idx <- K != j
      test_idx <- K == j
      
      weights_oc_T1 <- T * R * train_idx / pl
      weights_oc_T0 <- (1 - T) * R * train_idx / pl
      
      outcome_calibration_fm <- as.formula(
        paste0(outcome, " ~ . -", naive_imp, " -", label, " -", treatment, " -k",
               " + offset(logit(", naive_imp, "))")
      )
      
      outcome_calibration_T1 <- glm(outcome_calibration_fm, data = data_in,
                                    weights = weights_oc_T1, family = quasibinomial())
      
      Y_calibrated_T1[train_idx] <- predict(outcome_calibration_T1,
                                            newdata = data_in[train_idx, ],
                                            type = "response")
      
      outcome_calibration_T0 <- glm(outcome_calibration_fm, data = data_in,
                                    weights = weights_oc_T0, family = quasibinomial())
      
      Y_calibrated_T0[train_idx] <- predict(outcome_calibration_T0,
                                            newdata = data_in[train_idx, ],
                                            type = "response")
      
      outcome_model_T1 <- glm(
        as.formula(paste0("Y_calibrated_T1 ~ . -", naive_imp,
                          " -", label, " -", treatment, " -k")),
        data = data_noY, weights = T, family = quasibinomial(),
        subset = train_idx
      )
      
      mhat_T1[test_idx] <- predict(outcome_model_T1,
                                   newdata = data_in[test_idx, ],
                                   type = "response")
      
      outcome_model_T0 <- glm(
        as.formula(paste0("Y_calibrated_T0 ~ . -", naive_imp,
                          " -", label, " -", treatment, " -k")),
        data = data_noY, weights = 1 - T, family = quasibinomial(),
        subset = train_idx
      )
      
      mhat_T0[test_idx] <- predict(outcome_model_T0,
                                   newdata = data_in[test_idx, ],
                                   type = "response")
    }
    
    U_T1 <- T / pd
    imp1 <- glm(Y_obs ~ -1 + U_T1 + offset(logit(Y_imp_naive)),
                weights = R / pl, family = quasibinomial())
    
    Y_pred_imp_T1 <- expit(logit(Y_imp_naive) + coef(imp1) / pd)
    
    U_T0 <- (1 - T) / (1 - pd)
    imp0 <- glm(Y_obs ~ -1 + U_T0 + offset(logit(Y_imp_naive)),
                weights = R / pl, family = quasibinomial())
    
    Y_pred_imp_T0 <- expit(logit(Y_imp_naive) + coef(imp0) / (1 - pd))
    
    Y_original <- ifelse(is.na(Y_obs), 0, Y_obs)
    
    arm1 <- compute_arm_stats(T, pd, R, pl, Y_original,
                              Y_imp_naive, Y_pred_imp_T1, mhat_T1, N)
    
    arm0 <- compute_arm_stats(1 - T, 1 - pd, R, pl, Y_original,
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
    
    list(results.mu = results.mu, results.var = results.var)
  }
  
  set.seed(mainfit_seed)
  main_fit <- run_once(data)
  results.mu <- main_fit$results.mu
  results.var <- main_fit$results.var
  
  if (!return_bootstrap_var) {
    if (return_analytic_var)
      return(list("Estimates" = results.mu, "Analytic Variance" = results.var))
    else
      return(results.mu)
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
        boot_fit <- run_once(data[idx, , drop = FALSE])
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
        boot_fit <- run_once(data[idx, , drop = FALSE])
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
  
  if (return_analytic_var)
    list("Estimates" = results.mu, "Analytic Variance" = results.var, "Bootstrap Variance" = bootstrap_var_df)
  else
    list("Estimates" = results.mu, "Analytic Variance" = NULL, "Bootstrap Variance" = bootstrap_var_df)
}
