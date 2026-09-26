# Summarise a sienaBayes run from its periodic checkpoint (PartialBayesResult.RData) rather
# than from the final object. Written 2026-09-26: the 19-school run (sienabayes.R, launched
# 2026-09-24 21:37) advanced normally to iteration ~500 of 600 and then stopped consuming
# CPU — 36 seconds of compute in the following 6.7 hours, with the machine awake. The
# checkpoint sienaBayes writes every saveFreq=100 iterations holds 500 completed draws
# (100 warm + 400 main), which is the posterior reported in docs/M5_RESULTS.md.
#   Rscript benchmarks/baerveldt/sienabayes_extract.R [PartialBayesResult.RData] [nwarm]
suppressPackageStartupMessages({ library(RSienaTest); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[1] else "PartialBayesResult.RData"
e <- new.env(); load(path, envir = e); z <- get(ls(e)[1], envir = e)
nwarm <- if (length(args) >= 2) as.integer(args[2]) else z$nwarm
mu <- z$ThinPosteriorMu
keep <- which(apply(mu, 1, function(r) all(is.finite(r))))
keep <- keep[keep > nwarm]
cat(sprintf("%s: %d completed draws, using %d after %d warm-up; %d groups\n",
            path, max(which(apply(mu, 1, function(r) all(is.finite(r))))), length(keep), nwarm, z$nGroup))
mu <- mu[keep, , drop = FALSE]
sig <- z$ThinPosteriorSigma[keep, , , drop = FALSE]
tau <- t(apply(sig, 1, function(S) sqrt(diag(matrix(S, nrow = ncol(mu))))))
res <- data.frame(parameter = z$effectName,
                  mu_mean = colMeans(mu), mu_q05 = apply(mu, 2, quantile, 0.05),
                  mu_q95 = apply(mu, 2, quantile, 0.95),
                  tau_mean = colMeans(tau), tau_q05 = apply(tau, 2, quantile, 0.05),
                  tau_q95 = apply(tau, 2, quantile, 0.95))
print(res, row.names = FALSE, digits = 3)
write_json(list(rsienatest_version = as.character(packageVersion("RSienaTest")),
                source = path, groups = z$nGroup, nwarm = nwarm, n_draws = length(keep),
                note = "from the periodic checkpoint; the run stalled at iteration ~542 of 600",
                mu = as.list(setNames(res$mu_mean, res$parameter)),
                mu_q05 = as.list(setNames(res$mu_q05, res$parameter)),
                mu_q95 = as.list(setNames(res$mu_q95, res$parameter)),
                tau = as.list(setNames(res$tau_mean, res$parameter)),
                tau_q05 = as.list(setNames(res$tau_q05, res$parameter)),
                tau_q95 = as.list(setNames(res$tau_q95, res$parameter))),
           "benchmarks/baerveldt/sienabayes_network2w.json", auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("wrote benchmarks/baerveldt/sienabayes_network2w.json\n")
