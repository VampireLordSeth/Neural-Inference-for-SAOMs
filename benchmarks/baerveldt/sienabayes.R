# sienaBayes on the 19 Baerveldt schools: the established multilevel Bayesian SAOM
# (Koskinen & Snijders 2007, 2023) fitted to the same 19 groups our amortized estimator
# reads, so the population means and between-school spreads can be compared directly with
# benchmarks/population_stage.py (docs/M5_RESULTS.md). Written 2026-09-24.
#   Rscript benchmarks/baerveldt/sienabayes.R [nwarm] [nmain] [nrunMHBatches] [schools]
# Defaults 50, 250, 20, all 19. sienaBayes is in RSienaTest (R-Forge source); built here
# with Rtools45 and one patch — src/siena07setup.cpp calls set_terminate unqualified,
# which libstdc++ 14 no longer provides transitively, so #include <exception> and std::
# were added.
#
# Model: the network-only layout of prepare_and_fit.R part (a) — density, recip, transTrip,
# cycle3, altX/egoX of ln-delinquency, sameX of sex — with rates and evaluation effects
# random across schools. That is the model siena08 meta-analyses and the one our
# network estimator fits, so all three routes to (mu, tau) are on the same footing.
suppressPackageStartupMessages({ library(RSienaTest); library(jsonlite) })
out <- "benchmarks/baerveldt"; if (!dir.exists(out)) stop("run from the project root")
args <- commandArgs(trailingOnly = TRUE)
nwarm <- if (length(args) >= 1) as.integer(args[1]) else 50L
nmain <- if (length(args) >= 2) as.integer(args[2]) else 250L
nbatch <- if (length(args) >= 3) as.integer(args[3]) else 20L
schools <- c(1, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23)
if (length(args) >= 4) schools <- as.integer(strsplit(args[4], ",")[[1]])

mk <- function(s) {
  tag <- sprintf("school%02d", s)
  X1 <- as.matrix(read.csv(file.path(out, paste0(tag, "_net1.csv")), header = FALSE))
  X2 <- as.matrix(read.csv(file.path(out, paste0(tag, "_net2.csv")), header = FALSE))
  cov <- read.csv(file.path(out, paste0(tag, "_covariates.csv")))
  n <- nrow(X1)
  net <- sienaDependent(array(c(X1, X2), dim = c(n, n, 2)))
  sienaDataCreate(net = net, v = coCovar(cov$delinq_ln_centred), g = coCovar(cov$sex))
}
dats <- lapply(schools, mk)
names(dats) <- sprintf("school%02d", schools)
cat("groups:", length(dats), " sizes:", sapply(dats, function(d) dim(d$depvars[[1]])[1]), "\n")

grp <- sienaGroupCreate(dats)
eff <- getEffects(grp)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, altX, egoX, interaction1 = "v", verbose = FALSE)
eff <- includeEffects(eff, sameX, interaction1 = "g", verbose = FALSE)
# By default only the rates and density vary across groups and the rest are held constant
# ("eta"). We want mu and tau for every mechanism, which is what our population stage
# reports, so every included effect is made a random effect.
eff$randomEffects[eff$include & eff$shortName != "Rate"] <- TRUE

alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, seed = 20240924, maxlike = TRUE)

cat(sprintf("sienaBayes: %d groups, nwarm=%d nmain=%d nrunMHBatches=%d  %s\n",
            length(dats), nwarm, nmain, nbatch, format(Sys.time())))
t0 <- Sys.time()
bay <- sienaBayes(grp, effects = eff, algo = alg, nwarm = nwarm, nmain = nmain,
                  nrunMHBatches = nbatch, silentstart = TRUE)
el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("sienaBayes done in %.0f s (%.2f h)\n", el, el / 3600))

# population mean (mu) and between-group sd (tau) draws
mu <- bay$ThinPosteriorMu
sig <- bay$ThinPosteriorSigma
lab <- bay$requestedEffects
nm <- paste0(lab$name, ":", lab$shortName)
# bay$effectName has one entry per distinct effect (the per-group rate rows collapsed),
# in the order of the varying ("mu") block; eta holds any effect left non-random.
nmv <- bay$effectName
if (length(nmv) != ncol(mu)) nmv <- nm[as.logical(lab$randomEffects)][seq_len(ncol(mu))]
tau <- t(apply(sig, 1, function(S) sqrt(diag(matrix(S, nrow = ncol(mu))))))
res <- data.frame(parameter = nmv,
                  mu_mean = colMeans(mu), mu_q05 = apply(mu, 2, quantile, 0.05),
                  mu_q95 = apply(mu, 2, quantile, 0.95),
                  tau_mean = colMeans(tau), tau_q05 = apply(tau, 2, quantile, 0.05),
                  tau_q95 = apply(tau, 2, quantile, 0.95))
print(res, row.names = FALSE, digits = 3)
write_json(list(rsienatest_version = as.character(packageVersion("RSienaTest")),
                groups = length(dats), nwarm = nwarm, nmain = nmain,
                nrunMHBatches = nbatch, seconds = el, n_draws = nrow(mu),
                mu = as.list(setNames(res$mu_mean, res$parameter)),
                mu_q05 = as.list(setNames(res$mu_q05, res$parameter)),
                mu_q95 = as.list(setNames(res$mu_q95, res$parameter)),
                tau = as.list(setNames(res$tau_mean, res$parameter)),
                tau_q05 = as.list(setNames(res$tau_q05, res$parameter)),
                tau_q95 = as.list(setNames(res$tau_q95, res$parameter))),
           file.path(out, "sienabayes_network2w.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
saveRDS(bay, file.path(out, "sienabayes_network2w.rds"))
cat("wrote benchmarks/baerveldt/sienabayes_network2w.{json,rds}\n")
