# RSiena method-of-moments estimate for s501 -> s502 with the benchmark model,
# unconditional (cond = FALSE) so the rate is estimated too. Written to
# rsiena_estimate.json for comparison with saomsim's estimators and the
# amortized posterior (benchmarks/npe_s50.py).
#
#   Rscript benchmarks/rsiena_estimate.R
#
# RSiena's convergence rule: all |t-conv| < 0.1 and overall max. convergence
# ratio < 0.25. The script re-runs from the previous answer up to three times.

suppressPackageStartupMessages({
  library(RSiena)
  library(jsonlite)
})

out_dir <- "benchmarks"
if (!dir.exists(out_dir)) stop("run from the project root")

net <- sienaDependent(array(c(s501, s502), dim = c(50, 50, 2)))
alc <- coCovar(s50a[, 1])
smk <- coCovar(s50s[, 1])
dat <- sienaDataCreate(net, alc, smk)

eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, altX, egoX, interaction1 = "alc", verbose = FALSE)
eff <- includeEffects(eff, sameX, interaction1 = "smk", verbose = FALSE)

alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, seed = 20240911, n3 = 3000)
t0 <- Sys.time()
ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE, returnDeps = FALSE)
runs <- 1
while (ans$tconv.max >= 0.25 && runs < 4) {
  ans <- siena07(alg, data = dat, effects = eff, prevAns = ans, batch = TRUE, silent = TRUE)
  runs <- runs + 1
}
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

inc <- as.data.frame(eff)[eff$include, ]
labels <- ifelse(inc$interaction1 == "", inc$shortName,
                 paste0(inc$shortName, "(", inc$interaction1, ")"))
labels[labels == "Rate"] <- "rate"

res <- data.frame(parameter = labels, estimate = ans$theta, se = sqrt(diag(ans$covtheta)),
                  tconv = ans$tconv)
print(res, row.names = FALSE)
cat(sprintf("overall max convergence ratio %.3f after %d run(s), %.0fs\n", ans$tconv.max, runs, elapsed))

write_json(
  list(rsiena_version = as.character(packageVersion("RSiena")),
       cond = FALSE, runs = runs, seconds = elapsed,
       tconv_max = ans$tconv.max,
       estimate = as.list(setNames(ans$theta, labels)),
       se = as.list(setNames(sqrt(diag(ans$covtheta)), labels)),
       tconv = as.list(setNames(ans$tconv, labels))),
  file.path(out_dir, "rsiena_estimate.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA
)
