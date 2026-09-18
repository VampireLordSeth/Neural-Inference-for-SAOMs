# RSiena maximum-likelihood estimate of the s50 network x alcohol co-evolution model,
# three waves — the same model as rsiena_coevolution_estimate.R, fitted by
# likelihood (MCMC) instead of the method of moments. Written to adjudicate the stable
# 1.5-1.7 sd difference between the amortized posterior and the MoM point on the
# influence (avAlt) / quad pair (docs/M3_RESULTS.md, 10^7 clean section).
#   Rscript benchmarks/rsiena_coevolution_maxlike.R [n3] [mult] [init json]
# Defaults 3000, 5 and the MoM estimate. A first pass (3000, 5) reached per-parameter
# t-ratios < 0.25 but an overall ratio of 0.42 after four runs (2.4 h); the reported fit
# restarts from that pass with longer chains: 5000 10 benchmarks/rsiena_coevolution_maxlike.json.
suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
out_dir <- "benchmarks"; if (!dir.exists(out_dir)) stop("run from the project root")
net <- sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3)))
alc <- sienaDependent(s50a, type = "behavior")
dat <- sienaDataCreate(net, alc)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, simX, interaction1 = "alc", verbose = FALSE)
eff <- includeEffects(eff, name = "alc", avAlt, interaction1 = "net", verbose = FALSE)
# start from the MoM estimate so the ML phases begin near the optimum
args <- commandArgs(trailingOnly = TRUE)
n3 <- if (length(args) >= 1) as.integer(args[1]) else 3000L
mult <- if (length(args) >= 2) as.numeric(args[2]) else 5
init <- if (length(args) >= 3) args[3] else file.path(out_dir, "rsiena_coevolution_estimate.json")
mom <- fromJSON(file.path(out_dir, "rsiena_coevolution_estimate.json"))
eff$initialValue[eff$include] <- unlist(fromJSON(init)$estimate)
alg <- sienaAlgorithmCreate(projname = NULL, maxlike = TRUE, cond = FALSE, seed = 20240917,
                            n3 = n3, mult = mult)
t0 <- Sys.time()
ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE); runs <- 1
while (ans$tconv.max >= 0.25 && runs < 4) {
  ans <- siena07(alg, data = dat, effects = eff, prevAns = ans, batch = TRUE, silent = TRUE)
  runs <- runs + 1
}
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
df <- as.data.frame(eff)[eff$include, ]
labels <- paste0(df$name, ":", df$shortName)
isr <- df$shortName == "Rate"
labels[isr] <- paste0(df$name[isr], ":rate_", ave(seq_along(labels), df$name, FUN = seq_along)[isr])
res <- data.frame(parameter = labels, ml = ans$theta, ml_se = sqrt(diag(ans$covtheta)),
                  mom = unlist(mom$estimate), mom_se = unlist(mom$se), tconv = ans$tconv)
print(res, row.names = FALSE)
cat(sprintf("overall max convergence ratio %.3f after %d run(s), %.0fs\n", ans$tconv.max, runs, elapsed))
write_json(list(rsiena_version = as.character(packageVersion("RSiena")), waves = 3, method = "maxlike",
                cond = FALSE, n3 = n3, mult = mult, init = init, runs = runs, seconds = elapsed, tconv_max = ans$tconv.max,
                estimate = as.list(setNames(ans$theta, labels)),
                se = as.list(setNames(sqrt(diag(ans$covtheta)), labels))),
           file.path(out_dir, "rsiena_coevolution_maxlike.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
