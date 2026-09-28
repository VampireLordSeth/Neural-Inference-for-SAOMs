# Does adding a truncated out-degree effect repair the Glasgow goodness-of-fit failure?
#
#   Rscript benchmarks/glasgow/fit_outtrunc.R
#
# docs/GOF_RESULTS.md found that the canonical effect set cannot reproduce Glasgow's
# out-degree distribution, its triad census or its geodesic distribution, under any of
# three estimators, and that truncating the *simulated* networks at the survey's cap of
# six does not repair it: the observed distribution is a hump peaking at three to five
# that declines into the ceiling, not an uncapped process seen through a truncating
# instrument. The conclusion there was that the remedy has to change the dynamics rather
# than the observation, and RSiena has the effect for it -- outTrunc(c), whose statistic
# is min(x_i+, c), so that ties up to c are worth having and further ones are not.
#
# This fits the same three-wave network model with outTrunc(c) added and exports the
# estimate, so the goodness of fit can be recomputed at that point and compared with the
# canonical one. It is the experiment docs/GOF_RESULTS.md names as the obvious next step.
suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
out <- "benchmarks/glasgow"
args <- commandArgs(trailingOnly = TRUE)
cap <- if (length(args)) as.integer(args[1]) else 5L
cat("outTrunc cap c =", cap, "
")

X <- lapply(1:3, function(w) as.matrix(read.csv(file.path(out, sprintf("glasgow_net%d.csv", w)), header = FALSE)))
cov <- read.csv(file.path(out, "glasgow_covariates.csv"))
n <- nrow(X[[1]])
cat("n =", n, " ties per wave:", sapply(X, sum), "\n")
cat("observed max out-degree per wave:", sapply(X, function(m) max(rowSums(m))), "\n")

net <- sienaDependent(array(unlist(X), dim = c(n, n, 3)))
v <- coCovar(cov$alc1_centred)
g <- coCovar(cov$sex)
dat <- sienaDataCreate(net, v, g)

eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, altX, egoX, interaction1 = "v", verbose = FALSE)
eff <- includeEffects(eff, sameX, interaction1 = "g", verbose = FALSE)
# The one addition. Note the cap has to sit BELOW the observed maximum out-degree: the
# survey allowed six and nobody exceeded it, so min(x_i+, 6) equals x_i+ for every actor
# and the statistic is identical to density. At c = 6 the fit does not converge (ratio
# 0.844) and density and outTrunc take large offsetting values. Default 5 here.
# It has to go through setEffect: includeEffects silently ignores `parameter` and leaves
# the effect at RSiena's default cap, which is not the cap this survey used.
eff <- setEffect(eff, outTrunc, parameter = cap, verbose = FALSE)
edf <- as.data.frame(eff)
stopifnot(edf[eff$include & edf$shortName == "outTrunc", "parm"] == cap)

alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, seed = 20260927, n3 = 3000)
t0 <- Sys.time()
ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE); runs <- 1
while (ans$tconv.max >= 0.25 && runs < 5) {
  ans <- siena07(alg, data = dat, effects = eff, prevAns = ans, batch = TRUE, silent = TRUE)
  runs <- runs + 1
}
el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

df <- as.data.frame(eff)[eff$include, ]
lab <- paste0(df$name, ":", df$shortName)
isr <- df$shortName == "Rate"
lab[isr] <- paste0(df$name[isr], ":rate_", ave(seq_along(lab), df$name, FUN = seq_along)[isr])
print(data.frame(parameter = lab, estimate = round(ans$theta, 3),
                 se = round(sqrt(diag(ans$covtheta)), 3)), row.names = FALSE)
cat(sprintf("max convergence ratio %.3f after %d run(s), %.0fs\n", ans$tconv.max, runs, el))

write_json(list(rsiena_version = as.character(packageVersion("RSiena")), n = n, waves = 3,
                cond = FALSE, runs = runs, seconds = el, tconv_max = ans$tconv.max,
                outTrunc_parameter = cap,
                estimate = as.list(setNames(ans$theta, lab)),
                se = as.list(setNames(sqrt(diag(ans$covtheta)), lab))),
           file.path(out, sprintf("rsiena_network3w_outtrunc%d.json", cap)),
           auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("wrote", file.path(out, sprintf("rsiena_network3w_outtrunc%d.json", cap)), "\n")
