# RSiena dynamics benchmark: simulated statistics at fixed parameters.
#
# Matching target statistics (rsiena_compare.R) validates the effect
# definitions. This script validates the *simulator*: it runs RSiena's own
# simulation from s501 at a fixed parameter vector and writes the statistics of
# every simulated wave-2 network to rsiena_sims.csv, with the parameters in
# rsiena_sims.json. The Python side simulates from the same s501 with the same
# parameters and compares the two distributions.
#
# Run from the project root:
#   Rscript benchmarks/rsiena_simulate.R
#
# Set-up notes:
#   * cond = FALSE: unconditional simulation, so the number of ministeps is
#     Poisson(n * rate) as in saomsim (conditional simulation would instead run
#     until the observed number of changes is reached).
#   * simOnly = TRUE, nsub = 0: no estimation; phase 3 only, n3 simulations at
#     the initial values.
#   * ans$sf2[run, period, effect] holds the raw simulated statistics (the same
#     conventions as the targets, cycle3 counted once per cycle).

suppressPackageStartupMessages({
  library(RSiena)
  library(jsonlite)
})

out_dir <- "benchmarks"
if (!dir.exists(out_dir)) stop("run from the project root")

n3 <- 2000
seed <- 20240911

net <- sienaDependent(array(c(s501, s502), dim = c(50, 50, 2)))
alc <- coCovar(s50a[, 1])
smk <- coCovar(s50s[, 1])
dat <- sienaDataCreate(net, alc, smk)

eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, altX, egoX, interaction1 = "alc", verbose = FALSE)
eff <- includeEffects(eff, sameX, interaction1 = "smk", verbose = FALSE)

inc <- as.data.frame(eff)[eff$include, ]
labels <- ifelse(inc$interaction1 == "", inc$shortName,
                 paste0(inc$shortName, "(", inc$interaction1, ")"))
# order: Rate, density, recip, transTrip, cycle3, altX(alc), egoX(alc), sameX(smk)
theta <- c(5.0, -2.3, 2.2, 0.45, -0.25, 0.10, 0.05, 0.40)
stopifnot(length(theta) == nrow(inc))
eff$initialValue[eff$include] <- theta

alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, nsub = 0, n3 = n3,
                            simOnly = TRUE, seed = seed)
ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE)

sims <- ans$sf2[, 1, ]
colnames(sims) <- labels
write.csv(sims, file.path(out_dir, "rsiena_sims.csv"), row.names = FALSE)

write_json(
  list(
    rsiena_version = as.character(packageVersion("RSiena")),
    n3 = n3, seed = seed,
    start = "s501", n = 50,
    labels = labels,
    theta = as.list(setNames(theta, labels)),
    mean = as.list(setNames(colMeans(sims), labels)),
    sd = as.list(setNames(apply(sims, 2, sd), labels)),
    covariates = "alc = s50a[,1] centred; smk = s50s[,1]"
  ),
  file.path(out_dir, "rsiena_sims.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA
)

cat("RSiena", as.character(packageVersion("RSiena")), "-", n3, "simulations\n")
print(round(rbind(mean = colMeans(sims), sd = apply(sims, 2, sd)), 3))
