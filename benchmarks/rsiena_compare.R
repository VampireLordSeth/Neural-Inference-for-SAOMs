# RSiena parity benchmark: target statistics on the s50 data.
#
# Exports the two network waves and the actor covariates shipped with RSiena to
# CSV, builds the Siena data object with the effects saomsim implements, and
# writes RSiena's target statistics to rsiena_targets.json. The Python side
# (test_rsiena_parity.py) loads the same CSVs and compares saomsim.statistics().
#
# Run from the project root:
#   Rscript benchmarks/rsiena_compare.R
#
# Conventions recorded here so the comparison is interpretable:
#   * RSiena's target for an evaluation effect is the actor-summed statistic
#     s_k(x) of the *second* wave; the rate target is the number of tie changes
#     between waves.
#   * coCovar() centres covariates by default (subtracts the actor mean). The
#     centred values are exported so saomsim sees exactly what RSiena used.
#     sameX does not depend on centring.
#   * getTargets() is not exported from RSiena; it is what siena07 uses
#     internally to fill fit$targets. The script falls back to siena07 with
#     nsub = 0 if the internal function ever disappears.

suppressPackageStartupMessages({
  library(RSiena)
  library(jsonlite)
})

out_dir <- "benchmarks"
if (!dir.exists(out_dir)) stop("run from the project root")

# ---- data -------------------------------------------------------------------
x1 <- s501
x2 <- s502
alc_raw <- s50a[, 1]   # alcohol use, wave 1 (1..5)
smk_raw <- s50s[, 1]   # smoking, wave 1 (1..3)

net <- sienaDependent(array(c(x1, x2), dim = c(50, 50, 2)))
alc <- coCovar(alc_raw)            # centred = TRUE (default)
smk <- coCovar(smk_raw)
dat <- sienaDataCreate(net, alc, smk)

# the centred vector RSiena actually uses: raw minus the mean it stored
alc_centred <- alc_raw - mean(alc_raw)
smk_centred <- smk_raw - mean(smk_raw)

# ---- effects ----------------------------------------------------------------
eff <- getEffects(dat)                       # rate, density, recip by default
eff <- includeEffects(eff, transTrip, cycle3)
eff <- includeEffects(eff, altX, egoX, interaction1 = "alc")
eff <- includeEffects(eff, sameX, interaction1 = "smk")

inc <- as.data.frame(eff)[eff$include, ]
labels <- ifelse(inc$interaction1 == "", inc$shortName,
                 paste0(inc$shortName, "(", inc$interaction1, ")"))

# ---- targets ----------------------------------------------------------------
targets <- tryCatch(
  as.vector(RSiena:::getTargets(dat, eff)),
  error = function(e) {
    message("getTargets unavailable (", conditionMessage(e), "); using siena07")
    alg <- sienaAlgorithmCreate(projname = NULL, nsub = 0, n3 = 10, seed = 1)
    fit <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE)
    as.vector(fit$targets)
  }
)
names(targets) <- labels

# ---- write ------------------------------------------------------------------
write.table(x1, file.path(out_dir, "s501.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(x2, file.path(out_dir, "s502.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(
  data.frame(alc_raw = alc_raw, alc_centred = alc_centred,
             smk_raw = smk_raw, smk_centred = smk_centred),
  file.path(out_dir, "s50_covariates.csv"), sep = ",", row.names = FALSE
)

result <- list(
  rsiena_version = as.character(packageVersion("RSiena")),
  r_version = R.version.string,
  data = "s501 -> s502 (n = 50), covariates s50a[,1] (alc) and s50s[,1] (smk), centred",
  hamming = sum(x1 != x2),
  targets = as.list(targets),
  notes = c(
    "targets are RSiena's actor-summed statistics of wave 2; rate target = tie changes",
    "altX/egoX use the centred covariate; sameX is centring-invariant"
  )
)
write_json(result, file.path(out_dir, "rsiena_targets.json"), auto_unbox = TRUE,
           pretty = TRUE, digits = NA)

cat("RSiena", result$rsiena_version, "targets:\n")
print(targets)
