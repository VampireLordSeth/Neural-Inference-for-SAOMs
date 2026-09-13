# RSiena co-evolution gate (M3b): s50 network x alcohol, three waves.
# Exports the behaviour waves, RSiena's centring constants, and the per-period
# target statistics for network effects (incl. selection on alc) and behaviour
# effects (linear, quad, avAlt, avSim). Python (benchmarks/test_rsiena_coevolution.py)
# reproduces every number and thereby pins the conventions.
#   Rscript benchmarks/rsiena_coevolution.R

suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
out_dir <- "benchmarks"
if (!dir.exists(out_dir)) stop("run from the project root")
write.table(s50a, file.path(out_dir, "s50a.csv"), sep = ",", row.names = FALSE, col.names = FALSE)

net <- sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3)))
alc <- sienaDependent(s50a, type = "behavior")
dat <- sienaDataCreate(net, alc)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, simX, interaction1 = "alc", verbose = FALSE)
eff <- includeEffects(eff, name = "alc", avAlt, avSim, interaction1 = "net", verbose = FALSE)
df <- as.data.frame(eff)[eff$include, ]
labels <- paste0(df$name, ":", df$shortName)
tg <- RSiena:::getTargets(dat, eff)   # (effects x periods)
per_period <- lapply(seq_len(ncol(tg)), function(m) as.list(setNames(tg[, m], labels)))

write_json(list(
  rsiena_version = as.character(packageVersion("RSiena")),
  behaviour = "alc = s50a, waves 1..3, values 1..5",
  constants = list(simMean = attr(dat$depvars$alc, "simMean"), range = attr(dat$depvars$alc, "range"),
                   grand_mean = mean(s50a)),
  labels = labels,
  targets_by_period = per_period
), file.path(out_dir, "rsiena_coevolution_targets.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
print(cbind(labels, round(tg, 4)))
