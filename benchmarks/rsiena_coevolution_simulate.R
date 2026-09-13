# M3b dynamics gate: RSiena's joint network-behaviour simulator at fixed theta.
# s501 + alcohol wave 1 as the start (two-wave data object so the period is s501 -> s502),
# unconditional, simOnly; writes every simulation's statistics to rsiena_coevolution_sims.csv.
#   Rscript benchmarks/rsiena_coevolution_simulate.R

suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
out_dir <- "benchmarks"
if (!dir.exists(out_dir)) stop("run from the project root")
n3 <- 2000; seed <- 20240913

net <- sienaDependent(array(c(s501, s502), dim = c(50, 50, 2)))
alc <- sienaDependent(s50a[, 1:2], type = "behavior")
dat <- sienaDataCreate(net, alc)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, simX, interaction1 = "alc", verbose = FALSE)
eff <- includeEffects(eff, name = "alc", avAlt, interaction1 = "net", verbose = FALSE)
df <- as.data.frame(eff)[eff$include, ]
labels <- paste0(df$name, ":", df$shortName)
# order: net rate, density, recip, transTrip, cycle3, altX, egoX, simX, alc rate, linear, quad, avAlt
theta <- c(5.0, -2.3, 2.2, 0.45, -0.25, 0.10, 0.05, 1.0, 1.5, 0.2, -0.3, 1.5)
stopifnot(length(theta) == nrow(df))
eff$initialValue[eff$include] <- theta

alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, nsub = 0, n3 = n3, simOnly = TRUE, seed = seed)
ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE)
sims <- ans$sf2[, 1, ]
colnames(sims) <- labels
write.csv(sims, file.path(out_dir, "rsiena_coevolution_sims.csv"), row.names = FALSE)
write_json(list(rsiena_version = as.character(packageVersion("RSiena")), n3 = n3, seed = seed,
                labels = labels, theta = as.list(setNames(theta, labels)),
                constants = list(grand_mean = mean(s50a[, 1:2]), simMean = attr(dat$depvars$alc, "simMean"),
                                 range = attr(dat$depvars$alc, "range")),
                mean = as.list(setNames(colMeans(sims), labels)), sd = as.list(setNames(apply(sims, 2, sd), labels))),
           file.path(out_dir, "rsiena_coevolution_sims.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
print(round(rbind(mean = colMeans(sims), sd = apply(sims, 2, sd)), 3))
