# RSiena estimate of the s50 network x alcohol co-evolution model, three waves,
# unconditional. Benchmark for the M3b estimator.   Rscript benchmarks/rsiena_coevolution_estimate.R
suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
out_dir <- "benchmarks"; if (!dir.exists(out_dir)) stop("run from the project root")
net <- sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3)))
alc <- sienaDependent(s50a, type = "behavior")
dat <- sienaDataCreate(net, alc)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, simX, interaction1 = "alc", verbose = FALSE)
eff <- includeEffects(eff, name = "alc", avAlt, interaction1 = "net", verbose = FALSE)
alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, seed = 20240913, n3 = 3000)
t0 <- Sys.time()
ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE); runs <- 1
while (ans$tconv.max >= 0.25 && runs < 4) { ans <- siena07(alg, data = dat, effects = eff, prevAns = ans, batch = TRUE, silent = TRUE); runs <- runs + 1 }
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
df <- as.data.frame(eff)[eff$include, ]
labels <- paste0(df$name, ":", df$shortName)
labels[df$shortName == "Rate"] <- paste0(df$name[df$shortName == "Rate"], ":rate_", ave(seq_along(labels), df$name, FUN = seq_along)[df$shortName == "Rate"])
res <- data.frame(parameter = labels, estimate = ans$theta, se = sqrt(diag(ans$covtheta)), tconv = ans$tconv)
print(res, row.names = FALSE)
cat(sprintf("overall max convergence ratio %.3f after %d run(s), %.0fs\n", ans$tconv.max, runs, elapsed))
write_json(list(rsiena_version = as.character(packageVersion("RSiena")), waves = 3, cond = FALSE, runs = runs, seconds = elapsed,
                tconv_max = ans$tconv.max, estimate = as.list(setNames(ans$theta, labels)), se = as.list(setNames(sqrt(diag(ans$covtheta)), labels))),
           file.path(out_dir, "rsiena_coevolution_estimate.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
