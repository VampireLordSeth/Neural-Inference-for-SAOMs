# RSiena estimate for the three-wave s50 panel (s501 -> s502 -> s503), unconditional,
# two period rates and shared evaluation effects. Also exports s503.
#   Rscript benchmarks/rsiena_estimate_3w.R

suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
out_dir <- "benchmarks"
if (!dir.exists(out_dir)) stop("run from the project root")

write.table(s503, file.path(out_dir, "s503.csv"), sep = ",", row.names = FALSE, col.names = FALSE)

net <- sienaDependent(array(c(s501, s502, s503), dim = c(50, 50, 3)))
alc <- coCovar(s50a[, 1]); smk <- coCovar(s50s[, 1])
dat <- sienaDataCreate(net, alc, smk)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, altX, egoX, interaction1 = "alc", verbose = FALSE)
eff <- includeEffects(eff, sameX, interaction1 = "smk", verbose = FALSE)

alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, seed = 20240911, n3 = 3000)
t0 <- Sys.time()
ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE)
runs <- 1
while (ans$tconv.max >= 0.25 && runs < 4) {
  ans <- siena07(alg, data = dat, effects = eff, prevAns = ans, batch = TRUE, silent = TRUE); runs <- runs + 1
}
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
inc <- as.data.frame(eff)[eff$include, ]
labels <- ifelse(inc$interaction1 == "", inc$shortName, paste0(inc$shortName, "(", inc$interaction1, ")"))
labels[inc$shortName == "Rate"] <- paste0("rate_", seq_len(sum(inc$shortName == "Rate")))
res <- data.frame(parameter = labels, estimate = ans$theta, se = sqrt(diag(ans$covtheta)), tconv = ans$tconv)
print(res, row.names = FALSE)
cat(sprintf("overall max convergence ratio %.3f after %d run(s), %.0fs\n", ans$tconv.max, runs, elapsed))
write_json(list(rsiena_version = as.character(packageVersion("RSiena")), waves = 3, cond = FALSE,
                runs = runs, seconds = elapsed, tconv_max = ans$tconv.max,
                estimate = as.list(setNames(ans$theta, labels)), se = as.list(setNames(sqrt(diag(ans$covtheta)), labels)),
                tconv = as.list(setNames(ans$tconv, labels))),
           file.path(out_dir, "rsiena_estimate_3w.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
