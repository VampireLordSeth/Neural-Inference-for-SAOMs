# Glasgow Teenage Friends and Lifestyle Study (full panel): export the 129-pupil
# three-wave friendship network, alcohol (3 waves) and sex to CSV, and fit RSiena's
# (a) three-wave network model with the M2/M3 effect set (v = alcohol wave 1 centred,
# g = sex) and (b) the network x alcohol co-evolution model with the M3b effect set.
#   Rscript benchmarks/glasgow/prepare_and_fit.R <dir with Glasgow-*.RData>
suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
src <- if (length(args)) args[1] else "."
out <- "benchmarks/glasgow"
for (f in c("friendship", "substances", "demographic", "selections")) load(file.path(src, paste0("Glasgow-", f, ".RData")))
keep <- selection129
dich <- function(m) { m <- m[keep, keep]; stopifnot(!any(m == 10)); (m == 1 | m == 2) * 1L }
X <- list(dich(friendship.1), dich(friendship.2), dich(friendship.3))
alc <- alcohol[keep, ]; sex <- sex.F[keep]
# 26 of the 129 pupils miss an alcohol value in some wave. saomsim has no missing-data
# handling, so both RSiena and saomsim get the same completed matrix: carry the nearest
# observed wave (forward, else backward). Documented in docs/M4_RESULTS.md.
n_na <- sum(is.na(alc)); med <- apply(alc, 2, median, na.rm = TRUE)
n_none <- sum(rowSums(is.na(alc)) == 3)
for (i in seq_len(nrow(alc))) for (w in 1:3) if (is.na(alc[i, w])) {
  obs <- which(!is.na(alc[i, ]))
  alc[i, w] <- if (length(obs)) alc[i, obs[which.min(abs(obs - w) + 0.1 * (obs > w))]] else round(med[w])
}
cat("alcohol values imputed:", n_na, "of", length(alc), "(", n_none, "pupils with no observation -> wave median", round(med), ")
")
stopifnot(!anyNA(alc), !anyNA(sex))
n <- sum(keep); cat("n =", n, " ties per wave:", sapply(X, sum), " alcohol range", range(alc), "\n")
for (w in 1:3) write.table(X[[w]], file.path(out, sprintf("glasgow_net%d.csv", w)), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(alc, file.path(out, "glasgow_alcohol.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(data.frame(sex = sex, alc1_centred = alc[, 1] - mean(alc[, 1])), file.path(out, "glasgow_covariates.csv"), sep = ",", row.names = FALSE)

fit <- function(dat, eff, tag) {
  alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, seed = 20240914, n3 = 3000)
  t0 <- Sys.time(); ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE); runs <- 1
  while (ans$tconv.max >= 0.25 && runs < 4) { ans <- siena07(alg, data = dat, effects = eff, prevAns = ans, batch = TRUE, silent = TRUE); runs <- runs + 1 }
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  df <- as.data.frame(eff)[eff$include, ]
  lab <- paste0(df$name, ":", df$shortName)
  isr <- df$shortName == "Rate"
  lab[isr] <- paste0(df$name[isr], ":rate_", ave(seq_along(lab), df$name, FUN = seq_along)[isr])
  print(data.frame(parameter = lab, estimate = round(ans$theta, 3), se = round(sqrt(diag(ans$covtheta)), 3)), row.names = FALSE)
  cat(sprintf("[%s] max convergence ratio %.3f after %d run(s), %.0fs\n", tag, ans$tconv.max, runs, el))
  write_json(list(rsiena_version = as.character(packageVersion("RSiena")), n = n, waves = 3, cond = FALSE, runs = runs, seconds = el,
                  tconv_max = ans$tconv.max, estimate = as.list(setNames(ans$theta, lab)), se = as.list(setNames(sqrt(diag(ans$covtheta)), lab))),
             file.path(out, paste0("rsiena_", tag, ".json")), auto_unbox = TRUE, pretty = TRUE, digits = NA)
}
net <- sienaDependent(array(unlist(X), dim = c(n, n, 3)))
# (a) network with covariates, M3a layout
v <- coCovar(alc[, 1]); g <- coCovar(sex)
dat <- sienaDataCreate(net, v, g)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, altX, egoX, interaction1 = "v", verbose = FALSE)
eff <- includeEffects(eff, sameX, interaction1 = "g", verbose = FALSE)
fit(dat, eff, "network3w")
# (b) co-evolution with alcohol, M3b layout
alcB <- sienaDependent(alc, type = "behavior")
dat2 <- sienaDataCreate(net, alcB)
eff2 <- getEffects(dat2)
eff2 <- includeEffects(eff2, transTrip, cycle3, verbose = FALSE)
eff2 <- includeEffects(eff2, egoX, altX, simX, interaction1 = "alcB", verbose = FALSE)
eff2 <- includeEffects(eff2, name = "alcB", avAlt, interaction1 = "net", verbose = FALSE)
fit(dat2, eff2, "coevolution")
write_json(list(simMean = attr(dat2$depvars$alcB, "simMean"), range = attr(dat2$depvars$alcB, "range"), grand_mean = mean(alc)),
           file.path(out, "glasgow_behaviour_constants.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
