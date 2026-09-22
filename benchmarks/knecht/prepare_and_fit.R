# Knecht (2008) classroom 12b, as distributed on the Siena site (klas12b.zip): 26 pupils,
# four waves of friendship, delinquency at all four waves, alcohol at waves 2-4, sex,
# primary-school friendship. Export completed matrices to CSV and fit RSiena with our
# effect sets on the wave windows our estimators cover: three-wave network+covariates
# (M4 layout, v = delinquency wave 1 centred, g = sex) and three-wave network x
# delinquency co-evolution (M3b layout), on waves 1-3 and 2-4, plus the full four waves.
#   Rscript benchmarks/knecht/prepare_and_fit.R <dir with the unzipped klas12b.zip> [n3]
# Completion policy (saomsim has no missing-data or composition-change handling; RSiena
# gets the same completed data so the comparison is like for like): the one pupil who
# leaves after wave 2 (structural 10s in waves 3-4) is dropped from every wave (n = 25);
# the incidentally missing nomination rows (code 9: one in wave 2, two in wave 3) are
# carried from the nearest observed wave, forward first; the same rule fills the 7 missing
# delinquency values. Documented in docs.
suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
src <- if (length(args)) args[1] else "."
n3 <- if (length(args) >= 2) as.integer(args[2]) else 3000L
out <- "benchmarks/knecht"; if (!dir.exists(out)) stop("run from the project root")
rd <- function(f) as.matrix(read.table(file.path(src, f)))
W <- lapply(1:4, function(w) rd(sprintf("klas12b-net-%d.dat", w)))
deli <- rd("klas12b-delinquency.dat"); alc <- rd("klas12b-alcohol.dat"); demo <- rd("klas12b-demographics.dat"); prim <- rd("klas12b-primary.dat")
n0 <- nrow(W[[1]])
leaver <- which(sapply(seq_len(n0), function(i) any(sapply(W, function(m) all(m[i, -i] == 10)))))
cat("pupils:", n0, " structurally missing in some wave (dropped):", leaver, "\n")
keep <- setdiff(seq_len(n0), leaver); n <- length(keep)
W <- lapply(W, function(m) m[keep, keep]); deli <- deli[keep, ]; alc <- alc[keep, ]; demo <- demo[keep, ]; prim <- prim[keep, keep]
stopifnot(!any(sapply(W, function(m) any(m == 10))))
miss <- sapply(W, function(m) which(apply(m, 1, function(r) any(r == 9))))
cat("incidentally missing rows per wave:", sapply(miss, length), "\n")
for (w in 1:4) for (i in miss[[w]]) {
  obs <- setdiff(which(sapply(1:4, function(v) !any(W[[v]][i, ] == 9))), w)
  src_w <- obs[which.min(abs(obs - w) + 0.1 * (obs > w))]
  W[[w]][i, ] <- W[[src_w]][i, ]
}
stopifnot(!any(sapply(W, function(m) any(m == 9))))
# delinquency: 0 = missing; carry the nearest observed wave (forward first), as for Glasgow alcohol
n_na <- sum(deli == 0)
for (i in seq_len(n)) for (w in 1:4) if (deli[i, w] == 0) {
  obs <- which(deli[i, ] != 0); stopifnot(length(obs) > 0)
  deli[i, w] <- deli[i, obs[which.min(abs(obs - w) + 0.1 * (obs > w))]]
}
cat("delinquency values imputed:", n_na, "of", length(deli), "
")
stopifnot(!any(deli == 0))
W <- lapply(W, function(m) { diag(m) <- 0; m })   # one self-nomination in wave 2; RSiena ignores the diagonal too
cat("n =", n, " ties per wave:", sapply(W, sum), " delinquency range", range(deli), " alcohol missing:", sum(alc == 0), "\n")
for (w in 1:4) write.table(W[[w]], file.path(out, sprintf("klas12b_net%d.csv", w)), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(deli, file.path(out, "klas12b_delinquency.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(alc, file.path(out, "klas12b_alcohol_w2to4.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(prim, file.path(out, "klas12b_primary.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
write.table(data.frame(sex = demo[, 1] - 1, age = demo[, 2], ethnicity_minority = demo[, 3] - 1, delinq_w1_centred = deli[, 1] - mean(deli[, 1])),
            file.path(out, "klas12b_covariates.csv"), sep = ",", row.names = FALSE)

lab_of <- function(eff) {
  df <- as.data.frame(eff)[eff$include, ]; lab <- paste0(df$name, ":", df$shortName); isr <- df$shortName == "Rate"
  lab[isr] <- paste0(df$name[isr], ":rate_", ave(seq_along(lab), df$name, FUN = seq_along)[isr]); lab
}
fit <- function(dat, eff, tag) {
  alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, seed = 20240921, n3 = n3)
  t0 <- Sys.time(); ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE); runs <- 1
  while (ans$tconv.max >= 0.25 && runs < 4) { ans <- siena07(alg, data = dat, effects = eff, prevAns = ans, batch = TRUE, silent = TRUE); runs <- runs + 1 }
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs")); lab <- lab_of(eff)
  cat(sprintf("[%-22s] conv %.3f after %d run(s), %4.0fs\n", tag, ans$tconv.max, runs, el))
  print(data.frame(parameter = lab, estimate = round(ans$theta, 3), se = round(sqrt(diag(ans$covtheta)), 3)), row.names = FALSE)
  write_json(list(rsiena_version = as.character(packageVersion("RSiena")), n = n, waves = dim(dat$depvars[[1]])[3], cond = FALSE, runs = runs,
                  seconds = el, tconv_max = ans$tconv.max, estimate = as.list(setNames(ans$theta, lab)), se = as.list(setNames(sqrt(diag(ans$covtheta)), lab))),
             file.path(out, paste0("rsiena_", tag, ".json")), auto_unbox = TRUE, pretty = TRUE, digits = NA)
}
for (win in list(1:3, 2:4, 1:4)) {
  tag <- paste0("w", win[1], "to", win[length(win)])
  net <- sienaDependent(array(unlist(W[win]), dim = c(n, n, length(win))))
  # (a) network with covariates, M4 layout
  v <- coCovar(deli[, win[1]]); g <- coCovar(demo[, 1])
  dat <- sienaDataCreate(net, v, g)
  eff <- getEffects(dat)
  eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
  eff <- includeEffects(eff, altX, egoX, interaction1 = "v", verbose = FALSE)
  eff <- includeEffects(eff, sameX, interaction1 = "g", verbose = FALSE)
  tryCatch(fit(dat, eff, paste0("network_", tag)), error = function(e) cat("[network", tag, "] FAILED:", conditionMessage(e), "\n"))
  # (b) co-evolution with delinquency, M3b layout
  delB <- sienaDependent(deli[, win], type = "behavior")
  dat2 <- sienaDataCreate(net, delB)
  eff2 <- getEffects(dat2)
  eff2 <- includeEffects(eff2, transTrip, cycle3, verbose = FALSE)
  eff2 <- includeEffects(eff2, egoX, altX, simX, interaction1 = "delB", verbose = FALSE)
  eff2 <- includeEffects(eff2, name = "delB", avAlt, interaction1 = "net", verbose = FALSE)
  tryCatch(fit(dat2, eff2, paste0("coevolution_", tag)), error = function(e) cat("[coevolution", tag, "] FAILED:", conditionMessage(e), "\n"))
  write_json(list(simMean = attr(dat2$depvars$delB, "simMean"), range = attr(dat2$depvars$delB, "range"), grand_mean = mean(deli[, win])),
             file.path(out, paste0("klas12b_behaviour_constants_", tag, ".json")), auto_unbox = TRUE, pretty = TRUE, digits = NA)
}
cat("done", format(Sys.time()), "\n")
