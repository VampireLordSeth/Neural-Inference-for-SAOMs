# Dutch Social Behavior study (Chris Baerveldt; Snijders & Baerveldt 2003): 19 school
# networks, two waves, emotional-support ties, delinquency at both waves. Export every
# school to CSV and fit RSiena per school: (a) the two-wave network model with the M2
# covariate layout (v = ln-delinquency at wave 1, centred; g = sex) and (b) the network x
# delinquency co-evolution model with the M3b effect set. Then the siena08 meta-analysis
# of (a), the replication target of Snijders & Baerveldt (2003).
#   Rscript benchmarks/baerveldt/prepare_and_fit.R <dir with the unzipped CB_data.zip> [n3] [schools, e.g. 1,3]
# Data: https://www.stats.ox.ac.uk/~snijders/siena/CB_data.zip (public, teaching/research).
suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
src <- if (length(args)) args[1] else "."
n3 <- if (length(args) >= 2) as.integer(args[2]) else 3000L
out <- "benchmarks/baerveldt"; if (!dir.exists(out)) stop("run from the project root")
schools <- c(1, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23)
if (length(args) >= 3) schools <- as.integer(strsplit(args[3], ",")[[1]])
rd <- function(f) as.matrix(read.table(file.path(src, f)))
lab_of <- function(eff) {
  df <- as.data.frame(eff)[eff$include, ]
  lab <- paste0(df$name, ":", df$shortName)
  isr <- df$shortName == "Rate"
  lab[isr] <- paste0(df$name[isr], ":rate_", ave(seq_along(lab), df$name, FUN = seq_along)[isr])
  lab
}
fit <- function(dat, eff, tag, s) {
  alg <- sienaAlgorithmCreate(projname = NULL, cond = FALSE, seed = 20240921 + s, n3 = n3)
  t0 <- Sys.time(); ans <- siena07(alg, data = dat, effects = eff, batch = TRUE, silent = TRUE); runs <- 1
  while (ans$tconv.max >= 0.25 && runs < 4) { ans <- siena07(alg, data = dat, effects = eff, prevAns = ans, batch = TRUE, silent = TRUE); runs <- runs + 1 }
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs")); lab <- lab_of(eff)
  cat(sprintf("[school %2d %-11s] n=%3d conv %.3f after %d run(s), %4.0fs: %s\n", s, tag, dim(dat$depvars[[1]])[1], ans$tconv.max, runs, el,
              paste(sprintf("%s=%.2f", sub(".*:", "", lab), ans$theta), collapse = " ")))
  write_json(list(rsiena_version = as.character(packageVersion("RSiena")), school = s, n = dim(dat$depvars[[1]])[1], waves = 2, cond = FALSE,
                  runs = runs, seconds = el, tconv_max = ans$tconv.max,
                  estimate = as.list(setNames(ans$theta, lab)), se = as.list(setNames(sqrt(diag(ans$covtheta)), lab))),
             file.path(out, sprintf("rsiena_%s_%02d.json", tag, s)), auto_unbox = TRUE, pretty = TRUE, digits = NA)
  ans
}
summary_rows <- list(); net_fits <- list()
for (s in schools) {
  X1 <- rd(sprintf("N34_%d.DAT", s)); X2 <- rd(sprintf("HN34_%d.DAT", s))
  stopifnot(!any(X1 == 9), !any(X2 == 9))           # no missing ties in these 19 schools
  cov <- rd(sprintf("CBE%d.DAT", s)); colnames(cov) <- c("sex", "delinq_ln", "importance")   # sex 1=girl 2=boy; ln(offences+1)
  beh <- rd(sprintf("cbc%d.dat", s))                 # delinquency, 2 waves, 0..4
  eth <- rd(sprintf("cbe%d.sim", s))                 # same-ethnicity dyadic covariate
  n <- nrow(X1); stopifnot(ncol(X1) == n, nrow(cov) == n, nrow(beh) == n)
  if (nrow(eth) != n) { cat("school", s, ": coethnic matrix is", nrow(eth), "x", ncol(eth), "for", n, "pupils - not exported
"); eth <- NULL }
  tag <- sprintf("school%02d", s)
  write.table(X1, file.path(out, paste0(tag, "_net1.csv")), sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(X2, file.path(out, paste0(tag, "_net2.csv")), sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(beh, file.path(out, paste0(tag, "_delinquency.csv")), sep = ",", row.names = FALSE, col.names = FALSE)
  if (!is.null(eth)) write.table(eth, file.path(out, paste0(tag, "_coethnic.csv")), sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(data.frame(sex = cov[, "sex"] - 1, delinq_ln_centred = cov[, "delinq_ln"] - mean(cov[, "delinq_ln"]),
                         importance = cov[, "importance"], delinq_w1 = beh[, 1]),
              file.path(out, paste0(tag, "_covariates.csv")), sep = ",", row.names = FALSE)
  summary_rows[[length(summary_rows) + 1]] <- data.frame(school = s, n = n, ties1 = sum(X1), ties2 = sum(X2), deg1 = sum(X1) / n, deg2 = sum(X2) / n,
                                                         changed = sum(X1 != X2), delinq_mean1 = mean(beh[, 1]), delinq_mean2 = mean(beh[, 2]),
                                                         boys = mean(cov[, "sex"] == 2))
  net <- sienaDependent(array(c(X1, X2), dim = c(n, n, 2)))
  # (a) network with covariates, M2 layout: v = ln-delinquency (egoX, altX), g = sex (sameX)
  v <- coCovar(cov[, "delinq_ln"]); g <- coCovar(cov[, "sex"])
  dat <- sienaDataCreate(net, v, g)
  eff <- getEffects(dat)
  eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
  eff <- includeEffects(eff, altX, egoX, interaction1 = "v", verbose = FALSE)
  eff <- includeEffects(eff, sameX, interaction1 = "g", verbose = FALSE)
  net_fits[[tag]] <- tryCatch(fit(dat, eff, "network2w", s), error = function(e) { cat("[school", s, "network2w] FAILED:", conditionMessage(e), "\n"); NULL })
  # (b) co-evolution with delinquency, M3b layout
  delB <- sienaDependent(beh, type = "behavior")
  dat2 <- sienaDataCreate(net, delB)
  eff2 <- getEffects(dat2)
  eff2 <- includeEffects(eff2, transTrip, cycle3, verbose = FALSE)
  eff2 <- includeEffects(eff2, egoX, altX, simX, interaction1 = "delB", verbose = FALSE)
  eff2 <- includeEffects(eff2, name = "delB", avAlt, interaction1 = "net", verbose = FALSE)
  tryCatch(fit(dat2, eff2, "coevolution", s), error = function(e) cat("[school", s, "coevolution] FAILED:", conditionMessage(e), "\n"))
  write_json(list(simMean = attr(dat2$depvars$delB, "simMean"), range = attr(dat2$depvars$delB, "range"), grand_mean = mean(beh)),
             file.path(out, paste0(tag, "_behaviour_constants.json")), auto_unbox = TRUE, pretty = TRUE, digits = NA)
}
desc <- do.call(rbind, summary_rows)
write.csv(desc, file.path(out, "schools.csv"), row.names = FALSE)
print(desc, row.names = FALSE)
# siena08 meta-analysis of the network fits (Snijders & Baerveldt 2003 style)
ok <- Filter(Negate(is.null), net_fits)
if (length(ok) >= 2) {
  meta <- siena08(ok, bound = 5)
  lab <- lab_of(ok[[1]]$effects); rows <- list()
  for (k in seq_along(lab)) {
    m <- meta[[k]]
    rows[[k]] <- data.frame(parameter = lab[k], mu_ml = m$mu.ml, mu_ml_se = m$mu.ml.se, sigma_ml = m$sigma.ml,
                            mu_lo = m$mu.confint[1], mu_hi = m$mu.confint[2], sigma_lo = m$sigma.confint[1], sigma_hi = m$sigma.confint[2],
                            p_sigma_zero = m$pttilde, fisher_p_plus = m$cjplusp, fisher_p_minus = m$cjminusp, n_groups = m$n1)
  }
  meta_df <- do.call(rbind, rows); print(meta_df, row.names = FALSE)
  write.csv(meta_df, file.path(out, "siena08_network2w.csv"), row.names = FALSE)
}
cat("done", format(Sys.time()), "\n")
