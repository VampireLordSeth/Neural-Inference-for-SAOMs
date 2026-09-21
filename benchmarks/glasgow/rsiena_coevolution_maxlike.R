# RSiena maximum-likelihood estimate of the Glasgow network x alcohol co-evolution
# model (129 pupils, three waves) — the same model as prepare_and_fit.R part (b),
# fitted by likelihood (MCMC) instead of the method of moments. Written 2026-09-21
# to adjudicate the selection/influence/quad reads on which the amortized 10^7
# posterior differs from MoM by 1.8-2.0 sd (docs/M4_RESULTS.md, step 3 at 10^7);
# the s50 counterpart is benchmarks/rsiena_coevolution_maxlike.R.
#   Rscript benchmarks/glasgow/rsiena_coevolution_maxlike.R [n3] [mult] [init json] [nodes] [tag]
# Defaults 3000, 5, the MoM estimate (rsiena_coevolution.json), 1 process, "maxlike".
# Reads the CSVs prepare_and_fit.R wrote (alcohol already completed there).
suppressPackageStartupMessages({ library(RSiena); library(jsonlite) })
out <- "benchmarks/glasgow"; if (!dir.exists(out)) stop("run from the project root")
args <- commandArgs(trailingOnly = TRUE)
n3 <- if (length(args) >= 1) as.integer(args[1]) else 3000L
mult <- if (length(args) >= 2) as.numeric(args[2]) else 5
init <- if (length(args) >= 3) args[3] else file.path(out, "rsiena_coevolution.json")
nodes <- if (length(args) >= 4) as.integer(args[4]) else 1L
tag <- if (length(args) >= 5) args[5] else "maxlike"
X <- lapply(1:3, function(w) as.matrix(read.csv(file.path(out, sprintf("glasgow_net%d.csv", w)), header = FALSE)))
alc <- as.matrix(read.csv(file.path(out, "glasgow_alcohol.csv"), header = FALSE))
n <- nrow(alc); stopifnot(all(sapply(X, dim) == n))
net <- sienaDependent(array(unlist(X), dim = c(n, n, 3)))
alcB <- sienaDependent(alc, type = "behavior")
dat <- sienaDataCreate(net, alcB)
eff <- getEffects(dat)
eff <- includeEffects(eff, transTrip, cycle3, verbose = FALSE)
eff <- includeEffects(eff, egoX, altX, simX, interaction1 = "alcB", verbose = FALSE)
eff <- includeEffects(eff, name = "alcB", avAlt, interaction1 = "net", verbose = FALSE)
mom <- fromJSON(file.path(out, "rsiena_coevolution.json"))
eff$initialValue[eff$include] <- unlist(fromJSON(init)$estimate)
alg <- sienaAlgorithmCreate(projname = NULL, maxlike = TRUE, cond = FALSE, seed = 20240921,
                            n3 = n3, mult = mult)
cat(sprintf("Glasgow ML: n=%d n3=%d mult=%g nodes=%d init=%s  %s\n", n, n3, mult, nodes, init, format(Sys.time())))
t0 <- Sys.time()
run1 <- function(prev = NULL) siena07(alg, data = dat, effects = eff, prevAns = prev, batch = TRUE, silent = TRUE,
                                      useCluster = nodes > 1, nbrNodes = nodes, initC = nodes > 1)
ans <- run1(); runs <- 1
report <- function(ans, runs) {
  cat(sprintf("run %d: overall max convergence ratio %.3f, %.0fs elapsed, %s\n", runs, ans$tconv.max,
              as.numeric(difftime(Sys.time(), t0, units = "secs")), format(Sys.time())))
  print(data.frame(parameter = labels, ml = round(ans$theta, 3), se = round(sqrt(diag(ans$covtheta)), 3),
                   tconv = round(ans$tconv, 3)), row.names = FALSE)
}
df <- as.data.frame(eff)[eff$include, ]
labels <- paste0(df$name, ":", df$shortName)
isr <- df$shortName == "Rate"
labels[isr] <- paste0(df$name[isr], ":rate_", ave(seq_along(labels), df$name, FUN = seq_along)[isr])
report(ans, runs)
while (ans$tconv.max >= 0.25 && runs < 4) {
  ans <- run1(ans); runs <- runs + 1; report(ans, runs)
}
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
res <- data.frame(parameter = labels, ml = ans$theta, ml_se = sqrt(diag(ans$covtheta)),
                  mom = unlist(mom$estimate), mom_se = unlist(mom$se), tconv = ans$tconv)
print(res, row.names = FALSE)
cat(sprintf("overall max convergence ratio %.3f after %d run(s), %.0fs\n", ans$tconv.max, runs, elapsed))
write_json(list(rsiena_version = as.character(packageVersion("RSiena")), n = n, waves = 3, method = "maxlike",
                cond = FALSE, n3 = n3, mult = mult, init = init, nodes = nodes, runs = runs, seconds = elapsed,
                tconv_max = ans$tconv.max,
                estimate = as.list(setNames(ans$theta, labels)),
                se = as.list(setNames(sqrt(diag(ans$covtheta)), labels))),
           file.path(out, paste0("rsiena_coevolution_", tag, ".json")), auto_unbox = TRUE, pretty = TRUE, digits = NA)
