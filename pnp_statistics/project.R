library(MASS)


##############################################################################
# 1. Testing Goodness-of-Fit
##############################################################################

# task parameters
params <- list(
  N = 9,
  S = 9,
  I1 = 5,
  I2 = 8
)

# uniform distribution
theta0 <- c(-params$N, params$S + 4)

# determine mu0 and v0^2 for G0
a0  <- theta0[1]
b0  <- theta0[2]
mu0 <- (a0 + b0) / 2  # mean
v0sq <- (b0 - a0)^2 / 12  # variance
v0   <- sqrt(v0sq)        # standard deviation

cat("G0 = Unif(", a0, ",", b0, ")\n")
cat("mu0 =", mu0, "\n")
cat("v0^2 =", v0sq, ";  v0 =", v0, "\n\n")

# determine mu1, v1^2 for G1
a1b1_sum <- 2 * mu0
a1b1_diff <- sqrt(12 * params$N * v0sq)

# solve for a1, b1
a1 <- (a1b1_sum - a1b1_diff)/2
b1 <- (a1b1_sum + a1b1_diff)/2

theta1 <- c(a1, b1)
cat("G1 = Unif(", a1, ",", b1, ")\n")

# determine mu2, v2^2 for G2
a2b2_sum  <- 2*(mu0 + 2*v0)
a2b2_diff <- sqrt(12*(v0sq / params$S))

# solve for a2, b2
a2 <- (a2b2_sum - a2b2_diff)/2
b2 <- (a2b2_sum + a2b2_diff)/2

theta2 <- c(a2, b2)
cat("G2 = Unif(", a2, ",", b2, ")\n\n")

# define probabilities p1, p2
alpha1 <- 0.1
alpha2 <- 0.01
tau <- 1/(1 + params$I1)

p1 <- alpha1^(1 - tau) * alpha2^tau
p2 <- 5 * p1 / sqrt(params$S)  # S = 9 => sqrt(S) = 3 => p2 = 5*p1/3

cat("p1 =", p1, "\n")
cat("p2 =", p2, "\n\n")

# mixture: with probability p => Unif(a1,b1), else Unif(a0,b0)
rMixture <- function(n, p, a0, b0, a1, b1) {
  # vector of 0/1 for which component is chosen
  comp <- rbinom(n, size=1, prob=p)
  x <- numeric(n)
  # generate from G0 or G1
  idx1 <- which(comp == 1)
  idx0 <- which(comp == 0)
  # random draw from Uniform(a,b)
  x[idx1] <- runif(length(idx1), min=a1, max=b1)
  x[idx0] <- runif(length(idx0), min=a0, max=b0)
  return(x)
}

n1 <- 10 * (2 + params$N)
n2 <- 100 * (2 + params$N)

cat("Sample sizes:", n1, "and", n2, "\n\n")

set.seed(1337)

# G0 and G1
xA_n1 <- rMixture(n1, p1, a0, b0, a1, b1)
resA_n1 <- ks.test(xA_n1, "punif", min=a0, max=b0)$p.value

xA_n2 <- rMixture(n2, p1, a0, b0, a1, b1)
resA_n2 <- ks.test(xA_n2, "punif", min=a0, max=b0)$p.value

cat("Mixture (p1) results:\n")
cat("n =", n1, ":\n"); print(resA_n1)
cat("n =", n2, ":\n"); print(resA_n2)

# G0 and G2
xB_n1 <- rMixture(n1, p2, a0, b0, a2, b2)
resB_n1 <- ks.test(xB_n1, "punif", min=a0, max=b0)$p.value

xB_n2 <- rMixture(n2, p2, a0, b0, a2, b2)
resB_n2 <- ks.test(xB_n2, "punif", min=a0, max=b0)$p.value

cat("Mixture (p2) results:\n")
cat("n =", n1, ":\n"); print(resB_n1)
cat("n =", n2, ":\n"); print(resB_n2)


##############################################################################
# 2.1. Parametric bootstrap for testing complex GoF hypothesis
##############################################################################

# for uniform, MLE is (min(y), max(y))
mle_uniform <- function(y) {
  c(min(y), max(y))
}

parametric_bootstrap_test <- function(y, B, alpha=0.1) {
  n <- length(y)

  thetaN <- mle_uniform(y)  # min(y), max(y)
  aN <- thetaN[1]
  bN <- thetaN[2]

  # comparing EDF of y to unif(aN, bN)
  T_N <- ks.test(y, "punif", min=aN, max=bN)$statistic

  T_star <- numeric(B)
  for (b in 1:B) {
    # generate bootstrap sample
    y_star <- runif(n, min=aN, max=bN)

    # estimate parameter from y_star
    theta_star <- mle_uniform(y_star)
    a_star <- theta_star[1]
    b_star <- theta_star[2]

    # compute T_(N,b)* comparing y_star to unif(a_star, b_star)
    T_star[b] <- ks.test(y_star, "punif", min=a_star, max=b_star)$statistic
  }

  p_star <- mean(T_star > T_N)

  # reject or not
  reject <- (p_star < alpha)

  list(T_N = T_N,
       p_boot = p_star,
       reject = reject,
       aN = aN, bN = bN,
       B = B)
}

set.seed(1337)

alpha <- 0.1    # significance level

# G0 and G1
cat("G0 and G1\n")
for (n in c(n1, n2)) {
  B_boot <- 100 * n

  # generate data
  yA <- rMixture(n, p1, a0, b0, a1, b1)

  # parametric bootstrap test
  resA <- parametric_bootstrap_test(yA, B=B_boot, alpha=alpha)

  cat(sprintf(" n=%d, T_N=%.5f, p_boot=%.5f, rejectH0=%s\n",
              n, resA$T_N, resA$p_boot, resA$reject))
}

# G0 and G2
cat("G0 and G2\n")
for (n in c(n1, n2)) {
  B_boot <- 100 * n

  # generate data
  yB <- rMixture(n, p2, a0, b0, a2, b2)

  # parametric bootstrap test
  resB <- parametric_bootstrap_test(yB, B=B_boot, alpha=alpha)

  cat(sprintf(" n=%d, T_N=%.5f, p_boot=%.5f, rejectH0=%s\n",
              n, resB$T_N, resB$p_boot, resB$reject))
}


##############################################################################
# 2.2.1. Checking bootstrap consistency for Gamma(0.5, 9), N=100
##############################################################################

set.seed(1337)

N <- 100
a_true <- 0.5   # shape
b_true <- 9     # scale
B_boot <- 2000  # number of bootstrap replications

# generate data
Y <- rgamma(N, shape=a_true, scale=b_true)

gamma_fit <- fitdistr(Y, densfun="gamma")  # shape, rate=1/scale

# parametric bootstrap
a_hat <- gamma_fit$estimate["shape"]
b_hat <- 1/gamma_fit$estimate["rate"]  # scale = 1/rate

ybar_param_boot <- numeric(B_boot)
for (b in 1:B_boot) {
  Ystar <- rgamma(N, shape=a_hat, scale=b_hat)
  ybar_param_boot[b] <- mean(Ystar)
}

# nonparametric bootstrap
ybar_nparam_boot <- numeric(B_boot)
for (b in 1:B_boot) {
  Ystar_np <- sample(Y, size=N, replace=TRUE)
  ybar_nparam_boot[b] <- mean(Ystar_np)
}

ks_distance_true <- function(x_sample, cdfF) {
  x_sorted <- sort(x_sample)
  n_s <- length(x_sample)
  Fn <- seq_len(n_s)/n_s

  F_theory <- sapply(x_sorted, cdfF)
  max(abs(Fn - F_theory))
}

cdf_mean_gamma <- function(x, shape, scale) {
  pgamma(x, shape=shape, scale=scale)
}

cdfYbar_true <- function(x) {
  cdf_mean_gamma(x, shape=N*a_true, scale=b_true/N)
}
ks_param_stats <- ks.test(ybar_param_boot, cdfYbar_true)

# similarly for nonparam
ks_nparam_stats <- ks.test(ybar_nparam_boot, cdfYbar_true)

cat("Gamma(0.5, 9), N=100\n")
cat("  KS (param)\n", ks_param_stats$statistic, ks_param_stats$p.value, "\n")
cat("  KS (nonparam)\n", ks_nparam_stats$statistic, ks_nparam_stats$p.value, "\n")


##############################################################################
# 2.2.2. Checking bootstrap consistency for Pareto(c=9, d=0.5), N=100
##############################################################################

set.seed(1337)

# Monte Carlo to approximate true distribution
N <- 100
c_true <- 9
d_true <- 0.5
M_large <- 2e5  # 200k for MC approximation

# Pareto(c, d) random generator
rpareto <- function(n, c, d) {
  U <- runif(n)
  c * ((1 - U)^(-1/d))
}

# generate matrix
data_all <- rpareto(M_large, c=c_true, d=d_true)  # 2000 blocks
mat_all <- matrix(data_all, nrow=2000, ncol=N)
means_all <- rowMeans(mat_all)  # mean of each block

# from one sample of size N=100
Y <- rpareto(N, c=c_true, d=d_true)

# parametric bootstrap
pareto_mle <- function(y) {
  c_ml <- min(y)
  d_ml <- length(y) / sum(log(y/c_ml))
  c(c_ml, d_ml)
}

mle_pareto <- pareto_mle(Y)
c_hat <- mle_pareto[1]
d_hat <- mle_pareto[2]

B_boot <- 2000
ybar_param <- numeric(B_boot)
for (b in 1:B_boot) {
  Ystar <- rpareto(N, c_hat, d_hat)
  ybar_param[b] <- mean(Ystar)
}

# nonparametric bootstrap
ybar_nparam <- numeric(B_boot)
for (b in 1:B_boot) {
  Ystar_np <- sample(Y, size=N, replace=TRUE)
  ybar_nparam[b] <- mean(Ystar_np)
}

# compare each to true distribution
ks_param_stats <- suppressWarnings(ks.test(ybar_param, means_all))
ks_nparam_stats <- suppressWarnings(ks.test(ybar_nparam, means_all))

cat("Pareto(c=9, d=0.5), N=100\n")
cat("  KS (param)\n", ks_param_stats$statistic, ks_param_stats$p.value, "\n")
cat("  KS (nonparam)\n", ks_nparam_stats$statistic, ks_nparam_stats$p.value, "\n")


##############################################################################
# 4.3.1. Bootstrap Confidence Intervals for PARETO(c=9, d=11) var(mean), N=100
##############################################################################

# some utils
# - t_hat: point estimate from the original sample
# - t_boot: bootstrap replicates of estimate
# - alpha: 1 - gamma
ci_normal <- function(t_hat, t_boot, alpha = 0.1) {
  z <- qnorm(1 - alpha/2)
  sd_t <- sd(t_boot)
  lower <- t_hat - z * sd_t
  upper <- t_hat + z * sd_t
  return(c(lower, upper))
}

ci_percentile <- function(t_boot, alpha = 0.1) {
  lower <- quantile(t_boot, probs = alpha/2)
  upper <- quantile(t_boot, probs = 1 - alpha/2)
  return(c(lower, upper))
}

ci_pivotal <- function(t_hat, t_boot, alpha = 0.1) {
  lower <- 2 * t_hat - quantile(t_boot, probs = 1 - alpha/2)
  upper <- 2 * t_hat - quantile(t_boot, probs = alpha/2)
  return(c(lower, upper))
}

# true variance of Pareto sample mean
var_mean_pareto <- function(d, c, N) {
  numerator <- c * d^2
  denominator <- (c - 1)^2 * (c - 2)
  return( numerator / denominator / N )
}

set.seed(1337)

N <- 100       # sample size
c_true <- 9         # scale param as per the task
d_true <- 11        # shape param as per the task
M_boot <- 2000      # number of bootstrap reps
M_mc <- 5000      # number of direct Monte Carlo reps
alpha <- 0.10      # significance level => gamma=0.90


# generate
y_sample <- rpareto(N, c_true, d_true)

# sample mean
ybar_hat <- mean(y_sample)

mle_pareto <- pareto_mle(y_sample)
c_hat <- mle_pareto[1]
d_hat <- mle_pareto[2]

# parametric bootstrap
param_boot_vals <- numeric(M_boot)

for (b in 1:M_boot) {
  yb <- rpareto(N, c_hat, d_hat)
  # sample mean
  param_boot_vals[b] <- mean(yb)
}

var_hat_param <- var(param_boot_vals)

varhat_boot_param <- numeric(M_boot)
for (b in 1:M_boot) {
  # re-sample from the param_boot_vals with replacement:
  subvals <- sample(param_boot_vals, replace=TRUE, size=length(param_boot_vals))
  varhat_boot_param[b] <- var(subvals)
}

# varhat_boot_param is a bootstrap distribution var_hat_param
T_hat_param <- mean(varhat_boot_param)
T_boot_param <- varhat_boot_param

ci_param_normal <- ci_normal(T_hat_param, T_boot_param, alpha)
ci_param_percentile <- ci_percentile(T_boot_param, alpha)
ci_param_pivotal <- ci_pivotal(T_hat_param, T_boot_param, alpha)

# nonparametric bootstrap
nparam_boot_vals <- numeric(M_boot)
for (b in 1:M_boot) {
  # sample with replacement from original data
  yb <- sample(y_sample, replace=TRUE, size=N)
  # statistic = sample mean
  nparam_boot_vals[b] <- mean(yb)
}
var_hat_nparam <- var(nparam_boot_vals)

# nested approach for the CI on var_hat_nparam
varhat_boot_nparam <- numeric(M_boot)
for (b in 1:M_boot) {
  subvals <- sample(nparam_boot_vals, replace=TRUE, size=length(nparam_boot_vals))
  varhat_boot_nparam[b] <- var(subvals)
}

T_hat_nparam  <- mean(varhat_boot_nparam)
T_boot_nparam <- varhat_boot_nparam

ci_nparam_normal <- ci_normal(T_hat_nparam, T_boot_nparam, alpha)
ci_nparam_percentile <- ci_percentile(T_boot_nparam, alpha)
ci_nparam_pivotal <- ci_pivotal(T_hat_nparam, T_boot_nparam, alpha)

# direct Monte Carlo
mc_vals <- numeric(M_mc)
for (m in 1:M_mc) {
  ym <- rpareto(N, c_true, d_true)
  mc_vals[m] <- mean(ym)
}
var_hat_mc <- var(mc_vals)  # direct MC estimate

# CI by bootstrap of MC sample
varhat_boot_mc <- numeric(M_boot)
for (b in 1:M_boot) {
  subvals <- sample(mc_vals, replace=TRUE, size=length(mc_vals))
  varhat_boot_mc[b] <- var(subvals)
}

T_hat_mc  <- mean(varhat_boot_mc)
T_boot_mc <- varhat_boot_mc

ci_mc_normal <- ci_normal(T_hat_mc, T_boot_mc, alpha)
ci_mc_percentile <- ci_percentile(T_boot_mc, alpha)
ci_mc_pivotal <- ci_pivotal(T_hat_mc, T_boot_mc, alpha)

# compare all CIs and the true value
true_var_mean <- var_mean_pareto(c_true, d_true, N = N)

cat("Task A: var(mean(Y)) results\n")
cat("True value =", true_var_mean, "\n\n")

cat("Parametric bootstrap:", round(T_hat_param, 5), "\n")
cat("   Normal CI     =", round(ci_param_normal, 5), "\n")
cat("   Percentile CI =", round(ci_param_percentile, 5), "\n")
cat("   Pivotal CI    =", round(ci_param_pivotal, 5), "\n\n")

cat("Nonparametric bootstrap:", round(T_hat_nparam, 5), "\n")
cat("   Normal CI     =", round(ci_nparam_normal, 5), "\n")
cat("   Percentile CI =", round(ci_nparam_percentile, 5), "\n")
cat("   Pivotal CI    =", round(ci_nparam_pivotal, 5), "\n\n")

cat("Direct MC:", round(T_hat_mc, 5), "\n")
cat("   Normal CI     =", round(ci_mc_normal, 5), "\n")
cat("   Percentile CI =", round(ci_mc_percentile, 5), "\n")
cat("   Pivotal CI    =", round(ci_mc_pivotal, 5), "\n")


##############################################################################
# 4.3.2. Bootstrap Confidence Intervals for PARETO(c=9, d=11) var(Y_(1)), N=100
##############################################################################

# we repeat analogous steps, but now the statistic is Y_(1), the minimum of the sample

# parametric bootstrap
param_boot_cML <- numeric(M_boot)
for (b in 1:M_boot) {
  yb <- rpareto(N, c_hat, d_hat)
  param_boot_cML[b] <- min(yb)
}
# best point estimate for var(Y_(1)) from param bootstrap is var(param_boot_cML)
var_cML_param <- var(param_boot_cML)

varcML_boot_param <- numeric(M_boot)
for (b in 1:M_boot) {
  subvals <- sample(param_boot_cML, replace=TRUE, size=length(param_boot_cML))
  varcML_boot_param[b] <- var(subvals)
}
T_hat_cML_param  <- mean(varcML_boot_param)
T_boot_cML_param <- varcML_boot_param

ci_param_cML_normal <- ci_normal(T_hat_cML_param, T_boot_cML_param, alpha)
ci_param_cML_percentile <- ci_percentile(T_boot_cML_param, alpha)
ci_param_cML_pivotal <- ci_pivotal(T_hat_cML_param, T_boot_cML_param, alpha)

# nonparametric bootstrap
nparam_boot_cML <- numeric(M_boot)
for (b in 1:M_boot) {
  yb <- sample(y_sample, replace=TRUE, size=N)
  nparam_boot_cML[b] <- min(yb)
}
var_cML_nparam <- var(nparam_boot_cML)

varcML_boot_nparam <- numeric(M_boot)
for (b in 1:M_boot) {
  subvals <- sample(nparam_boot_cML, replace=TRUE, size=length(nparam_boot_cML))
  varcML_boot_nparam[b] <- var(subvals)
}
T_hat_cML_nparam  <- mean(varcML_boot_nparam)
T_boot_cML_nparam <- varcML_boot_nparam

ci_nparam_cML_normal <- ci_normal(T_hat_cML_nparam, T_boot_cML_nparam, alpha)
ci_nparam_cML_percentile <- ci_percentile(T_boot_cML_nparam, alpha)
ci_nparam_cML_pivotal <- ci_pivotal(T_hat_cML_nparam, T_boot_cML_nparam, alpha)


# direct Monte Carlo
mc_cML <- numeric(M_mc)
for (m in 1:M_mc) {
  ym <- rpareto(N, c_true, d_true)
  mc_cML[m] <- min(ym)
}
var_cML_mc <- var(mc_cML)

varcML_boot_mc <- numeric(M_boot)
for (b in 1:M_boot) {
  subvals <- sample(mc_cML, replace=TRUE, size=length(mc_cML))
  varcML_boot_mc[b] <- var(subvals)
}
T_hat_cML_mc <- mean(varcML_boot_mc)
T_boot_cML_mc <- varcML_boot_mc

ci_mc_cML_normal <- ci_normal(T_hat_cML_mc, T_boot_cML_mc, alpha)
ci_mc_cML_percentile <- ci_percentile(T_boot_cML_mc, alpha)
ci_mc_cML_pivotal <- ci_pivotal(T_hat_cML_mc, T_boot_cML_mc, alpha)


cat("Task B: var(Y_(1)) results\n")

cat("Parametric bootstrap:", round(T_hat_cML_param, 7), "\n")
cat("   Normal CI     =", round(ci_param_cML_normal, 7), "\n")
cat("   Percentile CI =", round(ci_param_cML_percentile, 7), "\n")
cat("   Pivotal CI    =", round(ci_param_cML_pivotal, 7), "\n\n")

cat("Nonparametric bootstrap:", round(T_hat_cML_nparam, 7), "\n")
cat("   Normal CI     =", round(ci_nparam_cML_normal, 7), "\n")
cat("   Percentile CI =", round(ci_nparam_cML_percentile, 7), "\n")
cat("   Pivotal CI    =", round(ci_nparam_cML_pivotal, 7), "\n\n")

cat("Direct MC:", round(T_hat_cML_mc, 7), "\n")
cat("   Normal CI     =", round(ci_mc_cML_normal, 7), "\n")
cat("   Percentile CI =", round(ci_mc_cML_percentile, 7), "\n")
cat("   Pivotal CI    =", round(ci_mc_cML_pivotal, 7), "\n")
