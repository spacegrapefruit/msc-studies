# parameters
params <- list(
  N = 9,
  S = 9,
  I1 = 5,
  I2 = 8
)

# Uniform distribution
theta_0 <- c(-params$N, params$S + 4)
mu_0 <- (theta_0[1] + theta_0[2]) / 2
var_0 <- (theta_0[2] - theta_0[1])^2 / 12
cat("Theta_0: ", theta_0, "Mu_0: ", mu_0, "Var_0: ", var_0, "\n")

mu_1 <- mu_0
var_1 <- params$N * var_0
theta_1 <- c(-31, 35)
cat("Theta_1: ", theta_1, "Mu_1: ", mu_1, "Var_1: ", var_1, "\n")

mu_2 <- mu_0 + 2 * sqrt(var_0)
var_2 <- var_0 / params$S
theta_2 <- c(11.03135, 18.03135)
cat("Theta_2: ", theta_2, "Mu_2: ", mu_2, "Var_2: ", var_2, "\n")

alpha_1 <- 0.1
alpha_2 <- 0.01
tau <- 1 / (1 + params$I1)
p_1 <- alpha_1^(1 - tau) * alpha_2^tau
p_2 <- 5 * p_1 / sqrt(params$S)
cat("Tau: ", tau, "p_1: ", p_1, "p_2: ", p_2, "\n")

N_1 <- 10 * (2 + params$N)
N_2 <- 100 * (2 + params$N)

cdf_uniform <- function(theta, x) {
  ifelse(x < theta[1], 0, ifelse(x > theta[2], 1, (x - theta[1]) / (theta[2] - theta[1])))
}

cdf_0 <- cdf_uniform(theta_0, seq(-50, 50, 0.1))
# cdf_1 <- cdf_uniform(theta_1, seq(-50, 50, 0.1))
# cdf_uniform(theta_0, seq(-50, 50, 0.1)) - cdf_uniform(theta_1, seq(-50, 50, 0.1))






# # generate the uniformly distributed data
# set.seed(123)
# data_G0 <- runif(N_1, theta_0[1], theta_0[2])
# data_G1 <- runif(N_1, theta_1[1], theta_1[2])
# data_G2 <- runif(N_1, theta_2[1], theta_2[2])



# kolmogorov-smirnov test
# ks.test(