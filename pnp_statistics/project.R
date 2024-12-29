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


compute_test_stat <- function(data, a, b) {
  # Compare EDF of data with U(a,b)
  # ks.test returns a list that includes the test statistic (D)
  ks <- ks.test(data, "punif", min = a, max = b)
  return(ks$statistic)  # D statistic of KS
}

## Task 2: Bootstrap Technique
for (N in c(N_1, N_2)) {
  B <- 100 * N
  cat("N: ", N, "B: ", B, "\n")

  set.seed(123)
  # uniform distribution
  data_G0 <- runif(N, theta_0[1], theta_0[2])
  data_G1 <- runif(N, theta_1[1], theta_1[2])
  data_G2 <- runif(N, theta_2[1], theta_2[2])

  # p_l of selecting i-th element from G_l, otherwise i-th element from G_0
  data_F1 <- ifelse(runif(N) < p_1, data_G1, data_G0)  # (1 - p_1)*G0 + p_1*G1
  data_F2 <- ifelse(runif(N) < p_2, data_G2, data_G0)  # (1 - p_2)*G0 + p_2*G2

  for (data_test in list(data_F1, data_F2)) {
    # maximum likelihood estimate for a, b in uniform distribution
    a_hat <- min(data_test)
    b_hat <- max(data_test)

    T_N <- compute_test_stat(data_test, a_hat, b_hat)

    # Step 3: Parametric bootstrap
    # Under H0, we assume data_test came from U(a_hat,b_hat)
    # Generate B bootstrap samples from U(a_hat,b_hat)

    T_boot <- numeric(B)
    for (b in 1:B) {
      # Bootstrap sample
      Y_star <- runif(N, a_hat, b_hat)

      # Re-estimate parameters for the bootstrap sample
      a_star_hat <- min(Y_star)
      b_star_hat <- max(Y_star)

      # Compute test statistic for bootstrap sample
      T_boot[b] <- compute_test_stat(Y_star, a_star_hat, b_star_hat)
    }

    # Step 4: Bootstrap p-value
    p_star <- mean(T_boot > T_N)

    cat("For N =", N, ": Observed T_N =", T_N, ", Bootstrap p* =", p_star, "\n")

    # Interpretation:
    # If p_star < alpha (e.g., alpha=0.1), reject H0 (F_Y not in G(Theta))
    # Otherwise, fail to reject H0.
  }
}
