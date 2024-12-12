# Preliminaries
In order to individualize tasks the following special quantities are to be used:
* N = 9
* S = 9
* I_1 = 5
* I_2 = 8

Let G1, ..., Gm be given distribution functions and p1, ..., pm be probabilities that sum to 1.
The distribution function G defined by

G(u) := p1*G1(u) + ... + pm*Gm(u) = sum(p_k*G_k(u), k=1..m), u ∈ R, (1)

is called a mixture of distribution functions G1, ..., Gm with probabilities (or weights) p1, ..., pm.
G is the distribution function of the random variable Z generated in the following way:
1. Choose k ∈ [1, m] := {1, ..., m} at random with the corresponding probabilities p1, ..., pm. The chosen number is denoted by k∗.
2. Generate a random variable Z_k∗ according to the distribution function G_k∗ and assign Z <- Z_k∗.
A brief representation of this algorithm:
Z <- Z_k∗, k∗ ∼ Multinomm(1, p1, ..., pm), Z_k∗ ∼ G_k (k ∈ [1, m]). (2)
When m = 2, the algorithm reduces to
Z <- Z_(1+k)∗, k∗ ∼ Binomial(1, p2), Z_k∗ ∼ G_k (k = 1, 2). (3)

Let
𝒢(Θ) := {G(·| θ), θ ∈ Θ} (4)
be a given parametric family of absolutely continuous distribution functions G(·| θ) with the respective distribution densities g(·| θ) dependent on the unknown parameter θ ∈ Θ. It is assumed that θ is two-dimensional, i.e., θ = (θ1, θ2) ∈ R2. A specific parametric distribution family is given below.
For Y ∼ G(·| θ), denote its mean and variance by
µ(θ) := Eθ(Y) = integral(u dG(u| θ, -∞..∞), v^2(θ) := Varθ(Y) = integral(u^2 * dG(u| θ), -∞..∞) - µ^2(θ). (5)

# Parametric Distribution Family
The parametric family G5(Θ) contains distribution functions of random variables uniformly distributed on [θ1, θ2], θ1 < θ2. The parameter of the basic distribution function G0 is θ0 = (−N , S + 4).

# Task 1: Testing goodness-of-fit
(a) Distribution Functions. Let Gi(u) := G(u| θi) be different distribution functions from G(Θ), θi ∈ Θ, i = 0, 1, 2. The distribution function G0(u) := G(u| θ0) with the respective mean µ0 := µ(θ0) and variance v_0^2 := v^2(θ0) is the basic one. Its parameter θ0 depends on the parametric family given above.
Given µ0 and v_0^2, the parameters of the other two distribution functions are to be found from the equations µ0 = µ(θ1), N * v_0^2 = v^2(θ1) (6)
with respect to θ1 and the equations
µ0 + 2v0 = µ(θ2), v_0^2 = S * v^2(θ2) (7)
with respect to θ2. Define also probabilities
p1 := (α1)^(1−τ)*(α2)^τ, p2 := 5*p1/sqrt(S), τ := 1/(1 + I1), α1 := 0.1, α2 := 0.01. (8)
The second equation in (7) and the definition of p2 in (8) ensure that the maximums of the weighted densities p2*g(·| θ2) and (1 − p2)*g(·| θ0) are comparable.
(b) Testing Goodness of Fit. For the significance levels α1 and α2 defined in (8), sample sizes N = 10 * (2 + N ) and N = 100 * (2 + N), test the simple Goodness-of-Fit hypothesis
H0: FY = G0 versus H': F_Y != G0 (9)
by using the Kolmogorov-Smirnov, Cramer–von Mises, Anderson-Darling tests and Dvoretzky–Kiefer–Wolfowitz Inequality (see Lecture notes, Theorem 2.1.4) when SRS (Yt)_1^N is generated from FY = (1 − p1)G0 + p1*G1 and when it is generated from FY = (1 − p2)G0 + p2*G2. Here the distribution functions Gi are taken from the parametric family G5(Θ). The parametric family G5(Θ) is specified above.
Find the respective (approximate) p-values.
(c) Comment the results, compare the tests.

# Task 2: Applications of Bootstrap Technique
Labs in this section are devoted to
* testing Complex Goodness of Fit Hypothesis,
* checking bootstrap consistency,
* comparison of several bootstrap confidence interval construction methods.
It is recommended to take a bootstrap sample size B much greater than a initial sample size N, say B ≈ 10^2*N.

## Testing Goodness-of-Fit by Bootstrap
Testing Complex Goodness of Fit Hypothesis. This problem continues that of (b) in Task 1 (Testing Goodness-of-Fit).
Complex Goodness of Fit Hypothesis asserts that the unknown distribution function G0 belongs to some given parametric family G(Θ) of distribution functions:
H0: FY ∈ G(Θ) versus H': FY ∈ G / (Θ). (11)
The task is to test (11) by making use of parametric bootstrap. Let T denote any of goodness-of-fit statistics: Kolmogorov-Smirnov, Cramer–von Mises or Anderson-Darling. The significance level is α = 0.1.
* The parametric family G(Θ) of distribution functions, the basic distribution function G0, the distribution functions G1 and G2, the definitions of distribution function FY for the initial data simulation, as well as numeric values of the parameters and other quantities are the same as in Task 1.

Algorithm:
1. Select or generate the same (analogous) data Y^N := (Y_t)_1^N
as in item (b) of Task 1. The two different cases (depending, respectively, on G1 and G2) of the underlying distribution FY are to be considered.
2. Assume that FY ∈ G(Θ) and estimate the unknown parameter θ by maximum likelihood
(ML) or method of moments (MM). Let bθN be the estimate obtained and GbN (u) := G(u|
bθN ). Calculate value T_N of the goodness-of-fit statistic T when comparing the EDF of the data Y^N with GbN . Roughly speaking, T_N is the value of the statistic T for the simple testing problem
H0: FY = G_N versus H': F_Y != G_N , (12)
where G_N (actually the θ_N ) is treated as known and nonrandom.
3. Generate independent bootstrap samples Y
N∗
b
, b ∈ [1, B], Y
∗ ∼ GbN . For each Y
N∗
b
, estimate
the unknown parameter θ by the same method as in Step 2 (the estimate obtained is denoted
by bθ
∗
N,b = bθ(Y
N∗
b
)) and calculate values Tb∗
N,b of the test statistic T when comparing the EDF
of the ”bootstrapt” data Y
N∗
b with Gb∗
N,b. Here Gb∗
N,b(u) := G(u|
bθ
∗
N,b) (b ∈ [1, B]).
4. Find approximate bootstrap p-value for the test statistic T:
pb
∗ = pb
∗
(T, G, Y N , B) =
#{b : Tb∗
N,b > TbN }
B
. (13)
4
* Compare simulation results of testing complex goodness-of-fit hypothesis (11) by making use
of parametric bootstrap with the goodness-of-fit testing results of Task 1. Draw conclusions.
* Make recommendations regarding applications of simple nonparametric bootstrap for testing
(11).
4.2 Checking Bootstrap Consistency
Let Z ∼ Gamma(a, b) where a > 0 is the shape parameter and b > 0 is the scale parameter
of the Gamma distribution, and let F1 denote the distribution function of Z with parameters
a = N /(N + S), b = S.
It is well known that
Z1 + Z2 ∼ Gamma(a1 + a2, b), if Z` ∼ Gamma(a`
, b), ` = 1, 2, Z1, Z2 are independent. (14)
Task A: Let Y
N be SRS of Y ∼ F1, N = (1 + N ) · 10. Check the consistency of the parametric
and simple nonparametric bootstrap for the sample mean Y N by comparing its true distribution
derived analytically from (14) and the bootstrap estimators of this distribution in an appropriate
way. Use Kolmogorov-Smirnov, Cramer–von Mises or Anderson-Darling statistics as goodness-of-fit
measures.
Let F2 ∼ P areto(c, d) be the Pareto (Type I; see https://en.wikipedia.org/wiki/Pareto_
distribution) distribution function with the scale parameter c > 0 and the shape parameter
d > 0. Take c = S and d = N /(N + S),.
Task B: Let Y
N be SRS of Y ∼ F2, N = (1+N )·10. Check the consistency of the parametric and
simple nonparametric bootstrap for the sample mean Y N by comparing approximation of its true
distribution obtained by Monte Carlo simulations and the bootstrap estimators of this distribution
in an appropriate way. Use Kolmogorov-Smirnov, Cramer–von Mises or Anderson-Darling statistics
as goodness-of-fit measures.
Discuss the results and draw conclusions. The significance level is α = 0.1.
4.3 Bootstrap Confidence Intervals
Let Y
N be SRS of Y ∼ P areto(c, d) with c = S, d = 2 + N , N = (1 + N )· 10. The confidence level
is γ = 0.90.
Task A: The parameter of interest is the variance of the sample mean Y N .
Compare γ-confidence intervals – normal, pivotal, percentile – obtained by making use of parametric bootstrap, simple nonparametric bootstrap and (direct) Monte Carlo simulations with the
true values c = S and d = 2 + N of the parameters. Find (estimate) true coverages of the resulting
γ-confidence intervals (optional1
).
The maximum likelihood estimator of the Pareto distribution parameter c is given by
cbML := Y(1) = min
j∈[1,N]
Yj . (15)
Task B: The parameter of interest is the variance of Y(1), i.e., the variance of the maximum
likelihood estimator cbML of c.
For the variance of Y(1), compare γ-confidence intervals – normal, pivotal, percentile – obtained
by making use of the three mentioned methods: parametric bootstrap, simple nonparametric bootstrap and (direct) Monte Carlo simulations with the true values c = S and d = 2 + N of the
parameters. Find (estimate) true coverages of the resulting γ-confidence intervals (optional 1
).
Discuss the results and draw conclusions.
1Finding true coverage is much more computer intensive task. Therefore it is optional.

# Task 3: Nonparametric density estimation
The task is to compare performance of various nonparametric density estimators.
Simulated data. The data is generated as a SRS from a mixture (Section 1) with the respective
probabilities π`
, ` = 0, 1, 2 of the same distributions G`
, ` = 0, 1, 2, described in Sections 2 and 3:
G(u) := π0G0(u) + π1G1(u) + π2G2(u), π1 :=
S
1 + I1 + 2N + S
, π2 :=
N
1 + I1 + 2N + S
. (16)
Task: Estimate the distribution density g(u) := G0
(u) nonparametrically by making use of the
Least Squared Cross-Validation, the Refind Plug-in Method for Bandwidth Selection, the Smoothed
Bootstrap for Bandwidth Selection and k-Nearest Neighbours Density Estimator.
Compare estimators with each other and with the true density g and discuss the results.
