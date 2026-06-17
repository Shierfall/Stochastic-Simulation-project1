import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numpy.linalg as la

# Transition probability matrix
P = np.array([
    [0.9915, 0.005,  0.0025, 0,     0.001],
    [0,      0.986,  0.005,  0.004, 0.005],
    [0,      0,      0.992,  0.003, 0.005],
    [0,      0,      0,      0.991, 0.009],
    [0,      0,      0,      0,     1.0  ],
])

# Sub-matrix Ps: remove last row and column (death state)
Ps = P[:4, :4]

# ps: probability of dying from each transient state (last column, rows 0-3)
ps = P[:4, 4]

# pi: initial distribution over transient states (all start in state 1 = index 0)
pi = np.array([1, 0, 0, 0])

# --- Simulate 1000 lifetimes (reusing Task 1 logic) ---
def simulate_woman(P, rng):
    state = 0
    lifetime = 0
    while state != 4:
        state = rng.choice(5, p=P[state])
        lifetime += 1
    return lifetime

N = 1000
rng = np.random.default_rng(seed=42)
lifetimes = np.array([simulate_woman(P, rng) for _ in range(N)])

# --- Theoretical PMF: P(T=t) = pi @ Ps^t @ ps ---
def theoretical_pmf(t_values, pi, Ps, ps):
    pmf = np.zeros(len(t_values))
    for idx, t in enumerate(t_values):
        Ps_t = la.matrix_power(Ps, t)
        pmf[idx] = pi @ Ps_t @ ps
    return pmf

# --- Theoretical CDF ---
def theoretical_cdf(t_max, pi, Ps, ps):
    cdf = np.zeros(t_max + 1)
    cumulative = 0
    for t in range(1, t_max + 1):
        Ps_t = la.matrix_power(Ps, t)
        cumulative += pi @ Ps_t @ ps
        cdf[t] = cumulative
    return cdf

# Compute theoretical mean: E(T) = pi @ (I - Ps)^-1 @ 1
I = np.eye(4)
theoretical_mean = pi @ la.inv(I - Ps) @ np.ones(4)
print(f"Theoretical mean lifetime: {theoretical_mean:.2f} months ({theoretical_mean/12:.1f} years)")
print(f"Simulated  mean lifetime:  {lifetimes.mean():.2f} months ({lifetimes.mean()/12:.1f} years)")

# --- KS test: compare empirical CDF to theoretical CDF ---
# Build empirical CDF
t_max = int(lifetimes.max())
empirical_cdf = np.array([np.mean(lifetimes <= t) for t in range(t_max + 1)])
theoretical_cdf_vals = theoretical_cdf(t_max, pi, Ps, ps)

# KS statistic: max absolute difference between CDFs
ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf_vals))
# Use scipy's KS test against the theoretical CDF as a callable
def cdf_callable(t_array):
    return np.array([theoretical_cdf(int(t), pi, Ps, ps)[-1] for t in t_array])

ks_result = stats.kstest(lifetimes, cdf_callable)

print(f"\nKS statistic: {ks_stat:.4f}")
print(f"KS p-value:   {ks_result.pvalue:.4f}")
if ks_result.pvalue > 0.05:
    print("Conclusion: Fail to reject H0 — lifetimes are consistent with the phase-type distribution.")
else:
    print("Conclusion: Reject H0 — lifetimes differ from the phase-type distribution.")

# --- Plot: empirical vs theoretical CDF ---
t_plot = np.arange(0, min(t_max, 3000))  # plot up to 3000 months for clarity
theo_cdf_plot = theoretical_cdf(len(t_plot) - 1, pi, Ps, ps)
emp_cdf_plot  = np.array([np.mean(lifetimes <= t) for t in t_plot])

plt.figure(figsize=(10, 5))
plt.plot(t_plot, theo_cdf_plot, color='tomato',    linewidth=2,   label='Theoretical CDF (phase-type)')
plt.plot(t_plot, emp_cdf_plot,  color='steelblue', linewidth=1.5, linestyle='--', label='Empirical CDF (simulation)')
plt.xlabel("Lifetime after surgery (months)")
plt.ylabel("Cumulative probability")
plt.title("Task 3: Empirical vs theoretical lifetime CDF")
plt.legend()
plt.tight_layout()
plt.savefig("task3_cdf_comparison.png", dpi=150)
plt.show()
print("\nPlot saved to task3_cdf_comparison.png")