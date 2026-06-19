import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import expm

# Transition-rate matrix Q
Q = np.array([
    [-0.0085, 0.005,  0.0025, 0,     0.001],
    [0,      -0.014,  0.005,  0.004, 0.005],
    [0,       0,     -0.008,  0.003, 0.005],
    [0,       0,      0,     -0.009, 0.009],
    [0,       0,      0,      0,     0    ],
])

# Sub-matrix Qs: remove last row/column (death state)
Qs = Q[:4, :4]
p0 = np.array([1, 0, 0, 0])  # everyone starts in state 1

def theoretical_cdf(t):
    """F_T(t) = 1 - p0 @ expm(Qs * t) @ 1"""
    return 1 - p0 @ expm(Qs * t) @ np.ones(4)

# --- Reuse CTMC simulation from Task 7 ---
def simulate_ctmc_woman(Q, rng):
    state = 0
    t = 0.0
    while state != 4:
        rate = -Q[state, state]
        t += rng.exponential(1 / rate)
        probs = Q[state].copy()
        probs[state] = 0
        probs = probs / probs.sum()
        state = rng.choice(5, p=probs)
    return t

N = 1000
rng = np.random.default_rng(seed=42)
lifetimes = np.array([simulate_ctmc_woman(Q, rng) for _ in range(N)])

# --- KS test against theoretical CDF ---
def cdf_callable(t_array):
    return np.array([theoretical_cdf(t) for t in t_array])

ks_result = stats.kstest(lifetimes, cdf_callable)

print("=== Task 8: Empirical vs theoretical lifetime distribution (CTMC) ===\n")
print(f"KS statistic: {ks_result.statistic:.4f}")
print(f"p-value:      {ks_result.pvalue:.4f}")
if ks_result.pvalue > 0.05:
    print("Conclusion: Fail to reject H0 — empirical lifetimes are consistent")
    print("with the theoretical continuous phase-type distribution.")
else:
    print("Conclusion: Reject H0 — empirical lifetimes differ from theory.")

# --- Plot empirical vs theoretical CDF ---
t_grid = np.linspace(0, lifetimes.max(), 500)
theo_cdf_vals = np.array([theoretical_cdf(t) for t in t_grid])

sorted_lifetimes = np.sort(lifetimes)
empirical_cdf_vals = np.arange(1, N + 1) / N

plt.figure(figsize=(10, 5))
plt.plot(t_grid, theo_cdf_vals, color='tomato', linewidth=2, label='Theoretical CDF')
plt.step(sorted_lifetimes, empirical_cdf_vals, color='steelblue', linewidth=1.2,
         label='Empirical CDF', where='post')
plt.xlabel("Lifetime after surgery (months)")
plt.ylabel("Cumulative probability")
plt.title("Task 8: Empirical vs theoretical CDF (CTMC)")
plt.legend()
plt.tight_layout()
plt.savefig("task8_cdf_comparison.png", dpi=150)
plt.show()
print("\nPlot saved to task8_cdf_comparison.png")