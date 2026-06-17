import numpy as np
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

def simulate_woman_at_t(P, t, rng):
    """Simulate one woman and return her state at time t (or 4 if she dies before t)."""
    state = 0
    for _ in range(t):
        if state == 4:
            break
        state = rng.choice(5, p=P[state])
    return state

# --- Simulation ---
N = 1000
T = 120
rng = np.random.default_rng(seed=42)

simulated_states = [simulate_woman_at_t(P, T, rng) for _ in range(N)]
simulated_counts = np.bincount(simulated_states, minlength=5)
simulated_dist   = simulated_counts / N

# --- Theoretical distribution: p0 @ P^120 ---
p0 = np.array([1, 0, 0, 0, 0])  # all women start in state 1 (index 0)
P_t = la.matrix_power(P, T)
theoretical_dist = p0 @ P_t

# --- Chi-square goodness-of-fit test ---
# Expected counts = theoretical probabilities * N
# We only include states with expected count > 0 to avoid division by zero
expected_counts = theoretical_dist * N
mask = expected_counts > 0

chi2_stat, p_value = stats.chisquare(
    f_obs=simulated_counts[mask],
    f_exp=expected_counts[mask]
)

# --- Results ---
print("=== Task 2: State distribution at t=120 months ===\n")
print(f"{'State':<8} {'Simulated':>12} {'Theoretical':>14} {'Sim count':>12} {'Exp count':>12}")
print("-" * 60)
state_names = ["1 (none)", "2 (local)", "3 (distant)", "4 (both)", "5 (death)"]
for i in range(5):
    print(f"{state_names[i]:<12} {simulated_dist[i]:>10.4f} {theoretical_dist[i]:>14.4f} "
          f"{simulated_counts[i]:>10} {expected_counts[i]:>12.1f}")

print(f"\nChi-square statistic: {chi2_stat:.4f}")
print(f"p-value:              {p_value:.4f}")
print(f"Degrees of freedom:   {mask.sum() - 1}")

if p_value > 0.05:
    print("\nConclusion: Fail to reject H0 — simulated distribution is consistent")
    print("with the theoretical distribution at the 5% significance level.")
else:
    print("\nConclusion: Reject H0 — simulated distribution differs significantly")
    print("from the theoretical distribution at the 5% significance level.")