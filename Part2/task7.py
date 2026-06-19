import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Transition-rate matrix Q
Q = np.array([
    [-0.0085, 0.005,  0.0025, 0,     0.001],
    [0,      -0.014,  0.005,  0.004, 0.005],
    [0,       0,     -0.008,  0.003, 0.005],
    [0,       0,      0,     -0.009, 0.009],
    [0,       0,      0,      0,     0    ],
])

def simulate_ctmc_woman(Q, rng):
    """
    Simulate one woman through the CTMC until death (state 5, index 4).
    Returns total lifetime (continuous time) and whether she ever
    reached state 3 (distant metastasis, index 2) by 30.5 months.
    """
    state = 0
    t = 0.0
    reached_distant_by_30_5 = False

    while state != 4:
        rate = -Q[state, state]  # rate of leaving current state
        # Sojourn time is exponential with this rate
        sojourn = rng.exponential(1 / rate)
        t_next = t + sojourn

        # Check if distant metastasis (state index 2) was reached by 30.5 months
        # (state must be entered AND we must check before the next jump)
        if state == 2 and t <= 30.5:
            reached_distant_by_30_5 = True

        # Determine next state: probabilities proportional to off-diagonal rates
        probs = Q[state].copy()
        probs[state] = 0  # exclude self
        probs = probs / probs.sum()

        state = rng.choice(5, p=probs)
        t = t_next

    # Final check in case she entered distant state and that was her last state before 30.5
    return t, reached_distant_by_30_5

# Need to track more carefully: check state AT time 30.5, not just upon entry
def simulate_ctmc_woman_v2(Q, rng):
    """
    More careful version: tracks the full (time, state) trajectory so we can
    query the state at any specific time point (e.g. t=30.5).
    """
    state = 0
    t = 0.0
    trajectory = [(0.0, state)]  # (time entered, state)

    while state != 4:
        rate = -Q[state, state]
        sojourn = rng.exponential(1 / rate)
        t += sojourn
        probs = Q[state].copy()
        probs[state] = 0
        probs = probs / probs.sum()
        state = rng.choice(5, p=probs)
        trajectory.append((t, state))

    lifetime = trajectory[-1][0]
    return lifetime, trajectory

def state_at_time(trajectory, query_t):
    """Return the state a woman was in at a given query time."""
    state = trajectory[0][1]
    for (t_enter, s) in trajectory:
        if t_enter <= query_t:
            state = s
        else:
            break
    return state

# --- Run simulation ---
N = 1000
rng = np.random.default_rng(seed=42)

lifetimes = np.empty(N)
distant_at_30_5 = np.zeros(N, dtype=bool)

for i in range(N):
    lifetime, trajectory = simulate_ctmc_woman_v2(Q, rng)
    lifetimes[i] = lifetime
    s = state_at_time(trajectory, 30.5)
    distant_at_30_5[i] = (s == 2)  # state index 2 = distant metastasis

# --- Summary statistics ---
mean_life = lifetimes.mean()
sd_life   = lifetimes.std(ddof=1)
se_mean   = sd_life / np.sqrt(N)
ci_mean   = (mean_life - 1.96 * se_mean, mean_life + 1.96 * se_mean)

# CI for standard deviation using chi-square distribution
alpha = 0.05
chi2_low  = stats.chi2.ppf(alpha / 2, N - 1)
chi2_high = stats.chi2.ppf(1 - alpha / 2, N - 1)
sd_ci = (
    np.sqrt((N - 1) * sd_life**2 / chi2_high),
    np.sqrt((N - 1) * sd_life**2 / chi2_low)
)

prop_distant = distant_at_30_5.mean()
se_prop = np.sqrt(prop_distant * (1 - prop_distant) / N)
prop_ci = (prop_distant - 1.96 * se_prop, prop_distant + 1.96 * se_prop)

print("=== Task 7: CTMC simulation results ===\n")
print(f"Mean lifetime:       {mean_life:.2f} months")
print(f"95% CI for mean:     [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")
print(f"Std deviation:       {sd_life:.2f} months")
print(f"95% CI for std dev:  [{sd_ci[0]:.2f}, {sd_ci[1]:.2f}]")
print(f"\nProportion with distant metastasis at t=30.5 months: {prop_distant:.4f}")
print(f"95% CI: [{prop_ci[0]:.4f}, {prop_ci[1]:.4f}]")

# --- Histogram ---
plt.figure(figsize=(10, 5))
plt.hist(lifetimes, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
plt.axvline(mean_life, color='tomato', linestyle='--', label=f"Mean = {mean_life:.1f} mo")
plt.xlabel("Lifetime after surgery (months)")
plt.ylabel("Number of women")
plt.title("Task 7: Lifetime distribution — CTMC simulation (1000 women)")
plt.legend()
plt.tight_layout()
plt.savefig("task7_ctmc_histogram.png", dpi=150)
plt.show()
print("\nHistogram saved to task7_ctmc_histogram.png")