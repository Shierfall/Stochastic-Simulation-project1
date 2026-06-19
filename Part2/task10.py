import numpy as np
from scipy import stats

# Same Q matrices as Task 9
Q_no_treatment = np.array([
    [-0.0085, 0.005,  0.0025, 0,     0.001],
    [0,      -0.014,  0.005,  0.004, 0.005],
    [0,       0,     -0.008,  0.003, 0.005],
    [0,       0,      0,     -0.009, 0.009],
    [0,       0,      0,      0,     0    ],
])

Q_treatment = np.array([
    [0,      0.0025, 0.00125, 0,     0.001],
    [0,      0,      0,       0.002, 0.005],
    [0,      0,      0,       0.003, 0.005],
    [0,      0,      0,       0,     0.009],
    [0,      0,      0,       0,     0    ],
], dtype=float)
for i in range(5):
    Q_treatment[i, i] = -Q_treatment[i].sum()

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

def log_rank_test(times_a, times_b):
    """
    Log-rank test for comparing two survival distributions.
    Since there is no censoring in this model (everyone dies eventually),
    this simplifies considerably compared to the general censored case,
    but we implement the general formula for correctness/transferability.
    """
    # Combine and get unique event (death) times
    all_times = np.concatenate([times_a, times_b])
    event_times = np.unique(all_times)

    n_a, n_b = len(times_a), len(times_b)

    O_a_total, E_a_total, V_total = 0, 0, 0

    for t in event_times:
        # Number at risk just before time t (those with lifetime >= t)
        at_risk_a = np.sum(times_a >= t)
        at_risk_b = np.sum(times_b >= t)
        at_risk_total = at_risk_a + at_risk_b
        if at_risk_total == 0:
            continue

        # Number of deaths AT exactly time t
        d_a = np.sum(times_a == t)
        d_b = np.sum(times_b == t)
        d_total = d_a + d_b

        if at_risk_total <= 1:
            continue

        # Expected deaths in group A under H0
        E_a = d_total * at_risk_a / at_risk_total

        # Variance (hypergeometric)
        V = (d_total * (at_risk_total - d_total) * at_risk_a * at_risk_b) / \
            (at_risk_total**2 * (at_risk_total - 1))

        O_a_total += d_a
        E_a_total += E_a
        V_total += V

    chi2_stat = (O_a_total - E_a_total)**2 / V_total
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value

# --- Simulate both groups ---
N = 1000
rng = np.random.default_rng(seed=42)

lifetimes_no_treatment = np.array([simulate_ctmc_woman(Q_no_treatment, rng) for _ in range(N)])
lifetimes_treatment    = np.array([simulate_ctmc_woman(Q_treatment, rng) for _ in range(N)])

# NOTE: continuous lifetimes are essentially all unique, so the log-rank test
# as implemented (with exact tie matching) will treat almost every time as
# its own "event time" with one event from one group at a time. This works
# correctly but is computationally heavier; for large N consider binning
# times into discrete intervals (e.g. monthly) first.

# To keep this efficient, we round lifetimes to the nearest month for
# practical tie-handling (a common approach with continuous survival data):
lifetimes_no_treatment_rounded = np.round(lifetimes_no_treatment)
lifetimes_treatment_rounded    = np.round(lifetimes_treatment)

chi2_stat, p_value = log_rank_test(lifetimes_no_treatment_rounded, lifetimes_treatment_rounded)

print("=== Task 10: Log-rank test ===\n")
print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"p-value:              {p_value:.6f}")

if p_value < 0.05:
    print("\nConclusion: Reject H0 — the preventive treatment has a statistically")
    print("significant effect on the survival function (p < 0.05).")
else:
    print("\nConclusion: Fail to reject H0 — no statistically significant")
    print("difference in survival between treatment and no treatment.")