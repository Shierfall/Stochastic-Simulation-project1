import numpy as np
import matplotlib.pyplot as plt

# Original (no treatment) Q matrix
Q_no_treatment = np.array([
    [-0.0085, 0.005,  0.0025, 0,     0.001],
    [0,      -0.014,  0.005,  0.004, 0.005],
    [0,       0,     -0.008,  0.003, 0.005],
    [0,       0,      0,     -0.009, 0.009],
    [0,       0,      0,      0,     0    ],
])

# Treatment Q matrix (off-diagonal rates given; diagonal computed so rows sum to 0)
Q_treatment = np.array([
    [0,      0.0025, 0.00125, 0,     0.001],
    [0,      0,      0,       0.002, 0.005],
    [0,      0,      0,       0.003, 0.005],
    [0,      0,      0,       0,     0.009],
    [0,      0,      0,       0,     0    ],
], dtype=float)
# Fill diagonal: q_ii = -sum of off-diagonal entries in row i
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

def kaplan_meier(lifetimes):
    """
    Simple Kaplan-Meier estimator (no censoring in this model, since all
    women are followed until death). S(t) = (N - d(t)) / N at each death time.
    """
    sorted_times = np.sort(lifetimes)
    N = len(lifetimes)
    survival_prob = 1 - np.arange(1, N + 1) / N
    return sorted_times, survival_prob

# --- Simulate both groups ---
N = 1000
rng = np.random.default_rng(seed=42)

lifetimes_no_treatment = np.array([simulate_ctmc_woman(Q_no_treatment, rng) for _ in range(N)])
lifetimes_treatment    = np.array([simulate_ctmc_woman(Q_treatment, rng) for _ in range(N)])

t_no_treat, s_no_treat = kaplan_meier(lifetimes_no_treatment)
t_treat, s_treat       = kaplan_meier(lifetimes_treatment)

print("=== Task 9: Treatment effect on survival ===\n")
print(f"Mean lifetime (no treatment): {lifetimes_no_treatment.mean():.2f} months")
print(f"Mean lifetime (treatment):    {lifetimes_treatment.mean():.2f} months")
print(f"Difference:                   {lifetimes_treatment.mean() - lifetimes_no_treatment.mean():.2f} months")

# --- Plot Kaplan-Meier curves ---
plt.figure(figsize=(10, 5))
plt.step(t_no_treat, s_no_treat, where='post', color='tomato', linewidth=1.5, label='No treatment')
plt.step(t_treat, s_treat, where='post', color='seagreen', linewidth=1.5, label='Preventive treatment')
plt.xlabel("Time after surgery (months)")
plt.ylabel("Survival probability S(t)")
plt.title("Task 9: Kaplan-Meier survival curves — treatment vs no treatment")
plt.legend()
plt.tight_layout()
plt.savefig("task9_kaplan_meier.png", dpi=150)
plt.show()
print("\nPlot saved to task9_kaplan_meier.png")
print("\nVisually compare the two curves: if the treatment curve stays above")
print("the no-treatment curve, the treatment appears to extend survival.")