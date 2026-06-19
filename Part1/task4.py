import numpy as np

# Transition probability matrix
P = np.array([
    [0.9915, 0.005,  0.0025, 0,     0.001],
    [0,      0.986,  0.005,  0.004, 0.005],
    [0,      0,      0.992,  0.003, 0.005],
    [0,      0,      0,      0.991, 0.009],
    [0,      0,      0,      0,     1.0  ],
])

def simulate_woman_full(P, rng):
    """Simulate one woman, returning full state trajectory until death."""
    states = [0]
    while states[-1] != 4:
        states.append(rng.choice(5, p=P[states[-1]]))
    return np.array(states)

def meets_criteria(states):
    """
    Returns True if the woman:
    - Survives the first 12 months (is NOT in state 4 at t=12)
    - Has had cancer reappear (visited state 1,2,3 i.e. local or distant)
      within the first 12 months
    """
    # Must survive first 12 months
    if states[12] == 4:
        return False
    # Must have had recurrence (state 1=local, 2=distant, 3=both) in first 12 months
    had_recurrence = any(s in [1, 2, 3] for s in states[:13])
    return had_recurrence

# Rejection sampling until 1000 accepted simulations
N_target = 1000
rng = np.random.default_rng(seed=42)

accepted_lifetimes = []
total_simulated = 0

while len(accepted_lifetimes) < N_target:
    states = simulate_woman_full(P, rng)
    total_simulated += 1
    if len(states) > 12 and meets_criteria(states):
        accepted_lifetimes.append(len(states) - 1)  # lifetime = steps until death

accepted_lifetimes = np.array(accepted_lifetimes)

print("=== Task 4: Conditional expected lifetime ===\n")
print(f"Total simulations run:    {total_simulated}")
print(f"Accepted simulations:     {N_target}")
print(f"Acceptance rate:          {N_target/total_simulated*100:.2f}%\n")
print(f"Condition: survived first 12 months AND cancer reappeared within 12 months\n")
print(f"Conditional mean lifetime:   {accepted_lifetimes.mean():.2f} months ({accepted_lifetimes.mean()/12:.1f} years)")
print(f"Conditional median lifetime: {np.median(accepted_lifetimes):.2f} months ({np.median(accepted_lifetimes)/12:.1f} years)")
print(f"Std deviation:               {accepted_lifetimes.std():.2f} months")

# 95% confidence interval for the mean
se = accepted_lifetimes.std() / np.sqrt(N_target)
ci_low  = accepted_lifetimes.mean() - 1.96 * se
ci_high = accepted_lifetimes.mean() + 1.96 * se
print(f"95% CI for mean:             [{ci_low:.2f}, {ci_high:.2f}] months")