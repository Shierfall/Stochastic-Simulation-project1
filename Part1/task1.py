import numpy as np
import matplotlib.pyplot as plt

# Transition probability matrix
P = np.array([
    [0.9915, 0.005,  0.0025, 0,     0.001],
    [0,      0.986,  0.005,  0.004, 0.005],
    [0,      0,      0.992,  0.003, 0.005],
    [0,      0,      0,      0.991, 0.009],
    [0,      0,      0,      0,     1.0  ],
])

def simulate_woman(P, rng):
    """Simulate one woman from state 1 until death (state 5, index 4)."""
    state = 0  # state 1 = index 0
    lifetime = 0
    visited_local = False
    while state != 4:  # state 5 = index 4
        state = rng.choice(5, p=P[state])
        lifetime += 1
        if state == 1:  # state 2 = index 1 (local recurrence)
            visited_local = True
    return lifetime, visited_local

# Run simulation
N = 1000
rng = np.random.default_rng(seed=42)

lifetimes = []
local_count = 0

for _ in range(N):
    lifetime, visited_local = simulate_woman(P, rng)
    lifetimes.append(lifetime)
    if visited_local:
        local_count += 1

lifetimes = np.array(lifetimes)

# Summary statistics
print("=== Task 1 Results ===")
print(f"Mean lifetime:     {lifetimes.mean():.1f} months ({lifetimes.mean()/12:.1f} years)")
print(f"Median lifetime:   {np.median(lifetimes):.1f} months ({np.median(lifetimes)/12:.1f} years)")
print(f"Std deviation:     {lifetimes.std():.1f} months")
print(f"Min lifetime:      {lifetimes.min()} months")
print(f"Max lifetime:      {lifetimes.max()} months")
print(f"\nLocal recurrence:  {local_count} / {N} women ({100*local_count/N:.1f}%)")

# Histogram of lifetime distribution
plt.figure(figsize=(10, 5))
plt.hist(lifetimes, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
plt.xlabel("Lifetime after surgery (months)")
plt.ylabel("Number of women")
plt.title("Lifetime distribution — 1000 simulated women (discrete-time Markov chain)")
plt.axvline(lifetimes.mean(), color='tomato', linestyle='--', label=f"Mean = {lifetimes.mean():.0f} mo")
plt.axvline(np.median(lifetimes), color='orange', linestyle=':', label=f"Median = {np.median(lifetimes):.0f} mo")
plt.legend()
plt.tight_layout()
plt.savefig("task1_lifetime_histogram.png", dpi=150)
plt.show()
print("\nHistogram saved to task1_lifetime_histogram.png")