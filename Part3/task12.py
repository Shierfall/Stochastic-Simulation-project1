import numpy as np
import matplotlib.pyplot as plt


Q = np.array([
    [-0.0085, 0.005,  0.0025, 0,     0.001],
    [0,      -0.014,  0.005,  0.004, 0.005],
    [0,       0,     -0.008,  0.003, 0.005],
    [0,       0,      0,     -0.009, 0.009],
    [0,       0,      0,      0,     0    ],
])


def simulate_ctmc_full_trajectory(Q, rng):
    """
    Simulate one woman's full continuous-time trajectory until death.
    Returns a list of (time_entered, state) tuples, e.g.
    [(0.0, 0), (14.2, 1), (88.7, 3), (340.1, 4)]
    """
    state = 0
    t = 0.0
    trajectory = [(t, state)]
    while state != 4:
        rate = -Q[state, state]
        t += rng.exponential(1 / rate)
        probs = Q[state].copy()
        probs[state] = 0
        probs = probs / probs.sum()
        state = rng.choice(5, p=probs)
        trajectory.append((t, state))
    return trajectory


def state_at_time(trajectory, query_t):
    """Return the state a woman was in at a given query time."""
    state = trajectory[0][1]
    for (t_enter, s) in trajectory:
        if t_enter <= query_t:
            state = s
        else:
            break
    return state


def get_observations(trajectory, interval=48):
    """
    Extract the observed state every `interval` months: Y = (X(0), X(48),
    X(96), ...), stopping once death (state 4) has been observed.
    """
    death_time = trajectory[-1][0]
    observations = []
    t = 0
    while True:
        s = state_at_time(trajectory, t)
        observations.append(s)
        if s == 4:
            break
        t += interval
    return observations


# --- Simulate 1000 women and extract their observation time series ---
N = 1000
rng = np.random.default_rng(seed=42)

all_observations = []  # list of lists, one per woman
all_trajectories = []  # keep full trajectories too (useful for Task 13 validation)

for i in range(N):
    traj = simulate_ctmc_full_trajectory(Q, rng)
    obs = get_observations(traj, interval=48)
    all_observations.append(obs)
    all_trajectories.append(traj)


# --- Inspect results ---
print("=== Task 12: Sparse observations every 48 months ===\n")
print(f"Simulated {N} women.\n")
print("Example observation time series (first 5 women):")
for i in range(5):
    print(f"  Woman {i+1}: {all_observations[i]}")

obs_lengths = [len(obs) for obs in all_observations]
print(f"\nNumber of observations per woman:")
print(f"  Min:    {min(obs_lengths)}")
print(f"  Max:    {max(obs_lengths)}")
print(f"  Mean:   {np.mean(obs_lengths):.2f}")
print(f"  Median: {np.median(obs_lengths):.1f}")

# Sanity check: every time series should end in state 4 (death)
all_end_in_death = all(obs[-1] == 4 for obs in all_observations)
print(f"\nAll time series end in state 5 (death)? {all_end_in_death}")

# Save data for use in Task 13
np.save("task12_observations.npy", np.array(all_observations, dtype=object), allow_pickle=True)
print("\nObservations saved to task12_observations.npy (for use in Task 13)")




lifetimes = np.array([traj[-1][0] for traj in all_trajectories])
n_observations = np.array([len(obs) for obs in all_observations])
 
# =============================================================================
# Plot 1: Histogram of true lifetimes across all 1000 women
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
 
ax = axes[0, 0]
ax.hist(lifetimes, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
ax.axvline(lifetimes.mean(), color='tomato', linestyle='--',
           label=f"Mean = {lifetimes.mean():.0f} mo")
ax.set_xlabel("True lifetime (months)")
ax.set_ylabel("Number of women")
ax.set_title("True lifetime distribution (1000 women)")
ax.legend()
 
# =============================================================================
# Plot 2: Number of checkup observations per woman
# =============================================================================
ax = axes[0, 1]
max_obs = n_observations.max()
ax.hist(n_observations, bins=np.arange(1, max_obs + 2) - 0.5,
        color='seagreen', edgecolor='white', linewidth=0.5)
ax.set_xlabel("Number of checkup observations")
ax.set_ylabel("Number of women")
ax.set_title("Checkups per woman (every 48 months)")
ax.set_xticks(range(1, max_obs + 1, 2))
 
# =============================================================================
# Plot 3: Overlaid sample of individual trajectories (true vs observed)
# =============================================================================
ax = axes[1, 0]
sample_idx = rng.choice(N, size=25, replace=False)
for i in sample_idx:
    traj = all_trajectories[i]
    times = [t for t, s in traj]
    states = [s for t, s in traj]
    # step plot
    ax.step(times, states, where='post', alpha=0.3, color='steelblue', linewidth=1)
ax.set_xlabel("Time (months)")
ax.set_ylabel("State")
ax.set_yticks(range(5))
ax.set_yticklabels(["1: none", "2: local", "3: distant", "4: both", "5: death"])
ax.set_title("Sample of 25 true trajectories")
ax.set_xlim(0, 600)
 
# =============================================================================
# Plot 4: State distribution at each checkup time across population
# =============================================================================
ax = axes[1, 1]
checkup_times = np.arange(0, 600, 48)
state_props = np.zeros((len(checkup_times), 5))
 
for idx, ct in enumerate(checkup_times):
    states_at_ct = []
    for traj in all_trajectories:
        if ct <= traj[-1][0]:  # still alive or just died by this checkup
            states_at_ct.append(state_at_time(traj, ct))
        else:
            states_at_ct.append(4)  # already dead
    counts = np.bincount(states_at_ct, minlength=5)
    state_props[idx] = counts / N
 
state_names = ["1: none", "2: local", "3: distant", "4: both", "5: death"]
colors = ['#378ADD', '#1D9E75', '#F2A623', '#D85A30', '#888780']
bottom = np.zeros(len(checkup_times))
for s in range(5):
    ax.bar(checkup_times, state_props[:, s], bottom=bottom, width=40,
           label=state_names[s], color=colors[s])
    bottom += state_props[:, s]
ax.set_xlabel("Time (months)")
ax.set_ylabel("Proportion of population")
ax.set_title("State distribution over time (population level)")
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=9)
 
plt.tight_layout()
plt.savefig("task12_population_visualization.png", dpi=150, bbox_inches='tight')
plt.show()
 
print("=== Summary ===")
print(f"Simulated {N} women")
print(f"Mean lifetime: {lifetimes.mean():.1f} months")
print(f"Mean checkups per woman: {n_observations.mean():.2f}")
print(f"Min/Max checkups: {n_observations.min()} / {n_observations.max()}")
print("\nPlot saved to task12_population_visualization.png")
