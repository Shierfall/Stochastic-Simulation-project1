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

