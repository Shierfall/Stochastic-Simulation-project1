import numpy as np

# =============================================================================
# True Q (only used to GENERATE the Task 12 data — the algorithm itself
# never gets to see this; it only sees the sparse observations)
# =============================================================================
Q_true = np.array([
    [-0.0085, 0.005,  0.0025, 0,     0.001],
    [0,      -0.014,  0.005,  0.004, 0.005],
    [0,       0,     -0.008,  0.003, 0.005],
    [0,       0,      0,     -0.009, 0.009],
    [0,       0,      0,      0,     0    ],
])

N_STATES = 5
DEATH = 4


# =============================================================================
# Step 0 (from Task 12): simulate full trajectories and reduce to sparse
# observations every `interval` months.
# =============================================================================
def simulate_ctmc_full_trajectory(Q, rng, start_state=0):
    state = start_state
    t = 0.0
    trajectory = [(t, state)]
    while state != DEATH:
        rate = -Q[state, state]
        t += rng.exponential(1 / rate)
        probs = Q[state].copy()
        probs[state] = 0
        probs = probs / probs.sum()
        state = rng.choice(N_STATES, p=probs)
        trajectory.append((t, state))
    return trajectory


def state_at_time(trajectory, query_t):
    state = trajectory[0][1]
    for (t_enter, s) in trajectory:
        if t_enter <= query_t:
            state = s
        else:
            break
    return state


def get_observations(trajectory, interval=48):
    observations = []
    t = 0
    while True:
        s = state_at_time(trajectory, t)
        observations.append(s)
        if s == DEATH:
            break
        t += interval
    return observations


# =============================================================================
# Step 1a: simulate a single segment of FIXED duration, starting from a
# given state, using a given Q. Returns the local (time, state) path and
# the state reached at the end of the segment.
# =============================================================================
def simulate_segment(start_state, duration, Q, rng):
    state = start_state
    t = 0.0
    path = [(0.0, state)]

    while True:
        if state == DEATH:
            # Already dead - nothing more happens, stays dead for the rest
            # of the segment.
            break
        rate = -Q[state, state]
        dt = rng.exponential(1 / rate)
        if t + dt >= duration:
            # No more jumps before the segment ends
            break
        t += dt
        probs = Q[state].copy()
        probs[state] = 0
        probs = probs / probs.sum()
        state = rng.choice(N_STATES, p=probs)
        path.append((t, state))

    final_state = state
    return path, final_state


# =============================================================================
# Step 1b: rejection sampling wrapper. Keep simulating segments of fixed
# duration starting from `start_state` until the simulated end state matches
# the OBSERVED end state. Return the accepted path.
# =============================================================================
def simulate_segment_conditional(start_state, end_state, duration, Q, rng,
                                  max_attempts=200_000):
    # Special case: death is absorbing. If a woman is observed dead at the
    # end of an interval, ANY path that reaches death before/at `duration`
    # and stays dead is consistent - including paths that died exactly at
    # the start. We still just rejection-sample normally; this works fine
    # because simulate_segment naturally produces many death-reaching paths
    # when Q drives the process toward state 4 (death).
    for _ in range(max_attempts):
        path, final_state = simulate_segment(start_state, duration, Q, rng)
        if final_state == end_state:
            return path
    raise RuntimeError(
        f"Rejection sampling failed to match start={start_state}, "
        f"end={end_state}, duration={duration} after {max_attempts} attempts. "
        f"Q may be a poor/degenerate guess for this transition."
    )


# =============================================================================
# Step 1c: reconstruct one woman's FULL trajectory by stitching together
# accepted segments between each pair of consecutive observations.
# =============================================================================
def reconstruct_trajectory(observations, interval, Q, rng):
    """
    observations: list of states, e.g. [0, 0, 1, 3, 4]
                  observed at times 0, interval, 2*interval, ...
    Returns a full (time, state) trajectory consistent with all observations.
    """
    full_trajectory = [(0.0, observations[0])]
    t_offset = 0.0

    for k in range(len(observations) - 1):
        start_state = observations[k]
        end_state = observations[k + 1]

        if start_state == DEATH:
            # Already dead at the start of this interval - nothing to simulate,
            # she just stays dead (this only happens for "extra" observations
            # past death, which our get_observations() doesn't produce, but
            # guard against it anyway).
            t_offset += interval
            continue

        segment_path = simulate_segment_conditional(
            start_state, end_state, interval, Q, rng
        )

        # Append segment to full trajectory, shifting local times by t_offset.
        # Skip the first point of the segment (it's the same as the last
        # point already in full_trajectory).
        for (local_t, s) in segment_path[1:]:
            full_trajectory.append((t_offset + local_t, s))

        t_offset += interval

        if end_state == DEATH:
            break

    return full_trajectory


# =============================================================================
# Step 2: summarize a full trajectory into sojourn times S_i and
# transition counts N_ij.
# =============================================================================
def summarize_trajectory(trajectory, N_matrix, S_vector):
    """
    Accumulates into N_matrix (N_STATES x N_STATES) and S_vector (N_STATES,)
    IN PLACE, based on one full (time, state) trajectory.
    """
    for i in range(len(trajectory) - 1):
        t_curr, s_curr = trajectory[i]
        t_next, s_next = trajectory[i + 1]

        sojourn = t_next - t_curr
        S_vector[s_curr] += sojourn

        if s_curr != s_next:
            N_matrix[s_curr, s_next] += 1


# =============================================================================
# Step 3: compute Q(k+1) from N_ij and S_i using equation (2).
# =============================================================================
def update_Q(N_matrix, S_vector):
    Q_new = np.zeros((N_STATES, N_STATES))
    for i in range(N_STATES):
        if S_vector[i] > 0:
            for j in range(N_STATES):
                if i != j:
                    Q_new[i, j] = N_matrix[i, j] / S_vector[i]
        Q_new[i, i] = -Q_new[i].sum()
    return Q_new


# =============================================================================
# Main Monte Carlo EM loop
# =============================================================================
def monte_carlo_em(observations_list, interval, Q_init, rng,
                    tol=1e-3, max_iter=50, verbose=True):
    Q_k = Q_init.copy()

    for iteration in range(max_iter):
        N_matrix = np.zeros((N_STATES, N_STATES))
        S_vector = np.zeros(N_STATES)

        # --- Step 1 & 2: reconstruct trajectories for all women, summarize ---
        for obs in observations_list:
            traj = reconstruct_trajectory(obs, interval, Q_k, rng)
            summarize_trajectory(traj, N_matrix, S_vector)

        # --- Step 3: update Q ---
        Q_next = update_Q(N_matrix, S_vector)

        diff = np.max(np.abs(Q_k - Q_next))
        if verbose:
            print(f"Iteration {iteration+1}: max|Q_k - Q_(k+1)| = {diff:.6f}")

        Q_k = Q_next

        if diff < tol:
            if verbose:
                print(f"\nConverged after {iteration+1} iterations.")
            break
    else:
        if verbose:
            print(f"\nReached max_iter={max_iter} without converging below tol={tol}.")

    return Q_k


# =============================================================================
# Run everything: generate Task 12 data, then estimate Q back from it
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    # --- Step A: generate the 1000 women's sparse observation data (Task 12) ---
    N_WOMEN = 1000
    INTERVAL = 48

    observations_list = []
    for _ in range(N_WOMEN):
        traj = simulate_ctmc_full_trajectory(Q_true, rng)
        obs = get_observations(traj, interval=INTERVAL)
        observations_list.append(obs)

    print(f"Generated {N_WOMEN} women's observation time series.")
    print(f"Example: {observations_list[0]}\n")

    # --- Step B: choose an initial guess Q(0) ---
    # A simple, slightly-wrong starting guess: small uniform off-diagonal
    # rates that respect the same ZERO-PATTERN as Q_true (no point guessing
    # rates for transitions that are structurally impossible, e.g. state 5
    # back to state 1). Adjust this to whatever you find reasonable.
    Q_init = np.array([
        [0,      0.005, 0.005, 0,     0.005],
        [0,      0,     0.005, 0.005, 0.005],
        [0,      0,     0,     0.005, 0.005],
        [0,      0,     0,     0,     0.005],
        [0,      0,     0,     0,     0    ],
    ], dtype=float)
    for i in range(N_STATES):
        Q_init[i, i] = -Q_init[i].sum()

    # --- Step C: run the Monte Carlo EM algorithm ---
    Q_estimated = monte_carlo_em(
        observations_list, INTERVAL, Q_init, rng, tol=1e-3, max_iter=50
    )

    # --- Step D: compare estimated Q to the true Q used to generate the data ---
    print("\n=== True Q ===")
    print(np.round(Q_true, 5))
    print("\n=== Estimated Q ===")
    print(np.round(Q_estimated, 5))
    print("\n=== Absolute difference ===")
    print(np.round(np.abs(Q_true - Q_estimated), 5))