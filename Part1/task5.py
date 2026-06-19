import numpy as np

# Transition probability matrix
P = np.array([
    [0.9915, 0.005,  0.0025, 0,     0.001],
    [0,      0.986,  0.005,  0.004, 0.005],
    [0,      0,      0.992,  0.003, 0.005],
    [0,      0,      0,      0.991, 0.009],
    [0,      0,      0,      0,     1.0  ],
])

# Theoretical mean lifetime (from Task 3): E(T) = pi @ (I - Ps)^-1 @ 1
import numpy.linalg as la
Ps = P[:4, :4]
pi = np.array([1, 0, 0, 0])
theoretical_mean = pi @ la.inv(np.eye(4) - Ps) @ np.ones(4)

def simulate_batch(P, n, rng):
    """Simulate n women, return array of lifetimes."""
    lifetimes = np.empty(n, dtype=int)
    for i in range(n):
        state, t = 0, 0
        while state != 4:
            state = rng.choice(5, p=P[state])
            t += 1
        lifetimes[i] = t
    return lifetimes

# Parameters
n_women    = 200    # women per batch
n_batches  = 100    # repetitions
T_thresh   = 350    # months
rng = np.random.default_rng(seed=42)

crude_estimates    = np.empty(n_batches)
control_estimates  = np.empty(n_batches)
batch_means        = np.empty(n_batches)

for b in range(n_batches):
    lifetimes = simulate_batch(P, n_women, rng)
    # Crude Monte Carlo: fraction dying within 350 months
    crude_estimates[b] = np.mean(lifetimes <= T_thresh)
    # Control variate: mean lifetime of this batch
    batch_means[b] = lifetimes.mean()

# --- Estimate optimal coefficient c for control variate ---
# c* = -Cov(Y, Z) / Var(Z)  where Y = crude estimate, Z = batch mean lifetime
cov_matrix = np.cov(crude_estimates, batch_means)
c_star = -cov_matrix[0, 1] / cov_matrix[1, 1]

# Control variate estimator: Y_cv = Y + c*(Z - E[Z])
control_estimates = crude_estimates + c_star * (batch_means - theoretical_mean)

# --- Results ---
crude_mean   = crude_estimates.mean()
crude_var    = crude_estimates.var()
control_mean = control_estimates.mean()
control_var  = control_estimates.var()
var_reduction = (1 - control_var / crude_var) * 100

print("=== Task 5: Fraction of women dying within 350 months ===\n")
print(f"Theoretical mean lifetime (control variate mu): {theoretical_mean:.2f} months\n")
print(f"--- Crude Monte Carlo ---")
print(f"Mean estimate:  {crude_mean:.4f}")
print(f"Variance:       {crude_var:.6f}")
print(f"Std deviation:  {np.sqrt(crude_var):.6f}\n")
print(f"--- Control Variate Estimator ---")
print(f"Optimal c*:     {c_star:.6f}")
print(f"Mean estimate:  {control_mean:.4f}")
print(f"Variance:       {control_var:.6f}")
print(f"Std deviation:  {np.sqrt(control_var):.6f}\n")
print(f"Variance reduction: {var_reduction:.1f}%")
print(f"(Control variate variance is {crude_var/control_var:.1f}x smaller than crude)")