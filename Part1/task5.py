
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

P = [
    [0.9915, 0.005, 0.0025, 0,     0.001],
    [0,      0.986, 0.005,  0.004, 0.005],
    [0,      0,     0.992,  0.003, 0.005],
    [0,      0,     0,      0.991, 0.009],
    [0,      0,     0,      0,     1]
]

def simulate_event(P, initial_state):
    states = [initial_state]
    months = 0
    while states[-1] != 4:
        months += 1
        current_state = states[-1]
        next_state = np.random.choice(len(P), p=P[current_state])
        states.append(next_state)
    return states, months

def simulate_patients(P, num_patients, initial_state):
    all_states = []
    all_months = []
    for _ in range(num_patients):
        states, months = simulate_event(P, initial_state)
        all_states.append(states)
        all_months.append(months)
    return all_states, all_months


def theoretical_mean(P):
    Ps = np.array(P)[:4, :4]
    pi = np.array([1, 0, 0, 0], dtype=float)
    return pi @ np.linalg.inv(np.eye(4) - Ps) @ np.ones(4)

def control_variates(X, Z, mu_Z):
    cov_XZ = np.cov(X, Z)[0, 1]
    var_Z  = np.var(Z, ddof=1)
    c = -cov_XZ / var_Z
    Y = X + c * (Z - mu_Z)
    return Y, c

def task5():
    X = []  # fraction dying <= 350 per run
    Z = []  # mean lifetime per run (control variate)

    for _ in range(100):
        _, iter_months = simulate_patients(P, 200, 0)
        iter_months = np.array(iter_months)
        X.append(np.mean(iter_months <= 350))
        Z.append(np.mean(iter_months))

    X = np.array(X)
    Z = np.array(Z)
    mu_Z = theoretical_mean(P)

    Y, c = control_variates(X, Z, mu_Z)

    cmc_var = np.var(X, ddof=1)
    cv_var  = np.var(Y, ddof=1)
    reduction = (1 - cv_var / cmc_var) * 100

    print(f'Theoretical mean lifetime: {mu_Z:.4f} months')
    print(f'CMC estimate: {np.mean(X):.4f}  -  variance: {cmc_var:.6f}')
    print(f'CV  estimate: {np.mean(Y):.4f}  -  variance: {cv_var:.6f}')
    print(f'Optimal c: {c:.4f}')
    print(f'Variance reduction: {reduction:.1f}%')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(X, bins=20, alpha=0.6, label=f'CMC  (var={cmc_var:.5f})')
    ax.hist(Y, bins=20, alpha=0.6, label=f'CV   (var={cv_var:.5f})')
    ax.axvline(np.mean(X), color='blue',   linestyle='--')
    ax.axvline(np.mean(Y), color='orange', linestyle='--')
    ax.set_xlabel('Fraction dying within 350 months')
    ax.set_ylabel('Count')
    ax.set_title('CMC vs Control Variate Estimator')
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    task5()

