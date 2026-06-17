import math
import scipy.stats as stats
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

def simulate_event(P, initial_state, time):
    states = [initial_state]
    months = 0
    if time:
        for _ in range(time+1):
            months += 1
            current_state = states[-1]
            next_state = np.random.choice(len(P), p=P[current_state])
            states.append(next_state)
    else:
        while states[-1] != 4:
            months += 1
            current_state = states[-1]
            next_state = np.random.choice(len(P), p=P[current_state])
            states.append(next_state)
    return states, months

def simulate_patients(P, num_patients, initial_state, time=None):
    all_states = []
    all_months = []
    for _ in range(num_patients):
        states, months = simulate_event(P, initial_state, time)
        all_states.append(states)
        all_months.append(months)
    return all_states, all_months

def probability_distribution(P, time):
    pt = np.matmul(np.array([1, 0, 0, 0, 0]),np.linalg.matrix_power(np.matrix(P),time))
    return pt


def task3():
    initial_state = 0
    num_patients = 10000
    # TODO p_value for every month

    # simulate until death
    _, lifetimes = simulate_patients(P, num_patients, initial_state, time=None)
    lifetimes = np.array(lifetimes)

    # phase-type distribution components
    Ps = np.array(P)[:4, :4]
    ps = np.array(P)[:4, 4]
    pi = np.array([1, 0, 0, 0])

    t_max = lifetimes.max()
    t_vals = np.arange(1, t_max + 1)
    pmf = np.array([pi @ np.linalg.matrix_power(Ps, t) @ ps for t in t_vals])

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lifetimes, bins=60, density=True, alpha=0.6, label='Simulated', edgecolor='black')
    ax.plot(t_vals, pmf, color='red', linewidth=1.5, label='Theoretical PMF')
    ax.set_xlabel('Lifetime (months)')
    ax.set_ylabel('Probability')
    ax.set_title('Task 3: Simulated vs Theoretical Lifetime Distribution')
    ax.legend()
    plt.tight_layout()
    plt.show()

    counts, bin_edges = np.histogram(lifetimes, bins=60)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected_counts = np.array([
        num_patients * np.sum(pmf[(t_vals >= bin_edges[i]) & (t_vals < bin_edges[i+1])])
        for i in range(len(counts))
    ])

    mask = expected_counts >= 5
    obs_valid = np.append(counts[mask], counts[~mask].sum())
    exp_valid = np.append(expected_counts[mask], expected_counts[~mask].sum())

    exp_valid = exp_valid * obs_valid.sum() / exp_valid.sum()
    t_stat, p_val = stats.chisquare(f_obs=obs_valid, f_exp=exp_valid)
    print(50 * "-")
    print(f't_stat: {t_stat:.4f}')
    print(f'p_val:  {p_val:.4f}')
    if p_val < 0.05:
        print("REJECT NULL HYPOTHESIS")
    else:
        print("FAIL TO REJECT NULL HYPOTHESIS")

if __name__ == '__main__':
    task3()