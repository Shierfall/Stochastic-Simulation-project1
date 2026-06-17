
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


def control_variates():
    return

def task5():
    fractions = []
    mean_variates = []

    num_times = 100

    for _ in range(num_times):
        iter_states, iter_months = simulate_patients(P, 200, 0)
        iter_months = np.asarray(iter_months)
        fraction = np.mean(iter_months <= 350)
        fractions.append(fraction)
        mean_variates.append(np.mean(iter_months))

    print(fractions)
    print(mean_variates)

    print(f"Variance prior to reduction: {np.var(fractions):.4f}")


if __name__ == "__main__":
    task5()

