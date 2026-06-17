import math

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
    if months < 12:
        return None,None
    state12 = states[:12]
    for state in state12:
        if state in [1,2,3]:
            return states,months
    return None,None

def simulate_patients(P, target_patients, initial_state):
    all_states = []
    all_months = []
    while target_patients > len(all_months):
        states, months = simulate_event(P, initial_state)
        if states is None:
            continue
        all_states.append(states)
        all_months.append(months)
    return all_states, all_months

def task4():
    states, months = simulate_patients(P, 1000, 0)

    print(f'Average Lifetime where cancer has reappeared within the first 12 months: {np.mean(months)}')
    print(f'Median Lifetime where cancer has reappeared within the first 12 months: {np.median(months)}')

    plt.hist(np.array(months)/12, bins=30, edgecolor='black')
    plt.show()

if __name__ == "__main__":
    task4()


