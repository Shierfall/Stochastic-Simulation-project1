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


def task2():
    initial_state = 0
    t = 120 # time
    pt = probability_distribution(P, t)
    states, months = simulate_patients(P, 1000, initial_state, t)
    final_state = []
    for state in states:
        final_state.append(state[-1])
    print(f'sum state: {np.sum(final_state)}')
    #chi square

    observed = np.bincount(final_state, minlength=5)
    print(observed)
    expected = np.array(pt).flatten() * 1000
    print(expected)

    t_stat, p_val = stats.chisquare(f_exp=expected, f_obs=observed)
    print(50*"-")
    print(f't_stat: {t_stat}')
    print(f'p_val: {p_val}')
    if p_val < 0.05:
        print("REJECT NULL HYPOTHESIS")
    else :
        print("FAIL TO REJECT NULL HYPOTHESIS")


if __name__ == '__main__':
    task2()