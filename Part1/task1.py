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
    return states, months

def simulate_patients(P, num_patients, initial_state):
    all_states = []
    all_months = []
    for _ in range(num_patients):
        states, months = simulate_event(P, initial_state)
        all_states.append(states)
        all_months.append(months)
    return all_states, all_months

def task1():
    initial_state = 0
    patients = 1000

    states ,death_months = simulate_patients(P, patients, initial_state)

    death_years = np.round(np.array(death_months) / 12, 1)


    print(f'Initial state: {initial_state} and number of patients: {patients}')
    print("-"*50)
    print(f'Longest months lived : {np.max(death_months)}')
    print(f'Shortest months lived : {np.min(death_months)}')
    print(f'Mean months lived : {np.mean(death_months)}')
    print(f'Median months lived : {np.median(death_months)}')
    print("-" * 50)
    print(f'longest lived (years) : {max(death_months)}')
    print(f'shortest lived (years) : {min(death_months)}')
    print(f'mean time lived (years) : {np.mean(death_months)}')
    print("-"*50)
    state1 = 0
    for state in states:
        for event in state:
            if event == 1:
                state1 += 1
                break
    print(f'Proportion of women, where the cancer reappears locally : {(state1 / patients) * 100}%')
    plt.hist(death_years, bins=range(0, int(max(death_years) + 1), 1), edgecolor='black')
    plt.title('Distribution of Years Until Death')
    plt.xlabel('Years Until Death')
    plt.ylabel('Number of Patients')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    plt.hist(death_months, bins=20, edgecolor='black')
    plt.title('Distribution of Months Until Death')
    plt.xlabel('Months Until Death')
    plt.ylabel('Number of Patients')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


if __name__ == "__main__":
    task1() #TODO find proportion of state 1 to state 2


