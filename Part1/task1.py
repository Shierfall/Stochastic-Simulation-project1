import math
from scipy.stats import chisquare
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
    all_states, death_months = simulate_patients(P, 1000, initial_state)
    death_years = np.round(np.array(death_months) / 12, 1)
    print(np.max(death_months))
    print(np.min(death_months))
    print(np.mean(death_months))

    print(np.max(death_years))
    print(np.min(death_years))
    print(np.mean(death_years))

    count = 0
    for patient in all_states:
        for i in patient:
            if i == 1:
                count = count +1
                break
        
    print(f'Proportion of local reoccurence: {count / len(all_states)*100}%')
            

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
