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



#Task 2

death_state = 4  # 0-indexed
  
initial_state = 0
all_states, death_months = simulate_patients(P, 1000, initial_state)
states = []
for patient in all_states:
    if len(patient) > 120:
        states.append(patient[120])
    else:
        states.append(death_state)  
def evaluate_at_t():
    states = np.array(states)
    counts = np.bincount(states, minlength=5)
    p0 = np.array([1, 0, 0, 0, 0])
    p120 = p0 @ np.linalg.matrix_power(P, 120)
    expected = 1000 * p120
    chi2, pval = chisquare(f_obs=counts, f_exp=expected)
    print(f'Chi-squared: {chi2}\nP-value: {pval}')


evaluate_at_t()
P_s = np.array([
    [0.9915, 0.005, 0.0025, 0    ],
    [0,      0.986, 0.005,  0.004],
    [0,      0,     0.992,  0.003],
    [0,      0,     0,      0.991],
])
pi  = np.array([1, 0, 0, 0])                    # distribution at t=0
p_s = np.array([0.001, 0.005, 0.005, 0.009])    # death column of P
ones = np.ones(4)

def pmf(t):
    P_s = np.array([
    [0.9915, 0.005, 0.0025, 0    ],
    [0,      0.986, 0.005,  0.004],
    [0,      0,     0.992,  0.003],
    [0,      0,     0,      0.991],
    ])
    pi  = np.array([1, 0, 0, 0])
    p_s = np.array([0.001, 0.005, 0.005, 0.009])  
    return pi @ np.linalg.matrix_power(P_s, t) @ p_s

# Mean: E(T) = pi @ (I - P_s)^{-1} @ 1
mean = pi @ np.linalg.inv(np.eye(4) - P_s) @ ones
T_max = max(death_months)
#evaluate at all t
states = np.array(states)
counts = np.bincount(states, minlength=5)
dist_theory = np.array([0.0] + [pmf(m - 1) for m in range(1, T_max + 1)])

# empirical PMF indexed by death_month
def task2():
    bin_width = 24  
    edges = np.arange(0, T_max + bin_width, bin_width)

    # empirical: histogram normalized to a PMF over bins
    emp_counts, _ = np.histogram(death_months, bins=edges)
    dist_emp = emp_counts / emp_counts.sum()

    # theoretical: sum pmf over each bin (death_month = t+1)
    dist_theory_binned = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # months lo+1 .. hi inclusive map to pmf indices lo .. hi-1
        dist_theory_binned.append(sum(pmf(t) for t in range(lo, hi)))
    dist_theory_binned = np.array(dist_theory_binned)

    centers = (edges[:-1] + edges[1:]) / 2
    plt.bar(centers, dist_emp, width=bin_width*0.9, alpha=0.5, label="empirical")
    plt.plot(centers, dist_theory_binned, 'o-', label="theoretical")
    plt.xlabel("death month"); plt.ylabel("P(T in bin)"); plt.legend()
    plt.show()
task2()



# if __name__ == "__main__":
#     task1() #TODO find proportion of state 1 to state 2 
