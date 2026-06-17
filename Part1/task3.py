from task1 import *
from task2 import *
P_s = np.array([
    [0.9915, 0.005, 0.0025, 0    ],
    [0,      0.986, 0.005,  0.004],
    [0,      0,     0.992,  0.003],
    [0,      0,     0,      0.991],
])
pi  = np.array([1, 0, 0, 0])              # distribution at t=0
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
death_state = 4  # 0-indexed
initial_state = 0
all_states, death_months = simulate_patients(P, 1000, initial_state)
states = []
for patient in all_states:
    if len(patient) > 120:
        states.append(patient[120])
    else:
        states.append(death_state)  

states = np.array(states)
mean = pi @ np.linalg.inv(np.eye(4) - P_s) @ ones
T_max = max(death_months)
#evaluate at all t
states = np.array(states)
counts = np.bincount(states, minlength=5)
dist_theory = np.array([0.0] + [pmf(m - 1) for m in range(1, T_max + 1)])

# empirical PMF indexed by death_month
def task3():
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

def task3_test(bin_width=48):
    N = len(death_months)
    edges = np.arange(0, T_max + bin_width, bin_width)

    # observed counts per bin
    obs, _ = np.histogram(death_months, bins=edges)

    # expected prob per bin (death_month = t+1, so month m uses pmf(m-1))
    exp_prob = np.array([sum(pmf(m - 1) for m in range(lo, hi))
                         for lo, hi in zip(edges[:-1], edges[1:])])
    # dump leftover tail mass into last bin so probs sum to 1
    exp_prob[-1] += 1.0 - exp_prob.sum()
    exp = N * exp_prob

    # merge bins until every expected count >= 5
    obs, exp = list(obs), list(exp)
    i = 0
    while i < len(exp):
        if exp[i] < 5 and len(exp) > 1:
            j = i - 1 if i == len(exp) - 1 else i + 1
            exp[j] += exp[i]; obs[j] += obs[i]
            del exp[i]; del obs[i]
            i = max(i - 1, 0)
        else:
            i += 1
    obs, exp = np.array(obs), np.array(exp)
    exp *= obs.sum() / exp.sum()   # fix float drift so sums match

    chi2, pval = chisquare(f_obs=obs, f_exp=exp)
    print(f"bins: {len(obs)}  chi2: {chi2:.4f}  p: {pval:.4f}  dof: {len(obs)-1}")
    return chi2, pval

if "__name__" == "__main__":
    task3()
    task3_test()