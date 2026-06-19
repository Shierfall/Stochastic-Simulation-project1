from task9 import *
from scipy.stats import logrank
Q_treat = np.array([
    [-(0.0025 + 0.00125 + 0.001), 0.0025,  0.00125, 0,     0.001],
    [0,  -(0.002 + 0.005),         0,       0.002,   0.005],
    [0,   0,  -(0.003 + 0.005),    0.003,   0.005],
    [0,   0,   0,  -0.009,          0.009],
    [0,   0,   0,   0,               0   ],
]) 
P=Q
_, _, all_total_times_treat = simulate_patients_con(Q_treat,1000,0)
_, _, all_total_times = simulate_patients_con(Q,1000,0)


times_treat = np.arange(int(max(all_total_times_treat)))
times = np.arange(int(max(all_total_times)))

survival_treat = [
    np.mean(np.array(all_total_times_treat) > t)
    for t in times_treat
]

survial = [
    np.mean(np.array(all_total_times) > t)
    for t in times
]

result = logrank(survial,survival_treat)
print("*"*10, "testing with the log-rank","*"*10)




pval = result.pvalue
print(f'Test statistic: {result.statistic}')

if pval < 0.05:
    print(f'P-value: {pval} is less than 0.05, so we reject the null hypothesis')
else:
    print(f'P-value: {pval} is greater than 0.05, so we fail to reject the null hypothesis')