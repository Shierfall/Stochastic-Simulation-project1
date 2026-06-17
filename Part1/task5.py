import numpy as np
from task1 import simulate_event, P
P_s = np.array([
    [0.9915, 0.005, 0.0025, 0    ],
    [0,      0.986, 0.005,  0.004],
    [0,      0,     0.992,  0.003],
    [0,      0,     0,      0.991],
    ])
pi  = np.array([1, 0, 0, 0])
p_s = np.array([0.001, 0.005, 0.005, 0.009])  
n_women, n_batches = 200, 100
mu = pi @ np.linalg.inv(np.eye(4) - P_s) @ np.ones(4) + 1   # known control mean

fracs, means = [], []
for _ in range(n_batches):
    deaths = np.array([simulate_event(P, 0)[1] for _ in range(n_women)])
    fracs.append(np.mean(deaths <= 350))   # died within first 350 months
    means.append(np.mean(deaths))          # control variate

X, C = np.array(fracs), np.array(means)
cov = np.cov(X, C, ddof=1)
c = cov[0, 1] / cov[1, 1]
Z = X - c * (C - mu)

print(f"crude   var: {X.var(ddof=1):.3e}")
print(f"control var: {Z.var(ddof=1):.3e}  ({X.var(ddof=1)/Z.var(ddof=1):.1f}x reduction)")