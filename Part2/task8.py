import numpy as np
from task7 import Q,simulate_event_con,simulate_patients_con
from scipy.linalg import expm
import matplotlib.pyplot as plt
from scipy.stats import kstest
def con_pmf(Q,t):
    assert Q.shape[0] == 5

    Q_s = Q[0:4,0:4]
    ones = np.ones(4)
    expo = expm(Q_s*t)
    p0 = np.array([1,0,0,0])
    return 1 - p0@expo@ones
t=200
Q = Q
con_pmf(Q,t=t)

def observed(Q,t):
    count = 0
    all_states, all_times, all_total_times = simulate_patients_con(Q,1000,initial_state=0)
    for i in all_total_times:
        if t > i:
            count = count + 1
    return count / len(all_total_times)


_, _, all_total_times = simulate_patients_con(Q,1000,initial_state=0)
theo = [con_pmf(Q,t) for t in range(int(max(all_total_times)))]
obs = [observed(Q,t) for t in range(int(max(all_total_times)))]

print("*"*50)
print('Testing with Kolomogorov-smirnov')
ks_stat, p_val = kstest(theo,obs)
if p_val < 0.05:
    print(f'P-value is {p_val}, so we Reject the null-hypothesis')
else:
    print(f'P-value is {p_val}, so we fail to reject the null-hypothesis')
print(f'Test statistic is: {ks_stat}')
x = np.arange(len(obs))

plt.figure(figsize=(8,5))
plt.plot(x, theo, label="Theoretical")
plt.plot(x, obs, linestyle='', marker='o', label="Observed", markersize=1)
plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Observed vs Theoretical")
plt.legend()
plt.grid(True)

plt.show()
