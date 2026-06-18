import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from task7 import simulate_ctmc, Q, rng

Q_treat = np.array([
    [-(0.0025 + 0.00125 + 0.001), 0.0025,  0.00125, 0,     0.001],
    [0,  -(0.002 + 0.005),         0,       0.002,   0.005],
    [0,   0,  -(0.003 + 0.005),    0.003,   0.005],
    [0,   0,   0,  -0.009,          0.009],
    [0,   0,   0,   0,               0   ],
])

def logrank_test(t1, t2):
    all_times = np.unique(np.concatenate([t1, t2]))
    O_minus_E = 0.0
    var_sum    = 0.0
    for t in all_times:
        n1 = np.sum(t1 >= t)
        n2 = np.sum(t2 >= t)
        d1 = np.sum(t1 == t)
        d2 = np.sum(t2 == t)
        n  = n1 + n2
        d  = d1 + d2
        if n < 2:
            continue
        e1 = n1 * d / n
        O_minus_E += d1 - e1
        var_sum+= n1*n2*d*(n-d)/(n**2* (n - 1))
    stat= O_minus_E**2 / var_sum
    p_value = chi2.sf(stat, df=1)
    return stat, p_value

def kaplan_meier(lifetimes):
    sorted_t=np.sort(lifetimes)
    N=len(lifetimes)
    S=(N-np.arange(1, N + 1))/N
    return sorted_t, np.concatenate([[1.0], S])

def task9():
    n_women = 1000
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(99)

    lifetimes_no_treat, _, _ = simulate_ctmc(Q,n_women, rng1)
    lifetimes_treat,_, _ = simulate_ctmc(Q_treat, n_women, rng2)
#gang
    t_no, S_no = kaplan_meier(lifetimes_no_treat)
    t_tr, S_tr = kaplan_meier(lifetimes_treat)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(np.concatenate([[0], t_no]), S_no, where='post', label='No treatment', color='red')
    ax.step(np.concatenate([[0], t_tr]), S_tr, where='post', label='Treatment', linestyle='--', color='blue')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('S(t)')
    ax.set_title('Task 9: Kaplan-Meier Survival Functions')
    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    print(f'Median survival no treatment: {np.median(lifetimes_no_treat):.1f} months')
    print(f'Median survival treatment: {np.median(lifetimes_treat):.1f} months')
    print("-"*50)
    print(f'Mean survival no treatment: {np.mean(lifetimes_no_treat):.1f} months')
    print(f'Mean survival treatment: {np.mean(lifetimes_treat):.1f} months')
    print("-"*50)
    stat, p_val = logrank_test(lifetimes_no_treat, lifetimes_treat)
    print(f'Log-rank statistic: {stat:.4f}')
    print(f'p-value: {p_val:.2e}')
    if p_val < 0.05:
        print('REJECT H0: treatment has significant effect on survival.')
    else:
        print('FAIL TO REJECT H0.')


if __name__ == '__main__':
    task9()
