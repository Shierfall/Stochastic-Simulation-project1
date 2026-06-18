import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.linalg import expm
from task7 import *

Qs   = Q[:4, :4]
p0   = np.array([1, 0, 0, 0], dtype=float)
ones = np.ones(4)

def theoretical_cdf(t):
    t = np.atleast_1d(t)
    return np.array([1.0 - p0 @ expm(Qs * ti) @ ones for ti in t])

def task8():
    n_women = 1000
    lifetimes, _, _ = simulate_ctmc(Q, n_women, rng)

    # KS test
    ks_stat, p_val = stats.kstest(lifetimes, theoretical_cdf)
    print(50 * '-')
    print(f'KS statistic: {ks_stat:.4f}')
    print(f'p-value:      {p_val:.4f}')
    if p_val < 0.05:
        print('REJECT NULL HYPOTHESIS')
    else:
        print('FAIL TO REJECT NULL HYPOTHESIS')

    # plot empirical vs theoretical CDF
    t_vals = np.linspace(0, lifetimes.max(), 500)
    theoretical = np.array([theoretical_cdf(t) for t in t_vals])

    sorted_lifetimes = np.sort(lifetimes)
    empirical = np.arange(1, n_women + 1) / n_women

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(sorted_lifetimes, empirical, label='Empirical CDF', where='post')
    ax.plot(t_vals, theoretical, color='purple', linewidth=1, label='Theoretical CDF')
    ax.set_xlabel('Lifetime (months)')
    ax.set_ylabel('F(t)')
    ax.set_title('Task 8: Empirical vs Theoretical Lifetime CDF')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    task8()
