import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats

Q = np.array([[-0.0085, 0.005,  0.0025, 0,      0.001],
     [0,       -0.014, 0.005,  0.004,  0.005],
     [0,        0,     -0.008, 0.003,  0.005],
     [0,        0,      0,     -0.009, 0.009],
     [0,        0,      0,      0,      0    ]])
DEATH = 4
N_STATES = len(Q)
rng = np.random.default_rng(42)
def simulate_ctmc(Q, n_women, rng, start=0):
    lifetimes = np.zeros(n_women)
    all_states = []
    all_times  = []
    for i in range(n_women):
        states, times = [], []
        t, state = 0.0, start
        while state != DEATH:
            states.append(state)
            times.append(t)
            rate_out = -Q[state, state]
            t += rng.exponential(1.0 / rate_out)
            probs = Q[state].copy(); probs[state] = 0.0
            state = rng.choice(N_STATES, p=probs / rate_out)
        all_states.append(states)
        all_times.append(np.array(times))
        lifetimes[i] = t
    return lifetimes, all_states, all_times

def task7():

    number_people = 1000
    lifetimes, all_states, all_times = simulate_ctmc(Q, number_people, rng)
    print(f"lifetimes x: {lifetimes}")
    print(50 * '-')
    mean_lifetime = np.mean(lifetimes)
    print(f"Mean lifetime: {mean_lifetime}")
    std_lifetime = np.std(lifetimes)
    print(f"Standard deviation of lifetime: {std_lifetime}")
    print(50*'-')
    CI = stats.norm.interval(0.95, loc=mean_lifetime, scale=std_lifetime/np.sqrt(number_people))
    print(f"95% Confidence Interval for mean lifetime: {CI[0]} - {CI[1]}")
    print(50*'-')
    distant_count = 0
    for lifetime, states, times in zip(lifetimes, all_states, all_times):
        if lifetime > 30.5:
            idx = np.searchsorted(times, 30.5, side='right') - 1
            if states[idx] in (2, 3):
                distant_count += 1
    print(f'Proportion with distant recurrence at 30.5 months: {distant_count / number_people:.4f}')
    plt.hist(lifetimes, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
    plt.title('Histogram of Simulated Lifetimes')
    plt.xlabel('Lifetime')
    plt.ylabel('Density')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    print(task7())