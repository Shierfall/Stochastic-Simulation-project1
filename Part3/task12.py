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
def simulate_ctmc(Q, n_women, rng, start=0, grid=48):
    lifetimes = np.zeros(n_women)
    all_states = []
    all_times  = []
    Y_series   = []
    for i in range(n_women):
        states, times = [], []
        t, state = 0.0, start
        # build full trajectory: state held on [times[k], times[k+1])
        while state != DEATH:
            states.append(state)
            times.append(t)
            rate_out = -Q[state, state]
            t += rng.exponential(1.0 / rate_out)
            probs = Q[state].copy(); probs[state] = 0.0
            state = rng.choice(N_STATES, p=probs / rate_out)
        states.append(DEATH)
        times.append(t)# death time
        times = np.array(times)

        # sample on grid 0, 48, 96, ... until death is observed
        Y_serie = []
        k = 0
        while True:
            obs_t = k * grid
            idx = np.searchsorted(times, obs_t, side='right') - 1
            obs_state = states[idx]
            Y_serie.append(obs_state)
            if obs_state == DEATH:
                break
            k += 1

        all_states.append(states)
        all_times.append(times)
        Y_series.append(Y_serie)
        lifetimes[i] = times[-1]
    return lifetimes, all_states, all_times, Y_series

def task12():
    lifetimes, all_states, all_times, Y_series = simulate_ctmc(Q, 1000, rng)
    print(Y_series)
    return Y_series

if __name__ == '__main__':
    task12()