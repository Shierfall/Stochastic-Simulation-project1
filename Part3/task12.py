import os
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

    lengths = [len(y) for y in Y_series]
    print(f"Simulated {len(Y_series)} women, observed every 48 months.")
    print("Example observed series (1-indexed):")
    for y in Y_series[:5]:
        print("   ", [s + 1 for s in y])
    print(f"Mean observations per woman: {np.mean(lengths):.2f}")
    print(f"All series end in death (state 5): {all(y[-1] == DEATH for y in Y_series)}")

    # plot: a few example women, observed grid points over the true path
    examples = [i for i, y in enumerate(Y_series)
                if 5 <= len(y) <= 11 and max(y) >= 2][:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, idx in zip(axes.ravel(), examples):
        ax.step(all_times[idx], [s + 1 for s in all_states[idx]], where='post',
                color='dimgray', linewidth=1.4, label='true CTMC path')
        Y = Y_series[idx]
        ox = [k * 48 for k in range(len(Y))]
        oy = [s + 1 for s in Y]
        ax.scatter(ox[:-1], oy[:-1], color='tab:green', s=45, zorder=5,
                   label='observation (every 48 mo)')
        ax.scatter(ox[-1], oy[-1], color='tab:red', s=70, marker='X', zorder=6,
                   label='death observed')
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_ylim(0.5, 5.5)
        ax.set_xlabel('Time (months)')
        ax.set_ylabel('State')
        ax.set_title(f'Woman #{idx}')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Observed time series vs underlying continuous path', fontsize=13)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0))
    plt.show()
    return Y_series

if __name__ == '__main__':
    task12()