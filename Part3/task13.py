import os
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')

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

def simulate_segment(Q, start, target, duration, rng, max_tries=10000):
    for _ in range(max_tries):
        t, state = 0.0, start
        N = np.zeros((N_STATES, N_STATES))
        S = np.zeros(N_STATES)
        ok = True
        while True:
            rate_out = -Q[state, state]
            if rate_out <= 0:
                if t < duration:
                    S[state] += duration - t
                break
            dt = rng.exponential(1.0 / rate_out)
            if t + dt >= duration:
                S[state] += duration - t
                break
            S[state] += dt
            t += dt
            probs = Q[state].copy(); probs[state] = 0.0
            nxt = rng.choice(N_STATES, p=probs / rate_out)
            N[state, nxt] += 1
            state = nxt
        end_state = state
        if end_state == target:
            return N, S
    return None


def estimate_Q(Y_series, grid=48, tol=1e-3, max_iter=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    # initial guess: any valid generator with the right zero-pattern
    Qk = np.array([[-0.01, 0.004, 0.004, 0.0,  0.002],
                   [0.0,  -0.01,  0.004, 0.003, 0.003],
                   [0.0,   0.0,  -0.01,  0.005, 0.005],
                   [0.0,   0.0,   0.0,  -0.01,  0.01 ],
                   [0.0,   0.0,   0.0,   0.0,   0.0  ]])

    diffs = []
    for it in range(max_iter):
        N_tot = np.zeros((N_STATES, N_STATES))
        S_tot = np.zeros(N_STATES)

        for Y in Y_series:
            for k in range(len(Y) - 1):
                a, b = Y[k], Y[k+1]
                if a == DEATH:
                    continue
                res = simulate_segment(Qk, a, b, grid, rng)
                if res is None:
                    continue
                N, S = res
                N_tot += N
                S_tot += S

        # re-estimate
        Q_new = np.zeros((N_STATES, N_STATES))
        for i in range(N_STATES):
            if S_tot[i] > 0:
                for j in range(N_STATES):
                    if i != j:
                        Q_new[i, j] = N_tot[i, j] / S_tot[i]
                Q_new[i, i] = -Q_new[i].sum()

        diff = np.max(np.abs(Q_new - Qk))
        Qk = Q_new
        diffs.append(diff)
        print(f"iter {it}: max|dQ| = {diff:.5f}")
        if diff < tol:
            break

    return Qk, diffs

def task13():
    lifetimes, all_states, all_times, Y_series = simulate_ctmc(Q, 1000, rng)
    Q_est, diffs = estimate_Q(Y_series, grid=48, tol=1e-3, max_iter=100, rng=rng)
    print("Estimated Q:\n", Q_est)

    # convergence plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(range(len(diffs)), diffs, color='tab:purple', marker='o',
                markersize=6, linewidth=1.6)
    ax.axhline(1e-3, color='tab:orange', linestyle='--', linewidth=1.2,
               label=r'tolerance $10^{-3}$')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('max|dQ|')
    ax.set_title('Task 13: MCEM convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'task13_convergence.png'), dpi=160)

    # true vs estimated rates
    pairs = [(0, 1), (0, 2), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    labels = [f'$q_{{{i+1}{j+1}}}$' for i, j in pairs]
    true_v = [Q[i, j] for i, j in pairs]
    est_v = [Q_est[i, j] for i, j in pairs]
    x = np.arange(len(pairs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, true_v, w, color='tab:blue', label='true rate')
    ax.bar(x + w / 2, est_v, w, color='tab:green', label='MCEM estimate')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Transition rate')
    ax.set_title('Task 13: Recovered transition rates vs truth')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, 'task13_compare.png'), dpi=160)
    plt.show()
    return Q_est

if __name__ == '__main__':
    task13()