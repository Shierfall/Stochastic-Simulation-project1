from task7 import Q, simulate_patients_con, simulate_event_con
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(42)
def task9():
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

    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    ax.step(
        times_treat,
        survival_treat,
        where="post",
        linewidth=2,
        color="#FF6F61",   # coral
        label="Treatment"
    )

    ax.step(
        times,
        survial,
        where="post",
        linewidth=2,
        color="#2E86AB",   # blue
        label="Control"
    )

    ax.set_title("Survival Function Comparison", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")

    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend()
    plt.tight_layout()
    plt.show()

if "__name__" == "__main__":
    task9()
