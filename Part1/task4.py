import numpy as np
from task1 import simulate_event, P
import matplotlib.pyplot as plt



def simulate_task4(P, n_accept, initial_state=0):
    lifetimes = []
    n_tried = 0
    while len(lifetimes) < n_accept:
        states, months = simulate_event(P, initial_state)
        n_tried += 1
        if len(states) > 12 and states[12] in (1, 2, 3):
            lifetimes.append(months)
    return np.array(lifetimes), n_tried

lifetimes, n_tried = simulate_task4(P=P,n_accept=1000,initial_state=0)
plt.hist(lifetimes,bins=40)
plt.show()
print(f'Lifetime expectancy: {lifetimes.mean()}')
if __name__ == "__main__":
    simulate_task4()