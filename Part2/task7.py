import numpy as np

import numpy as np

Q = np.array([
    [-0.0085, 0.005 , 0.0025, 0    , 0.001],
    [0      ,-0.014 , 0.005 , 0.004, 0.005],
    [0      ,0      ,-0.008 , 0.003, 0.005],
    [0      ,0      ,0      ,-0.009, 0.009],
    [0      ,0      ,0      ,0     ,0]
])

def simulate_event(Q, initial_state=0):
    current_state = initial_state

    states = [current_state]
    times_in_states = []
    total_time = 0.0

    while current_state != 4:   # Death state

        # Time spent in current state
        rate = -Q[current_state, current_state]
        waiting_time = np.random.exponential(scale=1/rate)

        times_in_states.append(waiting_time)
        total_time += waiting_time

        # Transition probabilities
        probs = Q[current_state].copy()
        probs[current_state] = 0
        probs /= rate

        # Move to next state
        current_state = np.random.choice(len(Q), p=probs)
        states.append(current_state)

    return states, times_in_states, total_time

states, times_in_states, lifetime = simulate_event(Q)



def simulate_patients(Q, num_patients, initial_state):
    all_states = []
    all_times = []
    all_total_times = []

    for _ in range(num_patients):
        states, times, total_time = simulate_event(Q, initial_state)

        all_states.append(states)
        all_times.append(times)
        all_total_times.append(total_time)

    return all_states, all_times, all_total_times

#task7
n = 1000
count = 0

for _ in range(n):
    states, times, _ = simulate_event(Q)

    # Did the patient enter state 2 before 30.5 months?
    if any(state == 2 and t <= 30.5
           for state, t in zip(states, times)):
        count += 1

proportion = count / n

print(proportion)
