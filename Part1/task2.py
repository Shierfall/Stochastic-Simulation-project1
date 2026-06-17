from task1 import simulate_patients, simulate_event, P
from scipy.stats import chisquare
import numpy as np

def evaluate_at_t():
    death_state = 4  # 0-indexed
    initial_state = 0
    all_states, death_months = simulate_patients(P, 1000, initial_state)
    states = []
    for patient in all_states:
        if len(patient) > 120:
            states.append(patient[120])
        else:
            states.append(death_state)  
    states = np.array(states)
    counts = np.bincount(states, minlength=5)
    p0 = np.array([1, 0, 0, 0, 0])
    p120 = p0 @ np.linalg.matrix_power(P, 120)
    expected = 1000 * p120
    chi2, pval = chisquare(f_obs=counts, f_exp=expected)
    print(f'Chi-squared: {chi2}\nP-value: {pval}')

if __name__ == "__main__":
    evaluate_at_t()