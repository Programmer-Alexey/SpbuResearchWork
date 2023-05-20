import numpy as np

def chi_square_agreement(observed_values, target_distribution_func, bin_edges):
    observed_freq, _ = np.histogram(observed_values, bins=bin_edges)
    observed_prob = observed_freq / len(observed_values)

    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    expected_prob = [target_distribution_func(x) for x in bin_centers]

    chi_square_statistic = sum((observed_prob - expected_prob) ** 2 / expected_prob)

    return chi_square_statistic


def MCMC_estimation(v):
    sigma = v[0]
    sample = metropolis_hastings(mixture, sigma, n)
    return chi_square_agreement(sample, mixture, np.linspace(left, right, NUM_BINS + 1))


def simulated_annealing_optimization(start, fitness, temperature, alpha, iters):
    current = start
    best_state = current

    for i in range(iters):
        next = current + np.random.uniform(-1, 1)
        while next <= 0:
            next = current + np.random.uniform(-1, 1)

        current_energy = fitness(current)
        neighbor_energy = fitness(next)

        delta = neighbor_energy - current_energy

        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
            current = next

        if fitness(current) < fitness(best_state):
            best_state = current

        temperature *= alpha

    return best_state