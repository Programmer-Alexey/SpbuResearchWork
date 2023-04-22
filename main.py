import random
from ini import *
from charts import *


# print_mixture_density(mixture, [distributions[u].data() for u in range(k)], weights, left, right)
# Алгоритм Метрополиса-Гастингса
def metropolis_hastings(function, sigma, size):
    s = sum(weights)
    mean, std = 0, 1
    x0 = 0

    for ind in range(k):
        x0 += weights[ind] / s * means[ind]
    sample = [x0]

    N = NormalDistribution(mean, sigma, 1000)
    N1 = NormalDistribution(x0, sigma, 0)
    N2 = NormalDistribution(x0, sigma, 0)

    for iteration in range(size):

        x = x0 + N.generate()
        N1.mu = x

        a = function(x) * N2.density(x) / (function(x0) * N1.density(x0))
        # a = function(x) / function(x0)
        r = min(1, a)

        if random.random() < r:
            x0 = x
            N2.mu = x0
        sample.append(x0)
    return sample


# Далее идут функция критерия Хи-квадрат и функция оптимизации

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


n = 10000

weights = [0.1, 0.2, 0.3, 0.4]
means = [-2, 3, 7, 13]
stds = [1, 1, 1, 1]

k = len(weights)
distributions = []

# Вычисление левой и правой границы графика
l_ind = means.index(min(means))
left = means[l_ind] - 3 * stds[l_ind]

r_ind = means.index(max(means))
right = means[r_ind] + 3 * stds[r_ind]

# Создаем k распределений
for j in range(k):
    distributions.append(NormalDistribution(means[j], stds[j], n))

mix = Mixture(*((distributions[j].density, weights[j]) for j in range(k)))
mixture = mix.function

# Исправил Функцию графика плотности смеси
print_mixture_density(mixture, [distributions[u].data() for u in range(k)], weights, left, right)

# Оптимизация
NUM_BINS = 50
print(simulated_annealing_optimization(np.array([2]), MCMC_estimation, 10, 0.99, 500))
