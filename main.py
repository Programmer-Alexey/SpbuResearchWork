import random
from ini import *
from charts import *

# print_mixture_density(mixture, [distributions[u].data() for u in range(k)], weights, left, right)
# Алгоритм Метрополиса-Гастингса

def metropolis_hastings(function, sigma, size):
    s = sum(weights)
    mean, std = 0, 1
    x0 = 0

    # Выбор стартового положения
    for ind in range(k):
        x0 += weights[ind] / s * means[ind]
    sample = [x0]

    # Генератор следующих чисел
    N = NormalDistribution(mean, sigma, 1000)

    # От N1 и N2 нужны только функции плотности
    N1 = NormalDistribution(x0, sigma, 0)
    N2 = NormalDistribution(x0, sigma, 0)

    for iteration in range(size):

        x = x0 + N.generate()
        N1.mu = x

        a = function(x) * N2.density(x) / (function(x0) * N1.density(x0))
        r = min(1, a)

        if random.random() < r:
            x0 = x
            N2.mu = x0
        sample.append(x0)
    return sample


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
def get_ar(function, sigma, size):
    count = 0
    s = sum(weights)
    mean, std = 0, 1
    x0 = 0

    # Выбор стартового положения
    for ind in range(k):
        x0 += weights[ind] / s * means[ind]

    # Генератор следующих чисел
    N = NormalDistribution(mean, sigma, 1000)

    # От N1 и N2 нужны только функции плотности
    N1 = NormalDistribution(x0, sigma, 0)
    N2 = NormalDistribution(x0, sigma, 0)

    for iteration in range(size):

        x = x0 + N.generate()
        N1.mu = x

        a = function(x) * N2.density(x) / (function(x0) * N1.density(x0))
        r = min(1, a)

        if random.random() < r:
            x0 = x
            N2.mu = x0
            count += 1
    return count / size

def optimal_sigma(accept):
    AR = 0
    epsilon = 0.001
    sigma_left, sigma_right = 0, 80
    sigma = 0

    while abs(AR - accept) > epsilon:
        sigma = (sigma_left + sigma_right) / 2
        AR = get_ar(mixture, sigma, 10000)
        if AR > accept:
            sigma_left = sigma
        else:
            sigma_right = sigma
    return sigma


#print_EDF(s)
# print_mixture_density(mixture, [distributions[u].data() for u in range(k)], weights, left, right)
accept = 0.5
"""while accept <= 0.4:
    sigma = optimal_sigma(accept)
    s = metropolis_hastings(mixture, sigma, 10000)
    print(accept, sigma)
    print_density(mixture, s)
    accept += 0.1"""
sigma = optimal_sigma(accept)
s = metropolis_hastings(mixture, sigma, 10000)
print_density(mixture, s)