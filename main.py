import random

from ini import *
from charts import *

n = 10000

weights = [0.1, 0.2, 0.3, 0.4]
means = [-2, 3, 7, 13]
stds = [1, 1, 1, 1]

k = len(weights)
distributions = []

# Вычисление левой и правой границы графика
l_ind = means.index(min(means))
left = means[l_ind] - 3 * stds[l_ind] - 1

r_ind = means.index(max(means))
right = means[r_ind] + 3 * stds[r_ind] + 1

# Создаем m распределений
for j in range(k):
    distributions.append(NormalDistribution(means[j], stds[j], n))

mix = Mixture(*((distributions[j].density, weights[j]) for j in range(k)))
mixture = mix.function


# print_mixture_density(mixture, [distributions[u].data() for u in range(k)], weights, left, right)

def metropolis_hastings(function):
    s = sum(weights)
    mean, std = 0, 1
    x0 = 0
    max_iter = 10000

    for ind in range(k):
        x0 += weights[ind] / s * means[ind]

    sample = [x0]

    N = NormalDistribution(mean, 1, 1000)
    N1 = NormalDistribution(x0, 2, 0)
    N2 = NormalDistribution(x0, 2, 0)

    for iteration in range(max_iter):

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


sample = metropolis_hastings(mixture)
print_density(sample)
