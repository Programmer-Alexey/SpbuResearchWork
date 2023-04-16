from ini import *
from charts import *

n = 1000000

weights = [0.1, 0.2, 0.3, 0.4]
means = [0, 3, 7, 13]
stds = [1, 1, 1, 1]

k = len(weights)
distributions = []

l_ind = means.index(min(means))
left = means[l_ind] - 3 * stds[l_ind] - 1

r_ind = means.index(max(means))
right = means[r_ind] + 3 * stds[r_ind] + 1


for i in range(1):
    for j in range(k):
        distributions.append(NormalDistribution(means[j], stds[j], n))

    mix = Mixture(*((distributions[j].density, weights[j]) for j in range(k)))
    mixture = mix.function

    print_mixture_density(mixture, [distributions[u].data() for u in range(k)], weights, left, right)
