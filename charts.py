import matplotlib.pyplot as plt
import numpy as np


class RandomGenerator:
    def __init__(self, weights):
        from bisect import bisect_left
        from random import random

        self.weights = weights
        s = sum(weights)
        for ind in range(len(weights)):
            self.weights[ind] /= s

        self.index_map = []
        self.values = list(range(len(weights)))
        self.s = 0

        self.bisect_left = bisect_left
        self.random = random

        for p in weights:
            self.s += p
            self.index_map.append(self.s)

    def generate(self):
        probability = self.random()
        index = self.bisect_left(self.index_map, probability)

        return self.values[index]


def print_distribution(sample):
    plt.style.use('ggplot')

    # Сортируем выборку
    sorted_sample = np.sort(np.array(sample))

    # Рассчет частот элементов выборки
    proportions = np.arange(1, len(sorted_sample) + 1) / len(sorted_sample)

    # plt.step строит непрерывные графики по выборке
    plt.step(sorted_sample, proportions)

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Empirical Distribution Function')

    plt.show()


# Плотность смеси с гистограммой двух выборок
def print_mixture_density(func, arrays, weights, left, right):
    rgen = RandomGenerator(weights)
    n = len(arrays[0])

    x = np.linspace(left, right, 1000)

    plt.style.use('ggplot')
    plt.plot(x, np.vectorize(func)(x), label='Mixture Density')
    plt.hist(np.array([arrays[rgen.generate()][i] for i in range(n)]), bins=50, density=True, label="Mixture Histogram")

    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title(f'Mixture Density of {len(weights)} Normal Distributions')

    plt.legend()
    plt.show()


def print_density(func, sample):
    x = np.linspace(min(sample) - 1, max(sample) + 1, 1000)

    plt.style.use('ggplot')
    plt.plot(x, np.vectorize(func)(x), label='Mixture Density')
    plt.hist(np.array(sample), bins=100, density=True, label="Mixture Histogram")

    plt.xlabel('x')
    plt.ylabel('Density')

    plt.legend()
    plt.show()
