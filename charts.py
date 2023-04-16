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
"""
def print_hist(mas: np.array, mean, std, den=True):  # Выводит гистограмму плотности либо функции
    plt.style.use('ggplot')

    hist, bins = np.histogram(mas, bins='auto', density=den)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    plt.step(bin_centers, hist, where='mid', label='Histogram')

    density = 1 / (np.sqrt(2 * np.pi * std ** 2)) * np.exp(-((bin_centers - mean) ** 2) / (2 * std ** 2))

    plt.plot(bin_centers, density, label='Normal Distribution')
    plt_print(coord_y="Density", title="Empirical Density Function")
"""

def print_distribution(mas: np.array):  # Только функция распределения
    plt.style.use('ggplot')
    x = np.sort(mas)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.plot(x, y)
    plt_print()


def plt_print(coord_y="F(x)", title="Empirical Distribution Function"):
    plt.style.use('ggplot')
    plt.xlabel("x")
    plt.ylabel(coord_y)
    plt.title(title)

    fig = plt.gcf()
    fig.set_size_inches(10 / 2.54, 8 / 2.54)

    plt.show()


# Плотность по выборке
def print_density(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    plt.plot(x, y, label='Плотность распределения')
    plt.hist(sample, density=True, bins=20, alpha=0.5, label='Выборка')
    plt.legend()
    plt.show()


# Плотность смеси с гистограммой двух выборок
def print_mixture_density(func, arrays, weights, left, right):
    rgen = RandomGenerator(weights)
    n = len(arrays[0])


    x = np.linspace(left, right, 1000)

    plt.plot(x, np.vectorize(func)(x), label='Mixture Density')
    plt.hist(np.array([arrays[rgen.generate()][i] for i in range(n)]), bins=50, density=True, label="Mixture Histogram")

    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture Density of n Normal Distributions')

    plt.legend()
    plt.show()
