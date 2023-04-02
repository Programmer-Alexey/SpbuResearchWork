from ini import *
import numpy as np
import matplotlib.pyplot as plt


def print_hist(mas: np.array, den=True):  # Выводит гистограмму плотности либо функции
    plt.style.use('ggplot')

    hist, bins = np.histogram(mas, bins='auto', density=den)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    plt.step(bin_centers, hist, where='mid', label='Histogram')

    density = 1 / (np.sqrt(2 * np.pi * std ** 2)) * np.exp(-((bin_centers - mean) ** 2) / (2 * std ** 2))

    plt.plot(bin_centers, density, label='Normal Distribution')
    plt_print(coord_y="Density", title="Empirical Density Function")


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


def print_density(sample):
    mu = np.mean(sample)
    sigma = np.std(sample)

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    plt.plot(x, y, label='Плотность распределения')
    plt.hist(sample, density=True, bins=20, alpha=0.5, label='Выборка')
    plt.legend()
    plt.show()


def print_mixture_density(func):
    x = np.linspace(-5, 10, 1000)

    plt.plot(x, np.vectorize(func)(x), label='Mixture Density')

    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Mixture Density of Two Normal Distributions')

    plt.legend()
    plt.show()


n = 10000
mean = 5
std = 2

for i in range(6):  # Демонстрация работы
    norm_ptr1 = init_normal(mean, std, n)
    fill(norm_ptr1)
    density1 = NormalDensity(mean=mean, std=std)
    norm1 = np.array(norm_ptr1.contents.values[:n])

    norm_ptr2 = init_normal(0, 1, n)
    fill(norm_ptr2)
    density2 = NormalDensity()
    norm2 = np.array(norm_ptr2.contents.values[:n])

    mix = Mixture([density1.func, 0.5], [density2.func, 0.5])
    mixture = mix.mixture
    print_mixture_density(mixture)
