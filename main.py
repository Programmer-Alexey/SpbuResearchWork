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


n = 1000
mean = 5
std = 2

for i in range(6):  # Демонстрация работы
    norm_ptr = init_normal(mean, std, n)
    fill(norm_ptr)

    norm = np.array(norm_ptr.contents.values[:n])

    print_distribution(norm)