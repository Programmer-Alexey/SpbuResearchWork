import ctypes
from math import exp, pi


# Структура, где хранятся все данные о (нормальном) распределении
class Normal(ctypes.Structure):
    _fields_ = [("second", ctypes.c_char),
                ("mu", ctypes.c_double),
                ("sigma", ctypes.c_double),
                ("u", ctypes.c_double),  # u,v - рандомные нормально распределенные числа, s = u^2 + v^2
                ("v", ctypes.c_double),
                ("s", ctypes.c_double),
                ("n", ctypes.c_int),
                ("values", ctypes.POINTER(ctypes.c_double))]  # Значения


# Обертка для функции плотности
class NormalDensity:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def func(self, x):
        return 1 / (exp(((x - self.mean) / self.std) ** 2 / 2) * (2 * pi) ** 0.5 * self.std)


class Mixture:
    """
        Вход: Функция плотности произвольного распределения и ее вес
        Выход: Функция смеси
    """
    def __init__(self, *args):
        self.args = args

    def function(self, x):
        s = 0
        for pair in self.args:
            f, w = pair
            s += w * f(x)
        return s


def normal_distribution(mu, sigma, s_ptr):
    return lib.normal_distribution(mu, sigma, s_ptr)


def fill(s_ptr):
    lib.fill(s_ptr)


def init_normal(mu, sigma, n):
    return lib.init_normal(mu, sigma, n)


# Загрузка библиотеки
lib = ctypes.CDLL("./distributions.dll")

# Определение ctypes для функций из dll
lib.normal_distribution.restype = ctypes.c_double
lib.normal_distribution.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.POINTER(Normal)]

lib.fill.argtypes = [ctypes.POINTER(Normal)]

lib.init_normal.restype = ctypes.POINTER(Normal)
lib.init_normal.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
