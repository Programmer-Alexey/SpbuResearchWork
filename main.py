from ini import *
from charts import *

n = 10000
mean = 10
std = 2
weights = [0.5, 0.5]

for i in range(1):
    # Генерация первой выборки и функции плотности
    norm1_cls = NormalDistribution(mean, std, n)
    norm1 = norm1_cls.data()
    density1 = NormalDensity(mean=mean, std=std)

    # Вторая выборка
    norm_ptr2 = init_normal(0, 1, n)
    fill(norm_ptr2)
    norm2 = np.array(norm_ptr2.contents.values[:n])

    # Её плотность (по умолчанию mean=0, std=1)
    density2 = NormalDensity()

    mix = Mixture([density1.func, weights[0]], [density2.func, weights[1]])
    mixture = mix.function

    print_mixture_density(mixture, (norm1, norm2))
