#include "distributions.h"


void fill(normal *s) {
    for (int i = 0; i < s->n; ++i) {
        s->values[i] = normal_distribution(s->mu, s->sigma, s);
    }
}


normal *init_normal(double mu, double sigma, int n) {
    normal *s = (normal *) malloc(sizeof(normal));
    s->second = 1;
    s->mu = mu;
    s->sigma = sigma;
    s->n = n;
    s->values = (double *) malloc(n * sizeof(double));
    fill(s);
    return s;
}


double normal_distribution(double mu, double sigma, normal *st) {
    st->second = (st->second + 1) % 2;
    switch (st->second) {
        case 0:
            while (1) {
                st->u = (double) (-10000 + rand() % 20000) / 10000.0;
                st->v = (double) (-10000 + rand() % 20000) / 10000.0;
                st->s = st->u * st->u + st->v * st->v;
                if (st->s > 1 || st->s == 0) continue;

                st->u = st->u * sqrt(-2 * log(st->s) / st->s);
                st->v = st->v * sqrt(-2 * log(st->s) / st->s);

                return mu + sigma * st->u;
            }

        case 1:
            return mu + sigma * st->v;
    }
}

