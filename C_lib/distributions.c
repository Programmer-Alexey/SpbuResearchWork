#include "distributions.h"


void fill(normal *s) {
    for (int i = 0; i < s->n; ++i) {
        s->values[i] = normal_distribution(s->mu, s->sigma, s);
    }
}


normal *init_normal(double mu, double sigma, int n) {
    srand(time(NULL));
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
            do {
                st->u = 2.0 * ((double) rand() / RAND_MAX) - 1.0;
                st->v = 2.0 * ((double) rand() / RAND_MAX) - 1.0;
                st->s = st->u * st->u + st->v * st->v;

            } while (st->s >= 1.0 || st->s == 0.0);

            st->u = st->u * sqrt(-2.0 * log(st->s) / st->s);
            st->v = st->v * sqrt(-2.0 * log(st->s) / st->s);

            return mu + sigma * st->u;

        case 1:
            return mu + sigma * st->v;
    }
}

