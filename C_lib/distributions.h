#pragma once
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct nd{
    char second;
    double mu, sigma;
    double u, v, s;
    int n;
    double *values;
} normal;
double normal_distribution(double mu, double sigma, normal* s);
void fill(normal *s);
normal *init_normal(double mu, double sigma, int n);