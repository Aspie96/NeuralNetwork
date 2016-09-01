#ifndef NN_H
#define NN_H

#include <math.h>
#include <mem.h>
#include <stdlib.h>
#include <time.h>

#define NN_DELTA 0.05
#define NN_MAX_ERR 0.01
#define NN_MAX_STEPS 1000
#define NN_MAX_WEIGHT 1000

struct NN;
typedef struct NN NN;

NN *nn_create(int inputs, int levels, int npl, int outputs);
void nn_forwardPropagate(NN *network, double *outputs, const double *inputs);
double nn_getError(NN *network, double *error, const double *inputs, const double *expected);
void nn_backPropagate(NN *network, const double *error);
void nn_refreshWeights(NN *network);
void nn_learn(NN *network, const double **inputs, const double **outputs, int entries);
void nn_destroy(NN *network);

#endif /* NN_H */
