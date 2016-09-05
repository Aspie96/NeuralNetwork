#ifndef NN_H
#define NN_H

#define NN_DELTA 0.3
#define NN_MOMENTUM 0.01	///TODO: Consider adding momentum.
#define NN_MAX_ERR 0.01
#define NN_MAX_STEPS 50000

#define NN_AF(x) (1 / (1 + exp(-(x))))
#define NN_AF_DER(fx) ((fx) * (1 - (fx)))

struct NN {
	int inputs;
	int layers;
	int npl;
	int outputs;
	double ***weights;
	double **neurons;
	double ***deltas;
};
typedef struct NN NN;

NN *nn_create(int inputs, int layers, int npl, int outputs);
void nn_forwardPropagate(NN *network, double *outputs, const double *inputs);
double nn_getError(NN *network, double *error, const double *inputs, const double *expected);
void nn_backPropagate(NN *network, const double *error);
void nn_refreshWeights(NN *network);
void nn_learn(NN *network, int entries, const double inputs[entries][network->inputs], const double outputs[entries][network->outputs]);
void nn_destroy(NN *network);

#endif /* NN_H */
