#include "nn.h"
#include <math.h>
#include <mem.h>
#include <stdlib.h>
#include <time.h>

double *nn_randomWeights(int size) {
	double *retVal;
	int i;
	retVal = (double*)malloc(sizeof(double) * size);
	for(i = 0; i < size; i++) {
		do {
			retVal[i] = rand() * 2.0d / RAND_MAX - 1;
		} while(!retVal[i]);
	}
	return retVal;
}

NN *nn_create(int inputs, int layers, int npl, int outputs) {
	NN *retVal;
	int i, j;
	srand(time(NULL) + rand());
	retVal = (NN*)malloc(sizeof(NN));
	retVal->inputs = inputs;
	retVal->layers = layers;
	retVal->npl = npl;
	retVal->outputs = outputs;

	retVal->deltas = (double***)malloc(sizeof(double**) * (layers + 2));
	retVal->weights = (double***)malloc(sizeof(double**) * (layers + 2));
	retVal->weights[0] = (double**)malloc(sizeof(double*) * (inputs + 1));
	retVal->deltas[0] = (double**)malloc(sizeof(double*) * (inputs + 1));
	for(j = 0; j < inputs + 1; j++) {
		retVal->weights[0][j] = nn_randomWeights(npl);
		retVal->deltas[0][j] = (double*)calloc(npl, sizeof(double));
	}
	for(i = 1; i < layers + 1; i++) {
		retVal->weights[i] = (double**)malloc(sizeof(double*) * (npl + 1));
		retVal->deltas[i] = (double**)malloc(sizeof(double*) * (npl + 1));
		for(j = 0; j < npl + 1; j++) {
			retVal->weights[i][j] = nn_randomWeights(npl);
			retVal->deltas[i][j] = (double*)calloc(npl, sizeof(double));
		}
	}
	retVal->weights[layers + 1] = (double**)malloc(sizeof(double*) * (npl + 1));
	retVal->deltas[layers + 1] = (double**)malloc(sizeof(double*) * (npl + 1));
	for(j = 0; j < npl + 1; j++) {
		retVal->weights[layers + 1][j] = nn_randomWeights(outputs);
		retVal->deltas[layers + 1][j] = (double*)calloc(outputs, sizeof(double));
	}

	retVal->neurons = (double**)malloc(sizeof(double*) * (layers + 2));
	retVal->neurons[0] = (double*)malloc(sizeof(double) * (inputs + 1));
	for(i = 1; i < layers + 1; i++) {
		retVal->neurons[i] = (double*)malloc(sizeof(double) * (npl + 1));
	}
	retVal->neurons[layers + 1] = (double*)malloc(sizeof(double) * outputs);
	return retVal;
}

double nn_getError(NN *network, double *error, const double *inputs, const double *expected) {
	int i;
	double output[network->outputs], aux, retVal;
	nn_forwardPropagate(network, output, inputs);
	retVal = 0;
	for(i = 0; i < network->outputs; i++) {
		aux = expected[i] - output[i];
		retVal += aux * aux;
		if(error) {
			error[i] = aux;
		}
	}
	retVal = sqrt(retVal);
	return retVal;
}


static inline void nn_forwardPropagateFunc(NN *network, int *layer, int inc, int outc) {
	int i, j;
	for(i = 0; i < outc; i++) {
		network->neurons[*layer][i] = 0;
		for(j = 0; j < inc; j++) {
			network->neurons[*layer][i] += network->neurons[*layer - 1][j] * network->weights[*layer - 1][j][i];
		}
		network->neurons[*layer][i] = NN_AF(network->neurons[*layer][i]);
	}
	network->neurons[*layer][i] = 1;
	(*layer)++;
}

void nn_forwardPropagate(NN *network, double *outputs, const double *inputs) {
	int layer;
	memcpy(network->neurons[0], inputs, sizeof(double) * network->inputs);
	network->neurons[0][network->inputs] = 1;
	layer = 1;
	nn_forwardPropagateFunc(network, &layer, network->inputs + 1, network->npl);
	while(layer < network->layers + 1) {
		nn_forwardPropagateFunc(network, &layer, network->npl + 1, network->npl);
	}
	nn_forwardPropagateFunc(network, &layer, network->npl + 1, network->outputs);
	memcpy(outputs, network->neurons[network->layers + 1], sizeof(double) * network->outputs);
}


static inline void nn_backPropagateFunc(NN *network, int *layer, double *deltas, double *deltasC, int l1count, int l2count) {
	int i, j;
	double delta;
	memcpy(deltasC, deltas, sizeof(double) * (network->npl + 1));
	for(i = 0; i < l1count; i++) {
		deltas[i] = 0;
		for(j = 0; j < l2count; j++) {
			delta = NN_AF_DER(network->neurons[*layer][i]) * network->weights[*layer][i][j] * deltasC[j];
			deltas[i] += delta;
			network->deltas[*layer][i][j] += network->neurons[*layer][i] * deltasC[j];
		}
	}
	(*layer)--;
}

void nn_backPropagate(NN *network, const double *error) {
	int layer, i;
	double deltas[network->npl + 1], deltasC[network->npl + 1], delta;
	layer = network->layers + 1;
	for(i = 0; i < network->outputs; i++) {
		delta = (NN_AF_DER(network->neurons[layer][i]) * error[i]);
		deltas[i] = delta;
	}
	layer--;
	nn_backPropagateFunc(network, &layer, deltas, deltasC, network->npl + 1, network->outputs);
	while(layer > 0) {
		nn_backPropagateFunc(network, &layer, deltas, deltasC, network->npl + 1, network->npl);
	}
	nn_backPropagateFunc(network, &layer, deltas, deltasC, network->inputs + 1, network->npl);
}


static inline void nn_refreshWeightsFunc(NN *network, int *layer, int l1count, int l2count) {
	int i, j;
	for(i = 0; i < l1count + 1; i++) {
		for(j = 0; j < l2count; j++) {
			network->weights[*layer][i][j] += NN_DELTA * network->deltas[*layer][i][j];
			network->deltas[*layer][i][j] = 0;
		}
	}
	(*layer)++;
}

void nn_refreshWeights(NN *network) {
	int layer;

	layer = 0;
	nn_refreshWeightsFunc(network, &layer, network->inputs, network->npl);
	while(layer < network->layers) {
		nn_refreshWeightsFunc(network, &layer, network->npl, network->npl);
	}
	nn_refreshWeightsFunc(network, &layer, network->npl, network->outputs);
}



void nn_learn(NN *network, int entries, const double inputs[entries][network->inputs], const double outputs[entries][network->outputs]) {
	int tries, i;
	double error, deltas[network->outputs];

	tries = 0;
	do {
		error = 0;
		for(i = 0; i < entries; i++) {
			error += nn_getError(network, deltas, inputs[i], outputs[i]);
			nn_backPropagate(network, deltas);
		}
		nn_refreshWeights(network);
		error /= entries;
	} while(error > NN_MAX_ERR && ++tries < NN_MAX_STEPS);
}

void nn_destroy(NN *network) {
	/*int i, j;
	for(j = 0; j < network->inputs; j++) {
		free(network->weights[0][j]);
	}
	free(network->weights[0]);
	for(i = 1; i < network->layers + 1; i++) {
		for(j = 0; j < network->npl + 1; j++) {
			free(network->weights[i][j]);
		}
		free(network->weights[i]);
	}
	free(network->weights);
	for(i = 0; i < network->layers + 2; i++) {
		free(network->neurons[i]);
	}
	free(network->neurons);
	free(network);*/
}
