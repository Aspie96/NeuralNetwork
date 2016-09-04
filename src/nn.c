/// Uso: http://courses.cs.washington.edu/courses/cse599/01wi/admin/Assignments/bpn.html
/// Asko for validation to: http://ai.stackexchange.com/
/// Valori tipici da: https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Neural_Network_Basics

/// Considero l'idea di tornare a salvare i dw per ogni peso, anziché per ogni neurone. Questo permette, tra l'altro, di implementare, oltre alla modalità batch, anche il momento.
/// Per il momento, se lo implemento, ad ogni giro scambio i puntatori oldDeltas e deltas: soluzione intelligente per risparmiare operazioni.

/// http://stackoverflow.com/questions/13095938/can-somebody-please-explain-the-backpropagation-algorithm-to-me

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

	retVal->weights = (double***)malloc(sizeof(double**) * (layers + 2));
	retVal->weights[0] = (double**)malloc(sizeof(double*) * (inputs + 1));
	for(j = 0; j < inputs + 1; j++) {
		retVal->weights[0][j] = nn_randomWeights(npl);
	}
	for(i = 1; i < layers + 1; i++) {
		retVal->weights[i] = (double**)malloc(sizeof(double*) * (npl + 1));
		for(j = 0; j < npl + 1; j++) {
			retVal->weights[i][j] = nn_randomWeights(npl);
		}
	}
	retVal->weights[layers + 1] = (double**)malloc(sizeof(double*) * (npl + 1));
	for(j = 0; j < npl + 1; j++) {
		retVal->weights[layers + 1][j] = nn_randomWeights(outputs);
	}

	retVal->neurons = (double**)malloc(sizeof(double*) * (layers + 2));
	retVal->deltas = (double**)malloc(sizeof(double*) * (layers + 2));
	retVal->neurons[0] = (double*)malloc(sizeof(double) * (inputs + 1));
	retVal->deltas[0] = (double*)calloc(inputs + 1, sizeof(double));
	for(i = 1; i < layers + 1; i++) {
		retVal->neurons[i] = (double*)malloc(sizeof(double) * (npl + 1));
		retVal->deltas[i] = (double*)calloc(npl + 1, sizeof(double));
	}
	retVal->neurons[layers + 1] = (double*)malloc(sizeof(double) * outputs);
	retVal->deltas[layers + 1] = (double*)calloc(outputs, sizeof(double));
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
			error[i] += aux;
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
		delta = 0;
		for(j = 0; j < l2count; j++) {
			delta += network->weights[*layer][i][j] * deltasC[j];
		}
		deltas[i] = NN_AF_DER(network->neurons[*layer][i]) * delta;
		network->deltas[*layer][i] += deltas[i];
	}
	(*layer)--;
}

void nn_backPropagate(NN *network, const double *error) {
	int layer, i;
	/// Quando implementerò il deltas a tre puntatori, qui non cambierà molto e continuerò ad utilizzare deltasC e deltas, vettori così come sono ora (e avrà ancora più senso).
	double deltas[network->npl + 1], deltasC[network->npl + 1], delta;
	layer = network->layers + 1;
	for(i = 0; i < network->outputs; i++) {
		delta = (NN_AF_DER(network->neurons[layer][i]) * error[i]);
		deltas[i] = delta;
		network->deltas[layer][i] += delta;
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
			network->weights[*layer][i][j] += NN_DELTA * network->neurons[*layer][i] * network->deltas[*layer + 1][j];
		}
	}
	// Copiare formule senza capirle: http://stackoverflow.com/questions/13095938/can-somebody-please-explain-the-backpropagation-algorithm-to-me
	/*for(j = 0; j < l2count; j++) {
		network->weights[*layer][i][j] = NN_DELTA * NN_AF(network->neurons[*layer][i]) * network->deltas[*layer + 1][j];
	}*/
	memset(network->deltas[*layer + 1], 0, sizeof(double) * l2count + 1);
	(*layer)++;
}

void nn_refreshWeights(NN *network) {
	int layer;

	memset(network->deltas[0], 0, sizeof(double) * (network->inputs + 1));
	layer = 0;
	nn_refreshWeightsFunc(network, &layer, network->inputs, network->npl);
	while(layer < network->layers) {
		nn_refreshWeightsFunc(network, &layer, network->npl, network->npl);
	}
	nn_refreshWeightsFunc(network, &layer, network->npl, network->outputs);
}



/*void nn_learn(NN *network, const double **inputs, const double **outputs, int entries) {
	int tries;
	double error;
	do {

	}while(error > NN_MAX_ERR && ++tries < NN_MAX_STEPS);
}*/

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
