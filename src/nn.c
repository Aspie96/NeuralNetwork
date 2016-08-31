#include "nn.h"

struct NN {
	int inputs;
	int levels;
	int npl;
	int outputs;
	double ***weights;
	double **neurons;
};

double *nn_randomWeights(int size) {
	double *retVal;
	int i;
	retVal = (double*)malloc(sizeof(double) * size);
	for(i = 0; i < size; i++) {
		do {
			retVal[i] = rand() * 2.0D / RAND_MAX - 1;
		} while(!retVal[i]);
	}
	return retVal;
}

void nn_vectorTimesMatrix(double *vectout, int rows, int cols, const double *vectin, const double **matrix) {
	int i, j;
	for(i = 0; i < cols; i++) {
		vectout[i] = 0;
		for(j = 0; j < rows; j++) {
			vectout[i] += vectin[j] * matrix[j][i];
		}
	}
}

void nn_activate(double *nodes, int count, int bias) {
	int i;
	if(bias) {
		nn_activate(nodes, count - 1, 0);
		nodes[count - 1] = 1;
	} else {
		for(i = 0; i < count; i++) {
			if(nodes[i] < 0) {
				nodes[i] = 0;
			}
		}
	}
}

double nn_incrementWeight(double *weight, double increment) {
	*weight += increment;
	if(*weight < -NN_MAX_WEIGHT) {
		*weight = -NN_MAX_WEIGHT;
	}
	if(*weight > NN_MAX_WEIGHT) {
		*weight = NN_MAX_WEIGHT;
	}
	return *weight;
}

NN *nn_create(int inputs, int levels, int npl, int outputs) {
	NN *retVal;
	int i, j;
	srand(time(NULL) + rand());
	retVal = (NN*)malloc(sizeof(NN));
	retVal->inputs = inputs;
	retVal->levels = levels;
	retVal->npl = npl;
	retVal->outputs = outputs;

	retVal->weights = (double***)malloc(sizeof(double**) * (levels + 1));
	retVal->weights[0] = (double**)malloc(sizeof(double*) * inputs + 1);
	for(j = 0; j < inputs + 1; j++) {
		retVal->weights[0][j] = nn_randomWeights(npl);
	}
	for(i = 1; i < levels; i++) {
		retVal->weights[i] = (double**)malloc(sizeof(double*) * npl + 1);
		for(j = 0; j < npl + 1; j++) {
			retVal->weights[i][j] = nn_randomWeights(npl);
		}
	}
	retVal->weights[levels] = (double**)malloc(sizeof(double*) * npl + 1);
	for(j = 0; j < npl + 1; j++) {
		retVal->weights[levels][j] = nn_randomWeights(outputs);
	}

	retVal->neurons = (double**)malloc(sizeof(double*) * (levels + 2));
	retVal->neurons[0] = (double*)malloc(sizeof(double) * inputs + 1);
	for(i = 1; i < levels + 1; i++) {
		retVal->neurons[i] = (double*)malloc(sizeof(double) * npl + 1);
	}
	retVal->neurons[levels + 1] = (double*)malloc(sizeof(double) * outputs);
	return retVal;
}

double nn_getError(NN *network, double *error, const double *inputs, const double *expected) {
	int i;
	double output[network->outputs], aux, retVal;
	nn_forwardPropagate(network, output, inputs);
	retVal = 0;
	for(i = 0; i < network->outputs; i++) {
		retVal += aux = error[i] = expected[i] - output[i];
		if(error) {
			error[i] = aux;
		}
	}
	retVal = sqrt(retVal);
	return retVal;
}

void nn_forwardPropagate(NN *network, double *outputs, const double *inputs) {
	int i;
	memcpy(network->neurons[0], inputs, sizeof(double) * network->inputs);
	network->neurons[0][network->inputs] = 1;
	nn_vectorTimesMatrix(network->neurons[1], network->inputs + 1, network->npl, network->neurons[0], (const double**)network->weights[0]);
	nn_activate(network->neurons[1], network->npl + 1, 1);
	for(i = 2; i < network->levels + 1; i++) {
		nn_vectorTimesMatrix(network->neurons[i], network->npl + 1, network->npl, network->neurons[i - 1], (const double**)network->weights[i - 1]);
		nn_activate(network->neurons[i], network->npl + 1, 1);
	}
	nn_vectorTimesMatrix(network->neurons[network->levels + 1], network->npl + 1, network->outputs, network->neurons[network->levels], (const double**)network->weights[network->levels]);
	memcpy(outputs, network->neurons[network->levels + 1], sizeof(double) * network->outputs);
}

void nn_backPropagate(NN *network, const double *error) {
	int i, j, k, level, aux, delta;
	double weights[network->npl + 1], weight;
	memcpy(network->neurons[network->levels + 1], error, sizeof(double) * network->outputs);
	for(i = 0; i < network->outputs; i++) {
		level = network->levels;
		memset(weights, 0, sizeof(double) * (network->npl + 1));
		for(j = 0; j < network->npl + 1; j++) {
			aux = network->weights[level][j][i];
			nn_incrementWeight(&network->weights[level][j][i], weights[j] = NN_DELTA * network->neurons[level][j] * network->neurons[level + 1][i]);
			weights[j] *= aux;
		}
		for(; level > 0; level--) {
			for(j = 0; j < network->npl + 1; j++) {
				weight = 0;
				for(k = 0; k < network->npl; k++) {
					aux = network->weights[level][j][k];
					delta = NN_DELTA * network->neurons[level][j] * weights[k];
					nn_incrementWeight(&network->weights[level][j][k], delta);
					weight += delta * aux;
				}
				weights[i] = weight;
			}
		}
		for(j = 0; j < network->npl + 1; j++) {
			nn_incrementWeight(&network->weights[level][j][i], NN_DELTA * network->neurons[1][j] * network->neurons[0][j]);
		}
	}
}

void nn_learn(NN *network, const double **inputs, const double **outputs, int entries) {
	int tries;
	double error;
	do {

	}while(error > NN_MAX_ERR && ++tries < NN_MAX_STEPS);
}

void nn_destroy(NN *network) {
	int i, j;
	for(j = 0; j < network->inputs; j++) {
		free(network->weights[0][j]);
	}
	free(network->weights[0]);
	for(i = 1; i < network->levels + 1; i++) {
		for(j = 0; j < network->npl + 1; j++) {
			free(network->weights[i][j]);
		}
		free(network->weights[i]);
	}
	free(network->weights);
	for(i = 0; i < network->levels + 2; i++) {
		free(network->neurons[i]);
	}
	free(network->neurons);
	free(network);
}
