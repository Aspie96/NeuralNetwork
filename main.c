#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

int main() {
	NN *myNet = nn_create(2, 2, 3, 1);
	double input1[2] = {0, 0};
	double input2[2] = {1, 1};
	double error[1];
	double output1[1] = { 0 };
	double output2[1] = { 1 };
	int i;
	for(i = 0; i < 1; i++) {
		nn_getError(myNet, error, input1, output1);
		nn_backPropagate(myNet, error);
		nn_getError(myNet, error, input2, output2);
		nn_backPropagate(myNet, error);
	}
	nn_forwardPropagate(myNet, output1, input1);
	nn_forwardPropagate(myNet, output2, input2);
	printf("Out1: %f\n", output1[0]);
	printf("Out2: %f\n", output2[0]);

	//nn_destroy(myNet);	//TODO: Correggo errore.
    return 0;
}
