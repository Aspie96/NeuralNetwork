#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

int main() {
	NN *myNet = nn_create(2, 3, 30, 1);
	/*double input0[2] = {0, 0};
	double input1[2] = {0, 1};
	double input2[2] = {1, 0};
	double input3[2] = {1, 1};
	double error[1] = { 0 };
	double output0[1] = { 1 };
	double output1[1] = { 1 };
	double output2[1] = { 0 };
	double output3[1] = { 0 };
	int i;
	for(i = 0; i < 500; i++) {
		nn_getError(myNet, error, input0, output0);
		nn_backPropagate(myNet, error);
		//nn_refreshWeights(myNet);
		nn_getError(myNet, error, input1, output1);
		nn_backPropagate(myNet, error);
		//nn_refreshWeights(myNet);
		nn_getError(myNet, error, input2, output2);
		nn_backPropagate(myNet, error);
		//nn_refreshWeights(myNet);
		nn_getError(myNet, error, input3, output3);
		nn_backPropagate(myNet, error);
		nn_refreshWeights(myNet);
	}
	nn_forwardPropagate(myNet, output0, input0);
	nn_forwardPropagate(myNet, output1, input1);
	nn_forwardPropagate(myNet, output2, input2);
	nn_forwardPropagate(myNet, output3, input3);
	printf("Out0: %f\n", output0[0]);
	printf("Out1: %f\n", output1[0]);
	printf("Out2: %f\n", output2[0]);
	printf("Out3: %f\n", output3[0]);*/

	double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
	double outputs[4][1] = {{0}, {0}, {0}, {1}};
	nn_learn(myNet, 4, inputs, outputs);

	nn_forwardPropagate(myNet, outputs[0], inputs[0]);
	nn_forwardPropagate(myNet, outputs[1], inputs[1]);
	nn_forwardPropagate(myNet, outputs[2], inputs[2]);
	nn_forwardPropagate(myNet, outputs[3], inputs[3]);
	printf("Out0: %f\n", outputs[0][0]);
	printf("Out1: %f\n", outputs[1][0]);
	printf("Out2: %f\n", outputs[2][0]);
	printf("Out3: %f\n", outputs[3][0]);

	nn_destroy(myNet);
    return 0;
}
