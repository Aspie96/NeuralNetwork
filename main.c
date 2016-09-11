#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

int main() {
	/*NN *myNet = nn_create(2, 1, 1, 1);
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
	printf("Out3: %f\n", outputs[3][0]);*/

	/*NN *myNet = nn_create(30, 2, 7, 1);
	double inputs[200][30];
	double outputs[200][1];
	FILE *fp = fopen("./data/train1.csv", "r");
	int i;
	for(i = 0; i < 200; i++) {
		fscanf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\r\n", &inputs[i][0], &inputs[i][1], &inputs[i][2], &inputs[i][3], &inputs[i][4], &inputs[i][5], &inputs[i][6], &inputs[i][7], &inputs[i][8], &inputs[i][9], &inputs[i][10], &inputs[i][11], &inputs[i][12], &inputs[i][13], &inputs[i][14], &inputs[i][15], &inputs[i][16], &inputs[i][17], &inputs[i][18], &inputs[i][19], &inputs[i][20], &inputs[i][21], &inputs[i][22], &inputs[i][23], &inputs[i][24], &inputs[i][25], &inputs[i][26], &inputs[i][27], &inputs[i][28], &inputs[i][29], &outputs[i][0]);
	}
	fclose(fp);
	nn_learn(myNet, 200, inputs, outputs);

	double inputsT[30];
	double outputsT[1];
	double expected;
	fp = fopen("./data/test1.csv", "r");
	int correct = 0;
	int wrong = 0;
	while(fscanf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\r\n", &inputsT[0], &inputsT[1], &inputsT[2], &inputsT[3], &inputsT[4], &inputsT[5], &inputsT[6], &inputsT[7], &inputsT[8], &inputsT[9], &inputsT[10], &inputsT[11], &inputsT[12], &inputsT[13], &inputsT[14], &inputsT[15], &inputsT[16], &inputsT[17], &inputsT[18], &inputsT[19], &inputsT[20], &inputsT[21], &inputsT[22], &inputsT[23], &inputsT[24], &inputsT[25], &inputsT[26], &inputsT[27], &inputsT[28], &inputsT[29], &expected) != EOF) {
		nn_forwardPropagate(myNet, outputsT, inputsT);
		if((expected == 1 && outputsT[0] > 0.5) || (expected == 0 && outputsT[0] < 0.5)) {
			correct++;
		} else {
			wrong++;
		}
	}
	fclose(fp);
	printf("Success rate: %d%%\n", correct * 100 / (correct + wrong));*/

	NN *myNet = nn_create(4, 2, 2, 3);
	double inputs[30][4];	///60
	double outputs[30][3];	///60
	FILE *fp = fopen("./data/train2.csv", "r");
	int i;
	for(i = 0; i < 30; i++) {	///60
		fscanf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\r\n", &inputs[i][0], &inputs[i][1], &inputs[i][2], &inputs[i][3], &outputs[i][0], &outputs[i][1], &outputs[i][2]);
	}
	///fclose(fp);
	nn_learn(myNet, 30, inputs, outputs);	///60

	double inputsT[4];
	double outputsT[3];
	double expected[3];
	///fp = fopen("./data/test2.csv", "r");
	int correct = 0;
	int wrong = 0;
	int category;
	int res;
	while(fscanf(fp, "%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\r\n", &inputsT[0], &inputsT[1], &inputsT[2], &inputsT[3], &expected[0], &expected[1], &expected[2]) != EOF) {
		nn_forwardPropagate(myNet, outputsT, inputsT);
		for(i = 0; i < 3; i++) {
			if(expected[i]) {
				category = i;
			}
		}
		res = 1;
		for(i = 0; i < 3; i++) {
			if(i != category && outputsT[i] >= outputsT[category]) {
				res = 0;
			}
		}
		if(res) {
			correct++;
		} else {
			wrong++;
		}
	}
	fclose(fp);
	printf("Success rate: %d%%\n", correct * 100 / (correct + wrong));

	nn_destroy(myNet);
    return 0;
}
