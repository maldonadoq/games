#include <iostream>
#include <string>

#include "neurals/neural_network.cuh"
#include "layers/linear_layer.cuh"
#include "layers/linear_relu.cuh"
#include "layers/linear_softmax.cuh"
#include "layers/relu_activation.cuh"
#include "utils/nn_exception.cuh"
#include "utils/cce_cost.cuh"
#include "layers/softmax_activation.cuh"
#include "neurals/snake_dataset.cuh"

#define num_batches_train 250
#define num_batches_test 20
#define batch_size 1000

#define input_size 7
#define output_size 3

int computeAccuracy(const Matrix &predictions, const Matrix &targets, int k){
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++){
		float _max = 0.0;
		float _maxt = 0.0;
		int label = 0;
		int labely = 0;
		for (int j = 0; j < k; j++){
			if (predictions[j * m + i] > _max){
				_max = predictions[j * m + i];
				label = j;
			}
			if (targets[j * m + i] > _maxt){
				_maxt = targets[j * m + i];
				labely = j;
			}
		}
		if (label == labely)
			correct_predictions++;
	}
	return correct_predictions;
}

// nvcc main.cu neurals/*.cu layers/*.cu utils/*.cu -o main.out
int main(int argc, char *argv[]){
	Matrix Y;

	int epochs = 5;
	int epoch, batch;
	float loss;

	if(argc == 2){
		epochs = std::stoi(argv[1]);
	}

	CCECost cce_cost;
	NeuralNetwork nn;

	nn.addLayer(new LinearLayer("linear_1", Shape(input_size, 256)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(256, output_size)));
	nn.addLayer(new softmaxActivation("softmax_output"));

	SnakeDataset snake(num_batches_train, batch_size, "data/trainX.csv", "data/trainY.csv");
	
	for (epoch = 0; epoch < epochs; epoch++){
		loss = 0.0;
		for (batch = 0; batch < num_batches_train; batch++){
			Y = nn.forward(snake.getBatches().at(batch));
			nn.backprop(Y, snake.getTargets().at(batch));
			loss += cce_cost.cost(Y, snake.getTargets().at(batch));
		}

		std::cout << "Epoch: " << epoch << ", Loss: " << loss / snake.getNumOfBatches() << std::endl;
	}

	int correct_predictions = 0;
	SnakeDataset snake_test(num_batches_test, batch_size, "data/testX.csv", "data/testY.csv");

	for (batch = 0; batch < num_batches_test; batch++){
		Y = nn.forward(snake_test.getBatches().at(batch));
		Y.copyDeviceToHost();
		correct_predictions += computeAccuracy(Y, snake_test.getTargets().at(batch), output_size);
	}

	float accuracy = (float)correct_predictions / (num_batches_test * batch_size);
	std::cout << "Accuracy: " << accuracy * 100 << std::endl;

	return 0;
}