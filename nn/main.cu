#include <iostream>
#include <string>

#include "layers/linear_layer.cuh"
#include "layers/linear_relu.cuh"
#include "layers/linear_softmax.cuh"
#include "layers/relu_activation.cuh"
#include "layers/softmax_activation.cuh"
#include "neurals/neural_network.cuh"
#include "neurals/snake_dataset.cuh"
#include "utils/cce_loss.cuh"

#define num_batches_train 100
#define num_batches_test 10
#define batch_size 10

#define input_size 7
#define output_size 3

int main(int argc, char *argv[]){
	Tensor Y;

	int epochs = 5;
	int epoch, batch;
	float loss;

	if(argc == 2){
		epochs = std::stoi(argv[1]);
	}

	CCELoss cce_cost;
	NeuralNetwork nn;

	nn.addLayer(new LinearLayer("Linear_1", Shape(input_size, 256)));
	nn.addLayer(new ReLUActivation("Relu_1"));
	nn.addLayer(new LinearLayer("Linear_2", Shape(256, output_size)));
	nn.addLayer(new softmaxActivation("Softmax"));

	nn.summary();

	SnakeDataset snake(num_batches_train, batch_size, "data/testX.csv", "data/testY.csv");
	
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