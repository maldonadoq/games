#include <iostream>
#include <string>

#include "../nn/neurals/neural_network.cuh"
#include "../nn/layers/linear_layer.cuh"
#include "../nn/layers/linear_relu.cuh"
#include "../nn/layers/linear_softmax.cuh"
#include "../nn/layers/relu_activation.cuh"
#include "../nn/utils/cce_loss.cuh"
#include "../nn/layers/softmax_activation.cuh"
#include "../nn/neurals/snake_dataset.cuh"

#include "src/result.cuh"

#define num_batches_train 175
#define batch_size 256

#define input_size 7
#define output_size 3

int main(int argc, char *argv[]){
	int epochs = 10;
	int epoch, batch;
	float loss;

	if(argc == 2){
		epochs = std::stoi(argv[1]);
	}

	CCELoss cce_cost;	
	NeuralNetwork nn;
	Matrix Y;

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

		if(epoch % 10 == 0){
			std::cout << "Epoch: " << epoch << ", Loss: " << loss / snake.getNumOfBatches() << std::endl;
		}		
	}
	std::cout << std::endl;
	
	Matrix X(Shape(1, input_size), {0,0,0,0.9818,0.0,0.19,1.0});

	Y = nn.forward(X);
	Y.copyDeviceToHost();

	std::vector<int> resInt = firstResultInt(Y, output_size);
	std::vector<float> resFloat = firstResultFloat(Y, output_size);

	printVector<int>({1,0,0});
	printVector<int>(resInt);
	printVector<float>(resFloat);	
	
	return 0;
}