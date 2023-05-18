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
#include "utils/gnuplot.h"

#define num_batches_train 175
#define num_batches_test 10
#define batch_size 256

#define input_size 7
#define output_size 3

using plotcpp::Lines;
using plotcpp::Plot;
using plotcpp::Points;

void PlotXY(string title, string xlabel, string ylabel, const std::vector<float> &x, const std::vector<float> &y){

	Plot plt(true);
    plt.SetTerminal("qt");
    plt.SetTitle(title);
    plt.SetXLabel(xlabel);
    plt.SetYLabel(ylabel);

    plt.Draw2D(Lines(x.begin(), x.end(), y.begin(), "Value"));

    plt.Flush();
}

// generate dataset https://github.com/TheAILearner/Snake-Game-with-Deep-learning
int main(int argc, char *argv[]){
	Tensor Y;

	int epochs = 5;
	int epoch, batch;
	float loss;
	int correct = 0;

	if(argc == 2){
		epochs = std::stoi(argv[1]);
	}

	CCELoss cce_cost;
	NeuralNetwork nn;

	nn.addLayer(new LinearLayer("Linear_1", Shape(input_size, 256)));
	nn.addLayer(new ReLUActivation("Relu_1"));
	nn.addLayer(new LinearLayer("Linear_2", Shape(256, 128)));
	nn.addLayer(new ReLUActivation("Relu_2"));
	nn.addLayer(new LinearLayer("Linear_3", Shape(128, output_size)));
	nn.addLayer(new softmaxActivation("Softmax"));

	nn.summary();

	SnakeDataset snake(num_batches_train, batch_size, "data/trainX.csv", "data/trainY.csv");

	vector<float> vect_epochs;
	vector<float> vect_loss;
	
	cout << "Training" << endl;
	for (epoch = 0; epoch < epochs; epoch++){
		loss = 0.0;
		for (batch = 0; batch < num_batches_train; batch++){
			Y = nn.forward(snake.getBatches().at(batch));
			nn.backprop(Y, snake.getTargets().at(batch));

			correct += computeAccuracy(Y, snake.getTargets().at(batch), output_size);
			loss += cce_cost.cost(Y, snake.getTargets().at(batch));
		}

		std::cout << " Epoch: " << epoch << ", Loss: " << loss / snake.getNumOfBatches() << std::endl;
		
		vect_epochs.push_back(epoch);
		vect_loss.push_back(loss / snake.getNumOfBatches());
	}

	PlotXY("Loss", "Epochs", "Loss", vect_epochs, vect_loss);
	
	cout << endl << "Testing" << endl;
	SnakeDataset snake_test(num_batches_test, batch_size, "data/testX.csv", "data/testY.csv");
	correct = 0;

	for (batch = 0; batch < num_batches_test; batch++){
		Y = nn.forward(snake_test.getBatches().at(batch));
		Y.copyDeviceToHost();
		correct += computeAccuracy(Y, snake_test.getTargets().at(batch), output_size);
	}

	float accuracy = (float)correct / (num_batches_test * batch_size);
	std::cout << " Accuracy: " << accuracy << std::endl;

	return 0;
}