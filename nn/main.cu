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

#define num_batches_train 10
#define num_batches_test 20
#define batch_size 100

#define input_size 7
#define output_size 3

int computeAccuracy(const Matrix& predictions, const Matrix& targets, int k) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float _max = 0.0;
		float _maxt = 0.0;
		int label = 0;
		int labely = 0;
		for(int j = 0 ; j < k ; j++){
			if(predictions[j * m + i] > _max){
				_max = predictions[j * m + i];
				label = j;
			}
			if(targets[j * m + i] > _maxt){
				_maxt = targets[j * m + i];
				labely = j;
			}
		}
		if(label == labely)	correct_predictions++;
	}
	return correct_predictions;
}

int main(int argc, char* argv[]) {

	srand( time(NULL) );
	Matrix Y;

	CCECost cce_cost;
	NeuralNetwork nn;

	nn.addLayer(new LinearLayer("linear_1", Shape(28*28, 256)));
	nn.addLayer(new ReLUActivation("relu_1"));

	nn.addLayer(new LinearLayer("linear_8", Shape(256, output_size)));
	nn.addLayer(new softmaxActivation("softmax_output"));

	/* MNISTDataset mnist(num_batches_train, batch_size, output_size);

	for (int epoch = 0; epoch < 51; epoch++) {
		float cost = 0.0;
		for(int batch = 0 ; batch < num_batches_train ; batch++){
			Y = nn.forward(mnist.getBatches().at(batch));
			nn.backprop(Y, mnist.getTargets().at(batch));
			cost += cce_cost.cost(Y, mnist.getTargets().at(batch));
		}
	
		if(epoch%10 == 0){
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / mnist.getNumOfBatches()
						<< std::endl;
		}
	}

	int correct_predictions = 0;
	MNISTDataset mnist_test(num_batches_test, batch_size, output_size, TEST);

	for(int batch = 0 ; batch < num_batches_test;  batch++){
		Y = nn.forward(mnist_test.getBatches().at(batch));
		Y.copyDeviceToHost();
		correct_predictions += computeAccuracy(Y, mnist_test.getTargets().at(batch), output_size);
	}

	float accuracy = (float)correct_predictions / (num_batches_test * batch_size);
	std::cout << "Accuracy: " << accuracy*100 << std::endl; */

	return 0;
}