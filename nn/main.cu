// nvcc main.cu neurals/*.cu layers/*.cu utils/*.cu -std=c++11 -o main.out
// ./main.out

#include <iostream>
#include <string>
#include <time.h>

#include "neurals/neural_network.h"
#include "layers/linear_layer.h"
#include "layers/linear_relu.h"
#include "layers/linear_softmax.h"
#include "layers/relu_activation.h"
#include "utils/cce_cost.h"
#include "layers/softmax_activation.h"
#include "neurals/mnist_dataset.h"

#define num_batches_train 100
#define batch_size 100

#define num_batches_test 20
#define classes 10

float computeAccuracy_mnist(const Matrix &predictions, const Matrix &targets)
{
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++)
	{
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i])
		{
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}

int computeAccuracyClasses_mnist(const Matrix &predictions, const Matrix &targets, int k)
{
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++)
	{
		float _max = 0.0;
		float _maxt = 0.0;
		int label = 0;
		int labely = 0;
		for (int j = 0; j < k; j++)
		{
			if (predictions[j * m + i] > _max)
			{
				_max = predictions[j * m + i];
				label = j;
			}
			if (targets[j * m + i] > _maxt)
			{
				_maxt = targets[j * m + i];
				labely = j;
			}
		}
		if (label == labely)
			correct_predictions++;
	}
	return correct_predictions;
}

int main(int argc, char *argv[])
{
	srand(time(NULL));

	Matrix Y;
	CCECost cce_cost;
	NeuralNetwork nn;

	nn.addLayer(new LinearLayer("linear_1", Shape(28 * 28, 256)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_8", Shape(256, 10)));
	nn.addLayer(new softmaxActivation("softmax_output"));

	MNISTDataset mnist(num_batches_train, batch_size, classes);

	std::cout << "Training" << std::endl;

	int epochs = 11;
	for (int epoch = 0; epoch < epochs; epoch++)
	{
		float cost = 0.0;

		for (int batch = 0; batch < num_batches_train; batch++)
		{

			Y = nn.forward(mnist.getBatches().at(batch));
			nn.backprop(Y, mnist.getTargets().at(batch));
			cost += cce_cost.cost(Y, mnist.getTargets().at(batch));
		}

		std::cout << "Epoch: " << epoch
				  << ", Cost: " << cost / mnist.getNumOfBatches()
				  << std::endl;
	}

	std::cout << "Testing" << std::endl;
	int correct_predictions = 0;

	MNISTDataset mnist_test(num_batches_test, batch_size, classes, TEST);

	for (int batch = 0; batch < num_batches_test; batch++)
	{
		Y = nn.forward(mnist_test.getBatches().at(batch));
		Y.copyDeviceToHost();
		correct_predictions += computeAccuracyClasses_mnist(Y, mnist_test.getTargets().at(batch), classes);
	}

	float accuracy = (float)correct_predictions / (num_batches_test * batch_size);
	std::cout << "Accuracy: " << accuracy * 100 << std::endl;

	return 0;
}