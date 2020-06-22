#include "neural_network.cuh"

void printTensor(const Tensor& m){
	cout << cout.precision();

	for(int i = 0 ; i < m.shape.x ; i++){
		for(int j = 0 ; j < m.shape.y ; j++)
			cout << m[j * m.shape.x + i] << " ";
		cout << endl;
	}
}

NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Tensor NeuralNetwork::forward(Tensor X) {
	Tensor Z = X;

	for (auto layer : layers) {
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Tensor predictions, Tensor target) {
	dY.allocateMemoryIfNotAllocated(predictions.shape);

	Tensor error = cce_cost.dCost(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learning_rate);
	}

	cudaDeviceSynchronize();
}

void NeuralNetwork::summary(){
	for (auto it = this->layers.begin(); it != this->layers.end(); it++) {
		cout << "Layer: " << (*it)->getName() << endl;
		cout << "---------------------" <<endl;
	}
	cout <<endl;
}																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																															

vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}
