#pragma once

#include <vector>
#include "../layers/layer.cuh"
#include "../utils/cce_loss.cuh"
#include "../utils/exception.cuh"

using std::string;
using std::vector;
using std::cout;
using std::endl;

class NeuralNetwork {
private:
	vector<NNLayer*> layers;
	CCELoss cce_cost;

	Tensor Y;
	Tensor dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Tensor forward(Tensor);
	void backprop(Tensor, Tensor);

	void addLayer(NNLayer *);
	std::vector<NNLayer*> getLayers() const;

	bool save(string);
	bool load(string);

	void summary();
};
