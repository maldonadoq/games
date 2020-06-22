#pragma once

#include <vector>
#include "../layers/layer.cuh"
#include "../utils/cce_loss.cuh"
#include "../utils/exception.cuh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	CCELoss cce_cost;

	Tensor Y;
	Tensor dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Tensor forward(Tensor X);
	void backprop(Tensor predictions, Tensor target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};
