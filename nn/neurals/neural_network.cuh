#pragma once

#include <vector>
#include "../layers/layer.cuh"
#include "../utils/cce_loss.cuh"
#include "../utils/exception.cuh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	CCELoss cce_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};
