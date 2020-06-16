#pragma once

#include "nn_layer.cuh"
#include "../utils/nn_exception.cuh"
#include <iostream>

class softmaxActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:
	softmaxActivation(std::string name);
	~softmaxActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
