#pragma once

#include "layer.cuh"
#include "../utils/exception.cuh"

class ReLUActivation : public NNLayer {
private:
	Tensor A;

	Tensor Z;
	Tensor dZ;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	Tensor& forward(Tensor& Z);
	Tensor& backprop(Tensor& dA, float learning_rate = 0.01);
};
