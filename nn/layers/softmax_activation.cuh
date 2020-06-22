#pragma once

#include "layer.cuh"
#include "../utils/exception.cuh"
#include <iostream>

class softmaxActivation : public NNLayer {
private:
	Tensor A;

	Tensor Z;
	Tensor dZ;

public:
	softmaxActivation(std::string name);
	~softmaxActivation();

	Tensor& forward(Tensor& Z);
	Tensor& backprop(Tensor& dA, float learning_rate = 0.01);
};
