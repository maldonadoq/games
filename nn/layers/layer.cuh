#pragma once

#include <iostream>

#include "../utils/tensor.cuh"

class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual Tensor& forward(Tensor& A) = 0;
	virtual Tensor& backprop(Tensor& dZ, float learning_rate) = 0;

	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}
