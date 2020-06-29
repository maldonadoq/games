#pragma once

#include "../utils/blas.cuh"
#include "layer.cuh"

class Softmax : public Layer{
 public:
	explicit Softmax(int dim = 1) : dim(dim) {}
	void forward();
	void backward();

 private:
	int dim;
};