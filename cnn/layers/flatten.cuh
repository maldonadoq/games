#pragma once

#include "layer.cuh"
#include "../utils/tensor.cuh"
#include "../utils/utils.cuh"

class Flatten : public Layer {
 public:
	Flatten(bool inplace) : inplace(inplace) {}

	void forward();
	void backward();

	Tensor *get_grad() {
		return this->inplace ? this->next->get_grad() : this->grad.get();
	}
	Tensor *get_output() {
		return this->inplace ? this->pre->get_output() : this->output.get();
	}

 private:
	std::vector<int> in_shape;
	bool inplace;
};