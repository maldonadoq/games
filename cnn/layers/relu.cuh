#pragma once

#include "layer.cuh"
#include "../utils/blas.cuh"

class ReLU : public Layer {
 public:
	ReLU(bool inplace) : inplace(inplace) {}

	void forward();
	void backward();

	Tensor *get_grad() {
		return this->inplace ? this->next->get_grad() : this->grad.get();
	}
	Tensor *get_output() {
		return this->inplace ? this->pre->get_output() : this->output.get();
	}

 private:
	bool inplace;
};
