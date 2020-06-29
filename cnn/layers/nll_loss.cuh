#pragma once

#include "../utils/blas.cuh"
#include "layer.cuh"
#include <unordered_map>
#include <memory>

class NLLLoss : public Layer {
 public:
	NLLLoss() { this->output.reset(new Tensor({1, 1})); }
	void forward(const Tensor *y);
	void backward();

 private:
	const Tensor *y;  // backup

	std::unordered_map<std::string, std::unique_ptr<Tensor>> temp;
};