#pragma once

#include "../utils/blas.cuh"
#include "layer.cuh"

#include <memory>
#include <unordered_map>

class Linear : public Layer {
 public:
	explicit Linear(int in_size, int out_size, bool is_bias);

	std::vector<std::pair<Tensor *, Tensor *>> parameters();
	void forward();
	void backward();

 private:
	std::unique_ptr<Tensor> weights;
	std::unique_ptr<Tensor> weights_grad;
	std::unique_ptr<Tensor> bias;
	std::unique_ptr<Tensor> bias_grad;

	std::unordered_map<std::string, std::unique_ptr<Tensor>> temp;

	int in_size;
	int out_size;
	bool is_bias;
};
