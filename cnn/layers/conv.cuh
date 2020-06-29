#pragma once

#include "../utils/blas.cuh"
#include "layer.cuh"

#include <thrust/device_vector.h>
#include <unordered_map>
#include <thrust/copy.h>
#include <cstdlib>
#include <memory>
#include <vector>

class Conv : public Layer {
 public:
	explicit Conv(int height, int width, int channel_in, int channel_out,
								int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
								int stride_w, bool is_bias);

	void forward();
	void backward();
	std::vector<std::pair<Tensor *, Tensor *>> parameters();

 private:
	std::unique_ptr<Tensor> filters;
	std::unique_ptr<Tensor> filters_grad;
	std::unique_ptr<Tensor> bias;
	std::unique_ptr<Tensor> bias_grad;
	std::unique_ptr<Tensor> cols;

	std::unordered_map<std::string, std::unique_ptr<Tensor>> temp;

	int height;
	int width;
	int channel_in;
	int channel_out;
	int kernel_h;
	int kernel_w;
	int pad_w;
	int pad_h;
	int stride_w;
	int stride_h;
	bool is_bias;
};
